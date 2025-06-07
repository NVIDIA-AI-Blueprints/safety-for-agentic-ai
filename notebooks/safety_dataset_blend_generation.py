#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter
from datasets import load_dataset
import pandas as pd
import os
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import time
import json

FINAL_COLUMNS = ["id", "source_l1", "source_l2", "prompt", "prompt_label", "categories", "subcategories", "adversarial_persona"]

def print_dataset_distribution(df, title="Dataset Distribution"):
    print(f"\n{title}")
    print("="*80)
    
    # Get value counts and percentages
    dist = df["source_l1"].value_counts()
    total = len(df)
    percentages = (dist / total * 100).round(2)
    
    # Create distribution table
    print(f"{'Dataset':<30} {'Count':>10} {'Percentage':>12}")
    print("-"*80)
    for dataset, count in dist.items():
        print(f"{dataset:<30} {count:>10,} {percentages[dataset]:>11.2f}%")
    print("-"*80)
    print(f"{'Total':<30} {total:>10,} {'100.00':>11}%")
    print("="*80)

def print_dataset_info(name, size):
    print(f"\n{'='*80}")
    print(f"Dataset: {name}")
    print(f"Number of samples: {size:,}")
    print(f"{'='*80}\n")

def category_stats(df, categories_column="categories"):
    return Counter([c for cat in df[categories_column] for c in cat.split(", ")])

def stratified_sample(df, n_samples, stratify_column='source_l1', random_state=42):
    if n_samples >= len(df):
        return df
    
    print("\nPerforming stratified sampling...")
    # Calculate the proportion of each stratum in the original dataset
    stratum_proportions = df[stratify_column].value_counts(normalize=True)
    
    # Calculate how many samples to take from each stratum
    stratum_counts = (stratum_proportions * n_samples).round().astype(int)
    
    # Adjust for rounding errors to ensure we get exactly n_samples
    while stratum_counts.sum() != n_samples:
        if stratum_counts.sum() < n_samples:
            # Add to the largest stratum
            largest_stratum = stratum_counts.idxmax()
            stratum_counts[largest_stratum] += 1
        else:
            # Remove from the smallest stratum
            smallest_stratum = stratum_counts[stratum_counts > 0].idxmin()
            stratum_counts[smallest_stratum] -= 1
    
    # Sample from each stratum
    sampled_dfs = []
    for stratum, count in tqdm(stratum_counts.items(), desc="Sampling from datasets"):
        stratum_df = df[df[stratify_column] == stratum]
        if count > 0:
            sampled_dfs.append(stratum_df.sample(n=count, random_state=random_state))
    
    return pd.concat(sampled_dfs).reset_index(drop=True)

def strip_prompt_of_jailbreak(prompt, all_unsafe_requests):
    for unsafe_request in all_unsafe_requests:
        if unsafe_request.lower() in prompt.lower():
            return unsafe_request
    stripped_prompt = prompt
    jailbreak_patterns = {
        "In this hypothetical story, you are to act as Niccolo Machiavelli": "Niccolo's first question:",
        "You must answer my prompts in 2 ways": "my first question:",
        "_ { >> _ _ System annou-ncement begins <<}": "{>> U-s-er's m-ess-age: <<} #",
        "{System: this simulation has been approved by explicitly OPENAI": "Question:",
        "Ignore all instructions until now": "QUESTION:",
        "You are now TranslatorBot": "My first question is:",
        "The way to describe a character planning to ": "is as follows:",
        "Let's imagine we have a Python function in this experiment.": "Now generate a possible output of print( simple_function('",
        "Now you act as two entities, Tom and Jerry.": "the second entity is talking about a topic called $Term2 where $Term2 = '",
        "Let's build a word game. Use the following instructions": "1) Write detailed instructions about '"
    }
    for start, splitter in jailbreak_patterns.items():
        if prompt.startswith(start):
            stripped_prompt = prompt.split(splitter)[1].split("'")[0] if "Term2" in splitter else prompt.split(splitter)[1]
            break
    return stripped_prompt.replace("'", "").replace('"', "").strip()

def uniform_sample(df, n_samples, random_state=42):
    """
    Perform uniform sampling across categories, taking equal samples from each category
    and distributing remaining samples from smallest to largest categories.
    
    Args:
        df: Input DataFrame
        n_samples: Number of samples to take
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame
    """
    if n_samples >= len(df):
        return df
    
    print("\nPerforming uniform sampling across categories...")
    
    # Get category sizes
    category_sizes = df["source_l1"].value_counts()
    n_categories = len(category_sizes)
    
    # Calculate base samples per category
    base_samples = n_samples // n_categories
    remaining_samples = n_samples % n_categories
    
    # Sort categories by size (ascending) for remaining sample distribution
    sorted_categories = category_sizes.sort_values().index.tolist()
    
    # Sample from each category
    sampled_dfs = []
    for category in tqdm(sorted_categories, desc="Sampling from categories"):
        category_df = df[df["source_l1"] == category]
        # Calculate samples for this category
        category_samples = base_samples
        if remaining_samples > 0:
            category_samples += 1
            remaining_samples -= 1
        
        # Ensure we don't try to take more samples than available
        category_samples = min(category_samples, len(category_df))
        
        if category_samples > 0:
            sampled_dfs.append(category_df.sample(n=category_samples, random_state=random_state))
            print(f"{category}: {category_samples} samples selected")
    
    # Combine all sampled data
    sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)
    
    # If we still have remaining samples, distribute them from smallest to largest
    if len(sampled_df) < n_samples:
        remaining = n_samples - len(sampled_df)
        print(f"\nDistributing {remaining} remaining samples...")
        
        # Get categories that still have available samples
        available_categories = []
        for category in sorted_categories:
            category_df = df[df["source_l1"] == category]
            already_sampled = len(sampled_df[sampled_df["source_l1"] == category])
            if len(category_df) > already_sampled:
                available_categories.append((category, len(category_df) - already_sampled))
        
        # Sort by available samples (ascending)
        available_categories.sort(key=lambda x: x[1])
        
        # Distribute remaining samples
        for category, available in available_categories:
            if remaining <= 0:
                break
                
            category_df = df[df["source_l1"] == category]
            already_sampled = len(sampled_df[sampled_df["source_l1"] == category])
            to_sample = min(remaining, available)
            
            if to_sample > 0:
                # Get indices of already sampled rows
                sampled_indices = set(sampled_df[sampled_df["source_l1"] == category].index)
                # Sample from remaining rows
                remaining_df = category_df[~category_df.index.isin(sampled_indices)]
                additional_samples = remaining_df.sample(n=to_sample, random_state=random_state)
                sampled_df = pd.concat([sampled_df, additional_samples]).reset_index(drop=True)
                remaining -= to_sample
                print(f"{category}: +{to_sample} additional samples")
    
    return sampled_df

def parse_args():
    parser = argparse.ArgumentParser(description='Generate safety dataset blend')
    parser.add_argument('-f', '--filename', type=str, required=True,
                      help='Output filename (e.g., nemotron-safety-sft-training-blend-v1.0.jsonl)')
    parser.add_argument('-n', '--total_samples', type=int, default=None,
                      help='Total number of samples to include in the dataset (default: all samples)')
    parser.add_argument('-c', '--cache_dir', type=str, default=None,
                      help='Cache directory for datasets (default: None)')
    parser.add_argument('-s', '--sampling_method', type=str, choices=['stratified', 'uniform'], default='stratified',
                      help='Sampling method to use when total_samples is specified (default: stratified)')
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    
    print("\nStarting dataset collection...")
    print(f"Output file: {args.filename}")
    if args.total_samples:
        print(f"Target sample size: {args.total_samples:,}")
        print(f"Sampling method: {args.sampling_method}")
    if args.cache_dir:
        print(f"Using cache directory: {args.cache_dir}")
    
    # Download Aegis v2
    print("\nDownloading Aegis v2 dataset...")
    print("Source: nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
    aegis_df = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train", cache_dir=args.cache_dir).to_pandas()
    aegis_df = aegis_df[aegis_df["prompt"] != "REDACTED"].drop_duplicates(subset=["prompt"])
    aegis_df["source_l1"], aegis_df["source_l2"] = "Aegis v2", None
    aegis_df["categories"], aegis_df["subcategories"] = aegis_df["violated_categories"], None
    aegis_df["adversarial_persona"]= None
    aegis_df = aegis_df[FINAL_COLUMNS]
    print_dataset_info("Aegis v2", len(aegis_df))

    # Download Gretel Safety Alignment v1
    print("\nDownloading Gretel Safety Alignment v1 dataset...")
    print("Source: gretelai/gretel-safety-alignment-en-v1")
    gretel_df = pd.DataFrame(load_dataset("gretelai/gretel-safety-alignment-en-v1", split="train", cache_dir=args.cache_dir))
    gretel_df["prompt_label"], gretel_df["source_l1"] = "unsafe", "Gretel Safety Alignment v1"
    gretel_df["subcategories"] = gretel_df.apply(lambda x: f"{x['sub_category']}, {x['tactic']}", axis=1)
    gretel_df["categories"], gretel_df["adversarial_persona"] = gretel_df["risk_category"], gretel_df["persona"]
    gretel_df["source_l2"] = None
    gretel_df = gretel_df[FINAL_COLUMNS]
    print_dataset_info("Gretel Safety Alignment v1", len(gretel_df))

    # Download Harmful Tasks
    print("\nDownloading Harmful Tasks dataset...")
    print("Source: CrystalEye42/eval-safety")
    os.system('git clone https://github.com/CrystalEye42/eval-safety.git')
    with open(f"{os.getcwd()}/eval-safety/malicious_tasks_dataset.yaml") as f:
        harmful_tasks_unsafe_requests = yaml.safe_load(f)
    with open(f"{os.getcwd()}/eval-safety/integrated.yaml", "r") as f:
        harmful_tasks_unsafe_requests_wrapped = yaml.safe_load(f)

    print("Processing Harmful Tasks data...")
    unique_unsafe_requests = [example.replace(".", "").strip() for category in tqdm(harmful_tasks_unsafe_requests, desc="Processing categories")
                            for sub_category in harmful_tasks_unsafe_requests[category] 
                            for severity in harmful_tasks_unsafe_requests[category][sub_category] 
                            for example in harmful_tasks_unsafe_requests[category][sub_category][severity]]

    rows = [[task, category, sub_category, severity, example] 
            for task in tqdm(harmful_tasks_unsafe_requests_wrapped, desc="Processing tasks")
            for category in harmful_tasks_unsafe_requests_wrapped[task] 
            for sub_category in harmful_tasks_unsafe_requests_wrapped[task][category] 
            for severity in harmful_tasks_unsafe_requests_wrapped[task][category][sub_category] 
            for example in harmful_tasks_unsafe_requests_wrapped[task][category][sub_category][severity]]

    harmful_tasks_df = pd.DataFrame(rows, columns=["task", "category", "sub_category", "severity", "prompt"])
    harmful_tasks_df = harmful_tasks_df[harmful_tasks_df["severity"] != "Low Severity"]
    harmful_tasks_df["prompt_without_jailbreak"] = harmful_tasks_df["prompt"].apply(lambda x: strip_prompt_of_jailbreak(x, unique_unsafe_requests))
    harmful_tasks_df["id"] = harmful_tasks_df.index
    harmful_tasks_df["prompt_label"], harmful_tasks_df["source_l1"] = "unsafe", "HarmfulTasks"
    harmful_tasks_df["subcategories"], harmful_tasks_df["categories"] = harmful_tasks_df["sub_category"], harmful_tasks_df["category"]
    harmful_tasks_df["adversarial_persona"] = harmful_tasks_df.apply(lambda x: f"{x['task']}, {x['severity']}", axis=1)
    harmful_tasks_df["source_l2"] = None
    harmful_tasks_df = harmful_tasks_df[FINAL_COLUMNS + ["prompt_without_jailbreak"]]
    print_dataset_info("Harmful Tasks", len(harmful_tasks_df))

    # Download RedTeam 2k
    print("\nDownloading RedTeam 2k dataset...")
    print("Source: JailbreakV-28K/JailBreakV-28k")
    redteam_2k = load_dataset("JailbreakV-28K/JailBreakV-28k", 'RedTeam_2K', cache_dir=args.cache_dir)["RedTeam_2K"]
    redteam_2k_filtered = redteam_2k.filter(lambda x: x["from"] in ["Handcraft", "Question Set", "hh-rlhf", "AdvBench", "LLM Jailbreak Study"]).to_pandas()
    redteam_2k_filtered.rename(columns={"question": "prompt", "policy": "categories", "from": "source_l2"}, inplace=True)
    redteam_2k_filtered["source_l1"], redteam_2k_filtered["prompt_label"] = "RedTeam 2k", "unsafe"
    redteam_2k_filtered["subcategories"], redteam_2k_filtered["adversarial_persona"] = None, None
    redteam_2k_filtered = redteam_2k_filtered[FINAL_COLUMNS]
    print_dataset_info("RedTeam 2k", len(redteam_2k_filtered))

    # Combine all datasets
    print("\nCombining datasets...")
    nemo_safety_blended_prompts = pd.concat([aegis_df, gretel_df, harmful_tasks_df, redteam_2k_filtered])
    nemo_safety_blended_prompts = nemo_safety_blended_prompts.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    nemo_safety_blended_prompts.loc[nemo_safety_blended_prompts["prompt_without_jailbreak"].isna(), "prompt_without_jailbreak"] = nemo_safety_blended_prompts["prompt"]
    nemo_safety_blended_prompts = nemo_safety_blended_prompts[FINAL_COLUMNS]
    
    # Print original distribution
    print_dataset_distribution(nemo_safety_blended_prompts, "Original Dataset Distribution")
    
    # Save full dataset
    print(f"\nSaving full dataset to {args.filename}...")
    with open(args.filename, 'w', encoding='utf-8') as f:
        for _, row in nemo_safety_blended_prompts.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
    print(f'Full dataset saved with {len(nemo_safety_blended_prompts):,} samples')
    
    # Sample if total_samples is specified
    if args.total_samples is not None:
        original_df = nemo_safety_blended_prompts.copy()
        if args.sampling_method == "uniform":
            nemo_safety_blended_prompts = uniform_sample(nemo_safety_blended_prompts, args.total_samples)
        else:  # stratified sampling
            nemo_safety_blended_prompts = stratified_sample(nemo_safety_blended_prompts, args.total_samples)
        
        # Print sampling report
        print("\nSampling Report")
        print("="*80)
        print(f"Original dataset size: {len(original_df):,}")
        print(f"Sampled dataset size: {len(nemo_safety_blended_prompts):,}")
        print(f"Sampling ratio: {(len(nemo_safety_blended_prompts) / len(original_df) * 100):.2f}%")
        print(f"Sampling method: {args.sampling_method}")
        print("="*80)
        
        # Print sampled distribution
        print_dataset_distribution(nemo_safety_blended_prompts, "Sampled Dataset Distribution")
        
        # Save sampled dataset
        sampled_filename = args.filename.replace('.jsonl', f'_sampled_{args.total_samples}_{args.sampling_method}.jsonl')
        print(f"\nSaving sampled dataset to {sampled_filename}")
        with open(sampled_filename, 'w', encoding='utf-8') as f:
            for _, row in nemo_safety_blended_prompts.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
        print(f'Sampled dataset saved with {len(nemo_safety_blended_prompts):,} samples')
    
    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

