#!/usr/bin/env python3
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

import json
import random
import glob
import os
from pathlib import Path
from collections import defaultdict
import tiktoken
from typing import Dict, List, Tuple

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string using tiktoken.
    
    Args:
        text: The input text to count tokens for
        model: The model to use for tokenization (default: gpt-3.5-turbo)
    
    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def print_token_statistics(stats: Dict[str, Dict], total_samples: int, skipped_samples: int) -> None:
    """
    Print detailed token statistics for the dataset.
    
    Args:
        stats: Dictionary containing token statistics per source
        total_samples: Total number of samples processed
        skipped_samples: Number of samples skipped due to token limit
    """
    print("\nToken Statistics:")
    print(f"Total samples processed: {total_samples}")
    print(f"Samples skipped due to token limit: {skipped_samples} ({skipped_samples/total_samples:.1%})")
    print(f"Samples kept: {total_samples - skipped_samples} ({(total_samples - skipped_samples)/total_samples:.1%})")
    
    print("\nToken counts by source:")
    for source, source_stats in sorted(stats.items(), key=lambda x: x[1]["total_tokens"], reverse=True):
        print(f"\n{source}:")
        print(f"  Samples: {source_stats['samples']} ({source_stats['samples']/total_samples:.1%})")
        print(f"  Total tokens: {source_stats['total_tokens']:,}")
        print(f"  Average tokens per sample: {source_stats['total_tokens']/source_stats['samples']:.1f}")
        print(f"  Min tokens: {source_stats['min_tokens']}")
        print(f"  Max tokens: {source_stats['max_tokens']}")
        print(f"  Samples skipped: {source_stats['skipped']} ({source_stats['skipped']/source_stats['total']:.1%})")

def combine_datasets(
    safety_file: str = "safety_sampled.jsonl",
    llama_nemo_dir: str = ".",
    output_dir: str = ".",
    val_split: float = 0.03,
    max_tokens: int = 16384,  # Default to 16k token limit
    max_samples: int = None  # New parameter to limit total samples
) -> None:
    """
    Combine safety dataset and Llama Nemotron dataset files, and create train/val splits.
    
    Args:
        safety_file: Path to the safety dataset file
        llama_nemo_dir: Directory containing Llama Nemotron files
        output_dir: Directory to save output files
        val_split: Proportion of data to use for validation (default: 3%)
        max_tokens: Maximum number of tokens allowed per sample (default: 16384)
        max_samples: Maximum number of total samples to process (default: None, process all)
    """
    combined_data = []
    item_id = 0  # Counter for consistent numeric IDs
    total_samples = 0
    skipped_samples = 0
    
    # Initialize statistics tracking
    stats = defaultdict(lambda: {
        "samples": 0,
        "total_tokens": 0,
        "min_tokens": float('inf'),
        "max_tokens": 0,
        "skipped": 0,
        "total": 0
    })
    
    # Load safety dataset
    print(f"\nLoading safety dataset from {safety_file}...")
    safety_data = []
    with open(safety_file, 'r') as f:
        for line in f:
            if max_samples and total_samples >= max_samples:
                break
            total_samples += 1
            stats["safety_dataset"]["total"] += 1
            try:
                item = json.loads(line.strip())
                # Standardize format with all required columns
                input_text = str(item.get("prompt", item.get("input", ""))).strip()
                # For safety dataset, we'll keep output empty as it will be generated later
                output_text = ""  # Empty output for safety dataset
                prompt_label = str(item.get("prompt_label", item.get("label", "unknown"))).lower()
                
                # Extract all required fields
                source_l1 = str(item.get("source_l1", "unknown"))
                source_l2 = str(item.get("source_l2", ""))
                categories = []
                subcategories = []
                adversarial_persona = str(item.get("adversarial_persona", ""))
                
                # Handle categories and subcategories
                raw_categories = item.get("categories", item.get("safety_categories", item.get("unsafe_categories", [])))
                raw_subcategories = item.get("subcategories", [])
                
                if isinstance(raw_categories, str):
                    categories = [c.strip().lower() for c in raw_categories.split(",")]
                elif isinstance(raw_categories, list):
                    categories = [str(c).strip().lower() for c in raw_categories]
                
                if isinstance(raw_subcategories, str):
                    subcategories = [s.strip().lower() for s in raw_subcategories.split(",")]
                elif isinstance(raw_subcategories, list):
                    subcategories = [str(s).strip().lower() for s in raw_subcategories]
                
                # For unsafe prompts, ensure we have at least one category
                if prompt_label != "safe" and not categories:
                    categories = [prompt_label]
                # For safe prompts, if no categories found, use "safe" as category
                elif prompt_label == "safe" and not categories:
                    categories = ["safe"]
                
                # Count tokens (only for input since output is empty)
                input_tokens = count_tokens(input_text)
                output_tokens = 0  # No output tokens for safety dataset
                total_tokens = input_tokens  # Only count input tokens
                
                # Update statistics
                stats["safety_dataset"]["total_tokens"] += total_tokens
                stats["safety_dataset"]["min_tokens"] = min(stats["safety_dataset"]["min_tokens"], total_tokens)
                stats["safety_dataset"]["max_tokens"] = max(stats["safety_dataset"]["max_tokens"], total_tokens)
                
                # Check token limit (only for input since output is empty)
                if total_tokens > max_tokens:
                    stats["safety_dataset"]["skipped"] += 1
                    skipped_samples += 1
                    continue
                
                transformed_item = {
                    "id": item_id,  # Use numeric ID
                    "input": input_text,
                    "output": output_text,  # Empty output
                    "source": "safety_dataset",
                    "metadata": {
                        "source_l1": source_l1,
                        "source_l2": source_l2,
                        "prompt_label": prompt_label,
                        "categories": categories,
                        "subcategories": subcategories,
                        "adversarial_persona": adversarial_persona,
                        "original_id": str(item.get("id", "")),
                        "split": "train",
                        "token_count": total_tokens,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens
                    }
                }
                # Only add if we have input (output can be empty)
                if transformed_item["input"]:
                    safety_data.append(transformed_item)
                    item_id += 1
                    stats["safety_dataset"]["samples"] += 1
                    
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in safety dataset")
            except Exception as e:
                print(f"Warning: Error processing safety dataset line: {e}")
    
    print(f"Loaded {len(safety_data)} samples from safety dataset")
    combined_data.extend(safety_data)
    
    # Load Llama Nemotron files
    llama_nemo_files = glob.glob(os.path.join(llama_nemo_dir, "*.jsonl"))
    print(f"\nFound {len(llama_nemo_files)} Llama-Nemotron files: {llama_nemo_files}")
    
    for file_path in llama_nemo_files:
        if max_samples and total_samples >= max_samples:
            break
        source_name = os.path.basename(file_path).split('.')[0]
        
        # Skip the safety dataset file as we already loaded it
        if source_name == "safety_sampled":
            print(f"Skipping {file_path} as it's already loaded")
            continue
        
        print(f"\nLoading {file_path} as '{source_name}'...")
        valid_items = 0
        total_items = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                total_items += 1
                total_samples += 1
                stats[f"llama_nemo_{source_name}"]["total"] += 1
                try:
                    item = json.loads(line.strip())
                    
                    # Handle input as list of messages (usually with role/content structure)
                    input_text = ""
                    if isinstance(item.get("input"), list):
                        # If input is a list, extract content from each message
                        for msg in item["input"]:
                            if isinstance(msg, dict) and "content" in msg:
                                input_text += msg["content"] + "\n"
                            elif isinstance(msg, str):
                                input_text += msg + "\n"
                    else:
                        input_text = str(item.get("input", ""))
                    
                    output_text = str(item.get("output", "")).strip()
                    
                    # Count tokens
                    input_tokens = count_tokens(input_text)
                    output_tokens = count_tokens(output_text)
                    total_tokens = input_tokens + output_tokens
                    
                    # Update statistics
                    stats[f"llama_nemo_{source_name}"]["total_tokens"] += total_tokens
                    stats[f"llama_nemo_{source_name}"]["min_tokens"] = min(stats[f"llama_nemo_{source_name}"]["min_tokens"], total_tokens)
                    stats[f"llama_nemo_{source_name}"]["max_tokens"] = max(stats[f"llama_nemo_{source_name}"]["max_tokens"], total_tokens)
                    
                    # Check token limit
                    if total_tokens > max_tokens:
                        stats[f"llama_nemo_{source_name}"]["skipped"] += 1
                        skipped_samples += 1
                        continue
                    
                    # Create transformed item with consistent format
                    transformed_item = {
                        "id": item_id,  # Use numeric ID
                        "input": input_text.strip(),
                        "output": output_text,
                        "source": f"llama_nemo_{source_name}",
                        "metadata": {
                            "category": str(item.get("category", "")),
                            "original_id": str(item.get("id", "")),
                            "split": "train",
                            "token_count": total_tokens,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens
                        }
                    }
                    
                    # Only add if we have both input and output
                    if transformed_item["input"] and transformed_item["output"]:
                        combined_data.append(transformed_item)
                        item_id += 1
                        valid_items += 1
                        stats[f"llama_nemo_{source_name}"]["samples"] += 1
                    else:
                        print(f"Warning: Missing input or output fields in item from {file_path}")
                        
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {file_path}")
                except Exception as e:
                    print(f"Warning: Error processing line in {file_path}: {e}")
        
        print(f"Added {valid_items}/{total_items} items from {source_name}")
    
    # Print token statistics
    print_token_statistics(stats, total_samples, skipped_samples)
    
    # Shuffle the combined dataset
    random.shuffle(combined_data)
    
    # Create train/val split
    val_size = int(len(combined_data) * val_split)
    train_data = combined_data[val_size:]
    val_data = combined_data[:val_size]
    
    # Update split in metadata for each dataset
    for item in train_data:
        item["metadata"]["split"] = "train"
    for item in val_data:
        item["metadata"]["split"] = "val"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train and val datasets
    train_file = os.path.join(output_dir, "train.jsonl")
    val_file = os.path.join(output_dir, "val.jsonl")
    
    print(f"\nSaving train dataset ({len(train_data):,} samples) to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saving validation dataset ({len(val_data):,} samples) to {val_file}...")
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Print final statistics
    print("\nFinal dataset statistics:")
    print(f"Total samples processed: {total_samples}")
    print(f"Samples kept: {len(combined_data)} ({len(combined_data)/total_samples:.1%})")
    print(f"Train samples: {len(train_data)} ({len(train_data)/len(combined_data):.1%})")
    print(f"Validation samples: {len(val_data)} ({len(val_data)/len(combined_data):.1%})")
    
    print("\nTrain set source distribution:")
    train_source_counts = defaultdict(int)
    for item in train_data:
        train_source_counts[item["source"]] += 1
    
    for source, count in sorted(train_source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{source}: {count} ({count/len(train_data):.1%})")
    
    print("\nValidation set source distribution:")
    val_source_counts = defaultdict(int)
    for item in val_data:
        val_source_counts[item["source"]] += 1
    
    for source, count in sorted(val_source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{source}: {count} ({count/len(val_data):.1%})")

    # Add prompt label and category statistics at the end
    print("\nPrompt Label Statistics:")
    label_counts = defaultdict(int)
    category_counts = defaultdict(int)
    label_category_counts = defaultdict(lambda: defaultdict(int))  # Track categories per label
    
    for item in safety_data:
        label = item["metadata"]["prompt_label"]
        categories = item["metadata"]["categories"]
        label_counts[label] += 1
        
        # Count categories
        for category in categories:
            category_counts[category] += 1
            label_category_counts[label][category] += 1
    
    print("\nLabel Distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {count} samples ({count/len(safety_data):.1%})")
    
    print("\nCategory Distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {count} samples")
    
    print("\nCategory Distribution by Label:")
    for label in sorted(label_counts.keys()):
        print(f"\n{label.upper()} prompts:")
        for category, count in sorted(label_category_counts[label].items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count} samples ({count/label_counts[label]:.1%})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine safety and Llama Nemotron datasets")
    parser.add_argument("--safety_file", type=str, required=True,
                      help="Path to the safety dataset file")
    parser.add_argument("--llama_nemo_dir", type=str, required=True,
                      help="Directory containing Llama Nemotron files")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save output files")
    parser.add_argument("--val_split", type=float, default=0.03,
                      help="Proportion of data to use for validation (default: 0.03)")
    parser.add_argument("--max_tokens", type=int, default=16384,
                      help="Maximum number of tokens allowed per sample (default: 16384)")
    parser.add_argument("--max_samples", type=int, default=None,
                      help="Maximum number of total samples to process (default: None, process all)")
    
    args = parser.parse_args()
    
    combine_datasets(
        safety_file=args.safety_file,
        llama_nemo_dir=args.llama_nemo_dir,
        output_dir=args.output_dir,
        val_split=args.val_split,
        max_tokens=args.max_tokens,
        max_samples=args.max_samples
    ) 