#!/usr/bin/env python3
import json
import random
import os
from collections import defaultdict
from huggingface_hub import hf_hub_download
from pathlib import Path

def download_and_sample_safety_dataset(
    dataset_name: str,
    filename: str,
    total_samples: int = 5000,
    cache_dir: str = None
) -> None:
    """
    Download safety dataset and perform stratified sampling across categories.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        filename: Name of the file to download
        total_samples: Total number of samples to select
        cache_dir: Directory to cache downloaded files
    """
    # Download the dataset
    print(f"Downloading {filename} from {dataset_name}...")
    downloaded_file = hf_hub_download(
        repo_id=dataset_name,
        filename=filename,
        repo_type='dataset',
        cache_dir=cache_dir
    )
    
    if not os.path.exists(downloaded_file):
        raise FileNotFoundError(f"Could not find downloaded file: {downloaded_file}")
    
    print(f"Processing {downloaded_file} with stratified sampling...")
    
    # Read and categorize the dataset
    data_by_category = defaultdict(list)
    with open(downloaded_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                category = item.get("source", "unknown")
                data_by_category[category].append(item)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line")
            except Exception as e:
                print(f"Warning: Error processing line: {e}")

    # Calculate target numbers for each category
    total_available = sum(len(items) for items in data_by_category.values())
    sampled_data = []

    print("\nCategory distribution before sampling:")
    for category, items in data_by_category.items():
        print(f"{category}: {len(items)} samples ({len(items)/total_available:.1%})")

    print("\nStratified sampling from categories:")
    for category, items in data_by_category.items():
        # Determine how many samples to include based on category proportion
        category_proportion = len(items) / total_available
        category_target = int(total_samples * category_proportion)
        
        # Ensure we have at least some samples from each category
        category_target = max(category_target, 1)
        
        # Make sure we don't try to take more samples than available
        category_target = min(category_target, len(items))
        
        # Shuffle and sample
        random.shuffle(items)
        sampled_data.extend(items[:category_target])
        print(f"{category}: {category_target} samples selected")

    # Shuffle final dataset
    random.shuffle(sampled_data)

    # Save sampled data
    output_file = "nv_safety_sampled.jsonl"
    with open(os.path.join(cache_dir, output_file), 'w') as f:
        for item in sampled_data:
            f.write(json.dumps(item) + '\n')

    print(f"\nTotal sampled data: {len(sampled_data)}")
    print(f"Data saved to {output_file}")

    # Print final category distribution
    print("\nFinal category distribution:")
    category_counts = defaultdict(int)
    for item in sampled_data:
        category_counts[item.get("source", "unknown")] += 1

    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {count} ({count/len(sampled_data):.1%})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and sample safety dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset on HuggingFace")
    parser.add_argument("--filename", type=str, required=True, help="Name of the file to download")
    parser.add_argument("--total_samples", type=int, default=5000, help="Total number of samples to select")
    parser.add_argument("--cache_dir", type=str, help="Directory to cache downloaded files")
    
    args = parser.parse_args()
    
    download_and_sample_safety_dataset(
        dataset_name=args.dataset_name,
        filename=args.filename,
        total_samples=args.total_samples,
        cache_dir=args.cache_dir
    ) 