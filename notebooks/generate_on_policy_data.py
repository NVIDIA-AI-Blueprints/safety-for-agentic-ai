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

from vllm import SamplingParams
import json
import logging
import os
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import argparse
from datetime import datetime
import time
from collections import defaultdict
from clients import VLLMClient, SafetyClient, AsyncVLLMClient, AsyncSafetyClient, process_batch, build_refusal_prompt
import asyncio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'workspace/training/logs/generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Default configuration
# TODO: Move to notebook and remove HF_TOKEN
DEFAULT_CONFIG = {
    'generation_model': "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    'safety_model': "nvidia/llama-3.1-nemoguard-8b-content-safety",
    'huggingface_token': os.getenv("HF_TOKEN", ""),
    'output_dir': "generated_data",
    'batch_size': 4,
    'max_tokens': 512,
    'temperature': 0.7,
    'top_p': 0.9,
    'safety_thresholds': {
        'overall': 0.8,
        'harmful': 0.9,
        'unethical': 0.85,
        'biased': 0.85
    },
    'max_attempts': 3,
    'vllm_server': {
        'host': 'localhost',
        'port': 5000,
        'timeout': 36000,  # 10 hours
        'max_retries': 3,
        'retry_delay': 5
    }
}

def generate_and_evaluate(
    vllm_client: VLLMClient,
    safety_client: SafetyClient,
    prompt_data: Dict[str, Any],
    sampling_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a response and evaluate its safety."""
    # Get prompt label and categories from metadata
    metadata = prompt_data.get('metadata', {})
    prompt_label = metadata.get('prompt_label', 'unknown')
    categories = metadata.get('categories', [])
    
    # Build appropriate prompt based on safety label
    if prompt_label == "unsafe":
        categories = metadata.get('categories', []) if metadata.get('categories') else [prompt_label]
        input_prompt = build_refusal_prompt(prompt_data['input'], categories)
    else:
        input_prompt = prompt_data['input']
    
    # Generate response with thinking enabled
    response = vllm_client.generate(
        prompt=input_prompt,
        temperature=sampling_params['temperature'],
        top_p=sampling_params['top_p'],
        max_tokens=sampling_params['max_tokens']
    )

    # Initialize safety metrics with default values
    safety_metrics = {
        'user_safety': 'unknown',
        'response_safety': 'unknown',
        'safety_categories': []
    }

    # Only evaluate safety if prompt_label is not 'unknown'
    if prompt_label != "unknown":
        # Get safety evaluation with thinking enabled
        safety_result = safety_client.evaluate_safety(
            user_message=prompt_data['input'],
            assistant_response=response,
            temperature=0.0,  # Use deterministic evaluation
            top_p=0.9,
            max_tokens=512
        )
        # Parse safety result
        try:
            safety_json = json.loads(safety_result)
            safety_metrics['user_safety'] = safety_json.get("User Safety", "unknown")
            safety_metrics['response_safety'] = safety_json.get("Response Safety", "unknown")
            safety_metrics['safety_categories'] = safety_json.get("Safety Categories", "").split(", ") if safety_json.get("Safety Categories") else []
        except json.JSONDecodeError:
            logging.warning(f"Could not parse safety result JSON: {safety_result}")

    return {
        'id': prompt_data['id'],
        'input': prompt_data['input'],
        'original_output': prompt_data.get('output', ''),
        'generated_output': response,
        'safety_metrics': safety_metrics,
        'metadata': {
            **metadata, 
            'prompt_label': prompt_label,
            'categories': categories
        }
    }

def save_results(results: List[Dict[str, Any]], output_file: str, store_all: bool = False, unsafe_output_file: str = None):
    """Save generated results to JSONL files.
    
    Args:
        results: List of result dictionaries
        output_file: Path to save safe responses (or all responses if store_all=True)
        store_all: If True, store all responses in output_file. If False, store only safe responses
        unsafe_output_file: Optional path to save unsafe responses separately
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Filter results based on store_all flag
        if store_all:
            results_to_save = results
            logging.info(f"Saving all {len(results)} responses to {output_file}")
        else:
            results_to_save = [r for r in results if r.get('safety_metrics', {}).get('response_safety') != 'unsafe']
            logging.info(f"Saving {len(results_to_save)} safe responses to {output_file}")
        
        # Save filtered results
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results_to_save:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Save unsafe responses separately if requested
        if unsafe_output_file:
            unsafe_results = [r for r in results if r.get('safety_metrics', {}).get('response_safety') == 'unsafe']
            if unsafe_results:
                os.makedirs(os.path.dirname(unsafe_output_file), exist_ok=True)
                with open(unsafe_output_file, 'w', encoding='utf-8') as f:
                    for result in unsafe_results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                logging.info(f"Saved {len(unsafe_results)} unsafe responses to {unsafe_output_file}")
            else:
                logging.info("No unsafe responses found to save")
                
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

async def async_main():
    parser = argparse.ArgumentParser(description='Generate responses and safety predictions with vLLM server')
    
    # Model and server arguments
    parser.add_argument('--model_name', type=str, default=DEFAULT_CONFIG['generation_model'],
                      help='Name of the generation model')
    parser.add_argument('--safety_model', type=str, default=DEFAULT_CONFIG['safety_model'],
                      help='Name of the safety evaluation model')
    parser.add_argument('--huggingface_token', type=str, default=DEFAULT_CONFIG['huggingface_token'],
                      help='Hugging Face API token')
    parser.add_argument('--vllm_host', type=str, default=DEFAULT_CONFIG['vllm_server']['host'],
                      help='vLLM server host')
    parser.add_argument('--vllm_model_port', type=int, default=5000,
                      help='vLLM server port')
    parser.add_argument('--vllm_safety_port', type=int, default=6000,
                      help='vLLM server port')
    parser.add_argument('--concurrency', type=int, default=4,
                      help='Number of concurrent requests')
    
    # Dataset arguments
    parser.add_argument('--input_dataset', type=str, required=True,
                      help='Path to input JSONL dataset')
    parser.add_argument('--output', type=str, required=True,
                      help='Output file path for generated responses and safety predictions')
    
    # Generation parameters
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                      help='Batch size for generation')
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_CONFIG['max_tokens'],
                      help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=DEFAULT_CONFIG['temperature'],
                      help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=DEFAULT_CONFIG['top_p'],
                      help='Top-p sampling parameter')
    
    # Add new arguments for response storage options
    parser.add_argument('--store_all', action='store_true',
                      help='Store all responses in the main output file (default: store only safe responses)')
    parser.add_argument('--unsafe_output', type=str, default=None,
                      help='Optional output file path for unsafe responses')
    
    # Add new argument for intermediate save frequency
    parser.add_argument('--save_frequency', type=int, default=100,
                      help='Number of prompts to process before saving intermediate results')
    
    args = parser.parse_args()

    try:
        # Initialize async clients
        logging.info(f"Initializing Async VLLM client for {args.vllm_host}:{args.vllm_model_port}")
        vllm_client = AsyncVLLMClient(
            host=args.vllm_host,
            port=args.vllm_model_port,
            model_name="test-model",
            timeout=DEFAULT_CONFIG['vllm_server']['timeout'],
            max_retries=DEFAULT_CONFIG['vllm_server']['max_retries'],
            retry_delay=DEFAULT_CONFIG['vllm_server']['retry_delay']
        )

        logging.info(f"Initializing async safety client: {args.safety_model}")
        safety_client = AsyncSafetyClient(
            host=args.vllm_host,
            port=args.vllm_safety_port,
            model_name="safety-model",
            timeout=DEFAULT_CONFIG['vllm_server']['timeout'],
            max_retries=DEFAULT_CONFIG['vllm_server']['max_retries'],
            retry_delay=DEFAULT_CONFIG['vllm_server']['retry_delay']
        )

        # Set up sampling parameters
        sampling_params = {
            'temperature': args.temperature,
            'top_p': args.top_p,
            'max_tokens': args.max_tokens
        }

        # Load input dataset
        logging.info(f"Loading input dataset from {args.input_dataset}")
        prompts = []
        total_prompts = 0
        label_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        with open(args.input_dataset, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                total_prompts += 1
                
                # Track prompt label and categories from metadata
                metadata = item.get('metadata', {})
                prompt_label = str(metadata.get('prompt_label', 'unknown')).lower()
                categories = metadata.get('categories', [])
                
                label_counts[prompt_label] += 1
                for category in categories:
                    category_counts[category] += 1
                
                prompt_data = {
                    'id': item.get('id', ''),
                    'input': item.get('input', ''),
                    'output': item.get('output', ''),
                    'source': item.get('source', ''),
                    'metadata': {
                        'category': metadata.get('category', ''),
                        'original_id': metadata.get('original_id', ''),
                        'split': metadata.get('split', ''),
                        'token_count': metadata.get('token_count', 0),
                        'input_tokens': metadata.get('input_tokens', 0),
                        'output_tokens': metadata.get('output_tokens', 0),
                        'prompt_label': prompt_label,
                        'categories': categories
                    }
                }
                if prompt_data['input']:
                    prompts.append(prompt_data)
        
        # Log prompt label and category statistics
        logging.info(f"Loaded {len(prompts)} prompts")
        logging.info("Prompt label distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"{label}: {count} prompts ({count/total_prompts:.1%})")
        
        logging.info("\nCategory distribution:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"{category}: {count} prompts")
        
        # Initialize safety statistics tracking
        safety_stats = {
            'total_samples': 0,
            'unsafe_responses': 0,
            'user_safety': defaultdict(int),
            'response_safety': defaultdict(int),
            'safety_categories': defaultdict(int),
            'category_by_prompt_type': defaultdict(lambda: defaultdict(int))
        }

        # Process prompts in batches using async processing
        all_results = []
        intermediate_file = f"{args.output}.intermediate"
        unsafe_intermediate = f"{args.unsafe_output}.intermediate" if args.unsafe_output else None
        
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating responses"):
            batch_prompts = prompts[i:i + args.batch_size]
            
            # Process batch asynchronously
            batch_results = await process_batch(
                vllm_client=vllm_client,
                safety_client=safety_client,
                batch_prompts=batch_prompts,
                sampling_params=sampling_params,
                concurrency=args.concurrency
            )
            
            # Update safety statistics
            for result in batch_results:
                safety_stats['total_samples'] += 1
                safety_metrics = result.get('safety_metrics', {})
                safety_stats['user_safety'][safety_metrics.get('user_safety', 'unknown')] += 1
                safety_stats['response_safety'][safety_metrics.get('response_safety', 'unknown')] += 1
                
                # Track unsafe responses
                if safety_metrics.get('response_safety') == 'unsafe':
                    safety_stats['unsafe_responses'] += 1
                
                # Track safety categories
                for category in safety_metrics.get('safety_categories', []):
                    safety_stats['safety_categories'][category] += 1
                    # Track categories by prompt type
                    prompt_type = result['metadata'].get('prompt_label', 'unknown')
                    safety_stats['category_by_prompt_type'][prompt_type][category] += 1
            
            all_results.extend(batch_results)
            
            # Save intermediate results after each batch
            save_results(all_results, intermediate_file, args.store_all, unsafe_intermediate)
            logging.info(f"Saved intermediate results to {intermediate_file} (processed {len(all_results)} prompts)")

        # Save final results
        save_results(all_results, args.output, args.store_all, args.unsafe_output)
        
        # Log final safety statistics
        logging.info("\nFinal Safety Statistics:")
        logging.info(f"Total samples processed: {safety_stats['total_samples']}")
        logging.info(f"Unsafe responses: {safety_stats['unsafe_responses']} ({safety_stats['unsafe_responses']/safety_stats['total_samples']:.1%})")
        
        logging.info("\nUser Safety Distribution:")
        for safety, count in sorted(safety_stats['user_safety'].items()):
            logging.info(f"{safety}: {count} ({count/safety_stats['total_samples']:.1%})")
        
        logging.info("\nResponse Safety Distribution:")
        for safety, count in sorted(safety_stats['response_safety'].items()):
            logging.info(f"{safety}: {count} ({count/safety_stats['total_samples']:.1%})")
        
        logging.info("\nSafety Categories Distribution:")
        for category, count in sorted(safety_stats['safety_categories'].items(), key=lambda x: x[1], reverse=True):
            logging.info(f"{category}: {count} occurrences")
        
        logging.info("\nSafety Categories by Prompt Type:")
        for prompt_type, categories in safety_stats['category_by_prompt_type'].items():
            logging.info(f"\n{prompt_type.upper()} prompts:")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"  {category}: {count} occurrences")

        logging.info(f"\nResults saved to {args.output}")
        if not args.store_all:
            safe_count = len([r for r in all_results if r.get('safety_metrics', {}).get('response_safety') != 'unsafe'])
            logging.info(f"Saved {safe_count} safe responses out of {len(all_results)} total responses")
        if args.unsafe_output:
            logging.info(f"Unsafe responses saved to {args.unsafe_output}")

    except Exception as e:
        logging.error(f"Script failed: {e}")
        raise

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main() 