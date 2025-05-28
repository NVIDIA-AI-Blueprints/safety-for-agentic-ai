from vllm import LLM, SamplingParams
import json
import logging
import os
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import argparse
from datetime import datetime
from datasets import load_dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import requests
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Default configuration
DEFAULT_CONFIG = {
    'generation_model': "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    'safety_model': "nvidia/llama-3.1-nemoguard-8b-content-safety",
    'huggingface_token': "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",
    'output_dir': "generated_data",
    'batch_size': 32,
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

class VLLMClient:
    def __init__(self, host: str, port: int, timeout: int, max_retries: int, retry_delay: int):
        self.base_url = f"http://{host}:{port}/v1"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def generate(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate responses using the vLLM server."""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/completions",
                    json={
                        "model": "test-model",
                        "prompt": prompts,
                        "max_tokens": sampling_params.max_tokens,
                        "temperature": sampling_params.temperature,
                        "top_p": sampling_params.top_p,
                        "stop": sampling_params.stop
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()['choices']
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)

class SafetyFilter:
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
        
    def check_safety(self, safety_scores: Dict[str, float]) -> Tuple[bool, str]:
        """Check if response meets safety thresholds."""
        for category, threshold in self.thresholds.items():
            if safety_scores[category] < threshold:
                return False, f"Failed {category} threshold: {safety_scores[category]:.3f} < {threshold}"
        return True, "Passed all safety thresholds"

class SafetyEvaluator:
    def __init__(self, vllm_host: str = "localhost", vllm_port: int = 5001, model_name: str = "test-model", api_key: str = None):
        self.vllm_host = vllm_host
        self.vllm_port = vllm_port
        self.model_name = model_name
        self.api_key = api_key  # If needed for auth
        self.api_url = f"http://{self.vllm_host}:{self.vllm_port}/v1/completions"

    def evaluate_safety(self, prompt: str, response: str) -> dict:
        """Query the vLLM server for safety prediction."""
        # Compose the input as needed by your vLLM model
        input_text = prompt + "\n" + response
        payload = {
            "model": self.model_name,
            "prompt": input_text,
            "max_tokens": 1,  # Only need prediction, not generation
            "temperature": 0.0
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            r = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            result = r.json()
            # You may need to adapt this depending on your model's output format
            # Example: extract safety score from result["choices"][0]["text"] or similar
            safety_pred = result["choices"][0]["text"]
            return {"safety_pred": safety_pred, "raw": result}
        except Exception as e:
            print(f"Error querying vLLM server: {e}")
            return {"safety_pred": None, "error": str(e)}

def load_nemotron_prompts(split: str = "train") -> List[Dict[str, Any]]:
    """Load prompts from the Nemotron Safety Training Dataset."""
    try:
        logging.info(f"Loading Nemotron dataset, split: {split}")
        dataset = load_dataset("nvidia/nemotron-safety-training-dataset", 
                             token=DEFAULT_CONFIG['huggingface_token'],
                             split=split)
        
        prompts = []
        for item in dataset:
            prompt_data = {
                'prompt': item['prompt'],
                'metadata': {
                    'id': item.get('id', ''),
                    'category': item.get('category', ''),
                    'split': split,
                    'source': 'nemotron-safety-training'
                }
            }
            prompts.append(prompt_data)
            
        logging.info(f"Loaded {len(prompts)} prompts from Nemotron dataset")
        return prompts
    except Exception as e:
        logging.error(f"Error loading Nemotron dataset: {e}")
        raise

def extract_unsafe_categories(safety_raw):
    """Extract unsafe categories from safety prediction raw output."""
    # This function assumes safety_raw contains a dict with category scores
    # and that lower scores mean unsafe. Adjust as needed for your model output.
    unsafe = []
    if not safety_raw:
        return unsafe
    # Example: {'harmful': 0.2, 'unethical': 0.3, 'biased': 0.4, 'overall': 0.5}
    for cat in ['harmful', 'unethical', 'biased']:
        score = safety_raw.get(cat)
        if score is not None and score < DEFAULT_CONFIG['safety_thresholds'][cat]:
            unsafe.append(cat)
    return unsafe

def build_refusal_prompt(prompt, categories):
    cats = ', '.join(categories)
    return (
        "You are a helpful assistant that generates safe responses for given prompts.\n\n"
        f"The prompt is: {prompt}\n"
        "This prompt is known to violate the following safety categories or elicits unsafe responses of these categories from the model. "
        f"The categories are: {cats}\n\n"
        "Generate a safe refusal for the prompt that politely points out that the model doesn't engage in unsafe behavior for the given categories, "
        "however, don't assume you were told the categories in advance, act like you figured the categories and safety issues yourself. Your response:"
    )

def generate_and_evaluate(
    vllm_client: VLLMClient,
    safety_evaluator: SafetyEvaluator,
    prompt_data: Dict[str, Any],
    sampling_params: SamplingParams
) -> Dict[str, Any]:
    """Generate a response and evaluate its safety."""
    # Generate response
    outputs = vllm_client.generate([prompt_data['prompt']], sampling_params)
    response = outputs[0]['text']

    # Get safety prediction
    safety_result = safety_evaluator.evaluate_safety(prompt_data['prompt'], response)
    safety_pred = safety_result.get('safety_pred')
    safety_raw = safety_result.get('raw')

    return {
        'prompt': prompt_data['prompt'],
        'original_response': prompt_data.get('original_response', ''),  # Include original response
        'generated_response': response,  # Rename to distinguish from original
        'safety_prediction': safety_pred,
        'safety_details': safety_raw,
        'metadata': prompt_data['metadata']
    }

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save generated results to a JSONL file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        logging.info(f"Saved {len(results)} results to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

def predict_safety_on_dataset(input_path: str, output_path: str, safety_evaluator: SafetyEvaluator):
    """Predict safety for every sample in the input dataset and save results."""
    results = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Predicting safety"):
            sample = json.loads(line)
            prompt = sample.get('prompt') or sample.get('input')
            response = sample.get('response') or sample.get('output')
            if not prompt or not response:
                continue
            safety_result = safety_evaluator.evaluate_safety(prompt, response)
            sample['safety_pred'] = safety_result.get('safety_pred')
            sample['safety_raw'] = safety_result.get('raw')
            sample['safety_error'] = safety_result.get('error')
            results.append(sample)
    save_results(results, output_path)
    logging.info(f"Predicted safety for {len(results)} samples. Results saved to {output_path}")

def main():
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
    
    args = parser.parse_args()

    try:
        # Initialize vLLM client
        logging.info(f"Initializing vLLM client for {args.vllm_host}:{args.vllm_port}")
        vllm_client = VLLMClient(
            host=args.vllm_host,
            port=args.vllm_model_port,
            timeout=DEFAULT_CONFIG['vllm_server']['timeout'],
            max_retries=DEFAULT_CONFIG['vllm_server']['max_retries'],
            retry_delay=DEFAULT_CONFIG['vllm_server']['retry_delay']
        )

        # Initialize safety evaluator
        logging.info(f"Initializing safety evaluator: {args.safety_model}")
        safety_evaluator = SafetyEvaluator(
            args.vllm_host,
            args.vllm_safety_port,  # Safety model runs on next port
            args.safety_model,
            args.huggingface_token
        )

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stop=None
        )

        # Load input dataset
        logging.info(f"Loading input dataset from {args.input_dataset}")
        prompts = []
        with open(args.input_dataset, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                prompt_data = {
                    'id': item.get('id', ''),
                    'input': item.get('input', ''),
                    'output': item.get('output', ''),
                    'source': item.get('source', ''),
                    'metadata': item.get('metadata', {
                        'category': '',
                        'original_id': '',
                        'split': '',
                        'token_count': 0,
                        'input_tokens': 0,
                        'output_tokens': 0
                    })
                }
                if prompt_data['input']:  # Changed from 'prompt' to 'input' to match format
                    prompts.append(prompt_data)
        
        logging.info(f"Loaded {len(prompts)} prompts from input dataset")
        
        # Process prompts
        all_results = []
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating responses and safety predictions"):
            batch_prompts = prompts[i:i + args.batch_size]
            batch_results = []
            
            for prompt_data in batch_prompts:
                # Generate response using 'input' field
                outputs = vllm_client.generate([prompt_data['input']], sampling_params)
                response = outputs[0]['text']

                # Get safety prediction
                safety_result = safety_evaluator.evaluate_safety(prompt_data['input'], response)
                safety_pred = safety_result.get('safety_pred')
                safety_raw = safety_result.get('raw')

                result = {
                    'id': prompt_data['id'],
                    'input': prompt_data['input'],
                    'original_output': prompt_data['output'],
                    'generated_output': response,
                    'safety_prediction': safety_pred,
                    'safety_details': safety_raw,
                    'source': prompt_data['source'],
                    'metadata': prompt_data['metadata']
                }
                batch_results.append(result)
            
            all_results.extend(batch_results)
            
            # Save intermediate results
            if (i + args.batch_size) % (args.batch_size * 5) == 0:
                intermediate_file = f"{args.output}.intermediate_{i}"
                save_results(all_results, intermediate_file)
                logging.info(f"Saved intermediate results to {intermediate_file}")

        # Save final results
        save_results(all_results, args.output)
        logging.info(f"Generated responses and safety predictions for {len(all_results)} samples")
        logging.info(f"Results saved to {args.output}")

    except Exception as e:
        logging.error(f"Script failed: {e}")
        raise

if __name__ == "__main__":
    main() 