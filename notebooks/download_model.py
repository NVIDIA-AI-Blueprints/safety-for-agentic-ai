import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_path = "/lustre/fsw/portfolios/llmservice/users/ahazare/cache/huggingface/llama-3.1-nemoguard-8b-content-safety"
# Load base model
from peft import PeftModel
from transformers import AutoModelForCausalLM



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, "nvidia/llama-3.1-nemoguard-8b-content-safety")
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer.save_pretrained(model_path)  