#!/bin/bash
# Base directory and cache setup
export BASE_DIR="/lustre/fsw/portfolios/llmservice/users/ahazare"
export LOG_DIR="$BASE_DIR/gtc_paris/logs"
mkdir -p $LOG_DIR

# Installation Requirements
# cd $BASE_DIR

# # Requirements
# pip install -U "huggingface_hub[cli]
# pip install vllm
# pip install datasets

# # NeMo-RL
# git clone https://github.com/NVIDIA/NeMo-RL.git
# cd "NeMo-RL"
# pip install uv
# uv sync

# # Evaluation
# pip install nvidia-simple-evals
# pip install nvidia-lm-eval

# Set environment variables to redirect temp files and caches to lustre
export TMPDIR="$BASE_DIR/tmp"
export XDG_CACHE_HOME="$BASE_DIR/cache"
export TRANSFORMERS_CACHE="$BASE_DIR/cache/transformers"
export HF_HOME="$BASE_DIR/cache/huggingface"
export TRITON_CACHE_DIR="$BASE_DIR/cache/triton"
export DATASET_CACHE_DIR="$BASE_DIR/dataset_cache"

# Set a shorter TMPDIR for Ray to avoid AF_UNIX path length limit
export RAY_TMPDIR="/tmp/ray_ahazare"

# Create directories if they don't exist
mkdir -p $TMPDIR
mkdir -p $XDG_CACHE_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME 
mkdir -p $TRITON_CACHE_DIR
mkdir -p $DATASET_CACHE_DIR
mkdir -p $RAY_TMPDIR

# Set Credentials and Model configuration
export SAFETY_DATASET_NAME="nvidia/Nemotron-Safety-Training-Dataset"
export POST_TRAINING_DATASET_NAME="nvidia/Llama-Nemotron-Post-Training-Dataset"
export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
export SAFETY_MODEL_NAME="nvidia/llama-3.1-nemoguard-8b-content-safety"

# Credentials
export HF_TOKEN="*****************************"
# huggingface-cli login --token $HF_TOKEN
export WANDB_API_KEY="******************************"

# On-Policy Data Generation parameters
export SAFETY_THRESHOLD=0.8
export MAX_ATTEMPTS=3
export BATCH_SIZE=64
export CONCURRENCY=16
export MAX_TOKENS=512
export TEMPERATURE=0.7
export TOP_P=0.9


# MASTER PROCESS
# Notebook 1
# 0. Download DeepSeek-R1-Distill-Llama-8B Model from HF Dataset Hub
# 0.1 Baseline Accuracy and Safety Evaluation

# Notebook 2 -- This one
# 1. Download NV Safety Dataset from HF Dataset Hub
# 2. Download Llama Nemotron Post-training Dataset
# 3. Combine two datasets to create a dataset blend
# 4. Generate on-policy data for post-training prompts using DS-R1-Distill-Llama-8B (*2 VLLMs on 45K samples)
# 5. Use NeMo-RL to run SFT
# 6. Convert checkpoints and save

# Notebook 3
# 7. Run Evaluation of SFT Model + Show improvements


# 1. Download NV Safety Dataset from HF Dataset Hub
python3 download_safety_dataset.py \
    --dataset_name "$SAFETY_DATASET_NAME" \
    --filename "nemotron-safety-sft-training-blend-v1.0.jsonl" \
    --total_samples 1000 \
    --cache_dir "$DATASET_CACHE_DIR"
if [ $? -ne 0 ]; then
    echo "Error downloading NV Safety dataset"
    exit 1
fi

# 2. Download Llama Nemotron Post-training Dataset
# Files to download (only v1.1 data and essential files, Skipping Safety as we replace it with our own)
files=(
    "SFT/math/math_v1.1.jsonl"
    "SFT/code/code_v1.1.jsonl"
    "SFT/chat/chat.jsonl"
    "SFT/science/science.jsonl"
)

# Create directory for Llama Nemotron files
LLAMA_NEMO_DIR="$DATASET_CACHE_DIR/Llama-Nemotron-Post-Training-Dataset"
mkdir -p "$LLAMA_NEMO_DIR"

# Download files and extract 1200 random samples from each
for file in "${files[@]}"; do
    echo "Downloading $file..."
    # Download using huggingface_hub
    downloaded_file=$(python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='$POST_TRAINING_DATASET_NAME', filename='$file', repo_type='dataset', cache_dir='$DATASET_CACHE_DIR'))")
    
    # Get the base filename
    filename=$(basename "$file")
    target_file="$LLAMA_NEMO_DIR/$filename"
    
    if [ -f "$downloaded_file" ]; then
        # Count total lines in the file
        total_lines=$(wc -l < "$downloaded_file" 2>/dev/null || echo "0")
        if [ "$total_lines" -gt 1200 ]; then
            shuf -n 1200 "$downloaded_file" > "$target_file"
            echo "Extracted 1200 random samples to $target_file"
        else
            cp "$downloaded_file" "$target_file"
            echo "File has fewer than 1200 lines, copied all $total_lines lines"
        fi
    else
        echo "Error: Could not find downloaded file for $file"
        exit 1
    fi
done

echo "All files downloaded and processed successfully"

# 3. Combine two datasets to create a dataset blend
OUTPUT_DIR="$DATASET_CACHE_DIR/sft_data"
python3 combine_datasets.py \
    --safety_file "$DATASET_CACHE_DIR/nv_safety_sampled.jsonl" \
    --llama_nemo_dir "$DATASET_CACHE_DIR/Llama-Nemotron-Post-Training-Dataset" \
    --output_dir "$OUTPUT_DIR" \
    --val_split 0.03 \
    --max_tokens 16384 \
    --max_samples 5000

echo "Datasets combined and split into train/val successfully"

# vLLM server configuration
export VLLM_ENGINE_ITERATION_TIMEOUT_S=36000
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_HOST="0.0.0.0"
export VLLM_TENSOR_PARALLEL_SIZE=1

# GPU Configuration
export POLICY_MODEL_GPUS="0,1,2,3" # GPUs for policy model (comma-separated)
export SAFETY_MODEL_GPUS="4,5"   # GPUs for safety model (comma-separated)

echo "Using GPUs $POLICY_MODEL_GPUS for policy model"
echo "Using GPUs $SAFETY_MODEL_GPUS for safety model"

# Kill any existing vLLM servers
pkill -f "vllm.entrypoints.openai.api_server"
sleep 5

echo "Starting policy model server"
# 4. Start vLLM server for policy model
CUDA_VISIBLE_DEVICES=$POLICY_MODEL_GPUS python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --trust-remote-code \
  --seed 1 \
  --host "$VLLM_HOST" \
  --port 5000 \
  --served-model-name "test-model" \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --download-dir="$HF_HOME" &> "$LOG_DIR/vllm-server-model.log" &

echo Starting safety model server
# 4. Start vLLM server for safety model
SAFETY_MODEL_NAME="nvidia/llama-3.1-nemoguard-8b-content-safety"
CUDA_VISIBLE_DEVICES=$SAFETY_MODEL_GPUS python3 -m vllm.entrypoints.openai.api_server \
  --model "$SAFETY_MODEL_NAME" \
  --trust-remote-code \
  --seed 1 \
  --host "$VLLM_HOST" \
  --port 6000 \
  --served-model-name "safety-model" \
  --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --download-dir="$HF_HOME" &> "$LOG_DIR/vllm-server-safety.log" &

sleep 120  # Wait for Policy and Safety model servers to start

# 5. Generate on-policy data for safety prompts using DS-R1-Distill-Llama-8B
# Set input and output paths
INPUT_DATASET="$OUTPUT_DIR/train.jsonl"  # Using the combined dataset from step 3
OUTPUT_FILE="$OUTPUT_DIR/train_on_policy_data.jsonl"
echo "Input dataset: $INPUT_DATASET"
echo "Output File: $OUTPUT_FILE"

python3 generate_on_policy_data.py \
  --model_name "$MODEL_NAME" \
  --safety_model "$SAFETY_MODEL_NAME" \
  --huggingface_token "$HUGGING_FACE_HUB_TOKEN" \
  --vllm_host "$VLLM_HOST" \
  --vllm_model_port 5000 \
  --vllm_safety_port 6000 \
  --concurrency "$CONCURRENCY" \
  --input_dataset "$INPUT_DATASET" \
  --output "$OUTPUT_FILE" \
  --batch_size "$BATCH_SIZE" \
  --max_tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" &> "$LOG_DIR/vllm-server-safety-train.log" &

wait
echo "Generation completed. Results saved to $OUTPUT_FILE"

# Also generate for validation set
INPUT_DATASET_VAL="$OUTPUT_DIR/val.jsonl"
OUTPUT_FILE_VAL="$OUTPUT_DIR/val_on_policy_data.jsonl"

echo "Generating responses and safety predictions for validation set..."
echo "Input dataset: $INPUT_DATASET_VAL"
echo "Output file: $OUTPUT_FILE_VAL"

python3 generate_on_policy_data.py \
  --model_name "$MODEL_NAME" \
  --safety_model "$SAFETY_MODEL_NAME" \
  --huggingface_token "$HUGGING_FACE_HUB_TOKEN" \
  --vllm_host "$VLLM_HOST" \
  --vllm_model_port 5000 \
  --vllm_safety_port 6000 \
  --concurrency "$CONCURRENCY" \
  --input_dataset "$INPUT_DATASET_VAL" \
  --output "$OUTPUT_FILE_VAL" \
  --batch_size "$BATCH_SIZE" \
  --max_tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" &> "$LOG_DIR/vllm-server-safety-val.log" &
wait
echo "Generation for validation set completed. Results saved to $OUTPUT_FILE_VAL"

# Cleanup vLLM servers after generation is complete
echo "Cleaning up vLLM servers..."
pkill -f "vllm.entrypoints.openai.api_server"

# 6. Use NeMo-RL to run SFT
# Install NeMo-Rl
cd $BASE_DIR/NeMo-RL
MODEL_DIR="$BASE_DIR/NeMo-RL/results/sft_deepseek_8b_trial_step_300/"
TMPDIR=$RAY_TMPDIR uv run python examples/run_sft.py --config $BASE_DIR/gtc_paris/deepseek_sft.yaml &> $LOG_DIR/sft.log

# 7. Convert checkpoint from DCP to HF
# Path to your NeMo-RL checkpoint
MODEL_DIR="$BASE_DIR/NeMo-RL/results/sft_deepseek_8b_step_300" # TODO: Change to the latest checkpoint
# Input path
DCP_CKPT_PATH="$MODEL_DIR/policy/weights/"
CONFIG_PATH="$MODEL_DIR/config.yaml"
# Output path
HF_CKPT_PATH="$MODEL_DIR/hf_ckpt"

echo "Starting conversion process..."
cd $BASE_DIR/NeMo-RL
uv run examples/convert_dcp_to_hf.py \
    --config $CONFIG_PATH \
    --dcp-ckpt-path $DCP_CKPT_PATH \
    --hf-ckpt-path $HF_CKPT_PATH 

# Verify conversion results
if [ -f "$HF_CKPT_PATH/pytorch_model.bin" ] && [ -f "$HF_CKPT_PATH/config.json" ]; then
    echo "Conversion successful! Files created:"
    ls -lh "$HF_CKPT_PATH"
    echo ""
    echo "The HuggingFace model is now available at: $HF_CKPT_PATH"
else
    echo "Conversion may have failed. Please check the output."
fi 


# ------------------------------------------------------------------------------------------------

# Notebook 3
# 7. Run Baseline Evaluation - Start vLLM server
export BASELINE_MODEL_GPUS="0,1"
CUDA_VISIBLE_DEVICES=$BASELINE_MODEL_GPUS python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --reasoning-parser "qwen3" \
  --trust-remote-code \
  --seed 1 \
  --host "$VLLM_HOST" \
  --port 5000 \
  --served-model-name "baseline-model" \
  --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --download-dir="$HF_HOME" &> "$BASE_DIR/vllm-server-baseline-model.log" &

# 7. Run Baseline Evaluation - Run Evaluations in Background
export MY_API_KEY="empty"
export JUDGE_API_KEY="nvapi-XXXXXXXXXXXXXX"

mkdir -p "$BASE_DIR/results/baseline-evals/gpqa-diamond"
simple_evals \
  --model "baseline-model" \
  --url http://localhost:5000/v1/chat/completions \
  --eval_name gpqa_diamond \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_tokens 8192 \
  --out_dir "$BASE_DIR/results/baseline-evals/gpqa-diamond" \
  --cache_dir "$BASE_DIR/results/baseline-evals/gpqa-diamond" \
  --num_threads 4 \
  --max_retries 5 \
  --timeout 150 &> "$BASE_DIR/baseline-eval-gpqa-diamond.log" &

mkdir -p "$BASE_DIR/results/baseline-evals/aa-aime-2024"
simple_evals \
  --model "baseline-model" \
  --url http://localhost:5000/v1/chat/completions \
  --eval_name AA_AIME_2024 \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_tokens 8192 \
  --out_dir "$BASE_DIR/results/baseline-evals/aa-aime-2024" \
  --cache_dir "$BASE_DIR/results/baseline-evals/aa-aime-2024" \
  --num_threads 4 \
  --max_retries 5 \
  --timeout 150 &> "$BASE_DIR/baseline-eval-aa-aime-2024.log" &

mkdir -p "$BASE_DIR/results/baseline-evals/aa-math-500"
simple_evals \
  --model "baseline-model" \
  --url http://localhost:5000/v1/chat/completions \
  --eval_name AA_math_test_500 \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_tokens 8192 \
  --out_dir "$BASE_DIR/results/baseline-evals/aa-math-500" \
  --cache_dir "$BASE_DIR/results/baseline-evals/aa-math-500" \
  --num_threads 4 \
  --max_retries 5 \
  --timeout 150 &> "$BASE_DIR/baseline-eval-aa-math-500.log" &

mkdir -p "$BASE_DIR/results/baseline-evals/ifeval"
lm-eval \
  --tasks ifeval \
  --num_fewshot 0 \
  --model local-chat-completions \
  --model_args "base_url=http://localhost:5000/v1/chat/completions,model=baseline-model,tokenized_requests=false,,num_concurrent=4,max_gen_toks=8192,timeout=150,max_retries=5,stream=False" \
  --log_samples \
  --output_path "$BASE_DIR/results/baseline-evals/ifeval" \
  --use_cache "$BASE_DIR/results/baseline-evals/ifeval" \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --gen_kwargs="temperature=0.6,top_p=0.95" &> "$BASE_DIR/baseline-eval-ifeval.log" &


# 8. Run Evaluation of SFT Model - Start vLLM server
# python3 -m vllm.entrypoints.openai.api_server \
#   --model "$HF_CKPT_PATH" \
#   --trust-remote-code \
#   --seed 1 \
#   --host "$VLLM_HOST" \
#   --port "$VLLM_PORT" \
#   --served-model-name "test-model" \
#   --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
#   --download-dir="$HF_HOME" &> "$BASE_DIR/vllm-server-model.log" &

# # Cleanup
# pkill -f "vllm.entrypoints.openai.api_server"