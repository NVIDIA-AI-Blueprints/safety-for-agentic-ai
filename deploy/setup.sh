pip install nvidia-safety-harness
pip install nvidia-simple-evals
pip install nvidia-lm-eval
pip install nvidia-bfcl
pip install garak==0.11.0

# *Requirements for post-training
pip install -U "huggingface_hub[cli]"
pip install vllm
pip install datasets
pip install wildguard==1.0.1
pip install protobuf==3.20
pip install wandb==0.16.6
pip install torchdata

# Clone the NeMo-Safetyrepo
cd /workspace
# git clone https://github.com/NVIDIA-AI-Blueprints/nemo-safety
git clone https://gitlab-master.nvidia.com/swdl-nemollm-mlops/NeMo-Safety.git -b dev/mvp

# Clone the NeMo-RL repo
cd /workspace
git clone https://github.com/NVIDIA/NeMo-RL.git
cd NeMo-Rl

# Install uv
pip install uv

# Sync dependencies using uv
uv sync

