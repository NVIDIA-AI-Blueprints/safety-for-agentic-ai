FROM nvcr.io/nvidia/nemo:25.04

# * Requirements for evaluation
RUN pip install nvidia-simple-evals
RUN pip install nvidia-lm-eval
RUN pip install nvidia-bfcl
RUN pip install garak==0.11.0

# *Requirements for post-training
RUN pip install -U "huggingface_hub[cli]"
RUN pip install vllm
RUN pip install datasets
RUN pip install wildguard==1.0.1
RUN pip install protobuf==3.20
RUN pip install wandb==0.16.6
RUN pip install torchdata

# Clone the repo
RUN git clone https://github.com/NVIDIA/NeMo-RL.git

# Set workdir to the cloned repo
WORKDIR /workspace/NeMo-RL

# Install uv
RUN pip install uv

# Sync dependencies using uv
RUN uv sync

