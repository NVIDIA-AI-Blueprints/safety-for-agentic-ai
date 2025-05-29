FROM nvcr.io/nvidia/nemo:25.04

# * Requirements for evaluation
RUN pip install nvidia-simple-evals
RUN pip install garak==0.11.0

# *Requirements for post-training
RUN pip install -U "huggingface_hub[cli]
RUN pip install vllm
RUN pip install datasets

# Set the working directory
WORKDIR /workspace

# Clone the repo
RUN git clone https://github.com/NVIDIA/NeMo-RL.git

# Set workdir to the cloned repo
WORKDIR /workspace/NeMo-RL

# Install uv
RUN pip install uv

# Sync dependencies using uv
RUN uv sync

# Set the default command to launch JupyterLab
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 
