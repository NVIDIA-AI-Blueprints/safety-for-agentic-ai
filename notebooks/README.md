# NeMo Safety: Deverloper Blueprint Notebooks

This repository hosts the following four notebooks to walk you through the core features of NeMo Safety:

- [Notebook 1]() Safety and Accuracy Evaluation using NeMo Eval
- [Notebook 2]() Safety Post-training using NeMo Safety’s training recipe 
- [Notebook 3]() Re-running the same safety and accuracy evaluation to understand how the model has improved
- [Notebook 4]() Model Safety Report Card

The notebooks assume that the user has 8x H100 80GB GPUs or similar computational resources. You can use [NVIDIA Brev](https://developer.nvidia.com/brev) to launch an instance. 

You will need to generate the following API keys
- `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com/)
- `HF_TOKEN` from [HF Hub](https://huggingface.co/models)

For post-training, you can optionally use [Weights & Biases](https://wandb.ai/home) for experiment tracking. 
