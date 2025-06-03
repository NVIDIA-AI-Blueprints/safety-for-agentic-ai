<h2><img align="center" src="https://github.com/user-attachments/assets/cbe0d62f-c856-4e0b-b3ee-6184b7c4d96f">NIM Agent Blueprint: NeMo Safety Toolkit</h2>

### Building Trust in AI Systems

Enterprises often find it challenging to deploy open-weight models to production because of concern about the safety and security of the models:

- Worry about risks from running powerful open models without proper controls.
- Concern about vulnerability to prompt injection attacks.
- Requirements for adequate security and safety measures.

NeMo Safety toolkit provides the tools to address the challenge.
Using the toolkit, enterprises can securely and confidently adopt open models.

This blueprint is a reference implementation that demonstrates how to take an open-weight model through the following tasks:

- Evaluate the base model for safety and accuracy to establish a baseline.
- Fine tune the model with safety-related data and SFT and RL data to preserve accuracy.
- Evaluate the fine-tuned model for safety and accuracy.

The notebooks in this blueprint use [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B).
You can substitute any preferred model.

### Key Software Components

- NVIDIA NeMo-RL - Post-training library
- LLM vulnerability scanner - `garak`
- Safety classification - `wildguard`
- Python packages to perform evaluation:
  - NVIDIA Safety Harness - `nvidia-safety-harness`
  - NVIDIA Simple Evaluations - `nvidia-simple-evals`
  - NVIDIA LM Evaluation - `nvidia-lm-eval`

### Target Audience

- AI developers and engineers
- Research teams focused on AI safety and security
- Product managers overseeing model deployment
- Compliance and regulatory teams

### Prerequisites

- A [personal NVIDIA API key](https://org.ngc.nvidia.com/setup/api-keys) with the `NGC catalog` and `Public API Endpoints` services selected.
- A [Hugging Face token](https://huggingface.co/settings/tokens) so that you can download models and datasets from the hub.

Refer to the [setup script](./deploy/setup.sh) to install key software components and dependencies.

### Hardware Requirements

One machine that meets the following requirements:

- 8 x H100 80 GB GPUs
- 128 GB RAM
- 100 GB disk space

### Quickstart Guide

Run the following notebooks:

- [Evaluating the Base Model for Safety and Accuracy](./notebooks/Step1_Evaluation.ipynb)
- [Fine-tuning for Safety and Accuracy](./notebooks/Step2_Safety_Post_Training.ipynb)
- [Evaluating the Fine-tuned Model](./notebooks/Step3_Post_Training_Eval.ipynb)

## License

This NVIDIA AI BLUEPRINT is licensed under the [Apache License, Version 2.0](./LICENSE).
This project downloads and installs additional third-party open source software projects and containers.
Review [the license terms of these open source projects](./LICENSE-3rd-party.txt) before use.

The software and materials are governed by the NVIDIA Software License Agreement (found at https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and the Product-Specific Terms for NVIDIA AI Products (found at https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).