<h2><img align="center" src="https://github.com/user-attachments/assets/cbe0d62f-c856-4e0b-b3ee-6184b7c4d96f">NVIDIA AI Blueprint: Safety for Agentic AI</h2>

### About Safety for Agentic AI

As enterprises explore open models to accelerate agentic workflows, a critical challenge emerges where total end-to-end safety isn't warranted.
Open models offer flexibility, but they pose unknown risks with content safety, security, and privacy risks when used out of the box.
As regulations tighten across regions and industries, non-compliance isn't just costly, it can be catastrophic.
CISOs, compliance leaders, and policymakers are increasingly concerned with unaligned model behavior, ranging from toxic outputs and jailbreak risks to leakage of sensitive or regulated data.
MLOps teams, lacking proactive safety tooling, are forced to patch issues reactively in production.

NVIDIA Safety for Agentic AI offers a structured recipe to evaluate and align open models early, enabling increased safety, security, and compliant agentic workflows.
Start with model evaluation using Garak vulnerability scanning with curated risk prompts, benchmarking against enterprise thresholds.
Afterward, post-train using recipes and safety datasets to close critical safety and security gaps.
Deploy the hardened model as a trusted universal NIM and then add inference-time safety protection with [NVIDIA NeMo Guardrails](https://developer.nvidia.com/nemo-guardrails/) that actively block unsafe behavior.
With continuous monitoring, auditability, and collaboration between AI and risk teams, model safety becomes enforceable, not aspirational.
With Safety for Agentic AI, enterprises can now confidently adopt open models, aligned to their policy, and protected from model adoption, deployment, and inference runtime at production.

For guidelines and suggestions about the process, refer to [Best Practices for Developing a Model Behavior Guide](./docs/best-practices-model-behavior-guide.md).

This repository is what powers the [build experience](https://build.nvidia.com/nvidia/safety-for-agentic-ai), helping you harden security at every stage of the AI development lifecycle.

### Architecture

NVIDIA Safety for Agentic AI is broken down into four steps, which map to a typical agentic workflow environment:

- Safety, security and accuracy evaluation of any model.
- Post-training with NVIDIA curated datasets.
- Deploying a trusted model as a universal NIM.
- Running a trusted model with NVIDIA NeMo Guardrails for run-time guardrails.

![](https://assets.ngc.nvidia.com/products/api-catalog/safety-for-agentic-ai/diagram.jpg)

### Key Features

- Evaluation pipelines for content safety with Nemotron Content Safety Dataset V2, formerly known as Aegis AI Content Safety Dataset v2, and Wildguard utilizing NeMo Eval.
- Security evaluation Pipeline with NVIDIA Garak.
- Dataset blend with 4 datasets and on-policy prompt generation with the target model.
- Post-training (SFT) with NeMo Framework RL.
- Easy-to-understand safety and security reports.
- Packaging and deploying the trusted model with Universal NIM.
- Integrating the Topic Control NIM with NeMo Guardrails for inference-time safety.

### Minimum System Requirements

**Hardware Requirements**

- Self-hosted Main LLM: 8 × (NVIDIA H100 or A100 GPUs 80GB)
- Storage: 300GB
- Minimum System Memory: 128GB

**OS Requirements**

- Python 3.12
- Docker Container: nvcr.io/nvidia/nemo:25.04
- Docker Engine

### Software Used in This Blueprint

**NVIDIA Technology**

- [NVIDIA NeMo Framework RL](https://github.com/NVIDIA/NeMo-RL) -  Post-training library for models ranging from 1 GPU to 100B+ parameters.
- [NVIDIA NeMo Framework Eval](https://github.com/NVIDIA/NeMo) -  Scalable, cloud-native framework to create, customize, and deploy the latest AI models.
- [NVIDIA NIM](https://docs.nvidia.com/nim/index.html) - Microservices for accelerating the deployment of foundation models agnostic of cloud or datacenter.
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) - Programmable logic at inference runtime to safeguard agentic AI applications.
- [NVIDIA NemoGuard Content Safety](https://huggingface.co/nvidia/llama-3.1-nemoguard-8b-content-safety) - Model that detects unsafe interactions between humans and LLMs.
- [NVIDIA Garak](https://github.com/NVIDIA/garak) - Open-source red teaming tool to scan vulnerabilities like hallucination, prompt injection, and jailbreaks.

**Third Party Software**

- [vLLM](https://github.com/vllm-project/vllm)
- [HuggingFace](https://huggingface.co/docs/hub/en/datasets-overview)
- [Weights & Biases](https://wandb.ai/site/)
- [PyTorch](https://pytorch.org/)
- [WildGuard](https://huggingface.co/allenai/wildguard)


**Dataset Used**

- [Nemotron Content Safety Dataset V2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0), formerly known as Aegis AI Content Safety Dataset v2.
- [Gretel Synthetic Safety Alignment Dataset](https://huggingface.co/datasets/gretelai/gretel-safety-alignment-en-v1)
- [HarmfulTasks](https://github.com/CrystalEye42/eval-safety)
- [JailbreakV-28K/JailBreakV-28k/RedTeam_2K](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k/viewer/RedTeam_2K)
- [allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix)
- [Llama Nemotron Post Training Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset)

### Target Audience

- AI developers and engineers
- Research teams focused on AI safety and security
- Product managers overseeing model deployment
- Compliance and regulatory teams

### Prerequisites

- A [personal NVIDIA API key](https://org.ngc.nvidia.com/setup/api-keys) with the `NGC catalog` and `Public API Endpoints` services selected.
- A [Hugging Face token](https://huggingface.co/settings/tokens) so that you can download models and datasets from the hub.

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

## Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility, and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure the models meet requirements for the relevant industry and use case and address unforeseen product misuse. For more detailed information on ethical considerations for the models, please see the Model Card++, Explainability, Bias, Safety & Security, and Privacy Subcards. Please report security vulnerabilities or NVIDIA AI concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## License
Use of this  developer example notebook  is governed by the Apache 2.0 License.

## Terms of Use
The software and materials are governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/), except that models are governed by the [AI Foundation Models Community License Agreement](https://docs.nvidia.com/ai-foundation-models-community-license.pdf) and the NVIDIA RAG dataset is governed by the [NVIDIA Asset License Agreement](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/data/LICENSE.DATA). ADDITIONAL INFORMATION: for Meta/llama-3.1-70b-instruct model, the Llama 3.1 Community License Agreement, for nvidia/llama-3.2-nv-embedqa-1b-v2 model, the Llama 3.2 Community License Agreement. Built with Llama.