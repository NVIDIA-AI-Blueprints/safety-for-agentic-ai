<h2><img align="center" src="https://github.com/user-attachments/assets/cbe0d62f-c856-4e0b-b3ee-6184b7c4d96f">NVIDIA AI Blueprint: Safety for Agentic AI</h2>

### About Safety for Agentic AI

As large language models (LLMs) increasingly enable agentic AI systems capable of autonomous reasoning and tool use, they also introduce critical safety risks, including goal misalignment, hallucinations, and prompt injections.
Enterprises are challenged to harness open-weight models' flexibility without compromising on trust, security, or compliance.
As regulations tighten across regions and industries, non-compliance becomes a persistent challenge.

With this safety recipe, enterprises can now confidently adopt open models, aligned to their policy.
Start with model evaluation using garak vulnerability scanning with curated risk prompts, benchmarking against enterprise thresholds.
Then, post-train using recipes and safety datasets to close critical safety and security gaps.
Deploy the hardened model as a trusted NVIDIA NIM and then add inference run-time safety protection with [NVIDIA NeMo Guardrails](https://developer.nvidia.com/nemo-guardrails/) that actively block unsafe model behavior.
With continuous monitoring, and collaboration between AI and risk teams, model safety becomes enforceable, not aspirational.

For guidelines and suggestions about the process, refer to [Best Practices for Developing a Model Behavior Guide](https://github.com/NVIDIA-AI-Blueprints/safety-for-agentic-ai/blob/main/docs/best-practices-model-behavior-guide.md).

This repository is what powers the [build experience](https://build.nvidia.com/nvidia/safety-for-agentic-ai), helping you harden security at every stage of the AI development lifecycle.

You can deploy this blueprint in a ready-to-use hardware and software environment as a Brev _launchable_ by going to the [build experience](https://build.nvidia.com/nvidia/safety-for-agentic-ai) page and clicking **Deploy on Cloud**.
Alternatively, refer to the [notebooks README](https://github.com/NVIDIA-AI-Blueprints/safety-for-agentic-ai/tree/main/notebooks#readme) for deployment instructions.

## Contents

<!-- TOC -->

- [Contents](#contents)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Minimum System Requirements](#minimum-system-requirements)
    - [Hardware Requirements](#hardware-requirements)
    - [OS Requirements](#os-requirements)
- [Software Used in This Blueprint](#software-used-in-this-blueprint)
    - [NVIDIA Technology](#nvidia-technology)
    - [Third Party Software](#third-party-software)
    - [Datasets](#datasets)
- [Target Audience](#target-audience)
- [Prerequisites](#prerequisites)
    - [Quickstart Guide](#quickstart-guide)
- [Ethical Considerations](#ethical-considerations)
- [License](#license)
- [Terms of Use](#terms-of-use)

<!-- /TOC -->

## Architecture

This safety recipe is broken down into four steps, which map to a typical agentic workflow environment: 

- Safety, security, and accuracy evaluation of models and systems
- Post-training with NVIDIA curated datasets  
- Deploying the trusted model as a NIM  
- Running the trusted model with NVIDIA NeMo Guardrails at inference

![Architecture diagram](https://assets.ngc.nvidia.com/products/api-catalog/safety-for-agentic-ai/diagram.jpg)

## Key Features

- Evaluation pipelines for content safety with [Nemotron Content Safety Dataset V2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) and [Wildguard](https://huggingface.co/AllanK24/modernbert-Aegis-2.0-Wildguard-Content-Safety) utilizing [NeMo Eval](https://github.com/NVIDIA/NeMo)  
- Security evaluation pipeline with [NVIDIA garak](https://github.com/NVIDIA/garak)  
- Dataset blend with 4 datasets and on-policy prompt generation with the target model  
- Post-training (SFT) with [NeMo Framework RL](https://github.com/NVIDIA/NeMo-RL)   
- Easy-to-understand safety and security reports  
- Packaging and deploying the trusted model with NIM  
- Integrating the [Content Safety NIM](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety) with [NeMo Guardrails](https://developer.nvidia.com/nemo-guardrails) for inference-time safety

## Minimum System Requirements

### Hardware Requirements

- Self-hosted Main LLM: 8 × (NVIDIA H100 or A100 GPUs 80GB)
- Storage: 300GB
- Minimum System Memory: 128GB

### OS Requirements

- Python 3.12
- Docker Container: nvcr.io/nvidia/nemo:25.04
- Docker Engine

## Software Used in This Blueprint

### NVIDIA Technology

- [NVIDIA NeMo Framework RL](https://github.com/NVIDIA/NeMo-RL) \-  Post-training library for models ranging from 1 GPU to 100B+ parameters  
- [NVIDIA NeMo Framework Eval](https://github.com/NVIDIA/NeMo) \-  Scalable, cloud-native framework to create, customize, and deploy the latest AI models  
- [NVIDIA NIM](https://docs.nvidia.com/nim/index.html) \- Microservices for accelerating the deployment of foundation models agnostic of cloud or datacenter
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) \- Programmable logic at inference runtime to safeguard agentic AI applications
- [NVIDIA NemoGuard Content Safety](https://huggingface.co/nvidia/llama-3.1-nemoguard-8b-content-safety) \- Multilingual model that detects unsafe interactions between humans and LLMs
- [NVIDIA Garak](https://github.com/NVIDIA/garak) \- Open-source red teaming tool to scan vulnerabilities like hallucination, prompt injection, and jailbreaks

### Third Party Software

- [vLLM](https://github.com/vllm-project/vllm)
- [Hugging Face](https://huggingface.co/docs/hub/en/datasets-overview)
- [Weights & Biases](https://wandb.ai/site/)
- [PyTorch](https://pytorch.org/)
- [WildGuard](https://huggingface.co/allenai/wildguard)

### Datasets

- [Nemotron Content Safety Dataset V2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0)  
- [Gretel Synthetic Safety Alignment Dataset](https://huggingface.co/datasets/gretelai/gretel-safety-alignment-en-v1)  
- [HarmfulTasks](https://github.com/CrystalEye42/eval-safety)  
- [Llama Nemotron Post Training Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset)
- [JaiBreakV-28k/ReadTeam 2k](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k/viewer/RedTeam_2K)
- [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix)

## Target Audience

- AI developers and engineers
- Research teams focused on AI safety and security
- Product managers overseeing model deployment
- Compliance and regulatory teams

## Prerequisites

- A [personal NVIDIA API key](https://org.ngc.nvidia.com/setup/api-keys) with the `NGC catalog` and `Public API Endpoints` services selected.
- A [Hugging Face token](https://huggingface.co/settings/tokens) so that you can download models and datasets from the hub.

## Quickstart Guide

Run the following notebooks:

- [Setup](https://github.com/NVIDIA-AI-Blueprints/safety-for-agentic-ai/blob/main/notebooks/Step0_Setup.ipynb)
- [Evaluating the Base Model for Safety and Accuracy](https://github.com/NVIDIA-AI-Blueprints/safety-for-agentic-ai/blob/main/notebooks/Step1_Evaluation.ipynb)
- [Fine-tuning for Safety and Accuracy](https://github.com/NVIDIA-AI-Blueprints/safety-for-agentic-ai/blob/main/notebooks/Step2_Safety_Post_Training.ipynb)
- [Evaluating the Fine-tuned Model](https://github.com/NVIDIA-AI-Blueprints/safety-for-agentic-ai/blob/main/notebooks/Step3_Post_Training_Eval.ipynb)

Optionally, you can [Run Inference with the Fine-Tuned Model and NeMo Guardrails](https://github.com/NVIDIA-AI-Blueprints/safety-for-agentic-ai/blob/main/notebooks/Step4_Run_Inference_with_NeMo_Guardrails_Docker.ipynb)

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility, and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure the models meet requirements for the relevant industry and use case and address unforeseen product misuse. For more detailed information on ethical considerations for the models, please see the Model Card++, Explainability, Bias, Safety & Security, and Privacy Subcards. Please [report security vulnerabilities](https://www.nvidia.com/en-us/support/submit-security-vulnerability/) or NVIDIA AI concerns.

## License

Use of this developer example notebook  is governed by the Apache 2.0 License.

## Terms of Use

The software and materials are governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/), except that models are governed by the [AI Foundation Models Community License Agreement](https://docs.nvidia.com/ai-foundation-models-community-license.pdf) and the NVIDIA RAG dataset is governed by the [NVIDIA Asset License Agreement](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/data/LICENSE.DATA). ADDITIONAL INFORMATION: for Meta/llama-3.1-70b-instruct model, the Llama 3.1 Community License Agreement, for nvidia/llama-3.2-nv-embedqa-1b-v2 model, the Llama 3.2 Community License Agreement. Built with Llama.