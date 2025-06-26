# Safety for Agentic AI: Developer Blueprint Notebooks

This directory has four notebooks to walk you through the core features of adding safety to an open model:

- [Set up the environment by configuring and installing required packages](./Step0_Setup.ipynb)
- [Evaluate the base model for safety and accuracy using NeMo Eval](./Step1_Evaluation.ipynb)
- [Post-train the base model to add content safety and preserve accuracy](./Step2_Safety_Post_Training.ipynb)
- [Rerun the same safety and accuracy evaluation and view the model safety report card to see how the model has improved](./Step3_Post_Training_Eval.ipynb)

These notebooks were developed using 8 x H100 80GB GPUs.
You must use the same or similar computational resources and follow the notebooks in sequential order (Step0->Step1->Step2->Step3).

You can use [NVIDIA Brev](https://developer.nvidia.com/brev) to launch an 8 x H100 instance.

You will need to generate the following API keys:

- `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com/)
- `HF_TOKEN` from [HF Hub](https://huggingface.co/models)

For post-training, you can optionally use [Weights & Biases](https://wandb.ai/home) for experiment tracking.
