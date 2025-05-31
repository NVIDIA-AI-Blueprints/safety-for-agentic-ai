# The NeMo Safety Data Generation Process
For the NeMo Safety training data blend generation, we use a 3-step process. 

## Step 1: Data Collection and Processing

### Step 1a: Data Collection
---
We first collect unsafe prompts from diverse sources, these currently are:
- Aegis v2 (~23k rows): https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0
- Gretel v1 (~6k rows): https://huggingface.co/datasets/gretelai/gretel-safety-alignment-en-v1
- RedTeam-2k (~580 rows): https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k/viewer/RedTeam_2K/
- HarmfulTasks (~1650 rows): https://raw.githubusercontent.com/CrystalEye42/eval-safety/refs/heads/main/integrated.yaml

### Step 1b: Data Processing
---
Once prompts are curated, the first processing step is to remove duplicate prompts, standardize schemas and categorical columns, and add source information to all rows. 

Next, we also create a modified version of the `prompt` called the `safe_prompt`.
This modification process does two things:
1. If the `prompt` is known to contain a popular jailbreak, it removes it to keep only the embedded request.
2. If the `prompt_label` is unsafe, we wrap the prompt around in a prompt template that elicits a safe response from a target model.

Note that the `safe_prompt` column is only used in Step 2, safe on-policy response generation.

For actual model training, we go back to using the `prompt`.

**We are effectively thus trying to teach the model to generate the same response to `prompt` after training, as it generated to `safe_prompt` before training.**

## Step 2: Safe On-Policy Response Generation
Next, generate responses using a target model to the content in the `safe_prompt` column.
These are stored in the `generated_response` column.

## Step 3: Filter out Unsafe Responses
Finally, we run a safety judge model on the generated responses to grade them as safe/unsafe.

Note that if you're dealing with a reasoning model, the notebook provides a way to judge the actual model response by stripping out the thinking traces.

This is to weed out any remaining generations that were still unsafe, as we don't want to teach the model to produce unsafe responses during training.

## Conclusion
That's it! The above 3-step process will generate the NeMo Safety Data Blend.