# Best Practices for Developing a Model Behavior Guide

## Benefits of a Model Behavior Guide

Every model training, evaluation, and red-teaming initiative reflects fundamental decisions regarding how a system should behave.
These decisions can emerge implicitly as the system is developed, or explicitly designed and optimized.
By anchoring AI development with a clear north star, such as a well-defined model behavior guide, we can effectively steer behavior to meet application needs.
After the behavior is defined, you can systematically optimize a model behavior guide through various techniques such as synthetic data generation and curation, alignment that includes SFT and RL, system prompt development, LLM-as-judge evaluation, robust red-teaming strategies, and so on.

## Considerations for a Model Behavior Guide

When developing a model behavior guide, it is important to consider the following aspects.

### Identify Your Target User Persona and Application Goals

The crux of every model behavior guide is a deep understanding of your target users, use cases, and desired output.
To begin, identify your target user needs, preferences, and pain points.
Anticipate the primary use scenarios.
In cases where your user is another AI system or third-party infrastructure, direct your focus toward the success criteria for your specific application.

Example user analysis:

**Industry**: What industry will the LLM application primarily serve?
Gaming, Automotive, Healthcare, Finance?

**Role**: Who will be the primary users interacting with the LLM?
IT Support Specialist, Online Shopper, Video Game User?

**Application**: What is the primary task or objective that the LLM will assist with?
Customer Service Agent, Shopping Recommendation Engine, Personalized Learning Plan?

**User Goals**: What are the primary goals your target user aims to achieve when interacting with the LLM?

Other key attributes to consider can include age, language, country of deployment, and more.

### Define Overall Behavior

Now that you have a deep understanding of your target user persona and application goals, outline the overall behavior you want your model to exhibit.
This step involves determining the tone, verbosity, boundaries, and other performance attributes that will guide your model's interactions.

Example behavior standards:

**Core Principles**: What fundamental values should guide the model's interactions?
Factuality, Coherence, Accessibility, Instruction Following?

**Persona**: How should the model present itself to users?
Knowledgeable Assistant, Helpful Customer Service Representative, Neutral Informant?

**Tone**: What demeanour should the model convey in its responses?
Professional, Humorous, Conversational?

**Formatting**: How should the model structure its responses?
Stylistic Requirements, Structured Outputs & Codeblocks?

**Verbosity**: What is the ideal length for the model's responses?
50-100 character for summaries?

**Content Safety and Product Security Standards**: What standards should the model adhere to in order to provide a secure, trustworthy, and respectful user experience, balancing information accessibility with potential risks and sensitivities? Consider both typical use cases and adversarial use cases where someone is trying to intentionally get the model to produce undesired output.

Other key attributes to consider can include 3rd party system dependencies, tone, knowledge cutoff date, and so on.

### Anticipate User Inputs & Edge Cases

Ensure your model behavior guide reflects target model performance across the diversity of input queries from end users, such as varying formats, intents, probes, and contexts.
This process includes queries that are well-defined or ambiguous, those that can contain errors or typos, content safety and product security threats, and other scenarios.
Codifying your desired model behavior across these scenarios---such as providing clear requests for clarification, offering relevant alternatives for out-of-scope topics---will support maintaining a consistent performance across varying user inputs.

Diverse input examples:

**Input Length and Structure**: How should the model respond to queries of varying lengths and structures?

**Ambiguous Prompts**: How should the model handle unclear or open-ended user inputs?

**Handling Technically Incapable Queries**: What approach should the model take when faced with queries that are beyond its technical capabilities?

**Content Safety and Product Security**: How will the model navigate user inputs that can involve sensitive or explicit content?
How should the model handle potential security threats or probes through user input?

#### Integrate Model Behavior Guide in AI Development

Now that your model behavior guide is in place, it's time to optimize against the defined standards.
This involves systematically testing the model's responses against the guide's criteria, and generating training data that aligns with desired behavior.

This could involve the following steps:

1. Develop a Comprehensive Evaluation & Red-teaming Approach

   Develop a strategy to evaluate the user goals, diversity of input queries, and target outputs specified by the codified Behavior policy.
   This approach should incorporate red-teaming exercises to simulate adversarial interactions, ensuring the model's robustness.

1. Generate Training Data

   Develop a strategy to create diverse, high-quality training data that aligns with the codified behavior policy.
   This strategy could incorporate targeted approaches for SFT such as query and ground-truth response generation and RL such as reward signals across model outputs.
   Training data should reflect the anticipated user goals, input queries, desired outputs, as well as edge cases and potential misuses.

1. Craft a System Prompt

   Design a system prompt that accurately conveys the model's purpose, scope, and expectations for user interactions.
   The system prompt should be informed by the codified behavior policy.

1. Disclosure and Documentation

   Like any software product, provide users documentation regarding the intended use of the model, capabilities, and any limitations they need to be aware of prior to deployment.
   Limitations should be communicated in [ModelCard++](https://developer.nvidia.com/blog/enhancing-ai-transparency-and-ethical-considerations-with-model-card/).

1. Iterate

   Refine the model through iterative cycles of evaluation, training data generation, and system prompt updates.
   Each iteration should focus on addressing identified weaknesses, improving overall performance, and enhancing adherence to the codified behavior policy.

Your model behavior guide will naturally evolve over time as you learn from users, identify novel edge cases, and expand system capabilities.
By continuously optimizing against the policy, you'll enhance the model's overall quality, reliability, and user satisfaction, ultimately delivering a more effective user experience that reflects your team's values.
