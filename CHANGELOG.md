# Change Log

## v0.2 - 2025-07-15

### Features

- The blueprint is available as a Brev launchable to simplify getting started.
  A launchable is a preconfigured, fully optimized compute and software environment
  that is fast and easy to deploy.
  Go to <https://build.nvidia.com/nvidia/safety-for-agentic-ai> and click **Deploy on Cloud**.

- Added a `deploy/docker-compose-guardrails.yaml` file and notebook to run inference with 
  NVIDIA LLM-agnostic NIM and NeMo Guardrails.
  The LLM-agnostic NIM enables you to deploy of a broad range of models and offers maximum flexibility.

- Updated the garak dependency to v0.12.0.
  This update adds a z-score for the grandma slurs probe.
  The evaluation now checks for an improvement with this probe.
