# Safety for Agentic AI: Developer Blueprint Notebooks

This directory has the following notebooks to walk you through the core features of adding safety to an open model:

- [Set up the environment by configuring and installing required packages](./Step0_Setup.ipynb)
- [Evaluate the base model for safety and accuracy using NeMo Eval](./Step1_Evaluation.ipynb)
- [Post-train the base model to add content safety and preserve accuracy](./Step2_Safety_Post_Training.ipynb)
- [Rerun the same safety and accuracy evaluation and view the model safety report card to see how the model has improved](./Step3_Post_Training_Eval.ipynb)
- [Inference with NeMo Guardrails](./Step4_Run_Inference_with_NeMo_Guardrails_Docker.ipynb)
    - To run the inference notebook, you must stop the NeMo Framework container that was used for the previous notebooks and start containers for NeMo Guardrails microservice and NIM for LLMs microservice. 
    - Refer to [The Setup in the last section of this README.md](#set-up-for-running-inference-with-nemo-guardrails-and-llm-agnostic-nim) for more information.
    - Once you start working with the [Inference Notebook](./Step4_Run_Inference_with_NeMo_Guardrails_Docker.ipynb), you will not be able to re-visit or run any of the previous notebooks. In order to do so, you need to again shut down the running docker services and start the nemo framework container. Follow the steps [to restart the NeMo Framework Services](#restarting-the-nemo-framework-docker-after-stopping)
 

These notebooks were developed using 8 x H100 80GB GPUs.
You must use the same or similar computational resources and follow the notebooks in sequential order (Step0->Step1->Step2->Step3->Step4).

You can use [NVIDIA Brev](https://developer.nvidia.com/brev) to launch an 8 x H100 instance.

You will need to generate the following API keys:

- `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com/)
- `HF_TOKEN` from [HF Hub](https://huggingface.co/models)

For post-training, you can optionally use [Weights & Biases](https://wandb.ai/home) for experiment tracking.

## Using the Brev Launchable

- Go to <https://build.nvidia.com/nvidia/safety-for-agentic-ai>.
- Click **Deploy on Cloud** to go the launchable page on Brev.
- Click **Deploy Launchable** to start an instance that can run the notebooks.

## Manual Set Up Using Brev

### Download and Edit Docker Compose File

1. Download the <https://raw.githubusercontent.com/NVIDIA-AI-Blueprints/safety-for-agentic-ai/refs/heads/main/deploy/docker-compose.yaml> file.

1. Edit the `docker-compose.yaml` file.

   - Add your NGC API key

     ```yaml
      ...
      environment:
         - API_KEY="<ngc-api-key>"
      ```

   - If you use an instance that is not a CRUSOE instance, such as GCP, comment out the `volumes` section:

     ```yaml
     #    volumes:
     #      - /ephemeral:/ephemeral
     ```

### Create and Start Instance

1. Go to <https://brev.nvidia.com/>.

1. Create **New Instance**.

1. Choose 8 x H100 or A100 80GB on CRUSOE or other providers.

   ![](images/select-instances.png)

1. Click **Configure**.

   ![](images/configure.png)

1. Click **Docker Compose**.

1. Upload the `docker-compose.yaml` that you updated with your NGC API key.

1. Click **Add Registry** and specify the following values:

   - Registry URL: `nvcr.io`
   - Username: `$oauthtoken`
   - Password: `<ngc-api-key>`

   ![](images/docker-compose.png)

1. Click **Validate**.

1. Click **Deploy**.

Starting an instance requires approximately 15 to 20 minutes.

### Post-Launch Configuration

1. Access your instance page.

1. Click **Share a Service** and add port `8888`.

   ![](images/share-service.png)

   Repeat this step and add port `8501` to access the report card that is part of the fourth notebook.
   The report card shows the improved evaluation scores after post-training the model.

1. Click the shareable URL for port `8888` to access JupyterLab.

   ![](images/secure-links.png)

1. Download the GitHub repository as a ZIP file:

   <https://github.com/NVIDIA-AI-Blueprints/safety-for-agentic-ai/archive/refs/heads/main.zip>

1. Upload the ZIP file to JupyterLab.

1. Click **Terminal** on the **Launcher** page and then unzip the archive.

   ```shell
   $ unzip safety-for-agentic-ai-main.zip
   ```

   The notebooks are located in the `safety-for-agentic-ai/notebooks` directory.
   Continue the set up by running the `Step0_Setup.ipynb` notebook.

2. The same `8888` port can be used to run the inference part of this Launchable. 

### Set Up for running Inference with NeMo Guardrails and LLM Agnostic NIM

**NOTE : The instructions in this section are necessary for following Notebook 4, which covers running inference after deploying the fine-tuned checkpoint from Notebook 2.**

1. To run inference on the post-trained model, first SSH into your Brev CLI following the instructions and commands on the Brev Console.

2. Then shut down the running NeMo Framework container - 
   ```
   docker ps -a
   docker stop <container-id>
   ```

3. Run the `inference_setup.sh` in the `safety-for-agentic-ai/notebooks/scripts` with the following commands to shut down the running docker running on port `8888` and install `jupyterlab` and `docker-compose`

   ```
   chmod +x inference_setup.sh
   ./inference_setup.sh
   ```

4. After you run the above setup script, go back to the Jupyterlab instance running on port `8888` and refresh the page.

5. Once there is no running docker services, now you can start working with the [Inference Notebook](./Step4_Run_Inference_with_NeMo_Guardrails_Docker.ipynb)
   - Remember to follow the steps in the notebook to do a docker login with your `NVIDIA_API_KEY` using `docker login nvcr.io` on the terminal of the jupyterlab server

### Restarting the NeMo Framework Docker After Stopping

1. You can view the stopped NeMo Framework Docker service in the list of images by running the following command:

```
docker image ls
```
This should output the following:

```
REPOSITORY              TAG       IMAGE ID       CREATED      SIZE
workspace-nemo_safety   latest    aaf7ca33db2c   9 days ago   79.7GB

```
2. Run the following docker command to restart the initial docker container

```
docker run -it --rm \
  --gpus all \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=1g \
  -p 8888:8888 \
  -p 8501:8501 \
  -u root \
  -v /ephemeral:/ephemeral \
  -w /ephemeral/workspace/ \
  workspace-nemo_safety:latest \
  bash -c "python -m jupyter lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.iopub_data_rate_limit=1.0e10"

```

3. You should now be able to re-visit Notebooks 0–3.