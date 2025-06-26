# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import subprocess

from openai import OpenAI


class VLLMServerProcess:
    def __init__(self, 
                 proc: subprocess.Popen,
                 log_filepath: str,
                 model_name: str = "test-model",
                 port: int = 5000):
        self.proc = proc
        self.log_filepath = log_filepath
        self.model_name = model_name
        self.port = port

    def is_alive(self):
        if self.proc.poll() is None:
            return True
        return False

    def print_log(self, n=10):
        subprocess.run(["tail", f"-{n}", self.log_filepath])

    def stop(self):
        self.proc.terminate()
        self.proc.wait()

    def send_query(self,
                   user_prompt: str,
                   system_prompt: str = "",
                   temperature: float = 0.6,
                   top_p: float = 0.95):
        # Set up your endpoint and model
        client = OpenAI(
            base_url=f"http://localhost:{str(self.port)}/v1",
            api_key="EMPTY"
        )
        
        # Send prompt and get response
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message.content.strip()


class VLLMLauncher:
    def __init__(self,
                 total_num_gpus: int):
        self.total_num_gpus = total_num_gpus

    def launch(self,
               model_name_or_path: str,
               gpu_devices: str,
               port: int,
               served_model_name: str,
               log_filepath: str,
               enable_reasoning: bool = False,
               use_v1: bool = False,
               max_model_len: int = 131072,
               seed: int = 1,
               vllm_host: str = "0.0.0.0"):
        """
        Launch a vLLM server for the given model.
        """
        num_gpu_devices = len([int(gpu) for gpu in gpu_devices.split(",")])
        assert num_gpu_devices <= self.total_num_gpus, "Number of GPUs requested is greater than the total number of GPUs available."

        os.environ.update({
            'VLLM_ENGINE_ITERATION_TIMEOUT_S': '36000',
            'VLLM_ALLOW_LONG_MAX_MODEL_LEN': '1',
            'VLLM_TENSOR_PARALLEL_SIZE': str(num_gpu_devices),
        })
        command = [
              'python3', '-m', 'vllm.entrypoints.openai.api_server',
              '--model', model_name_or_path,
              '--trust-remote-code',
              '--seed', str(seed),
              '--host', vllm_host,
              '--port', str(port),
              '--served-model-name', served_model_name,
              '--max-model-len', str(max_model_len),
              '--tensor-parallel-size', str(num_gpu_devices),
              '--download-dir', os.environ['HF_HOME']
        ]

        if enable_reasoning:
            command.extend([
                '--enable-reasoning',
                '--reasoning-parser', 'custom',
            ])

        vllm_server = subprocess.Popen(command,
                                       env={**os.environ,
                                            'VLLM_USE_V1': str(int(use_v1)),
                                            'CUDA_VISIBLE_DEVICES': gpu_devices},
                                            stdout=open(log_filepath, "w"),
                                            stderr=subprocess.STDOUT)
        
        return VLLMServerProcess(proc=vllm_server,
                                 log_filepath=log_filepath,
                                 model_name=served_model_name,
                                 port=port)

    def stop_all(self):
        subprocess.run(['pkill', '-f', 'vllm.entrypoints.openai.api_server'])