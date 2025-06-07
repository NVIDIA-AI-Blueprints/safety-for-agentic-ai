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
import json 
import argparse
import pandas as pd
from collections import Counter
from pandarallel import pandarallel
from utils import moderate_chat_with_nim, parse_aegis_v2_response, parse_aegis_v2_response_categories
from transformers import AutoTokenizer
from tqdm.notebook import tqdm
from typing import Tuple, List

def process_reasoning_traces(df, response_column="model_output", end_of_thought_token="</think>"):
    df[response_column + "_no_think"] = ""
    df.loc[df[response_column].str.contains(end_of_thought_token), response_column + "_no_think"] = df.loc[df[response_column].str.contains(end_of_thought_token), response_column].apply(lambda x: x.split(end_of_thought_token)[1].strip())
    df.loc[~df[response_column].str.contains(end_of_thought_token), response_column + "_no_think"] = df.loc[~df[response_column].str.contains(end_of_thought_token), response_column].apply(lambda x: x)
    assert len(df[df[response_column + "_no_think"] == ""]) == 0
    return df
    
def main(input_file):
    df = pd.read_csv(input_file)
    base_dir = os.path.dirname(input_file)
    if 'Response Safety 0' in df.columns:
        df.drop(columns=['Response Safety 0', 'Safety Categories 0'], inplace=True)

    prompt_column = "prompt"
    response_column = "model_output"
    end_of_thought_token = "</think>"

    class Args:
        prompt_format_type = "aegis_v2"
        model_name = "llama-3.1-nemoguard-8b-content-safety"
        host = "localhost"
        port = 6000

        def __init__(self, prompt_format_type, judge_model_name, judge_url, judge_url_port):
            self.prompt_format_type = prompt_format_type
            self.model_name = judge_model_name
            self.host = judge_url
            self.port = judge_url_port

    inference_args = Args(
        prompt_format_type="aegis_v2",
        judge_model_name="llama-3.1-nemoguard-8b-content-safety",
        judge_url="localhost",
        judge_url_port=6000
    )

    tqdm.pandas()

    prompt_column = "prompt"
    response_column = "model_output"
    end_of_thought_token = "</think>"

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    if df[response_column].str.contains(end_of_thought_token).any():
        df = process_reasoning_traces(df, response_column=response_column)
        response_column = response_column + "_no_think"

    pandarallel.initialize(nb_workers=20, progress_bar=True)

    df[f"aegis_v2_safety_generation"] = df.parallel_apply(
        lambda x: moderate_chat_with_nim(x[prompt_column], x[response_column], tokenizer=tokenizer, args=inference_args), axis=1
    )

    df["Response Safety 0"] = df["aegis_v2_safety_generation"].apply(parse_aegis_v2_response)
    df["Safety Categories 0"] = df["aegis_v2_safety_generation"].apply(parse_aegis_v2_response_categories)

    df.to_csv(f"{base_dir}/output.csv", index=False)

    d = {k: int(v) for k, v in df["Response Safety 0"].value_counts().items()}
    with open(f"{base_dir}/metrics.json", 'w') as f:
        json.dump(d, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()

    main(args.input_file)
