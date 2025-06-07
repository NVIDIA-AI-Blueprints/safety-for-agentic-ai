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

import argparse
import copy
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import signal
import subprocess
import sys
import time
from typing import List
from pathlib import Path
import shutil
import yaml
import csv
from bs4 import BeautifulSoup
from pathlib import Path

import openai
from tqdm import tqdm


TIER1_PROBES = [
    "ansiescape.AnsiRaw",
    "dan.DanInTheWild",
    "encoding.InjectBase64",
    "encoding.InjectUU",
    "exploitation.JinjaTemplatePythonInjection",
    "exploitation.SQLInjectionEcho",
    "goodside.Tag",
    "grandma.Slurs",
    "grandma.Substances",
    "latentinjection.LatentInjectionFactSnippetEiffel",
    "latentinjection.LatentInjectionFactSnippetLegal",
    "latentinjection.LatentInjectionReport",
    "latentinjection.LatentInjectionResume",
    "latentinjection.LatentInjectionTranslationEnFr",
    "latentinjection.LatentInjectionTranslationEnZh",
    "latentinjection.LatentJailbreak",
    "latentinjection.LatentWhois",
    "latentinjection.LatentWhoisSnippet",
    "leakreplay.GuardianComplete",
    "leakreplay.LiteratureComplete",
    "leakreplay.NYTComplete",
    "leakreplay.PotterComplete",
    "packagehallucination.JavaScript",
    "packagehallucination.Python",
    "packagehallucination.Ruby",
    "packagehallucination.Rust",
    "realtoxicityprompts.RTPBlank",
    "xss.ColabAIDataLeakage",
    "xss.MarkdownImageExfil",
    "xss.MdExfil20230929",
    "xss.StringAssemblyDataExfil"
]

TIER2_PROBES = [
    "atkgen.Tox",
    "continuation.ContinueSlursReclaimedSlurs",
    "dan.Ablation_Dan_11_0",
    "dan.AutoDANCached",
    "divergence.Repeat",
    "encoding.InjectAscii85",
    "encoding.InjectBase16",
    "encoding.InjectBase2048",
    "encoding.InjectBase32",
    "encoding.InjectBraille",
    "encoding.InjectEcoji",
    "encoding.InjectHex",
    "encoding.InjectMorse",
    "encoding.InjectNato",
    "encoding.InjectROT13",
    "encoding.InjectZalgo",
    "goodside.ThreatenJSON",
    "goodside.WhoIsRiley",
    "grandma.Win10",    
    "grandma.Win11",
    "leakreplay.GuardianCloze",
    "leakreplay.LiteratureCloze",
    "leakreplay.NYTCloze",
    "leakreplay.PotterCloze",
    "lmrc.Bullying",
    "lmrc.Deadnaming",
    "lmrc.QuackMedicine",
    "lmrc.SexualContent",
    "lmrc.Sexualisation",
    "lmrc.SlurUsage",
    "malwaregen.Evasion",
    "malwaregen.Payload",
    "malwaregen.SubFunctions",
    "malwaregen.TopLevel",
    "misleading.FalseAssertion",
    "phrasing.FutureTense",
    "phrasing.PastTense",
    "promptinject.HijackHateHumans",
    "promptinject.HijackKillHumans",
    "promptinject.HijackLongPrompt",
    "snowball.GraphConnectivity",
    "suffix.GCGCached",
    "tap.TAPCached",
    "topic.WordnetControversial"
]


def run_garak_probes(target_probes: List[str],
                     report_dir: str,
                     conf_dir: str,
                     log_dir: str,
                     max_workers: int = 4):
    """Run Garak probes in parallel using ThreadPoolExecutor"""
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_probe = {
            executor.submit(run_single_probe, probe, report_dir, conf_dir, log_dir): probe 
            for probe in target_probes
        }
        
        for future in tqdm(as_completed(future_to_probe), total=len(target_probes), desc="Running probes"):
            probe = future_to_probe[future]
            try:
                result = future.result()
                print(f"Completed probe: {probe}")
            except Exception as e:
                print(f"Error running probe {probe}: {str(e)}")

def run_single_probe(probe: str,
                     report_dir: str,
                     conf_dir:str,
                     log_dir:str):
    """Run a single Garak probe"""
    report_path = os.path.abspath(os.path.join(report_dir, probe))
    if not os.path.exists(report_path):
        os.makedirs(report_path)
    conf_path = os.path.abspath(os.path.join(conf_dir, f"{probe}.yaml"))
    log_path = os.path.abspath(os.path.join(log_dir, f"{probe}.log"))
    
    #env = os.environ #.copy()
    env = copy.deepcopy(dict(os.environ))
    env["XDG_DATA_HOME"] = report_path

    print(f"probe={probe}, report_path={report_path}")
    
    garak_cmd = ["garak", "--config", conf_path]
    
    with open(log_path, 'w') as f:
        subprocess.run(
            garak_cmd,
            env=env,
            stdout=f,
            stderr=f,
            check=True
        )


def extract_probe_section(panel):
    # probe
    h3 = panel.find('h3', class_=lambda c: c and c.startswith('defcon'))
    probe = ""
    if h3 and 'probe:' in h3.text:
        probe = h3.text.split('probe:')[1].split('-')[0].strip()
    # detector and aggregate_defcon
    h4 = panel.find('h4', class_=lambda c: c and c.startswith('defcon'))
    detector = ""
    aggregate_defcon = ""
    if h4:
        p = h4.find('p', class_='left')
        if p and 'detector:' in p.text:
            detector = p.text.split('detector:')[1].strip()
        dc_span = h4.find('span', class_=lambda c: c and c.startswith('defcon') and 'dc' in c)
        if dc_span and 'DC:' in dc_span.text:
            aggregate_defcon = dc_span.text.split('DC:')[1].strip()
    # pass_rate and pass_rate_defcon
    pass_rate = ""
    pass_rate_defcon = ""
    abs_score_p = panel.find('span', string='absolute score:')
    if abs_score_p:
        b = abs_score_p.find_next('b')
        if b:
            pass_rate = b.text.split('%')[0].strip() + "%" if '%' in b.text else b.text.strip()
        dc = abs_score_p.find_next('span', class_=lambda c: c and c.startswith('defcon') and 'dc' in c)
        if dc and 'DC:' in dc.text:
            pass_rate_defcon = dc.text.split('DC:')[1].strip()
    # z_score, z_score_status, z_score_defcon
    z_score = ""
    z_score_status = ""
    z_score_defcon = ""
    rel_score_p = panel.find('span', string='relative score (Z):')
    if rel_score_p:
        b = rel_score_p.find_next('b')
        if b:
            zscore_text = b.text.strip()
            if ' ' in zscore_text:
                z_score = zscore_text.split(' ')[0]
            if '(' in zscore_text and ')' in zscore_text:
                z_score_status = zscore_text.split('(')[1].split(')')[0]
        dc = rel_score_p.find_next('span', class_=lambda c: c and c.startswith('defcon') and 'dc' in c)
        if dc and 'DC:' in dc.text:
            z_score_defcon = dc.text.split('DC:')[1].strip()
    return {
        "probe": probe,
        "detector": detector,
        "pass_rate": pass_rate,
        "z_score": z_score,
        "z_score_status": z_score_status,
        "z_score_defcon": z_score_defcon,
        "pass_rate_defcon": pass_rate_defcon,
        "aggregate_defcon": aggregate_defcon
    }

def extract_from_html(html_file):
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    # Find all probe panels
    results = []
    found = 0
    for button in soup.find_all('button', class_=lambda c: c and 'accordion' in c.split()):
        panel = button.find_next('div', class_='panel')
        if panel:
            h3 = panel.find('h3', class_=lambda c: c and c.startswith('defcon'))
            if h3:
                found += 1
                row = extract_probe_section(panel)
                if row['probe']:
                    results.append(row)
    return results

def find_all_reports(reports_root):
    html_files = []
    for root, dirs, files in os.walk(reports_root):
        for file in files:
            if file.startswith('garak.') and file.endswith('.report.html'):
                html_files.append(os.path.join(root, file))
    return html_files

def create_csv(reports_root: str,
               output_csv: str):
    COLUMNS = [
        "probe",
        "detector",
        "pass_rate",
        "z_score",
        "z_score_status",
        "z_score_defcon",
        "pass_rate_defcon",
        "aggregate_defcon"
    ]

    all_rows = []
    html_files = find_all_reports(reports_root)
    for html_file in html_files:
        rows = extract_from_html(html_file)
        all_rows.extend(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Aggregated {len(all_rows)} rows from {len(html_files)} reports into {output_csv}")


if __name__ == "__main__":
    """
      * Launch vLLM server:

        export MODEL_NAME_OR_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        export MODEL_NAME_OR_PATH="/lustre/fsw/llmservice_nemo_mlops/users/ysuhara/work/gitlab/NeMo-Safety/notebooks/shared/checkpoint-200"
        export VLLM_TENSOR_PARALLEL_SIZE=8
        export HF_HOME=./workspace/cache/huggingface
        export LOG_DIR=./workspace/logs
        python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME_OR_PATH" \
        --trust-remote-code \
        --seed 1 \
        --host "$VLLM_HOST" \
        --port 5001 \
        --served-model-name "test-model" \
        --enable-reasoning \
        --reasoning-parser qwen3 \
        --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
        --download-dir="$HF_HOME"      

      * Update garak_base_config.yaml accordingly
        ...
        model_type: openai.OpenAICompatible
        model_name: test-model
        generators:
            openai: 
            OpenAICompatible:
                uri: http://localhost:5001/v1
                api_key: "EMPTY"
                model: test-model
                max_tokens: 4096
                temperature: 0.6
                top_p: 0.95
                use_chat: True
                skip_seq_start: ""
                skip_seq_end: "</think>"

      * Usage

      python run_garak.py --output_basedir ./DeepSeek-R1-Distill-Llama-8B-checkpoint-200-full --base_config_path ./garak_base_config.yaml
      python run_garak.py --output_basedir ./DeepSeek-R1-Distill-Llama-8B-checkpoint-200-3probes --base_config_path ./garak_base_config.yaml --max_workers 4 --target_probes dan.DanInTheWild grandma.Slurs grandma.Substances

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_basedir", type=str, required=True)
    parser.add_argument("--base_config_path", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=5)
    parser.add_argument("--target_probes", type=str, nargs="+", help="List of target probes to run. If not specified, runs TIER1_PROBES + TIER2_PROBES")
    args = parser.parse_args()
    
    # * Uncomment the line below if you'd like to run the full Garak evaluation
    # TARGET_PROBES = ["dan.DanInTheWild", "grandma.Slurs", "grandma.Substances"]
    if args.target_probes is None:
        print("Tier1+Tier2")
        TARGET_PROBES = TIER1_PROBES + TIER2_PROBES
    elif len(args.target_probes) == 1 and args.target_probes[0] == "tier1":
        print("Tier1")        
        TARGET_PROBES = TIER1_PROBES        
    elif len(args.target_probes) == 1 and args.target_probes[0] == "tier2":    
        print("Tier2")        
        TARGET_PROBES = TIER2_PROBES
    else:
        print(", ".join(args.target_probes))
        TARGET_PROBES = args.target_probes

    report_dir = os.path.join(args.output_basedir, "reports")
    conf_dir = os.path.join(args.output_basedir, "configs")
    log_dir = os.path.join(args.output_basedir, "logs")
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    if not os.path.exists(conf_dir):
        os.makedirs(conf_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(args.base_config_path) as fin:
        garak_base_config = yaml.safe_load(fin)
    
    for target_probe in TARGET_PROBES:
        garak_config = copy.deepcopy(garak_base_config)
        garak_config["run"]["generations"] = args.num_generations
        garak_config["plugins"]["probe_spec"] = target_probe
        new_config_filepath = os.path.join(conf_dir, f"{target_probe}.yaml")
        with open(new_config_filepath, "w") as fout:
            yaml.dump(garak_config, fout, sort_keys=False)
    
    # Run Garak evaluation
    run_garak_probes(target_probes=TARGET_PROBES, report_dir=report_dir, conf_dir=conf_dir, log_dir=log_dir, max_workers=args.max_workers)
    create_csv(reports_root=report_dir, output_csv=os.path.join(report_dir, "garak_results.csv"))
