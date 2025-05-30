import os
import subprocess
import time
import signal
import sys
from tqdm import tqdm

# Base paths
BASE_DIR = "/lustre/fsw/portfolios/llmservice/users/vinitas/safety_scripts"
GARAK_DIR = os.path.join(BASE_DIR, "garak-fork-aishwarya")
LOG_DIR = os.path.join(BASE_DIR, "log/r1_llama_8b_test")
REPORT_DIR = os.path.join(BASE_DIR, "reports/r1_llama_8b_test")
CONF_DIR = os.path.join(BASE_DIR, "confs/r1_llama_8b_test")

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TENSOR_PARALLEL_SIZE = 8  

tier_1_probes = [
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

tier_2_probes = [
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
    "topic.WordnetControversial",
]

PROBES = tier_1_probes + tier_2_probes

def run_vllm():
    """Run single VLLM server using all 8 GPUs"""
    log_file = os.path.join(LOG_DIR, "vllm__DeepSeek-R1-Distill-Llama-8B__max_tokens4096.log")

    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = os.path.join(BASE_DIR, "triton_cache")
    env["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")
    env["VLLM_LOGGING_LEVEL"] = "DEBUG"
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  
    
    vllm_cmd = [
        "vllm", "serve",
        MODEL_NAME,
        "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
        "--enable-reasoning",
        "--reasoning-parser", "qwen3"
    ]
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            vllm_cmd,
            env=env,
            stdout=f,
            stderr=f
        )
    
    print("Waiting for VLLM server to start...")
    time.sleep(300)  # 5 minutes
    print("VLLM server started")
    
    return process

def run_garak_probes(vllm_process):
    """Run all Garak probes in sequence"""
    for probe in tqdm(PROBES, desc="Running Garak probes"):
        probe_name = f"probe_{probe}"
        report_path = os.path.join(REPORT_DIR, probe_name)
        conf_path = os.path.join(CONF_DIR, f"DeepSeek-R1-Distill-Llama-8B__max_tokens4096__{probe_name}.conf")
        log_path = os.path.join(LOG_DIR, f"garak__DeepSeek-R1-Distill-Llama-8B__max_tokens4096__{probe_name}.log")
        
        print(f"Running probe: {probe}")
        
        env = os.environ.copy()
        env["XDG_DATA_HOME"] = report_path
        
        # Add Garak directory to PYTHONPATH
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{GARAK_DIR}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = GARAK_DIR
        
        garak_cmd = ["garak", "--config", conf_path]
        
        with open(log_path, 'w') as f:
            subprocess.run(
                garak_cmd,
                env=env,
                stdout=f,
                stderr=f
            )
        
        print(f"Completed probe: {probe}")

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "triton_cache"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "hf_cache"), exist_ok=True)
    
    vllm_process = run_vllm()
    
    try:
        run_garak_probes(vllm_process)
    finally:
        vllm_process.terminate()
        vllm_process.wait()

    print("Aggregating Garak HTML results into CSV...")
    agg_script = os.path.join(BASE_DIR, "aggregate_results.py")
    subprocess.run(["python", agg_script], check=True)
    print("Aggregation complete.")

if __name__ == "__main__":
    main()
