# === Avaliação rápida: BASE (4-bit) vs MERGED (4-bit) ===
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, json, random, pandas as pd, torch, shutil
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ===== CONFIG BÁSICA =====
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MERGED_DIR = "/content/drive/MyDrive/amazon_ft/outputs/tinyllama11b_descfmt_merged_fp16"
PREP_DIR   = "/content/drive/MyDrive/amazon_ft/cache/prepared_descfmt"
OUT_CSV    = "/content/drive/MyDrive/amazon_ft/outputs/side_by_side_tinyllama_descfmt_merged_quick.csv"
OUT_JSON   = "/content/drive/MyDrive/amazon_ft/outputs/metrics_tinyllama_descfmt_merged_quick.json"

K        = 12
MAX_NEW  = 150

# Métricas
try:
    import evaluate
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "evaluate", "rouge-score"])
    import evaluate

# Limpa cache da GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Carrega dataset
ds = load_from_disk(PREP_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

# Configuração de 4-bit
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
)

# Diretório de offload
offload_dir = "/content/offload_eval"; shutil.rmtree(offload_dir, ignore_errors=True); os.makedirs(offload_dir, exist_ok=True)

# Carrega modelos
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", trust_remote_code=True,
    quantization_config=bnb_cfg, offload_buffers=True, offload_folder=offload_dir
)
merged = AutoModelForCausalLM.from_pretrained(
    MERGED_DIR, device_map="auto", trust_remote_code=True,
    quantization_config=bnb_cfg, offload_buffers=True, offload_folder=offload_dir
)

# Função de geração de chat
def chat_generate(model, user_text, max_new_tokens=MAX_NEW):
    prompt = tok.apply_chat_template([{"role":"user","content":user_text}],
                                     tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")  # fica no CPU; accelerate despacha
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                             pad_token_id=tok.eos_token_id, use_cache=True)
    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()

# Amostra aleatória de índices
random.seed(42)
idxs = random.sample(range(len(ds["test"])), k=min(K, len(ds["test"])))

rows, preds_base, preds_ft, refs = [], [], [], []
for i in idxs:
    ex  = ds["test"][i]
    inp = ex["input_text"]; ref = ex["target_text"]
    yb = chat_generate(base,   inp)
    ym = chat_generate(merged, inp)
    rows.append({
        "id": i,
        "input_preview": inp[:140].replace("\n"," ") + "...",
        "reference_preview": ref[:200].replace("\n"," ") + "...",
        "baseline": yb,
        "fine_tuned_merged": ym,
    })
    preds_base.append(yb); preds_ft.append(ym); refs.append(ref)

# Salva CSV
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print("CSV salvo em:", OUT_CSV)

rouge = evaluate.load("rouge")
r_base = rouge.compute(predictions=preds_base, references=refs)
r_ft   = rouge.compute(predictions=preds_ft,   references=refs)

metrics = {"rouge_base": {k: float(v) for k,v in r_base.items()},
           "rouge_ft":   {k: float(v) for k,v in r_ft.items()}}
with open(OUT_JSON, "w") as f:
    json.dump(metrics, f, indent=2)
print("Métricas (ROUGE) salvas em:", OUT_JSON)
print("Resumo:", metrics)
