# === Merge do LoRA em fp16 (modelo único para inferência) ===
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===== CONFIG BÁSICA =====
MODEL_NAME  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "/content/drive/MyDrive/amazon_ft/outputs/tinyllama11b_descfmt_lora/adapter"
SAVE_DIR    = "/content/drive/MyDrive/amazon_ft/outputs/tinyllama11b_descfmt_merged_fp16"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== PREPARAÇÃO DO TOKENIZADOR =====
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

# Tenta GPU; se OOM, cai para CPU automaticamente
def try_gpu_then_cpu():
    try:
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map=None, torch_dtype=torch.float16, trust_remote_code=True
        ).to("cuda")
        ft = PeftModel.from_pretrained(base, ADAPTER_DIR, is_trainable=False)
        merged = ft.merge_and_unload()
        me10rged.save_pretrained(SAVE_DIR); tok.save_pretrained(SAVE_DIR)
        print("Merge em GPU concluído:", SAVE_DIR)
    except Exception as e:
        print("Aviso: GPU falhou, tentando em CPU (pode demorar). Detalhe:", e)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map=None, torch_dtype=torch.float32, trust_remote_code=True
        )  # CPU em fp32 para segurança
        ft = PeftModel.from_pretrained(base, ADAPTER_DIR, is_trainable=False)
        merged = ft.merge_and_unload()
        merged.save_pretrained(SAVE_DIR); tok.save_pretrained(SAVE_DIR)
        print("Merge em CPU concluído:", SAVE_DIR)

try_gpu_then_cpu()