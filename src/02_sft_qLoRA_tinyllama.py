# === Fine-tuning QLoRA no TinyLlama ===
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, types, torch
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          DataCollatorForLanguageModeling, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===== CONFIG BÁSICA =====
MODEL_NAME    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PREP_DIR      = "/content/drive/MyDrive/amazon_ft/cache/prepared_descfmt"
OUT_DIR       = "/content/drive/MyDrive/amazon_ft/outputs/tinyllama11b_descfmt_lora"
ADAPTER_DIR   = f"{OUT_DIR}/adapter"
os.makedirs(OUT_DIR, exist_ok=True)

ds = load_from_disk(PREP_DIR)

# Carrego o tiny já compactado em 4bit
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
)

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb, device_map="auto", trust_remote_code=True
)
base = prepare_model_for_kbit_training(base)

# LoRA abrangente (garante que o modelo "escute" o adapter)
lora_cfg = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules="all-linear", modules_to_save=["lm_head","embed_tokens"]
)
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()

# Formatação e tokenização
def format_chat(example):
    messages = [
        {"role":"user", "content": example["input_text"]},
        {"role":"assistant", "content": example["target_text"]},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

# ===== PREPARAÇÃO DO DATASET =====
train_txt = ds["train"].map(format_chat, remove_columns=ds["train"].column_names)
def tok_fn(e): return tok(e["text"], truncation=True, max_length=512)
train_tok = train_txt.map(tok_fn, batched=True, remove_columns=["text"])
collator = DataCollatorForLanguageModeling(tok, mlm=False)

# ===== ARGS DE TREINO =====
args = TrainingArguments(
    output_dir=OUT_DIR,
    max_steps=1300,
    num_train_epochs=1,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    gradient_accumulation_steps=8,
    per_device_train_batch_size=2,
    logging_steps=20,
    eval_strategy="no",
    save_steps=1300,
    save_total_limit=1,
    bf16=False, fp16=True,
    report_to="none",
    optim="adamw_torch",
)

trainer = Trainer(model=model, args=args, train_dataset=train_tok, data_collator=collator)

# Patch accelerate (evita erro "optimizer.train()")
try:
    from accelerate.optimizer import AcceleratedOptimizer
    _noop=lambda self,*a,**k: None
    AcceleratedOptimizer.train = types.MethodType(_noop, AcceleratedOptimizer)
    AcceleratedOptimizer.eval  = types.MethodType(_noop,  AcceleratedOptimizer)
except: pass

trainer.train()

# ===== SALVANDO O ADAPTER =====
os.makedirs(ADAPTER_DIR, exist_ok=True)
model.save_pretrained(ADAPTER_DIR)
print("Adapter salvo em:", ADAPTER_DIR)