"""Entrenamiento LoRA para Mistral 7B.

Este script entrena usando:
- Train: FineTuningDatos/dataTrain.json
- Validation: FineTuningDatos/dataValidation.json

Salida:
- Adaptador LoRA + tokenizer en la carpeta indicada por NEW_MODEL_NAME
"""

import torch
import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# === CONFIGURACIÓN ===
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
NEW_MODEL_NAME = "mistral-7b-fac-finetuned"
OUTPUT_DIR = "./results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verificar GPU
print("\n" + "=" * 70)
if torch.cuda.is_available():
    print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
else:
    print("❌ NO HAY GPU - Usa CPU (LENTO)")
print("=" * 70)

# Cargar datos
print("\n[1/5] Cargando datos...")
with open("FineTuningDatos/dataTrain.json", encoding="utf-8") as f:
    train_data = json.load(f)
with open("FineTuningDatos/dataValidation.json", encoding="utf-8") as f:
    val_data = json.load(f)

print(f"Train: {len(train_data)}")
print(f"Val: {len(val_data)}")

# Cargar tokenizador
print("\n[2/5] Cargando tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Formatear textos
print("\n[3/5] Preparando textos...")
train_texts = []
for d in train_data:
    if d.get("input"):
        t = f"### Instruction:\n{d['instruction']}\n\n### Input:\n{d['input']}\n\n### Response:\n{d['output']}"
    else:
        t = f"### Instruction:\n{d['instruction']}\n\n### Response:\n{d['output']}"
    train_texts.append(t)

val_texts = []
for d in val_data:
    if d.get("input"):
        t = f"### Instruction:\n{d['instruction']}\n\n### Input:\n{d['input']}\n\n### Response:\n{d['output']}"
    else:
        t = f"### Instruction:\n{d['instruction']}\n\n### Response:\n{d['output']}"
    val_texts.append(t)

# Tokenizar
print("\n[4/5] Tokenizando...")
train_tokens = tokenizer(
    train_texts, padding="max_length", max_length=2048, truncation=True
)
val_tokens = tokenizer(
    val_texts, padding="max_length", max_length=2048, truncation=True
)

train_dataset = Dataset.from_dict(train_tokens)
val_dataset = Dataset.from_dict(val_tokens)

# Cargar modelo
print("\n[5/5] Cargando Mistral 7B + LoRA...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

# Entrenar
print("\n" + "=" * 70)
print("INICIANDO ENTRENAMIENTO...")
print("=" * 70 + "\n")

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        warmup_steps=10,
        logging_steps=1,
        save_steps=100,
        eval_steps=100,
        save_total_limit=1,
        load_best_model_at_end=False,
        optim="adamw_torch",
        max_grad_norm=0.3,
        weight_decay=0.01,
        fp16=True,
        gradient_checkpointing=True,
        report_to=[],
    ),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# Guardar
print("\nGuardando...")
model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)

print("\n✅ ¡LISTO!")
