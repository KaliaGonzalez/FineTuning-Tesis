"""
Script de Fine-tuning para Mistral 7B con LoRA
"""

import torch
import json
import os
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

print("\n" + "=" * 70)
print("🚀 FINE-TUNING MISTRAL 7B CON LoRA")
print("=" * 70)

# === CONFIGURACIÓN ===
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
NEW_MODEL_NAME = "mistral-7b-fac-finetuned"
OUTPUT_DIR = "./results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. CARGAR DATOS CRUDOS ===
print("\n1️⃣ Cargando datos JSON...")
with open("FineTuningDatos/dataTrain.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open("FineTuningDatos/dataValidation.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)

print(f"✅ Train: {len(train_data)} ejemplos")
print(f"✅ Val: {len(val_data)} ejemplos")

# === 2. CARGAR TOKENIZADOR ===
print("\n2️⃣ Cargando tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizador cargado")

# === 3. FORMATEAR TEXTOS ===
print("\n3️⃣ Formateando datos...")


def format_text(example):
    inst = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")

    if inp and inp.strip():
        text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    else:
        text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"

    return text


train_texts = [format_text(ex) for ex in train_data]
val_texts = [format_text(ex) for ex in val_data]

print(f"✅ Textos formateados")

# === 4. TOKENIZAR ===
print("\n4️⃣ Tokenizando...")

train_encodings = tokenizer(
    train_texts,
    padding="max_length",
    max_length=2048,
    truncation=True,
    return_tensors=None,
)

val_encodings = tokenizer(
    val_texts,
    padding="max_length",
    max_length=2048,
    truncation=True,
    return_tensors=None,
)

# Convertir a Dataset
train_dataset = Dataset.from_dict(train_encodings)
val_dataset = Dataset.from_dict(val_encodings)

print(f"✅ Datos tokenizados")
print(f"   Train: {len(train_dataset)} ejemplos")
print(f"   Val: {len(val_dataset)} ejemplos")

# === 5. CARGAR MODELO ===
print("\n5️⃣ Cargando Mistral 7B...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
print("✅ Modelo cargado")

# === 6. CONFIGURAR LoRA ===
print("\n6️⃣ Configurando LoRA...")
lora_config = LoraConfig(
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
    inference_mode=False,
)
model = get_peft_model(model, lora_config)
print("✅ LoRA aplicado")

# === 7. ARGUMENTOS DE ENTRENAMIENTO ===
print("\n7️⃣ Configurando entrenamiento...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    warmup_steps=10,
    logging_steps=1,
    save_steps=50,
    eval_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    eval_strategy="steps",
    optim="adamw_torch",
    max_grad_norm=0.3,
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=True,
    report_to=[],
)
print("✅ Configuración lista")

# === 8. CREAR TRAINER ===
print("\n8️⃣ Creando Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
print("✅ Trainer listo")

# === 9. ENTRENAR ===
print("\n9️⃣ INICIANDO ENTRENAMIENTO...")
print("=" * 70)

try:
    result = trainer.train()
    print("\n✅ ¡ENTRENAMIENTO COMPLETADO!")
except KeyboardInterrupt:
    print("\n⚠️  Entrenamiento interrumpido")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()

# === 10. GUARDAR ===
print("\n🔟 Guardando modelo...")
try:
    model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)
    print(f"✅ Modelo guardado en: {NEW_MODEL_NAME}/")
except Exception as e:
    print(f"❌ Error al guardar: {e}")

print("\n" + "=" * 70)
print("🎉 ¡LISTO!")
print("=" * 70)
