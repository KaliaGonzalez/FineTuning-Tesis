"""
Script de Fine-tuning con LoRA para TinyLlama
Versión rápida y compatible con GPU local
Compatible con Python 3.13
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import os

# === CONFIGURACIÓN ===
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Modelo pequeño y rápido
NEW_MODEL_NAME = "tinylla-fac-finetuned"
OUTPUT_DIR = "./results"

# Crear directorio de salida si no existe
import os

os.makedirs(OUTPUT_DIR, exist_ok=True)

data_files = {
    "train": "FineTuningDatos/dataTrain.json",
    "validation": "FineTuningDatos/dataValidation.json",
}

print("=" * 60)
print("🚀 INICIANDO FINE-TUNING CON LORA")
print("=" * 60)

# === 1. CARGAR TOKENIZADOR ===
print("\n1️⃣ Cargando tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✅ Tokenizador cargado")

# === 2. CARGAR DATASET ===
print("\n2️⃣ Cargando dataset...")
dataset = load_dataset("json", data_files=data_files)
print(
    f"✅ Dataset cargado - Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}"
)

# === 3. CARGAR MODELO BASE ===
print("\n3️⃣ Cargando modelo base (esto puede tardar 2-3 minutos)...")

# === 4. CARGAR MODELO ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float32,
    trust_remote_code=True,
)
print("✅ Modelo cargado")

# === 5. CONFIGURACIÓN LORA ===
print("\n5️⃣ Configurando LoRA...")
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "v_proj",
    ],  # TinyLlama tiene menos capas
)
model = get_peft_model(model, peft_config)
print("✅ LoRA configurado")

# === 6. FUNCIÓN DE FORMATTEO ===
print("\n6️⃣ Preparando formato de datos...")


def formatting_prompts_func(example):
    """Convierte los datos al formato instruction/input/output"""
    output_texts = []
    for i in range(len(example["instruction"])):
        instruction = example["instruction"][i]
        input_text = example["input"][i]
        output = example["output"][i]

        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        output_texts.append(text)
    return {"text": output_texts}


# Aplicar formatteo
dataset = dataset.map(formatting_prompts_func, batched=True)


# Tokenizar
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        max_length=2048,
        truncation=True,
        padding="max_length",
    )


dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["instruction", "input", "output", "text"],
)
print("✅ Datos preparados")

# === 8. ARGUMENTOS DE TRAINING ===
print("\n8️⃣ Configurando argumentos de training...")
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # REDUCIDO para evitar OOM
    gradient_accumulation_steps=4,  # Efecto de batch=4
    optim="paged_adamw_32bit",
    save_steps=10000,  # Guardar solo al final para ahorrar espacio en disco (red universitaria)
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=2,  # Solo guardar 2 checkpoints para ahorrar disco
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)
print("✅ Argumentos configurados")

# === 9. TRAINER ===
print("\n9️⃣ Creando trainer...")
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
print("✅ Trainer creado")

# === 10. ENTRENAR ===
print("\n🔟 Iniciando entrenamiento...")
print("=" * 60)
try:
    trainer.train()
    print("\n✅ ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
except Exception as e:
    print(f"❌ Error durante entrenamiento: {e}")
    print("Guardando lo que se pudo entrenar...")
print("=" * 60)

# === 11. GUARDAR MODELO ===
print("\n✅ Guardando modelo...")
try:
    import shutil

    if os.path.exists(NEW_MODEL_NAME):
        shutil.rmtree(NEW_MODEL_NAME)
    trainer.model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)
    print(f"✅ Modelo guardado exitosamente en {NEW_MODEL_NAME}")
except Exception as e:
    print(f"❌ Error guardando: {e}")

print("\n" + "=" * 60)
print(f"🎉 ¡ENTRENAMIENTO COMPLETADO!")
print(f"📁 Modelo guardado en: {NEW_MODEL_NAME}/")
print("=" * 60)
