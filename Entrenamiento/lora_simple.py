"""
Script PROFESIONAL de Fine-tuning con LoRA para Mistral 7B
Entrenamiento robusto con validación correcta
Compatible con Python 3.13 y GPU local
"""

import torch
import json
import warnings
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import os

warnings.filterwarnings("ignore")

# === CONFIGURACIÓN ===
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
NEW_MODEL_NAME = "mistral-7b-fac-finetuned"
OUTPUT_DIR = "./results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 70)
print("🚀 FINE-TUNING MISTRAL 7B CON LoRA")
print("=" * 70)

# === 1. VERIFICAR Y CARGAR DATASETS ===
print("\n1️⃣ Cargando datasets...")
data_files = {
    "train": "FineTuningDatos/dataTrain.json",
    "validation": "FineTuningDatos/dataValidation.json",
}

# Validar que existen
for key, path in data_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No encontrado: {path}")

# Cargar
dataset = load_dataset("json", data_files=data_files)
print(f"✅ Train: {len(dataset['train'])} ejemplos")
print(f"✅ Val: {len(dataset['validation'])} ejemplos")

# === 2. CARGAR TOKENIZADOR ===
print("\n2️⃣ Cargando tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, add_eos_token=True
)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizador cargado")

# === 3. PREPARAR DATOS ===
print("\n3️⃣ Preparando datos...")


def prepare_data(examples):
    """Formatea y tokeniza en una sola pasada"""
    texts = []

    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        output = examples["output"][i]

        # Formatear
        if input_text and input_text.strip():
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        texts.append(prompt)

    # Tokenizar TODO de una vez
    tokenized = tokenizer(
        texts,
        padding="max_length",
        max_length=2048,
        truncation=True,
        return_tensors=None,
    )

    return tokenized


# Aplicar preparación
dataset = dataset.map(
    prepare_data,
    batched=True,
    batch_size=32,
    remove_columns=dataset["train"].column_names,
    desc="Preparando datos",
)

print(f"✅ Datos preparados")
print(f"   Train: {len(dataset['train'])} ejemplos")
print(f"   Val: {len(dataset['validation'])} ejemplos")

# === 4. CARGAR MODELO ===
print("\n4️⃣ Cargando Mistral 7B...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
print("✅ Modelo cargado")

# === 5. CONFIGURAR LoRA ===
print("\n5️⃣ Configurando LoRA...")
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

# === 6. ARGUMENTOS DE ENTRENAMIENTO ===
print("\n6️⃣ Configurando entrenamiento...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Batch muy pequeño para evitar OOM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # Efecto de batch 4
    learning_rate=5e-4,
    warmup_steps=20,
    logging_steps=1,  # Log cada paso (para ver progreso)
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
    report_to=[],  # Sin wandb/tensorboard para ir más rápido
    seed=42,
)
print("✅ Configuración lista")

# === 7. CREAR TRAINER ===
print("\n7️⃣ Creando Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
print("✅ Trainer listo")

# === 8. ENTRENAR ===
print("\n8️⃣ INICIANDO ENTRENAMIENTO...")
print("=" * 70)

try:
    result = trainer.train()
    print("\n✅ ¡ENTRENAMIENTO COMPLETADO!")
    print(f"   Loss final: {result.training_loss:.4f}")
except KeyboardInterrupt:
    print("\n⚠️  Entrenamiento interrumpido")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()

# === 9. GUARDAR ===
print("\n9️⃣ Guardando modelo...")
try:
    model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)
    print(f"✅ Modelo guardado en: {NEW_MODEL_NAME}/")
except Exception as e:
    print(f"❌ Error al guardar: {e}")

print("\n" + "=" * 70)
print("🎉 ¡LISTO!")
print("=" * 70)
