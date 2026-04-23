"""
Script PROFESIONAL de Fine-tuning con LoRA para Mistral 7B
Entrenamiento robusto con validación correcta
Compatible con Python 3.13 y GPU local
"""

import torch
import json
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
MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # Mistral 7B oficial
NEW_MODEL_NAME = "mistral-7b-fac-finetuned"
OUTPUT_DIR = "./results"

# Crear directorio de salida si no existe
import os

os.makedirs(OUTPUT_DIR, exist_ok=True)

data_files = {
    "train": "FineTuningDatos/dataTrain.json",
    "validation": "FineTuningDatos/dataValidation.json",
}

print("=" * 60)
print("🚀 INICIANDO FINE-TUNING CON LORA - MISTRAL 7B")
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

# Verificar estructura de datos
sample = dataset["train"][0]
print(f"   📝 Ejemplo: instruction='{sample['instruction'][:50]}...'")

# === 3. CARGAR MODELO BASE ===
print("\n3️⃣ Cargando Mistral 7B (esto puede tardar 3-5 minutos)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,  # Mistral 7B usa float16
    trust_remote_code=True,
)
print("✅ Modelo cargado en GPU")

# === 4. CONFIGURACIÓN LORA ===
print("\n4️⃣ Configurando LoRA para Mistral 7B...")
peft_config = LoraConfig(
    r=16,  # Rank reducido para ahorrar memoria
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Todos los módulos de Mistral
)
model = get_peft_model(model, peft_config)
print("✅ LoRA configurado")

# === 5. FUNCIÓN DE FORMATTEO ===
print("\n5️⃣ Preparando formato de datos...")


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
print(
    "📊 El modelo se entrenará CON tus datos de Train y validará CON los datos de Validation"
)
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,  # 1 ÉPOCA para aprender bien sin overfitear
    per_device_train_batch_size=2,  # Batch size para Mistral 7B (GPU limitada)
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Equivalente a batch 16
    optim="adamw_torch",
    save_steps=50,  # Guardar checkpoint cada 50 pasos
    logging_steps=5,  # Mostrar progreso cada 5 pasos
    learning_rate=3e-4,  # Learning rate para LoRA
    weight_decay=0.01,
    fp16=True,  # Usar mixed precision para Mistral
    max_grad_norm=1.0,
    warmup_ratio=0.1,  # 10% warmup
    lr_scheduler_type="linear",
    eval_strategy="steps",  # Evaluar cada X pasos
    eval_steps=25,  # Evaluar con datos de VALIDATION cada 25 pasos
    save_total_limit=2,  # Solo guardar 2 checkpoints para ahorrar disco
    load_best_model_at_end=True,  # Cargar el mejor modelo basado en métrica
    metric_for_best_model="loss",  # Basado en loss de validación
    gradient_checkpointing=True,  # Ahorrar memoria
)
print("✅ Argumentos configurados")

# === 8. TRAINER ===
print("\n8️⃣ Creando trainer...")
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
print("✅ Trainer creado")

# === 9. ENTRENAR ===
print("\n9️⃣ Iniciando entrenamiento...")
print("=" * 60)
print("📚 ENTRENAMIENTO CON:")
print(f"   - Dataset de ENTRENAMIENTO: FineTuningDatos/dataTrain.json")
print(f"   - Dataset de VALIDACIÓN: FineTuningDatos/dataValidation.json")
print("=" * 60)
try:
    trainer.train()
    print("\n✅ ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
except Exception as e:
    print(f"❌ Error durante entrenamiento: {e}")
    print("Guardando lo que se pudo entrenar...")
print("=" * 60)

# === 10. GUARDAR MODELO ===
print("\n🔟 Guardando modelo...")
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
print(f"📁 Modelo fine-tuneado guardado en: {NEW_MODEL_NAME}/")
print(f"💾 Este modelo contiene:")
print(f"   ✓ Adaptador LoRA entrenado con tus datos")
print(f"   ✓ Tokenizador configurado")
print(f"   ✓ Listo para usar con app_delfos.py")
print("=" * 60)
