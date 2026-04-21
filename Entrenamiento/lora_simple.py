"""
Script de Fine-tuning con LoRA para Mistral 7B
Versión simplificada sin dependencias de TRL problemáticas
Compatible con Python 3.13
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import os

# === CONFIGURACIÓN ===
MODEL_NAME = "unsloth/mistral-7b-v0.3-bnb-4bit"
NEW_MODEL_NAME = "mistral-7b-fac-finetuned"
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

# === 3. CONFIGURACIÓN QUANTIZATION (4-bit) ===
print("\n3️⃣ Configurando quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
print("✅ Quantization configurado")

# === 4. CARGAR MODELO ===
print("\n4️⃣ Cargando modelo base (esto puede tardar 5-10 minutos)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
print("✅ Modelo cargado")

# === 5. PREPARAR MODELO PARA TRAINING ===
print("\n5️⃣ Preparando modelo para K-bit training...")
model = prepare_model_for_kbit_training(model)
print("✅ Modelo preparado")

# === 6. CONFIGURACIÓN LORA ===
print("\n6️⃣ Configurando LoRA...")
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
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
    ],
)
model = get_peft_model(model, peft_config)
print("✅ LoRA configurado")

# === 7. FUNCIÓN DE FORMATTEO ===
print("\n7️⃣ Preparando formato de datos...")


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
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    evaluation_strategy="steps",
    eval_steps=25,
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
trainer.train()
print("=" * 60)

# === 11. GUARDAR MODELO ===
print("\n✅ Guardando modelo...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)

print("\n" + "=" * 60)
print(f"🎉 ¡ENTRENAMIENTO COMPLETADO!")
print(f"📁 Modelo guardado en: {NEW_MODEL_NAME}/")
print("=" * 60)
