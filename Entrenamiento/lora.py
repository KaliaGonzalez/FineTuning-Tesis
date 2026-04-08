import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import os

# Configuracion
MODEL_NAME = "unsloth/mistral-7b-v0.3-bnb-4bit"  # O tu modelo base de preferencia (ej. Llama-3-8B-bnb-4bit)
NEW_MODEL_NAME = "mistral-7b-fac-finetuned"
data_files = {
    "train": "FineTuningDatos/dataTrain.json",
    "validation": "FineTuningDatos/dataValidation.json",
}

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Importante para evitar problemas con fp16

#
dataset = load_dataset("json", data_files=data_files)

# Q-Lora
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Modelo
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepara el modelo
model = prepare_model_for_kbit_training(model)

# --- Configuración LoRA ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,  # Rango: Determina la cantidad de parámetros a entrenar (16, 32, 64)
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

# model = get_peft_model(model, peft_config)

# --- Argumentos de Entrenamiento ---
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # Ajustar según necesidad (1-3 suele ser suficiente)
    per_device_train_batch_size=1,
    evaluation_strategy="steps",
    eval_steps=25,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,  # Poner True si usas Ampere GPU (A100, RTX 3090, etc.)
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)


# --- Configurar Trainer ---
def formatting_prompts_func(example):
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
    return output_texts


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

# --- Entrenar ---
print("Iniciando entrenamiento...")
trainer.train()

# --- Guardar Adaptadores LoRA ---
print(f"Guardando modelo en {NEW_MODEL_NAME}...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
