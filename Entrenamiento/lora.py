import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig
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
training_arguments = SFTConfig(
    output_dir="./results",
    num_train_epochs=5,  # MÁS épocas = aprende mejor cada término
    per_device_train_batch_size=1,  # Mantener en 1 para conservar memoria
    eval_strategy="steps",
    eval_steps=5,  # Evalúa muy frecuentemente para detectar problemas
    gradient_accumulation_steps=4,  # Más acumulación = gradientes más estables
    optim="adamw_torch",
    save_steps=5,  # Guarda muy frecuentemente para no perder progreso
    logging_steps=2,  # Logs muy detallados
    learning_rate=5e-5,  # Learning rate MÁS BAJO = aprendizaje más preciso
    weight_decay=0.05,  # Más regularización para evitar alucinaciones
    fp16=False,
    bf16=False,
    max_grad_norm=0.1,  # Gradientes muy controlados (evita grandes saltos)
    max_steps=-1,
    warmup_ratio=0.2,  # Calentamiento más largo (20% de entrenamiento)
    group_by_length=True,
    lr_scheduler_type="cosine",  # Cosine decay es mejor que linear para ajuste fino
    max_length=2048,
    packing=False,
)


# --- Configurar Trainer ---
def formatting_prompts_func(example):
    # TRL puede pasar un solo ejemplo (diccionario de strings) o un lote (diccionario de listas)
    if isinstance(example.get("instruction"), list):
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
    else:
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]

        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return text


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_arguments,
)

# --- Entrenar ---
print("Iniciando entrenamiento...")
trainer.train()

# --- Guardar Adaptadores LoRA ---
print(f"Guardando modelo en {NEW_MODEL_NAME}...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
