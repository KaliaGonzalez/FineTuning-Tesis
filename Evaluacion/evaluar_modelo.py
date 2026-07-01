"""
evaluar_modelo.py — Módulo de carga y generación para el modelo FINE-TUNED (Qwen2.5 + LoRA)
=============================================================================================
Este módulo YA NO ejecuta una evaluación completa por sí mismo. Su única
responsabilidad es cargar el modelo base + adapter de fine-tuning (LoRA) y
generar respuestas dado un prompt. `processor.py` importa estas funciones
para construir el pipeline completo:

    orquestador.py  -->  processor.py (usa este módulo)  -->  metrics.py

Si necesitas hacer una prueba rápida y manual del modelo, puedes correr este
archivo directamente (ver bloque `if __name__ == "__main__":` al final).
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==================== CONFIGURACIÓN POR DEFECTO ====================
# Estos valores se usan solo si processor.py no pasa otros explícitamente.
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
ADAPTER_PATH = "../Entrenamiento/qwen-2.5-14b-fac-finetuned"


def cargar_modelo(base_model: str = BASE_MODEL, adapter_path: str = ADAPTER_PATH):
    """
    Carga el modelo base y el adapter LoRA (fine-tuning) junto con su tokenizador.

    Retorna: (model, tokenizer, device)
    """
    print("🔄 Cargando modelo base...")

    if torch.cuda.is_available():
        print(f"🎮 GPU detectada: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠️ GPU no detectada, usando CPU")
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path and os.path.exists(adapter_path):
        print(f"✅ Cargando adapter (fine-tuning) desde: {adapter_path}")
        try:
            # Cargar modelo base primero (evita problemas de memoria)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            # Aplicar el adapter LoRA
            model = PeftModel.from_pretrained(model, adapter_path)

            if device == "cuda":
                model = model.to(device)

        except Exception as e:
            print(f"⚠️ Error al cargar adapter con PEFT: {e}")
            print("   Intentando carga alternativa (modelo base, SIN adapter)...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        print(f"⚠️ Advertencia: No se encontró adapter en '{adapter_path}'")
        print("   Usando modelo base SIN fine-tuning")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()

    if hasattr(model, "peft_config"):
        print("✅ Adapter PEFT detectado - Modelo fine-tuned activo\n")
    else:
        print("⚠️ Modelo corriendo SIN adapter de fine-tuning\n")

    return model, tokenizer, device


def generar_respuesta(
    model,
    tokenizer,
    pregunta: str,
    device: str,
    temperature: float = 0.0,
    max_tokens: int = 300,
) -> str:
    """
    Genera la respuesta del modelo fine-tuned para una pregunta, usando el
    mismo formato de prompt utilizado durante el entrenamiento.
    """
    formatted_prompt = f"### Instruction:\n{pregunta}\n\n### Response:\n"

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
    ).to(device)

    with torch.no_grad():
        if temperature == 0.0:
            # Temperatura 0 -> generación determinística (greedy)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_tokens,
                min_new_tokens=20,
                do_sample=False,
                num_beams=1,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_tokens,
                min_new_tokens=20,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                num_beams=1,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
            )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraer solo lo que está después de "### Response:"
    if "### Response:" in full_response:
        response_only = full_response.split("### Response:")[-1].strip()
    else:
        response_only = full_response.strip()

    # Cortar si aparece otra sección "###" (evita arrastrar texto de más)
    if "###" in response_only:
        next_section = response_only.find("###")
        if next_section > 0:
            response_only = response_only[:next_section].strip()

    # Limpiar marcas residuales
    response_only = (
        response_only.replace("### Fuente:", "").replace("### Instruction:", "").strip()
    )

    if not response_only:
        response_only = "Sin respuesta"

    return response_only


if __name__ == "__main__":
    # Prueba rápida y manual: carga el modelo y genera una respuesta de ejemplo.
    modelo, tok, dev = cargar_modelo()
    pregunta_prueba = "¿Qué es la FAC?"
    print(f"\n🧪 Pregunta de prueba: {pregunta_prueba}")
    respuesta = generar_respuesta(modelo, tok, pregunta_prueba, dev, temperature=0.2)
    print(f"🤖 Respuesta: {respuesta}\n")