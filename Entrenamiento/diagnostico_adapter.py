"""
Script de diagnóstico para verificar si el LoRA adapter está funcionando
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

print("=" * 80)
print("DIAGNÓSTICO DEL ADAPTER LoRA")
print("=" * 80)

base_model = "unsloth/mistral-7b-v0.3-bnb-4bit"
adapter_path = "mistral-7b-fac-finetuned"

# 1. Verificar que el adapter existe
print("\n1️⃣ Verificando que el adapter existe...")
if os.path.exists(adapter_path):
    print(f"   ✅ Carpeta '{adapter_path}' ENCONTRADA")
    files = os.listdir(adapter_path)
    print(f"   📁 Archivos: {files}")
    for f in files:
        size_mb = os.path.getsize(os.path.join(adapter_path, f)) / (1024*1024)
        print(f"      - {f}: {size_mb:.1f} MB")
else:
    print(f"   ❌ Carpeta '{adapter_path}' NO ENCONTRADA")
    exit(1)

# 2. Cargar tokenizador
print("\n2️⃣ Cargando tokenizador...")
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizador cargado")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# 3. Cargar modelo base
print("\n3️⃣ Cargando modelo base...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("   ✅ Modelo base cargado")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# 4. Cargar adapter
print("\n4️⃣ Cargando adapter LoRA...")
try:
    model = PeftModel.from_pretrained(model, adapter_path)
    print("   ✅ Adapter cargado")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# 5. Poner en eval
model.eval()
print("   ✅ Modelo en modo EVAL")

# 6. Probar generación
print("\n5️⃣ Probando generación...")
test_prompt = "### Instruction:\n¿Qué es una barraca?\n\n### Response:\n"

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

print(f"   Prompt: {test_prompt[:50]}...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        num_beams=1,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n📝 RESPUESTA COMPLETA:\n{response}")

# 7. Verificar si la respuesta tiene sentido
print("\n6️⃣ Análisis de respuesta...")
if len(response) > 100:
    print("   ✅ Respuesta tiene longitud razonable")
else:
    print("   ❌ Respuesta muy corta (posible problema)")

if "### Fuente:" in response:
    print("   ✅ Respuesta incluye ### Fuente:")
else:
    print("   ⚠️ Respuesta NO incluye ### Fuente:")

print("\n" + "=" * 80)
print("FIN DEL DIAGNÓSTICO")
print("=" * 80)
