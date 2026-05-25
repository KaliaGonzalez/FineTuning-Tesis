"""
Script de evaluación del modelo fine-tuned con métricas BLEU, ROUGE, METEOR, BERTScore
Evalúa TODOS los GoldSets en la carpeta FineTuningDatos
Genera CSV con resultados para cada documento
"""

import torch
import json
import os
import sys
import csv
import glob
from datetime import datetime
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np

# Importar funciones de métricas
from metricas import calcular_todas_metricas

# ==================== CONFIGURACIÓN ====================
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
ADAPTER_PATH = "../Entrenamiento/qwen-2.5-14b-fac-finetuned"
GOLDSETS_DIR = "../FineTuningDatos"
RESULTS_DIR = "./resultados"

# Temperaturas a evaluar
TEMPERATURAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Número de iteraciones por temperatura
NUM_ITERACIONES = 10

# Patrones de GoldSets a evaluar
GOLDSET_PATTERNS = ["Goalset_FAC_*.json", "GoalSet_*.json", "GoldSet.json"]

# ==================== FUNCIONES ====================


def cargar_modelo():
    """Carga el modelo base + adapter LoRA"""
    print("🔄 Cargando modelo base...")

    # Detectar GPU
    if torch.cuda.is_available():
        print(f"🎮 GPU detectada: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠️ GPU no detectada, usando CPU")
        device = "cpu"

    # Cargar tokenizador
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Cargar adapter LoRA primero
    if os.path.exists(ADAPTER_PATH):
        print(f"✅ Cargando adapter desde: {ADAPTER_PATH}")
        try:
            # Cargar modelo base en CPU primero (evita problemas de memoria)
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            # Luego cargar el adapter
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)

            # Mover a device después de cargar adapter
            if device == "cuda":
                model = model.to(device)

        except Exception as e:
            print(f"⚠️ Error al cargar adapter con PEFT: {e}")
            print("   Intentando carga alternativa...")
            # Fallback: cargar sin adapter
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        print(f"⚠️ Advertencia: No se encontró adapter en {ADAPTER_PATH}")
        print("   Usando modelo base sin fine-tuning")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer, device


def encontrar_goldsets() -> Dict[str, str]:
    """Busca todos los GoldSets en la carpeta FineTuningDatos"""
    goldsets = {}

    for pattern in GOLDSET_PATTERNS:
        files = glob.glob(os.path.join(GOLDSETS_DIR, pattern))
        for file_path in files:
            # Extraer nombre del documento (sin extensión)
            filename = os.path.basename(file_path)
            doc_name = (
                filename.replace("Goalset_FAC_", "")
                .replace("GoalSet_", "")
                .replace(".json", "")
                .upper()
            )
            goldsets[doc_name] = file_path

    return goldsets


def cargar_goldset(filepath: str) -> List[Dict]:
    """Carga un GoldSet específico"""
    if not os.path.exists(filepath):
        print(f"❌ Error: No se encontró {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        goldset = json.load(f)

    return goldset


def guardar_resultados_csv(
    doc_name: str,
    resultados: List[Dict],
    metricas_promedio: Dict,
    temperatura: float,
    num_iteracion: int = 1,
):
    """Guarda los resultados en un archivo CSV con identificación de temperatura e iteración"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Formatear temperatura para el nombre del archivo (0.0 -> temp0, 0.2 -> temp0_2, etc)
    temp_str = str(temperatura).replace(".", "_")

    csv_filename = os.path.join(
        RESULTS_DIR,
        f"resultados_{doc_name}_temp{temp_str}_iter{num_iteracion}_{timestamp}.csv",
    )

    # Escribir CSV con resultados detallados
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "ID",
            "Pregunta",
            "Respuesta_Esperada",
            "Respuesta_Modelo",
            "BLEU",
            "BLEU_1",
            "BLEU_2",
            "BLEU_3",
            "BLEU_4",
            "ROUGE_1",
            "ROUGE_2",
            "ROUGE_L",
            "METEOR",
            "SemanticSim",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for resultado in resultados:
            row = {
                "ID": resultado["id"],
                "Pregunta": resultado["pregunta"],
                "Respuesta_Esperada": resultado["respuesta_esperada"],
                "Respuesta_Modelo": resultado["respuesta_modelo"],
                "BLEU": resultado["metricas"]["BLEU"],
                "BLEU_1": resultado["metricas"]["BLEU_1"],
                "BLEU_2": resultado["metricas"]["BLEU_2"],
                "BLEU_3": resultado["metricas"]["BLEU_3"],
                "BLEU_4": resultado["metricas"]["BLEU_4"],
                "ROUGE_1": resultado["metricas"]["ROUGE_1"],
                "ROUGE_2": resultado["metricas"]["ROUGE_2"],
                "ROUGE_L": resultado["metricas"]["ROUGE_L"],
                "METEOR": resultado["metricas"]["METEOR"],
                "SemanticSim": resultado["metricas"]["SemanticSim"],
            }
            writer.writerow(row)

    # Escribir CSV con promedios Y detalles
    summary_filename = os.path.join(
        RESULTS_DIR,
        f"resumen_{doc_name}_temp{temp_str}_iter{num_iteracion}_{timestamp}.csv",
    )
    with open(summary_filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Iteracion",
            "ID",
            "Pregunta",
            "Respuesta_Esperada",
            "Respuesta_Modelo",
            "BLEU",
            "BLEU_1",
            "BLEU_2",
            "BLEU_3",
            "BLEU_4",
            "ROUGE_1",
            "ROUGE_2",
            "ROUGE_L",
            "METEOR",
            "SemanticSim",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Escribir cada resultado con sus métricas
        for idx, resultado in enumerate(resultados, 1):
            row = {
                "Iteracion": idx,
                "ID": resultado["id"],
                "Pregunta": resultado["pregunta"],
                "Respuesta_Esperada": resultado["respuesta_esperada"],
                "Respuesta_Modelo": resultado["respuesta_modelo"],
                "BLEU": resultado["metricas"]["BLEU"],
                "BLEU_1": resultado["metricas"]["BLEU_1"],
                "BLEU_2": resultado["metricas"]["BLEU_2"],
                "BLEU_3": resultado["metricas"]["BLEU_3"],
                "BLEU_4": resultado["metricas"]["BLEU_4"],
                "ROUGE_1": resultado["metricas"]["ROUGE_1"],
                "ROUGE_2": resultado["metricas"]["ROUGE_2"],
                "ROUGE_L": resultado["metricas"]["ROUGE_L"],
                "METEOR": resultado["metricas"]["METEOR"],
                "SemanticSim": resultado["metricas"]["SemanticSim"],
            }
            writer.writerow(row)

    print(f"   ✅ CSV guardado: {csv_filename}")
    print(f"   ✅ Resumen guardado: {summary_filename}")

    return csv_filename, summary_filename


def generar_respuesta(
    model,
    tokenizer,
    pregunta: str,
    device: str,
    temperature: float = 0.0,
    max_tokens: int = 150,
) -> str:
    """Genera respuesta del modelo para una pregunta con una temperatura específica"""

    # Formatear prompt exactamente como en training
    formatted_prompt = f"### Instruction:\n{pregunta}\n\n### Response:\n"

    # Tokenizar
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
    ).to(device)

    # Generar con temperatura
    with torch.no_grad():
        # Si temperatura es 0, usar greedy (do_sample=False)
        # Si temperatura > 0, usar sampling
        if temperature == 0.0:
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

    # Decodificar
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Limpiar respuesta
    if "### Response:" in full_response:
        response_only = full_response.split("### Response:")[-1].strip()
    else:
        response_only = full_response.strip()

    # Detener en próximo ### si existe
    if "###" in response_only[10:]:
        next_section = response_only[10:].find("###")
        if next_section != -1:
            response_only = response_only[: 10 + next_section].strip()

    # Limpiar marcas residuales
    response_only = response_only.replace("### Fuente:", "").strip()
    response_only = response_only.replace("### Instruction:", "").strip()

    return response_only


def evaluar_modelo():
    """Función principal de evaluación - Evalúa TODOS los GoldSets con MÚLTIPLES TEMPERATURAS E ITERACIONES"""

    print("\n" + "=" * 100)
    print("🚀 INICIANDO EVALUACIÓN MÚLTIPLE CON TEMPERATURAS E ITERACIONES")
    print(f"   Temperaturas: {TEMPERATURAS}")
    print(f"   Iteraciones por temperatura: {NUM_ITERACIONES}")
    print("=" * 100 + "\n")

    # Crear directorio de resultados
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Cargar modelo
    model, tokenizer, device = cargar_modelo()

    # Encontrar todos los GoldSets
    goldsets_dict = encontrar_goldsets()

    if not goldsets_dict:
        print("❌ Error: No hay GoldSets para evaluar en", GOLDSETS_DIR)
        return

    print(f"📊 Se encontraron {len(goldsets_dict)} GoldSets:\n")
    for doc_name, filepath in goldsets_dict.items():
        print(f"   ✓ {doc_name}: {filepath}")
    print()

    # Diccionario para almacenar reportes
    todos_reportes_por_temp = {temp: {} for temp in TEMPERATURAS}
    resumen_general_por_temp = {temp: [] for temp in TEMPERATURAS}

    # ========== LOOP PRINCIPAL: ITERAR SOBRE TEMPERATURAS ==========
    for temperatura in TEMPERATURAS:
        print("\n" + "=" * 100)
        print(f"🌡️  EVALUANDO CON TEMPERATURA: {temperatura}")
        print("=" * 100 + "\n")

        # ========== LOOP SECUNDARIO: ITERAR SOBRE ITERACIONES POR TEMPERATURA ==========
        for num_iter in range(1, NUM_ITERACIONES + 1):
            print(f"\n🔁 ITERACIÓN {num_iter}/{NUM_ITERACIONES}")
            print("-" * 100 + "\n")

            # Evaluar cada GoldSet
            for doc_name, goldset_path in goldsets_dict.items():
                print("\n" + "-" * 80)
                print(
                    f"📋 DOCUMENTO: {doc_name} | TEMPERATURA: {temperatura} | ITERACIÓN: {num_iter}"
                )
                print("-" * 80)

                # Cargar GoldSet
                goldset = cargar_goldset(goldset_path)

                if not goldset:
                    print(f"⚠️  No hay preguntas en {doc_name}\n")
                    continue

                print(f"📊 Preguntas cargadas: {len(goldset)}\n")

                # Evaluar cada pregunta
                resultados = []

                for idx, item in enumerate(goldset, 1):
                    pregunta = item.get("pregunta") or item.get("instruction", "")
                    respuesta_esperada = (
                        item.get("respuesta")
                        or item.get("respuesta_esperada")
                        or item.get("output", "")
                    )

                    if not pregunta or not respuesta_esperada:
                        print(f"   ⚠️  Pregunta {idx} incompleta, saltando...")
                        continue

                    print(
                        f"   [{idx}/{len(goldset)}] {pregunta[:50]}...",
                        end="",
                        flush=True,
                    )

                    # Generar respuesta CON TEMPERATURA ESPECÍFICA
                    respuesta_modelo = generar_respuesta(
                        model, tokenizer, pregunta, device, temperature=temperatura
                    )

                    # Calcular métricas
                    metricas = calcular_todas_metricas(
                        respuesta_esperada, respuesta_modelo
                    )

                    # Guardar resultado
                    resultado = {
                        "id": item.get("id", idx),
                        "pregunta": pregunta,
                        "respuesta_esperada": respuesta_esperada,
                        "respuesta_modelo": respuesta_modelo,
                        "metricas": metricas,
                    }
                    resultados.append(resultado)

                    print(
                        f" ✓ BLEU:{metricas['BLEU']:.0f} ROUGE:{metricas['ROUGE_1']:.0f} METEOR:{metricas['METEOR']:.0f}"
                    )

                if not resultados:
                    print(f"⚠️  Sin resultados válidos para {doc_name}\n")
                    continue

                # Calcular promedios
                metricas_promedio = {
                    "BLEU": np.mean([r["metricas"]["BLEU"] for r in resultados]),
                    "BLEU_1": np.mean([r["metricas"]["BLEU_1"] for r in resultados]),
                    "BLEU_2": np.mean([r["metricas"]["BLEU_2"] for r in resultados]),
                    "BLEU_3": np.mean([r["metricas"]["BLEU_3"] for r in resultados]),
                    "BLEU_4": np.mean([r["metricas"]["BLEU_4"] for r in resultados]),
                    "ROUGE_1": np.mean([r["metricas"]["ROUGE_1"] for r in resultados]),
                    "ROUGE_2": np.mean([r["metricas"]["ROUGE_2"] for r in resultados]),
                    "ROUGE_L": np.mean([r["metricas"]["ROUGE_L"] for r in resultados]),
                    "METEOR": np.mean([r["metricas"]["METEOR"] for r in resultados]),
                    "SemanticSim": np.mean(
                        [r["metricas"]["SemanticSim"] for r in resultados]
                    ),
                }

                # Mostrar resultados del documento
                print(
                    f"\n   📊 RESULTADOS DE {doc_name} (TEMP={temperatura}, ITER={num_iter}):"
                )
                print(f"      BLEU:          {metricas_promedio['BLEU']:.2f}")
                print(f"      ROUGE-1:       {metricas_promedio['ROUGE_1']:.2f}")
                print(f"      ROUGE-2:       {metricas_promedio['ROUGE_2']:.2f}")
                print(f"      ROUGE-L:       {metricas_promedio['ROUGE_L']:.2f}")
                print(f"      METEOR:        {metricas_promedio['METEOR']:.2f}")
                print(f"      SemanticSim:   {metricas_promedio['SemanticSim']:.2f}")

                # Guardar resultados en CSV
                print(
                    f"\n   💾 Guardando resultados para {doc_name} (TEMP={temperatura}, ITER={num_iter})..."
                )
                guardar_resultados_csv(
                    doc_name, resultados, metricas_promedio, temperatura, num_iter
                )

                # Guardar reporte JSON
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_str = str(temperatura).replace(".", "_")
                json_file = os.path.join(
                    RESULTS_DIR,
                    f"evaluacion_{doc_name}_temp{temp_str}_iter{num_iter}_{timestamp}.json",
                )

                reporte = {
                    "fecha": datetime.now().isoformat(),
                    "documento": doc_name,
                    "temperatura": temperatura,
                    "iteracion": num_iter,
                    "modelo": BASE_MODEL,
                    "adapter": ADAPTER_PATH,
                    "total_preguntas": len(resultados),
                    "metricas_promedio": metricas_promedio,
                    "resultados_detallados": resultados,
                }

                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(reporte, f, indent=2, ensure_ascii=False)

                print(f"   ✅ JSON guardado: {json_file}")

                todos_reportes_por_temp[temperatura][doc_name] = reporte
                resumen_general_por_temp[temperatura].append(
                    {
                        "Documento": doc_name,
                        "Temperatura": temperatura,
                        "Iteracion": num_iter,
                        "Total_Preguntas": len(resultados),
                        **metricas_promedio,
                    }
                )

    # ========== CREAR RESUMEN GENERAL POR TEMPERATURA ==========
    print("\n" + "=" * 100)
    print("📈 RESUMEN FINAL DE TODAS LAS TEMPERATURAS")
    print("=" * 100 + "\n")

    # Crear un CSV resumen para cada temperatura
    for temperatura in TEMPERATURAS:
        temp_str = str(temperatura).replace(".", "_")
        resumen_csv = os.path.join(
            RESULTS_DIR,
            f"resumen_general_temp{temp_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )

        with open(resumen_csv, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "Documento",
                "Temperatura",
                "Iteracion",
                "Total_Preguntas",
                "BLEU",
                "BLEU_1",
                "BLEU_2",
                "BLEU_3",
                "BLEU_4",
                "ROUGE_1",
                "ROUGE_2",
                "ROUGE_L",
                "METEOR",
                "SemanticSim",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(resumen_general_por_temp[temperatura])

        print(f"✅ Resumen para TEMPERATURA {temperatura} guardado en: {resumen_csv}")

    print("✅ ¡EVALUACIÓN COMPLETADA!\n")
    return todos_reportes_por_temp


if __name__ == "__main__":
    evaluar_modelo()
