"""
processor.py — Adaptado para evaluación de FINE-TUNING (Qwen2.5 + LoRA)
=========================================================================
Genera respuestas del modelo fine-tuneado sobre UN solo GoldSet (JSON),
midiendo el tiempo de cada respuesta, con N iteraciones por pregunta
(por defecto 10).

Salidas (2 CSV):
    1) CSV detallado de tiempos:  ../results/<base_name>.csv
    2) CSV de resumen (por pregunta + global): ../results/<base_name>_summary.csv

También actualiza config.py con las rutas usadas para que metrics.py
pueda encontrar el CSV detallado y generar los archivos de métricas.

Uso:
    python processor.py --goalset ../FAC_Documents/rag_files/Goalset_FAC_MATER.json \
                         --temperature 0.4 --doc_name MATER
"""

import re
import os
import csv
import json
import time
import argparse
import unicodedata
from pathlib import Path

from evaluar_modelo import cargar_modelo, generar_respuesta


def limpiar_string(texto):
    texto = texto.lower()
    texto = (
        unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")
    )
    texto = re.sub(r"[^a-z0-9\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def eliminar_chain_of_thought(texto_respuesta):
    """Elimina bloques <think>...</think> que algunos modelos incluyen en la salida."""
    return re.sub(
        r"<think>.*?</think>", "", texto_respuesta, flags=re.DOTALL | re.IGNORECASE
    ).strip()


def sanitize_filename(name):
    return re.sub(r'[.<>:"/\\|?*]', "_", name)


def load_goalset(ruta_json):
    with open(ruta_json, "r", encoding="utf-8") as f:
        return json.load(f)


def normalizar_goldset(goldset):
    """
    Acepta distintos formatos de GoldSet (pregunta/respuesta o instruction/output)
    y los normaliza a una lista de dicts {"pregunta": ..., "respuesta": ...}.
    """
    preguntas_respuestas = []
    for item in goldset:
        pregunta = item.get("pregunta") or item.get("instruction", "")
        respuesta_esperada = (
            item.get("respuesta")
            or item.get("respuesta_esperada")
            or item.get("output", "")
        )
        if not pregunta or not respuesta_esperada:
            continue
        preguntas_respuestas.append({"pregunta": pregunta, "respuesta": respuesta_esperada})
    return preguntas_respuestas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera respuestas y mide tiempos del modelo Fine-Tuned sobre UN GoldSet."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Nombre/ruta del modelo base (Hugging Face) usado en el fine-tuning.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="../Entrenamiento/qwen-2.5-14b-fac-finetuned",
        help="Ruta al adapter LoRA del modelo fine-tuneado.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperatura de generación (0.0 a 1.0).",
    )
    parser.add_argument(
        "--goalset",
        type=str,
        required=True,
        help="Ruta al ÚNICO archivo GoldSet (JSON) que se va a evaluar.",
    )
    parser.add_argument(
        "--doc_name",
        type=str,
        default="",
        help="Nombre del documento/goldset (para tracking). Si se omite, se deriva del nombre del archivo.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Número de iteraciones por pregunta.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=300,
        help="Máximo de tokens nuevos generados por respuesta.",
    )
    args = parser.parse_args()

    model_name = args.base_model
    adapter_path = args.adapter_path
    temperature = args.temperature
    ruta_json = args.goalset
    doc_name = args.doc_name or Path(ruta_json).stem
    num_iterations = args.num_iterations
    max_tokens = args.max_tokens

    # Cargar y normalizar el GoldSet (UN solo archivo JSON)
    goldset = load_goalset(ruta_json)
    preguntas_respuestas = normalizar_goldset(goldset)

    if not preguntas_respuestas:
        print(f"❌ No se encontraron preguntas válidas en el GoldSet: {ruta_json}")
        raise SystemExit(1)

    # ========================================================================
    # NOMBRES DE ARCHIVOS DE SALIDA
    # ========================================================================
    nombre_para_archivo = os.path.basename(adapter_path.rstrip("/")) or model_name
    sanitized_model_name = sanitize_filename(nombre_para_archivo)
    temp_str = str(temperature).replace(".", "_")
    sanitized_doc_name = sanitize_filename(doc_name) if doc_name else "documento"

    csv_base_name = f"{sanitized_model_name}_temperatura_{temp_str}_{sanitized_doc_name}"

    results_dir = Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"../results/{csv_base_name}.csv"
    output_summary_filename = f"../results/{csv_base_name}_summary.csv"

    # ========================================================================
    # ACTUALIZAR config.py CON LOS VALORES USADOS (para que metrics.py los use)
    # ========================================================================
    config_content = f"""model_name = "{sanitized_model_name}"
temperature = {temperature}
doc_name = "{doc_name}"
output_filename = "{output_filename}"
metrics_output = "../results/{csv_base_name}_metrics.csv"
metrics_output_summary = "../results/{csv_base_name}_metrics_summary.csv"
"""
    with open("config.py", "w", encoding="utf-8") as config_file:
        config_file.write(config_content)

    print("✅ config.py actualizado")
    print(f"   Modelo base: {model_name}")
    print(f"   Adapter (fine-tuning): {adapter_path}")
    print(f"   Temperatura: {temperature}")
    print(f"   Documento/Goldset: {doc_name}  ({ruta_json})")
    print(f"   Preguntas cargadas: {len(preguntas_respuestas)}")
    print(f"   Iteraciones por pregunta: {num_iterations}")
    print(f"   CSV: {csv_base_name}.csv")

    # ========================================================================
    # CARGAR MODELO FINE-TUNED (UNA SOLA VEZ)
    # ========================================================================
    print("\n🚀 INICIALIZANDO MODELO FINE-TUNED...\n")
    model, tokenizer, device = cargar_modelo(model_name, adapter_path)

    # ========================================================================
    # GENERACIÓN DE RESPUESTAS + MEDICIÓN DE TIEMPOS
    # ========================================================================
    csv_headers = [
        "Pregunta_ID",
        "Iteracion_Num",
        "Pregunta",
        "Respuesta_Esperada",
        "Respuesta_Limpia",
        "Tiempo_Segundos",
    ]

    todos_los_tiempos_para_resumen_global = []
    datos_resumen_por_pregunta_lista = []

    with open(output_filename, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()

        for idx, item in enumerate(preguntas_respuestas, start=1):
            pregunta = item["pregunta"]
            respuesta_esperada = item["respuesta"]
            tiempos_iteraciones_pregunta_actual = []

            print(f"\n🟦 Procesando pregunta {idx} de {len(preguntas_respuestas)}")
            print(f"➡️  {pregunta}\n")

            for iter_num_actual in range(1, num_iterations + 1):
                print(
                    f"Ejecutando iteración {iter_num_actual} de {num_iterations} para pregunta {idx}"
                )

                start = time.time()
                respuesta_bruta = generar_respuesta(
                    model,
                    tokenizer,
                    pregunta,
                    device,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                tiempo_segundos = time.time() - start

                respuesta_limpia_final = eliminar_chain_of_thought(respuesta_bruta)

                tiempos_iteraciones_pregunta_actual.append(tiempo_segundos)
                todos_los_tiempos_para_resumen_global.append(tiempo_segundos)

                print(f"✅ Respuesta generada en {tiempo_segundos:.2f}s")

                writer.writerow(
                    {
                        "Pregunta_ID": idx,
                        "Iteracion_Num": iter_num_actual,
                        "Pregunta": pregunta,
                        "Respuesta_Esperada": respuesta_esperada,
                        "Respuesta_Limpia": respuesta_limpia_final,
                        "Tiempo_Segundos": round(tiempo_segundos, 2),
                    }
                )

            # --- Resumen por pregunta (después de todas sus iteraciones) ---
            if tiempos_iteraciones_pregunta_actual:
                promedio_q = sum(tiempos_iteraciones_pregunta_actual) / len(
                    tiempos_iteraciones_pregunta_actual
                )
                tiempo_max_q = max(tiempos_iteraciones_pregunta_actual)
                tiempo_min_q = min(tiempos_iteraciones_pregunta_actual)
                iteracion_max_q_num = (
                    tiempos_iteraciones_pregunta_actual.index(tiempo_max_q) + 1
                )
                iteracion_min_q_num = (
                    tiempos_iteraciones_pregunta_actual.index(tiempo_min_q) + 1
                )

                datos_resumen_por_pregunta_lista.append(
                    {
                        "Tipo_Resumen": "Pregunta",
                        "Pregunta_ID": idx,
                        "Num_Iteraciones": len(tiempos_iteraciones_pregunta_actual),
                        "Tiempo_Promedio_s": round(promedio_q, 2),
                        "Tiempo_Max_s": round(tiempo_max_q, 2),
                        "Iteracion_Mas_Lenta": iteracion_max_q_num,
                        "Tiempo_Min_s": round(tiempo_min_q, 2),
                        "Iteracion_Mas_Rapida": iteracion_min_q_num,
                    }
                )
                print(
                    f"Resumen para pregunta {idx}: Promedio={promedio_q:.2f}s, "
                    f"Max={tiempo_max_q:.2f}s (Iter {iteracion_max_q_num}), "
                    f"Min={tiempo_min_q:.2f}s (Iter {iteracion_min_q_num})"
                )

    # ========================================================================
    # ESCRITURA DEL CSV DE RESÚMENES (por pregunta + global)
    # ========================================================================
    summary_csv_headers = [
        "Tipo_Resumen",
        "Pregunta_ID",
        "Num_Iteraciones",
        "Tiempo_Promedio_s",
        "Tiempo_Max_s",
        "Iteracion_Mas_Lenta",
        "Tiempo_Min_s",
        "Iteracion_Mas_Rapida",
    ]

    with open(output_summary_filename, "w", encoding="utf-8", newline="") as csvfile_summary:
        writer_summary = csv.DictWriter(csvfile_summary, fieldnames=summary_csv_headers)
        writer_summary.writeheader()

        if datos_resumen_por_pregunta_lista:
            writer_summary.writerows(datos_resumen_por_pregunta_lista)

        if todos_los_tiempos_para_resumen_global:
            promedio_global = sum(todos_los_tiempos_para_resumen_global) / len(
                todos_los_tiempos_para_resumen_global
            )
            tiempo_max_global = max(todos_los_tiempos_para_resumen_global)
            tiempo_min_global = min(todos_los_tiempos_para_resumen_global)
            iteracion_max_global_num = (
                todos_los_tiempos_para_resumen_global.index(tiempo_max_global) + 1
            )
            iteracion_min_global_num = (
                todos_los_tiempos_para_resumen_global.index(tiempo_min_global) + 1
            )

            resumen_global_dict = {
                "Tipo_Resumen": "Global",
                "Pregunta_ID": "N/A",
                "Num_Iteraciones": len(todos_los_tiempos_para_resumen_global),
                "Tiempo_Promedio_s": round(promedio_global, 2),
                "Tiempo_Max_s": round(tiempo_max_global, 2),
                "Iteracion_Mas_Lenta": iteracion_max_global_num,
                "Tiempo_Min_s": round(tiempo_min_global, 2),
                "Iteracion_Mas_Rapida": iteracion_min_global_num,
            }
            writer_summary.writerow(resumen_global_dict)

            print("\n========== RESUMEN GLOBAL ==========")
            print(
                f"Total de iteraciones (LLM calls): {len(todos_los_tiempos_para_resumen_global)}"
            )
            print(f"Tiempo promedio global: {promedio_global:.2f} s")
            print(
                f"Iteración global más lenta: {iteracion_max_global_num} ({tiempo_max_global:.2f} s)"
            )
            print(
                f"Iteración global más rápida: {iteracion_min_global_num} ({tiempo_min_global:.2f} s)"
            )
            print("=" * 60 + "\n")
        else:
            print(
                "\n⚠️ No se ejecutaron iteraciones, no se pudo generar el resumen global."
            )

    print(f"\n✅ Proceso finalizado. Resultados detallados guardados en '{output_filename}'.")
    if todos_los_tiempos_para_resumen_global:
        print(f"📊 Resumen de rendimiento guardado en '{output_summary_filename}'.")