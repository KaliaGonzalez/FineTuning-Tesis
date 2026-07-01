"""
orquestador.py — Automatiza el pipeline de evaluación FINE-TUNING
====================================================================
Por cada temperatura (0.0 a 1.0, en pasos de 0.2):
    1) Ejecuta processor.py  -> genera respuestas del modelo fine-tuned + tiempos (2 CSV)
    2) Ejecuta metrics.py    -> calcula las métricas de calidad sobre esos resultados (2 CSV)

Solo se evalúa UN GoldSet (JSON) por corrida completa: configúralo abajo
en GOALSET_PATH.
"""

import subprocess
import sys

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Modelo base y adapter de fine-tuning (LoRA) que se van a evaluar
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
ADAPTER_PATH = "../Entrenamiento/qwen-2.5-14b-fac-finetuned"

# --------------------------------------------------------------------------
# 👉 Coloca aquí la ruta al ÚNICO GoldSet (JSON) que quieres evaluar
# --------------------------------------------------------------------------
GOALSET_PATH = "FineTuningDatos/prueba.json"
DOC_NAME = "MATER"  # Nombre para tracking / nombres de archivo (opcional)

# Temperaturas a evaluar: de 0.0 a 1.0, de 0.2 en 0.2
TEMPERATURAS = [0.0]

# Iteraciones por pregunta
NUM_ITERATIONS = 10

# ============================================================================
# LOGGING
# ============================================================================


def log(msg, level="INFO"):
    """Log message with level indicator"""
    icons = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "ERROR": "❌",
        "RUNNING": "🔄",
    }
    print(f"{icons.get(level, '')} {msg}")


# ============================================================================
# MAIN LOOP
# ============================================================================

log("🚀 INICIANDO ORQUESTADOR DE EVALUACIÓN (FINE-TUNING)", "RUNNING")
log(f"Modelo base: {BASE_MODEL}", "INFO")
log(f"Adapter: {ADAPTER_PATH}", "INFO")
log(f"GoldSet: {GOALSET_PATH}", "INFO")
log(f"Documento: {DOC_NAME}", "INFO")
log(f"Temperaturas a evaluar: {TEMPERATURAS}", "INFO")
log(f"Iteraciones por pregunta: {NUM_ITERATIONS}", "INFO")
log(f"Total de corridas: {len(TEMPERATURAS)}", "INFO")
print("=" * 80)

contador_total = 0
contador_exito = 0
contador_error = 0

try:
    # FLUJO: Temperatura -> processor.py -> metrics.py
    for temp in TEMPERATURAS:
        contador_total += 1

        print(f"\n{'='*80}")
        log(f"[{contador_total}/{len(TEMPERATURAS)}] Temperatura={temp}", "RUNNING")
        print(f"{'='*80}")

        # ============================================================
        # PASO 1: EJECUTAR PROCESSOR.PY (genera respuestas + tiempos)
        # ============================================================
        log("  Ejecutando processor.py...", "INFO")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "processor.py",
                    "--base_model",
                    BASE_MODEL,
                    "--adapter_path",
                    ADAPTER_PATH,
                    "--temperature",
                    str(temp),
                    "--goalset",
                    GOALSET_PATH,
                    "--doc_name",
                    DOC_NAME,
                    "--num_iterations",
                    str(NUM_ITERATIONS),
                ],
                check=True,
                capture_output=False,
            )
            log("  ✅ processor.py completado", "SUCCESS")
        except subprocess.CalledProcessError:
            log("  ❌ processor.py falló", "ERROR")
            contador_error += 1
            continue

        # ============================================================
        # PASO 2: EJECUTAR METRICS.PY (calcula métricas del CSV generado)
        # ============================================================
        log("  Ejecutando metrics.py...", "INFO")
        try:
            subprocess.run(
                [sys.executable, "metrics.py"],
                check=True,
                capture_output=False,
            )
            log("  ✅ metrics.py completado", "SUCCESS")
            contador_exito += 1
        except subprocess.CalledProcessError:
            log("  ❌ metrics.py falló", "ERROR")
            contador_error += 1

except KeyboardInterrupt:
    log("\n⚠️  Orquestador interrumpido por el usuario", "ERROR")
    sys.exit(1)
except Exception as e:
    log(f"❌ Error general: {e}", "ERROR")
    sys.exit(1)

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print(f"\n{'='*80}")
log("🎉 ORQUESTADOR FINALIZADO", "SUCCESS")
print(f"{'='*80}")
log(f"Total de corridas (temperaturas): {contador_total}", "INFO")
log(f"Exitosas: {contador_exito}", "SUCCESS")
log(f"Con error: {contador_error}", "ERROR")
print(f"{'='*80}\n")

if contador_error > 0:
    sys.exit(1)