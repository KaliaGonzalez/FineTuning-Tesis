# 📊 Sistema de Evaluación Múltiple - Fine-Tuning Qwen

## ✨ Descripción

Sistema avanzado para evaluar el desempeño del modelo fine-tuned contra **MÚLTIPLES GoldSets** simultáneamente.

El script detecta automáticamente TODOS los GoldSets en la carpeta `FineTuningDatos` y genera **CSV separados por documento** con métricas:

- **BLEU** (1, 2, 3, 4-gramas)
- **ROUGE** (1, 2, L)
- **METEOR**
- **Similitud Semántica**

## 🎯 Características

✅ **Detección automática** de todos los GoldSets  
✅ **CSV para cada documento** (resultados_EDAES.csv, resultados_MATER.csv, etc.)  
✅ **Resumen general comparativo** en tabla  
✅ **JSON detallado** con todos los resultados  
✅ **Tabla visual** en consola

## 📁 Estructura de carpetas

```
FineTuningDatos/
├── Goalset_FAC_EDAES.json         → CSV: resultados_EDAES.csv
├── Goalset_FAC_MATER.json         → CSV: resultados_MATER.csv
├── Goalset_FAC_PORFAC2.json       → CSV: resultados_PORFAC2.csv
├── GoalSet_Historia.json          → CSV: resultados_HISTORIA.csv
├── GoalSetNewMater.json           → CSV: resultados_NEWMATER.csv
└── ...

Evaluacion/
├── metricas.py              # Funciones de métricas
├── evaluar_modelo.py        # Script principal (DETECTA AUTOMÁTICAMENTE)
├── README.md                # Este archivo
└── resultados/              # Se crea automáticamente
    ├── resultados_EDAES_20260512_143022.csv
    ├── resumen_EDAES_20260512_143022.csv
    ├── evaluacion_EDAES_20260512_143022.json
    ├── resultados_MATER_20260512_143025.csv
    ├── resumen_MATER_20260512_143025.csv
    ├── evaluacion_MATER_20260512_143025.json
    ├── resultados_PORFAC2_20260512_143030.csv
    └── resumen_general_20260512_143030.csv  ← TABLA COMPARATIVA
```

## 🚀 Cómo usar

### Paso 1: Ejecutar

```bash
cd Evaluacion
python evaluar_modelo.py
```

El script automáticamente:

1. ✅ Busca TODOS los GoldSets en `../FineTuningDatos/`
2. ✅ Carga el modelo + adapter LoRA
3. ✅ Evalúa cada pregunta
4. ✅ Genera CSV, JSON y tabla

### Paso 2: Ver resultados

Se generan 4 tipos de archivos:

#### **📊 CSV Detallado** (`resultados_DOCUMENTO.csv`)

Todas las preguntas con sus métricas individuales:

```
ID,Pregunta,Respuesta_Esperada,Respuesta_Modelo,BLEU,ROUGE_1,METEOR,SemanticSim
1,"¿Qué es una barraca?","Espacio temporal...","Espacio temporal...",78.45,85.23,72.30,88.50
2,"¿Cuáles son...","Las formaciones...","Las formaciones...",62.10,71.45,58.90,75.20
...
```

#### **📈 CSV Resumen** (`resumen_DOCUMENTO.csv`)

Promedios de todas las métricas para ese documento:

```
BLEU,BLEU_1,BLEU_2,BLEU_3,BLEU_4,ROUGE_1,ROUGE_2,ROUGE_L,METEOR,SemanticSim
65.43,82.34,71.23,58.45,48.92,76.45,64.23,69.87,62.15,78.90
```

#### **📋 JSON Completo** (`evaluacion_DOCUMENTO.json`)

Reporte detallado con fecha, modelo, métricas y resultados.

#### **🎯 CSV Comparativo** (`resumen_general_TIMESTAMP.csv`)

Tabla que compara todos los documentos:

```
Documento,Total_Preguntas,BLEU,BLEU_1,ROUGE_1,ROUGE_2,METEOR,SemanticSim
EDAES,25,65.43,82.34,76.45,64.23,62.15,78.90
MATER,30,72.10,85.20,81.30,70.50,68.45,82.10
PORFAC2,20,58.90,78.50,72.10,61.20,55.30,75.40
HISTORIA,28,71.45,84.10,80.20,69.10,67.80,81.50
```

## 📊 Ejemplo de salida en consola

```
================================================================================
🚀 INICIANDO EVALUACIÓN MÚLTIPLE DEL MODELO FINE-TUNED
================================================================================

📊 Se encontraron 4 GoldSets:

   ✓ EDAES: ../FineTuningDatos/Goalset_FAC_EDAES.json
   ✓ MATER: ../FineTuningDatos/Goalset_FAC_MATER.json
   ✓ PORFAC2: ../FineTuningDatos/Goalset_FAC_PORFAC2.json
   ✓ HISTORIA: ../FineTuningDatos/GoalSet_Historia.json

================================================================================
📋 EVALUANDO: EDAES
================================================================================

📊 Preguntas cargadas: 25

   [1/25] ¿Qué es una barraca?... ✓ BLEU:78 ROUGE:85 METEOR:72
   [2/25] ¿Cuáles son los tipos... ✓ BLEU:62 ROUGE:71 METEOR:59
   [3/25] Define táctica aérea... ✓ BLEU:71 ROUGE:78 METEOR:65
   ...

   📊 RESULTADOS DE EDAES:
      BLEU:          65.43
      ROUGE-1:       76.45
      ROUGE-2:       64.23
      ROUGE-L:       69.87
      METEOR:        62.15
      SemanticSim:   78.90

   💾 Guardando resultados para EDAES...
   ✅ CSV guardado: ./resultados/resultados_EDAES_20260512_143022.csv
   ✅ Resumen guardado: ./resultados/resumen_EDAES_20260512_143022.csv
   ✅ JSON guardado: ./resultados/evaluacion_EDAES_20260512_143022.json

[Repite para MATER, PORFAC2, HISTORIA...]

================================================================================
📈 RESUMEN GENERAL DE TODOS LOS DOCUMENTOS
================================================================================

✅ Resumen general guardado en: ./resultados/resumen_general_20260512_143030.csv

┌────────────────────┬──────────┬────────┬──────────┬────────┬──────────┐
│ Documento          │ Preguntas│  BLEU  │ ROUGE-1  │ METEOR │ SemanticSim│
├────────────────────┼──────────┼────────┼──────────┼────────┼──────────┤
│ EDAES              │       25 │  65.4  │     76.5 │  62.2  │     78.9 │
│ MATER              │       30 │  72.1  │     81.3 │  68.4  │     82.1 │
│ PORFAC2            │       20 │  58.9  │     72.1 │  55.3  │     75.4 │
│ HISTORIA           │       28 │  71.5  │     80.2 │  67.8  │     81.5 │
└────────────────────┴──────────┴────────┴──────────┴────────┴──────────┘
```

## ⚙️ Configuración

Edita el inicio de `evaluar_modelo.py`:

```python
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
ADAPTER_PATH = "../Entrenamiento/qwen-2.5-14b-fac-finetuned"
GOLDSETS_DIR = "../FineTuningDatos"
RESULTS_DIR = "./resultados"

# Patrones a buscar (personalizable)
GOLDSET_PATTERNS = [
    "Goalset_FAC_*.json",
    "GoalSet_*.json",
    "GoldSet.json"
]
```

## 📋 Formatos aceptados

El script acepta dos formatos de GoldSet:

**Formato 1** (recomendado):

```json
[
  {
    "id": 1,
    "pregunta": "¿Qué es una barraca?",
    "respuesta_esperada": "Espacio temporal limitado..."
  }
]
```

**Formato 2** (compatible):

```json
[
  {
    "id": 1,
    "instruction": "¿Qué es una barraca?",
    "output": "Espacio temporal limitado..."
  }
]
```

## 📈 Interpretación de métricas

| Métrica         | Rango | Interpretación                                  |
| --------------- | ----- | ----------------------------------------------- |
| **BLEU**        | 0-100 | Precisión de n-gramas. >30 bueno, >50 excelente |
| **ROUGE-1**     | 0-100 | Overlap de palabras. >70 bueno                  |
| **ROUGE-2**     | 0-100 | Overlap de bigramas. >60 bueno                  |
| **ROUGE-L**     | 0-100 | Longest common subsequence. >65 bueno           |
| **METEOR**      | 0-100 | Similitud con stem. >50 bueno, >70 excelente    |
| **SemanticSim** | 0-100 | Similitud simple. >70 bueno                     |

## 🔍 Análisis comparativo

Abre el CSV de resumen general en Excel o Python:

```python
import pandas as pd

# Leer tabla comparativa
df = pd.read_csv("resultados/resumen_general_20260512_143030.csv")

# Ordenar por BLEU descendente
print(df.sort_values("BLEU", ascending=False))

# Ver cuál documento tiene mejor desempeño
print(f"Mejor documento: {df.loc[df['BLEU'].idxmax(), 'Documento']}")
```

## ✅ Requisitos

```bash
pip install torch transformers peft numpy pandas
```

## 🎯 Casos de uso

1. **Comparar desempeño por tema**: ¿Qué documento obtiene mejores resultados?
2. **Identificar debilidades**: ¿Cuál métrica es más baja?
3. **Mejora iterativa**: Reentrenar y comparar resultados
4. **Validación del modelo**: Verificar que el fine-tuning fue exitoso

## 💡 Tips

- Los resultados se generan en **tiempo real** por documento
- Puedes interrumpir el script sin perder los resultados parciales
- El CSV se abre directamente en Excel para análisis visual
- La tabla en consola muestra un resumen rápido al finalizar

---

**Sistema de Evaluación Múltiple**  
**Creado para:** Evaluación de Fine-Tuning FAC  
**Última actualización:** Mayo 2026
