# Instrucciones para Google Colab

Este archivo `lora.py` está configurado para entrenar un modelo Mistral 7B usando QLoRA.

## 1. Configuración del Entorno (Colab)

Asegúrate de cambiar el entorno de ejecución a **GPU T4** (o superior).

Ejecuta la siguiente celda para instalar las dependencias necesarias:

```python
!pip install -q -U torch bitsandbytes transformers peft accelerate datasets trl
```

## 2. Archivos y Rutas

Sube la estructura de carpetas a tu Google Drive o directamente al entorno de Colab.
La estructura debe ser:

```
/content/
  FineTuningDatos/
    dataTrain.json
    dataValidation.json
  lora.py
```

Si usas Google Drive, monta el drive:

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/Ruta/A/Tu/Proyecto') # Ajusta esto
```

## 3. Ejecutar el Entrenamiento

Una vez instaladas las librerías y ubicados los archivos, ejecuta:

```python
!python lora.py
```

## Notas Adicionales

- **Memoria:** El entrenamiento en 4-bit de un modelo 7B consume cerca de 6-8GB de VRAM. T4 tiene 16GB, así que debería funcionar bien.
- **Unsloth:** El script usa la librería `transformers` estándar, pero el modelo base es de `unsloth`. Si quieres entrenar mÁs rápido (2x) y con menos memoria, se recomienda usar la librería `unsloth` directamente, pero este script funcionará con `transformers`.
- **Formato de Texto:** Se ha añadido una función `formatting_prompts_func` para convertir tus jsons (instruction/input/output) al formato que espera el modelo.
