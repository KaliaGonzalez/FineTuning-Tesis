import os
import glob
import json
import re
import ollama  # pip install ollama
from tqdm import tqdm

# --- CONFIGURACIÓN ---
SOURCE_DIR = "../docs"  # Carpeta donde están tus .md
OUTPUT_FILE = "dataset_generation/dataset_raw.jsonl"  # Archivo de salida
MODEL_NAME = "llama3"  # Modelo local a usar (asegúrate de tenerlo: ollama pull mistral)


def read_markdown_files(directory):
    """Lee todos los archivos .md del directorio."""
    md_files = glob.glob(os.path.join(directory, "*.md"))
    documents = {}
    for filepath in md_files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                documents[filename] = f.read()
        except Exception as e:
            print(f"Error leyendo {filename}: {e}")
    return documents


def split_text_into_chunks(text, chunk_size=2000, overlap=200):
    """
    Divide el texto en chunks más grandes para que el modelo tenga contexto suficiente.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Intentar cortar en un salto de línea o punto para no cortar frases a la mitad
        if end < text_len:
            last_break = text.rfind("\n", start, end)
            if last_break != -1 and last_break > start + chunk_size * 0.7:
                end = last_break + 1
            else:
                last_period = text.rfind(".", start, end)
                if last_period != -1 and last_period > start + chunk_size * 0.7:
                    end = last_period + 1

        chunk = text[start:end].strip()
        if len(chunk) > 100:  # Solo agregar si tiene contenido sustancial
            chunks.append(chunk)

        start = end - overlap

    return chunks


def generate_qa_pair(chunk_text, source_name):
    """
    Usa Ollama local para generar pares QA en formato Alpaca.
    """

    # Prompt diseñado para forzar formato JSON válido
    prompt = f"""
    Actúa como un experto en creación de datasets para Fine-Tuning de LLMs.
    Tu tarea es leer el siguiente texto extraído del documento "{source_name}" y generar 
    EXACTAMENTE 3 pares de pregunta-respuesta (QA) de alta calidad.

    TEXTO:
    --------------------------------------------------
    {chunk_text}
    --------------------------------------------------

    INSTRUCCIONES DE FORMATO:
    1. Responde SOLO con un array JSON válido. Sin texto antes ni después.
    2. Usa este formato exacto para cada elemento:
       {{
         "instruction": "Pregunta clara y específica basada en el texto",
         "input": "",
         "output": "Respuesta detallada y correcta extraída del texto"
       }}
    3. Las preguntas deben ser autocontenidas (no uses frases como "según el texto anterior").
    
    SALIDA JSON:
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7},  # Algo de creatividad para variar preguntas
        )

        content = response["message"]["content"]

        # Limpieza básica por si el modelo sigue escribiendo texto extra (Markdown, ```json, etc)
        content_clean = re.sub(r"```json|```", "", content).strip()

        # Intentar parsear el JSON
        data = json.loads(content_clean)

        # Validar estructura y retornar
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # A veces envuelve todo en una clave raíz
            return list(data.values())[0] if data else []

    except json.JSONDecodeError:
        print(f"⚠️  Error de formato JSON en chunk de {source_name}. Saltando...")
        # Opcional: Podrías guardar el 'content' en un log para ver qué falló
    except Exception as e:
        print(f"❌ Error generando QA con Ollama: {e}")

    return []


def main():
    print(f"🚀 Iniciando generador de Dataset con Ollama ({MODEL_NAME})...")

    # 1. Leer documentos
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: No existe el directorio {SOURCE_DIR}")
        return

    docs = read_markdown_files(SOURCE_DIR)
    print(f"📄 Se procesarán {len(docs)} documentos.")

    all_pairs = []

    # 2. Procesar cada documento
    for filename, text in docs.items():
        print(f"\nProcesando: {filename}...")
        chunks = split_text_into_chunks(text)
        print(f"  -> Dividido en {len(chunks)} fragmentos.")

        # Procesar chunks con barra de progreso
        for chunk in tqdm(chunks, desc="Generando QA", unit="chunk"):
            pairs = generate_qa_pair(chunk, filename)
            if pairs:
                all_pairs.extend(pairs)

    # 3. Guardar resultado
    if all_pairs:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        print(f"\n💾 Guardando {len(all_pairs)} ejemplos en {OUTPUT_FILE}...")

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for pair in all_pairs:
                # Escribir línea por línea (JSONL)
                json.dump(pair, f, ensure_ascii=False)
                f.write("\n")
        print("✅ ¡Dataset generado con éxito!")
    else:
        print(
            "⚠️  No se generaron pares QA. Revisa si Ollama está corriendo o los documentos están vacíos."
        )


if __name__ == "__main__":
    main()
