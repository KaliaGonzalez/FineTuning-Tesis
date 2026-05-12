import streamlit as st
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import os

st.set_page_config(page_title="DELFOS FAC", page_icon="🛩️", layout="wide")
st.title("🛩️ DELFOS - Sistema de Inteligencia Aérea")
st.markdown(
    "**Bienvenido, oficial.** Soy DELFOS, el Sistema de Inteligencia y Doctrina de la Fuerza Aérea Colombiana. A tu servicio. ✈️"
)
st.divider()


# --- CARGAR DATASET PARA BUSCAR FUENTES ---
@st.cache_data
def load_training_data():
    """Carga el dataset de training para buscar fuentes por similitud"""
    try:
        # Intentar desde la carpeta Entrenamiento
        dataset_path = "FineTuningDatos/dataTrain.json"
        if not os.path.exists(dataset_path):
            # Si no existe, intentar desde la raíz
            dataset_path = "../FineTuningDatos/dataTrain.json"

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return []  # Retornar lista vacía si hay error


training_data = load_training_data()


# --- FUNCIÓN PARA BUSCAR FUENTE DEL DATASET ---
def find_fuente_in_training_data(instruction_prompt):
    """Busca la pregunta en el dataset y devuelve la fuente"""
    if not training_data:
        return None

    instruction_lower = instruction_prompt.lower().strip()

    for entry in training_data:
        entry_inst = entry.get("instruction", "").lower().strip()

        # Búsqueda exacta
        if entry_inst == instruction_lower:
            fuente = entry.get("fuente", "").strip()
            if fuente and fuente.lower() not in ["", "desconocida", "none"]:
                return fuente

        # Búsqueda parcial: primeras palabras clave
        instruction_words = instruction_lower.split()[:5]
        entry_words = entry_inst.split()[:5]

        if instruction_words == entry_words:
            fuente = entry.get("fuente", "").strip()
            if fuente and fuente.lower() not in ["", "desconocida", "none"]:
                return fuente

        # Búsqueda por similitud: si más del 70% de las palabras coinciden
        common_words = len(set(instruction_words) & set(entry_words))
        if common_words >= len(instruction_words) * 0.7:
            fuente = entry.get("fuente", "").strip()
            if fuente and fuente.lower() not in ["", "desconocida", "none"]:
                return fuente

    return None  # No encontrada


# --- FUNCIÓN PARA GUARDAR REGISTRO DE CONVERSACIONES ---
def save_conversation_log(pregunta, respuesta, tiempo_respuesta, fuente):
    """Guarda la pregunta, respuesta y tiempo en un archivo de texto"""
    import datetime

    # Crear carpeta de logs si no existe
    logs_dir = "chat_logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Nombre del archivo con fecha
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(logs_dir, f"conversaciones_{timestamp}.txt")

    # Crear o abrir el archivo
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(
            f"FECHA Y HORA: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write("=" * 80 + "\n")
        f.write(f"pregunta: {pregunta}\n")
        f.write(f"respuesta: {respuesta}\n")
        f.write(f"tiempo_respuesta: {tiempo_respuesta:.2f} segundos\n")
        f.write(f"fuente: {fuente if fuente else 'Desconocida'}\n")
        f.write("\n\n")


# --- CARGAR EL MODELO (En caché para no recargar cada vez) ---
@st.cache_resource
def load_model():
    """Carga Mistral 7B cuantizado en 4-bit - MUCHO MÁS RÁPIDO"""

    # Verificar GPU disponible
    if torch.cuda.is_available():
        st.info(
            f"🎮 GPU detectada: {torch.cuda.get_device_name(0)}. Cargando modelo en GPU...",
            icon="⏳",
        )
        device = "cuda"
    else:
        st.warning(
            "⚠️ GPU no detectada. Usando CPU (mucho más lento).",
            icon="⚠️",
        )
        device = "cpu"

    base_model_name = "Qwen/Qwen2.5-14B-Instruct"  # Modelo base Qwen
    adapter_name = "qwen-2.5-14b-fac-finetuned"  # Tu modelo fine-tuneado con tus datos

    try:
        # Cargamos el tokenizador
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Cargamos el modelo base en 4-bit (pre-cuantizado)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,  # Compatible con 4-bit
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Menos memoria RAM
        )

        # CRÍTICO: El adaptador LoRA DEBE existir
        if not os.path.exists(adapter_name):
            st.error(
                f"❌ **ADAPTER LoRA NO ENCONTRADO**\n\n"
                f"Carpeta esperada: `{adapter_name}/`\n\n"
                f"**SOLUCIÓN:**\n"
                f"1. En tu otra computadora potente:\n"
                f"   → Ejecuta: `python lora.py`\n"
                f"   → Espera a que genere: `qwen-2.5-14b-fac-finetuned/`\n\n"
                f"2. Transfiere la carpeta completa a esta computadora\n\n"
                f"3. Colócala aquí: `{os.getcwd()}/`\n\n"
                f"4. Recarga esta página (F5)"
            )
            return None, None, None

        # Cargar y MANTENER como PEFT (sin fusionar)
        try:
            model = PeftModel.from_pretrained(model, adapter_name)
            st.info(
                f"✅ Adapter LoRA cargado (SIN fusionar para máxima compatibilidad)"
            )
            st.info("📚 Modelo entrenado con datos FAC")

            # NO fusionar - mantener como PEFT model para mejor compatibilidad
            total_params = sum(p.numel() for p in model.parameters())
            st.caption(f"📊 Parámetros: {total_params:,}")

        except Exception as e:
            st.error(f"❌ Error al cargar adapter: {str(e)}")
            return None, None, None

        model.eval()  # Modo inferencia

        if device == "cuda":
            st.success(
                f"🚀 ¡Mistral 7B 4-bit en GPU cargado! Respuestas RÁPIDAS garantizadas ⚡",
                icon="✅",
            )
        else:
            st.warning(
                "⚠️ Modelo cargado en CPU. Las respuestas serán más lentas.", icon="⚠️"
            )

        return model, tokenizer, device

    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info(
            "📌 Asegúrate de que la carpeta 'qwen-2.5-14b-fac-finetuned' esté en la misma carpeta que este script."
        )
        return None, None, None


# Intentar cargar el modelo (puede fallar si no hay GPU o no encuentra los archivos)
model, tokenizer, device = load_model()

if model is None or tokenizer is None:
    st.error(
        "❌ No se pudo cargar el modelo. Verifica que la carpeta esté en el lugar correcto."
    )
    st.stop()

# --- HISTORIAL DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CONTROL DE LONGITUD DE RESPUESTA ---
st.divider()
col1, col2 = st.columns([3, 1])
with col1:
    response_length = st.select_slider(
        "📏 Longitud de respuesta:",
        options=["Corta", "Mediana", "Larga"],
        value="Corta",
        help="Las respuestas largas pueden tomar más tiempo",
    )
with col2:
    if response_length == "Larga":
        st.warning("⏱️ Puede demorar 30-45s", icon="⚠️")
    elif response_length == "Mediana":
        st.info("⏱️ ~10-15s", icon="ℹ️")
    else:
        st.success("⏱️ ~5-8s", icon="✅")

st.divider()

# Mostrar los mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- INPUT DEL USUARIO ---
if prompt := st.chat_input("🎯 Ingresa tu consulta táctica/doctrinaria aquí..."):
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Mostrar un placeholder de respuesta mientras se genera
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("⏳ Generando respuesta...")

        try:
            # Limpiar el prompt del usuario
            clean_prompt = prompt.strip()

            # Formatear EXACTAMENTE como se entrenó
            # Durante training: "### Instruction:\n{instruction}\n\n### Response:\n{output}\n\n### Fuente:\n{fuente}"
            # Para inferencia, comenzamos el patrón pero NO incluimos output/fuente
            # El modelo las generará
            formatted_prompt = f"### Instruction:\n{clean_prompt}\n\n### Response:\n"

            # Tokenizar
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                add_special_tokens=True,
            ).to(model.device)

            # Generar respuesta
            start_time = time.time()

            # Ajustar tokens según longitud elegida
            if response_length == "Corta":
                max_tokens = 120
                min_tokens = 20
            elif response_length == "Mediana":
                max_tokens = 250
                min_tokens = 50
            else:  # Larga
                max_tokens = 400
                min_tokens = 100

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_tokens,
                    min_new_tokens=min_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    num_beams=1,
                    early_stopping=True,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generation_time = time.time() - start_time

            # Decodificar la SALIDA COMPLETA
            full_response = tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )  # LIMPIAR RESPUESTA: Extraer solo lo después de "### Response:"
            if "### Response:" in full_response:
                response_only = full_response.split("### Response:")[-1].strip()
            else:
                response_only = full_response.strip()

            # CRÍTICO: Si hay otro "###" después, detener ahí (evita múltiples ejemplos)
            # Buscar el PRÓXIMO ### (que sería otra sección)
            if "###" in response_only[10:]:  # Saltar los primeros 10 caracteres
                # Encontrar la posición del próximo ###
                next_section = response_only[10:].find("###")
                if next_section != -1:
                    response_only = response_only[: 10 + next_section].strip()

            # Separar la respuesta y la fuente PRIMERO (antes de otras limpiezas)
            # Separar respuesta y fuente ANTES de cualquier limpieza
            fuente = None
            response_only = full_response

            # Extraer solo lo después de "### Response:"
            if "### Response:" in response_only:
                response_only = response_only.split("### Response:")[-1].strip()
            else:
                response_only = response_only.strip()

            # *** EXTRAER FUENTE PRIMERO (antes de truncar) ***
            if "### Fuente:" in response_only:
                parts = response_only.split("### Fuente:")
                response_only = parts[0].strip()
                if len(parts) > 1:
                    fuente_raw = parts[1].strip()
                    # Tomar solo la primera línea después de "### Fuente:"
                    fuente = fuente_raw.split("\n")[0].strip()
                    # Limpiar caracteres especiales
                    fuente = (
                        fuente.replace("*", "")
                        .replace("_", "")
                        .replace("#", "")
                        .replace("`", "")
                        .strip()
                    )
                    # Validar que tenga contenido
                    if fuente and len(fuente) >= 2:
                        # Fuente válida, se mantiene
                        pass
                    else:
                        fuente = None

            # Detener en el próximo ### para no incluir múltiples ejemplos (DESPUÉS de extraer fuente)
            if "###" in response_only[10:]:
                next_section = response_only[10:].find("###")
                if next_section != -1:
                    response_only = response_only[: 10 + next_section].strip()

            # Limpiar cualquier marca de instrucción que quedó
            response_only = response_only.replace("### Instruction:", "").strip()
            response_only = response_only.replace("Instruction:", "").strip()
            response_only = response_only.replace("### Response:", "").strip()
            response_only = response_only.replace("### Response:", "").strip()

            # Validación: debe tener al menos algo de contenido (no vacío)
            if not response_only or len(response_only.strip()) < 2:
                response_only = "No pude generar una respuesta válida. Intenta reformular la pregunta."

            # ESTRATEGIA: Si el modelo no generó fuente, buscar en el dataset
            # Si la pregunta está en training data, usar la fuente de ahí
            if not fuente:
                fuente_dataset = find_fuente_in_training_data(clean_prompt)
                if fuente_dataset:
                    fuente = fuente_dataset

            final_response = response_only

            # Mostrar la respuesta generada con formato militar
            response_placeholder.markdown(final_response)

            # Mostrar la fuente SIEMPRE que exista
            if fuente and fuente.lower() not in [
                "",
                "none",
                "null",
                "n/a",
                "desconocida",
            ]:
                st.info(f"📋 **Fuente:** {fuente}")
            else:
                st.info("📋 **Fuente:** Desconocida")

        except Exception as e:
            error_msg = f"❌ Error al generar la respuesta: {str(e)}"
            st.error(error_msg)
            response_placeholder.markdown(
                "Lo siento, ocurrió un error. Por favor intenta de nuevo."
            )
            st.session_state.messages.pop()  # Eliminar el mensaje del usuario si falló
            st.stop()

    # Guardar la respuesta del modelo en el historial
    st.session_state.messages.append({"role": "assistant", "content": final_response})

    # Guardar registro en archivo de texto
    save_conversation_log(
        pregunta=clean_prompt,
        respuesta=final_response,
        tiempo_respuesta=generation_time,
        fuente=fuente,
    )
