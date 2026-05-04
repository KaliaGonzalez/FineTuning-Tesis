import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import os

st.set_page_config(page_title="Delfos Chatbot", page_icon="🛩️", layout="wide")
st.title("🛩️ DELFOS - Sistema de Inteligencia Aérea")
st.markdown("**Bienvenido, oficial.** Soy DELFOS, el Sistema de Inteligencia y Doctrina de la Fuerza Aérea Colombiana. A tu servicio en cualquier momento. ✈️")

# Agregar un separador visual militar
st.divider()


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

    base_model_name = "unsloth/mistral-7b-v0.3-bnb-4bit"  # ¡4-BIT = MUCHO MÁS RÁPIDO!
    adapter_name = "mistral-7b-fac-finetuned"  # Tu modelo fine-tuneado con tus datos

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
                f"   → Espera a que genere: `mistral-7b-fac-finetuned/`\n\n"
                f"2. Transfiere la carpeta completa a esta computadora\n\n"
                f"3. Colócala aquí: `{os.getcwd()}/`\n\n"
                f"4. Recarga esta página (F5)"
            )
            return None, None, None
        
        # Cargar y fusionar el adapter
        try:
            model = PeftModel.from_pretrained(model, adapter_name)
            st.info(f"✅ Adapter LoRA cargado")
            
            # Fusionar
            model = model.merge_and_unload()
            st.success(f"✅ LoRA Adapter cargado y fusionado")
            st.info("📚 Modelo entrenado con datos FAC")
            
            # Verificar que el adapter se cargó (comparar parámetros)
            total_params = sum(p.numel() for p in model.parameters())
            st.caption(f"📊 Parámetros totales del modelo: {total_params:,}")
            
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
            "📌 Asegúrate de que la carpeta 'mistral-7b-fac-finetuned' esté en la misma carpeta que este script."
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

            # Formatear el prompt EXACTAMENTE como se entrenó en lora.py
            # El modelo fue entrenado con este patrón:
            # "### Instruction:\n{instruction}\n\n### Response:\n{output}\n\n### Fuente:\n{fuente}"
            # Para inferencia, terminamos sin la fuente para que el modelo la genere
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

            with torch.no_grad():  # Sin gradientes para ahorrar memoria
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=220,  # Moderado para velocidad y calidad
                    min_new_tokens=45,   
                    do_sample=False,  # CRÍTICO: determinístico, nunca aleatorio
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=1,  # Greedy = rápido
                    repetition_penalty=1.8,  # Penaliza repeticiones
                    early_stopping=True,
                )

            generation_time = time.time() - start_time

            # Decodificar la SALIDA COMPLETA
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # DIAGNÓSTICO: Si la respuesta parece basura, mostrar warning
            if len(full_response) < 50 or "###" not in full_response:
                st.warning(
                    "⚠️ **ADVERTENCIA**: La respuesta parece incompleta o incorrecta.\n\n"
                    "Esto indica que el LoRA adapter puede no estar bien entrenado.\n\n"
                    "**Verifica en la otra computadora:**\n"
                    "1. El entrenamiento completó correctamente\n"
                    "2. Los archivos en `mistral-7b-fac-finetuned/` se guardaron\n"
                    "3. Que sean > 1MB cada uno"
                )

            # LIMPIAR RESPUESTA: Extraer solo lo después de "### Response:"
            if "### Response:" in full_response:
                # Obtener todo después de "### Response:"
                response_only = full_response.split("### Response:")[-1].strip()
            else:
                response_only = full_response.strip()

            # Separar la respuesta y la fuente PRIMERO (antes de otras limpiezas)
            fuente = None
            if "### Fuente:" in response_only:
                # Dividir por la marca de fuente
                parts = response_only.split("### Fuente:")
                response_only = parts[0].strip()
                
                if len(parts) > 1:
                    fuente_raw = parts[1].strip()
                    # Tomar solo hasta el primer salto de línea o caracteres especiales
                    fuente = fuente_raw.split("\n")[0].split("\r")[0].strip()
                    # Limpiar caracteres especiales pero mantener espacios
                    fuente = fuente.replace("*", "").replace("_", "").replace("#", "").replace("**", "").strip()
                    # Si quedó vacío o es un marcador, ignorar
                    if not fuente or len(fuente) < 2:
                        fuente = None
            
            # Si no encontramos fuente con ese formato, intentar otros formatos
            if not fuente and "Fuente:" in response_only:
                parts = response_only.split("Fuente:")
                response_only = parts[0].strip()
                if len(parts) > 1:
                    fuente = parts[1].strip().split("\n")[0].strip()
                    fuente = fuente.replace("*", "").replace("_", "").replace("#", "").strip()
                    if not fuente or len(fuente) < 2:
                        fuente = None

            # Limpiar cualquier marca de instrucción que quedó
            response_only = response_only.replace("### Instruction:", "").strip()
            response_only = response_only.replace("Instruction:", "").strip()
            response_only = response_only.replace("### Response:", "").strip()

            # Validación: respuesta debe tener contenido mínimo
            if not response_only or len(response_only) < 15:
                response_only = "No pude generar una respuesta válida. Intenta reformular la pregunta."

            final_response = response_only

            # Mostrar la respuesta generada con formato militar
            response_placeholder.markdown(final_response)

            # Mostrar la fuente si existe con icono militar
            if fuente and fuente.lower() not in ["", "none", "null", "n/a"]:
                st.info(f"🎖️ **Clasificación de Fuente:** {fuente}", icon="�")

            # Mostrar tiempo de procesamiento (debug)
            st.caption(f"⏱️ Tiempo de procesamiento: {generation_time:.2f} segundos")

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
