import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import os

st.set_page_config(page_title="Delfos Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Delfos - Asistente Institucional")
st.markdown("¡Hola! Soy Delfos, tu asistente inteligente. Escribe tu pregunta abajo.")


# --- CARGAR EL MODELO (En caché para no recargar cada vez) ---
@st.cache_resource
def load_model():
    """Carga TinyLlama - MÁS RÁPIDO para respuestas inmediatas"""

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

    base_model_name = "mistralai/Mistral-7B-v0.1"  # Mistral 7B
    adapter_name = "mistral-7b-fac-finetuned"  # Tu modelo fine-tuneado con tus datos

    try:
        # Cargamos el tokenizador
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Cargamos el modelo base
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=device,
            torch_dtype=torch.float16,  # Mistral usa float16
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Menos memoria RAM
        )

        # IMPORTANTE: Cargar el adaptador LoRA con tus datos entrenados
        try:
            if os.path.exists(adapter_name):
                model = PeftModel.from_pretrained(model, adapter_name)
                st.success(f"✅ Adaptador LoRA cargado desde '{adapter_name}'")
                st.info("📚 El modelo usa TUS DATOS DE ENTRENAMIENTO")
            else:
                st.error(
                    f"❌ Carpeta '{adapter_name}' NO ENCONTRADA.\n\n"
                    f"⚠️ El modelo respondará en INGLÉS genérico.\n\n"
                    f"Solución: Ejecuta primero: python lora_simple.py"
                )
        except Exception as e:
            st.error(
                f"❌ Error cargando adaptador: {str(e)}\n\n"
                f"El modelo respondará en INGLÉS genérico."
            )

        model.eval()  # Modo inferencia

        if device == "cuda":
            st.success(
                f"🚀 ¡Mistral 7B en GPU cargado! Respuestas rápidas garantizadas 🎉",
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
if prompt := st.chat_input("Escribe tu pregunta para Delfos aquí..."):
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

            # Formatear el prompt EXACTAMENTE como se entrenó
            # (sin Input porque no lo tiene)
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
                    max_new_tokens=350,  # Limitado para evitar demasiado contenido
                    min_new_tokens=50,  # Mínimo para respuestas coherentes
                    do_sample=False,  # SIN sampling = determinístico, no alucina
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=1,
                    repetition_penalty=1.5,  # MÁS ALTO (1.5 vs 1.15) - penaliza fuertemente repeticiones
                    length_penalty=0.8,  # Penaliza longitudes extremas
                    early_stopping=True,  # Detiene temprano si alcanza EOS
                )

            generation_time = time.time() - start_time

            # Decodificar la SALIDA COMPLETA
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # LIMPIAR RESPUESTA: Extraer solo lo después de "### Response:"
            if "### Response:" in full_response:
                # Obtener todo después de "### Response:"
                response_only = full_response.split("### Response:")[-1].strip()
            else:
                response_only = full_response.strip()

            # Separar la respuesta y la fuente
            fuente = None
            if "### Fuente:" in response_only:
                parts = response_only.split("### Fuente:")
                response_only = parts[0].strip()
                fuente = parts[1].strip() if len(parts) > 1 else None

            # Limpiar cualquier marca de instrucción que quedó
            response_only = response_only.replace("### Instruction:", "").strip()
            response_only = response_only.replace("Instruction:", "").strip()

            # Si la respuesta está vacía, algo malo pasó
            if not response_only or len(response_only) < 5:
                response_only = "Lo siento, no pude generar una respuesta válida."

            final_response = response_only

            # Mostrar la respuesta generada
            response_placeholder.markdown(final_response)

            # Mostrar la fuente si existe
            if fuente:
                st.info(f"📚 **Fuente:** {fuente}", icon="📖")

            # Mostrar tiempo de procesamiento (debug)
            st.caption(f"⏱️ Tiempo de respuesta: {generation_time:.2f} segundos")

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
