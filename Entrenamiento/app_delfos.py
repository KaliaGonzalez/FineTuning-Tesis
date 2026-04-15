import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

st.set_page_config(page_title="Delfos Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Delfos - Asistente Institucional")
st.markdown("¡Hola! Soy Delfos, tu asistente inteligente. Escribe tu pregunta abajo.")


# --- CARGAR EL MODELO (En caché para no recargar cada vez) ---
@st.cache_resource
def load_model():
    """Carga el modelo TinyLlama optimizado para CPU"""
    st.info(
        "Cargando el modelo TinyLlama (optimizado para CPU)... Esto tardará 1-2 minutos la primera vez.",
        icon="⏳",
    )

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    try:
        # Cargamos el tokenizador
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configurar el pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Cargamos el modelo en CPU (más lento pero sin GPU)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        # Muy importante: modo de inferencia (no entrena, solo predice)
        model.eval()

        st.success("¡Modelo TinyLlama cargado con éxito en CPU!", icon="✅")
        return model, tokenizer

    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None, None


# Intentar cargar el modelo (puede fallar si no hay GPU o no encuentra los archivos)
model, tokenizer = load_model()

if model is None or tokenizer is None:
    st.error(
        "❌ No se pudo cargar el modelo. Verifica tu conexión a internet e intenta nuevamente."
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
            # Formatear el prompt como lo entrenaste
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

            # Tokenizar con límite de longitud
            inputs = tokenizer(
                formatted_prompt, return_tensors="pt", max_length=512, truncation=True
            ).to(model.device)

            # Generar respuesta con tiempo límite
            start_time = time.time()

            with torch.no_grad():  # Sin gradientes para ahorrar memoria
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,  # Reducido para ser más rápido en CPU
                    min_length=10,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=1,  # Beam search desactivado (más rápido)
                )

            generation_time = time.time() - start_time

            # Decodificar respuesta
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extraer solo la parte de la respuesta
            if "### Response:" in response_text:
                final_response = response_text.split("### Response:")[-1].strip()
            else:
                final_response = response_text.strip()

            # Mostrar la respuesta generada
            response_placeholder.markdown(final_response)

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
