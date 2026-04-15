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
    """Carga el modelo Mistral 7B entrenado con LoRA - OPTIMIZADO PARA GPU NVIDIA"""

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

    base_model_name = "unsloth/mistral-7b-v0.3-bnb-4bit"
    adapter_path = "mistral-7b-fac-finetuned"  # La carpeta que descargaste de Colab

    try:
        # Cargamos el tokenizador desde los archivos entrenados
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Cargamos el modelo base (en 4 bits para ahorrar memoria)
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device,  # Usa GPU si está disponible, sino CPU
            trust_remote_code=True,
        )

        # Fusionamos el modelo base con tus pesos LoRA entrenados
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()  # Modo inferencia

        if device == "cuda":
            st.success(
                f"🎉 ¡Tu modelo en GPU cargado exitosamente! {torch.cuda.get_device_name(0)}",
                icon="✅",
            )
        else:
            st.warning(
                "⚠️ Modelo cargado en CPU. Las respuestas serán lentas.", icon="⚠️"
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
                    max_new_tokens=120,  # REDUCIDO: Menos tokens = más rápido
                    min_length=5,  # REDUCIDO para respuestas más cortas
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_beams=1,
                    repetition_penalty=1.1,  # Evita repeticiones
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
