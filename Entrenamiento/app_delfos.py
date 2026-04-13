import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

st.set_page_config(page_title="Delfos Chatbot", page_icon="🤖")
st.title("🤖 Delfos - Asistente Institucional")
st.markdown("¡Hola! Soy Delfos, tu asistente inteligente. Escribe tu pregunta abajo.")


# --- CARGAR EL MODELO (En caché para no recargar cada vez) ---
@st.cache_resource
def load_model():
    st.info(
        "Cargando el modelo Mistral entrenado... Esto tardará unos minutos.", icon="⏳"
    )

    base_model_name = "unsloth/mistral-7b-v0.3-bnb-4bit"
    adapter_path = (
        "mistral-7b-fac-finetuned"  # Esta es la carpeta generada por tu entrenamiento
    )

    # 1. Cargamos el tokenizador desde tu adaptador
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # 2. Configuración para cargarlo en 4-bits
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # 3. Cargamos el modelo base usando GPU automáticamente
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 4. Le inyectamos tu conocimiento (Los pesos LoRA entrenados)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()  # <--- MUY IMPORTANTE: Poner el modelo en modo lectura/inferencia, no en entrenamiento

    st.success("¡Modelo Mistral Finetuned cargado con éxito!", icon="✅")
    return model, tokenizer


# Intentar cargar el modelo (puede fallar si no hay GPU o no encuentra los archivos)
try:
    model, tokenizer = load_model()
except Exception as e:
    st.error(
        f"Error al cargar el modelo: {e}. Asegúrate de haberlo entrenado primero y tener los recursos necesarios."
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

    # Formatear el prompt como lo entrenaste
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Delfos está pensando..."):
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,  # Límite de la respuesta
                do_sample=True,  # Para dar variabilidad a la respuesta
                temperature=0.3,  # 0.1 a 0.3 es bueno para respuestas certeras y serias
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,  # Evita que el modelo se enrede en un bucle infinito
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decodificar el texto generado, ignorando el prompt inicial
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extraer solo la parte correspondiente a la respuesta
            final_response = response_text.split("### Response:\n")[-1].strip()

            st.markdown(final_response)

    # Guardar la respuesta del modelo en el historial
    st.session_state.messages.append({"role": "assistant", "content": final_response})
