
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


# =========================================
# 1. CONFIGURACIÓN GENERAL
# =========================================

st.set_page_config(
    page_title="PneumoAI | Análisis de Radiografías",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "modelos"
CONFIG_PATH = MODELOS_DIR / "configuracion_modelo.json"


@st.cache_data
def cargar_configuracion():
    with open(CONFIG_PATH, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


@st.cache_resource
def cargar_modelo(ruta):
    import tensorflow as tf

    return tf.keras.models.load_model(ruta, compile=False)


config = cargar_configuracion()
UMBRAL = float(config["umbral"])
IMG_SIZE = tuple(config["tamano_imagen"])
MODEL_PATH = MODELOS_DIR / config["modelo"]




# =========================================
# 2. ESTILOS VISUALES - TEMA CLARO
# =========================================

st.markdown("""
<style>
    /* Fondo principal */
    .stApp {
        background: #F6F9FC;
        color: #203A53;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Títulos y textos */
    h1, h2, h3 {
        color: #163C62 !important;
        letter-spacing: -0.5px;
    }

    p {
        color: #536E85;
    }

    /* Contenedores principales */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border: 1px solid #D7E5F0;
        border-radius: 16px;
    }

    /* Tarjetas de indicadores */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D7E5F0;
        border-radius: 14px;
        padding: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: #60788F;
    }

    div[data-testid="stMetricValue"] {
        color: #164B73;
    }

    /* Carga de radiografías */
    div[data-testid="stFileUploader"] {
        background: #EFF7FC;
        border: 1px dashed #72B7D1;
        border-radius: 12px;
        padding: 12px;
    }

    /* Botón de análisis */
    div.stButton > button[kind="primary"] {
        background: #148FAF;
        border: none;
        color: #FFFFFF;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.7rem 1.5rem;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #0E7190;
        color: #FFFFFF;
    }

    /* Nombre de la plataforma */
    .marca {
        font-size: 29px;
        font-weight: 800;
        color: #163C62;
        margin-bottom: 0;
    }

    .marca span {
        color: #148FAF;
    }

    .submarca {
        color: #60788F;
        font-size: 12px;
        letter-spacing: 1.6px;
    }

    /* Encabezado */
    .eyebrow {
        color: #148FAF;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-top: 30px;
    }

    .hero-title {
        color: #163C62;
        font-size: 40px;
        font-weight: 800;
        line-height: 1.15;
        margin: 12px 0;
    }

    .hero-desc {
        color: #536E85;
        font-size: 16px;
        max-width: 750px;
        line-height: 1.6;
    }

    /* Títulos de secciones */
    .section-title {
        color: #163C62;
        font-size: 19px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* Resultado: PNEUMONIA */
    .result-positive {
        background: #FFF2DA;
        border: 1px solid #E6BA6F;
        color: #94600F;
        padding: 14px;
        border-radius: 10px;
        font-weight: 700;
        margin: 12px 0;
    }

    /* Resultado: NORMAL */
    .result-negative {
        background: #E3F5EA;
        border: 1px solid #86CFA9;
        color: #176B46;
        padding: 14px;
        border-radius: 10px;
        font-weight: 700;
        margin: 12px 0;
    }

    /* Valores de predicción */
    .result-value {
        color: #163C62;
        font-size: 38px;
        font-weight: 800;
        margin: 4px 0;
    }

    .result-label {
        color: #60788F;
        font-size: 13px;
        margin-top: 15px;
    }

    /* Aviso médico */
    .notice {
        background: #EAF4FA;
        color: #37566F;
        padding: 17px;
        border: 1px solid #CDDFEA;
        border-radius: 12px;
        font-size: 13px;
        line-height: 1.6;
        margin-top: 18px;
    }

    /* Pie de página */
    .footer {
        color: #71889A;
        text-align: center;
        font-size: 12px;
        margin-top: 35px;
    }

    /* Separadores */
    hr {
        border-color: #D7E5F0;
    }

    /* Adaptación a pantallas pequeñas */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 29px;
        }

        .block-container {
            padding-top: 1rem;
        }

        .result-value {
            font-size: 30px;
        }
    }
</style>
""", unsafe_allow_html=True)



# =========================================
# 3. PREPROCESAMIENTO
# =========================================

def preparar_radiografia(imagen):
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    imagen = imagen.convert("RGB")
    imagen = np.asarray(imagen)

    imagen = tf.convert_to_tensor(imagen)
    imagen = tf.image.resize(imagen, IMG_SIZE)
    imagen = tf.cast(imagen, tf.float32)

    # Misma normalización del entrenamiento
    imagen = preprocess_input(imagen)

    return tf.expand_dims(imagen, axis=0)


# =========================================
# 4. ENCABEZADO
# =========================================

st.markdown("""
<div class="marca">✚ Pneumo<span>AI</span></div>
<div class="submarca">MEDICAL IMAGING INTELLIGENCE</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="eyebrow">ANÁLISIS DE RADIOGRAFÍAS</div>
<div class="hero-title">
Detección inteligente<br>de neumonía
</div>
<div class="hero-desc">
Plataforma experimental de inteligencia artificial
para el análisis de radiografías de tórax mediante
redes neuronales convolucionales y Transfer Learning.
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================================
# 5. INDICADORES DEL MODELO
# =========================================

k1, k2, k3 = st.columns(3)

with k1:
    st.metric("Arquitectura", "MobileNetV2")

with k2:
    st.metric("Recall en prueba", "98,72 %")

with k3:
    st.metric("AUC en prueba", "0,9596")

st.write("")
st.divider()


# =========================================
# 6. PANEL DE ANÁLISIS
# =========================================

col_izq, col_der = st.columns([1.15, 1], gap="large")

with col_izq:
    with st.container(border=True):

        st.markdown(
            '<div class="section-title">🩻 Radiografía de tórax</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Carga una imagen para realizar el análisis."
        )

        archivo = st.file_uploader(
            "Seleccionar radiografía",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        imagen = None

        if archivo is not None:
            try:
                imagen = Image.open(archivo)
                imagen.load()

                st.image(
                    imagen,
                    caption="Imagen seleccionada",
                    width="stretch"
                )

            except (UnidentifiedImageError, OSError):
                st.error("El archivo no contiene una imagen válida.")

        analizar = st.button(
            "Analizar radiografía",
            type="primary",
            width="stretch",
            disabled=imagen is None
        )


# =========================================
# 7. RESULTADOS
# =========================================

with col_der:
    with st.container(border=True):

        st.markdown(
            '<div class="section-title">◎ Resultado del análisis</div>',
            unsafe_allow_html=True
        )

        if analizar and imagen is not None:

            with st.spinner("Procesando radiografía..."):
                tensor = preparar_radiografia(imagen)

                modelo = cargar_modelo(str(MODEL_PATH))
                salida = modelo.predict(
                    tensor,
                    verbose=0
                )

                prob_neumonia = float(salida[0][0])

            if prob_neumonia >= UMBRAL:
                clase = "PNEUMONIA"

                st.markdown(
                    '<div class="result-positive">'
                    '⚠ Posible neumonía detectada'
                    '</div>',
                    unsafe_allow_html=True
                )

            else:
                clase = "NORMAL"

                st.markdown(
                    '<div class="result-negative">'
                    '✓ Clasificación NORMAL'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                '<div class="result-label">CLASE PREDICHA</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="result-value">{clase}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="result-label">'
                'PROBABILIDAD ESTIMADA DE NEUMONÍA'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="result-value">'
                f'{prob_neumonia * 100:.2f}%'
                '</div>',
                unsafe_allow_html=True
            )

            st.progress(
                max(0.0, min(1.0, prob_neumonia))
            )

            st.caption(
                f"Umbral de clasificación: {UMBRAL:.2f}"
            )

            st.info(
                "Esta probabilidad representa la salida "
                "del modelo y no una certeza clínica calibrada."
            )

        else:
            st.info(
                "Carga una radiografía y presiona "
                "'Analizar radiografía' para visualizar "
                "el resultado."
            )


# =========================================
# 8. INFORMACIÓN Y ADVERTENCIAS
# =========================================

st.write("")

with st.expander("Información técnica del modelo"):
    st.write("**Arquitectura:** MobileNetV2")
    st.write("**Estrategia:** Transfer Learning")
    st.write("**Clasificación:** NORMAL / PNEUMONIA")
    st.write("**Entrada:** 224 × 224 píxeles")
    st.write(f"**Umbral:** {UMBRAL:.2f}")
    st.write("**Accuracy en prueba:** 80,29 %")
    st.write("**Recall en prueba:** 98,72 %")
    st.write("**AUC en prueba:** 0,9596")
    st.caption(
        "Resultados experimentales sobre el conjunto "
        "de prueba de Kaggle. La independencia por "
        "paciente no está completamente verificada."
    )

st.markdown("""
<div class="notice">
<strong>⚕ Aviso importante</strong><br>
PneumoAI es un prototipo académico de Deep Learning.
Sus resultados no constituyen un diagnóstico médico
ni sustituyen la valoración de un profesional de salud.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
PneumoAI · Proyecto académico de Inteligencia Artificial<br>
MobileNetV2 · TensorFlow · Streamlit
</div>
""", unsafe_allow_html=True)
