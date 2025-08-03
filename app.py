import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Cargar el modelo (asegúrate que el archivo .h5 esté en el mismo repo)
model = load_model("resnet50_binary_classifier_final.h5")

# Interfaz
st.title("Detector de Retinopatía")
st.write("Sube una imagen de ojo para analizar si hay retinopatía.")

# Subir imagen
uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagen cargada", use_column_width=True)

    # Preprocesar imagen
    img = img.resize((224, 224))  # Ajusta al tamaño usado en tu ResNet50
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    pred = model.predict(img_array)
    resultado = "Con retinopatía" if pred[0][0] > 0.5 else "Sin retinopatía"

    st.subheader(f"Resultado: {resultado}")
