import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

# Descargar el modelo desde Google Drive si no está presente
if not os.path.exists("binary_classifier_final.h5"):
    url = "https://drive.google.com/uc?id=1tGYr4EWuO0mAQNh9LB6q0w2pzdYT4WaQ" 
    gdown.download(url, "binary_classifier_final.h5", quiet=False)

# Cargar el modelo
model = load_model("binary_classifier_final.h5")

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






