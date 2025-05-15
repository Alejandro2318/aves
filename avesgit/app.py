import streamlit as st
from keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import preprocess_input
import os
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# Nombres de las aves
names = [
    'Ramphocelus sp', 'Tangara Cabeza Gris', 'Tangara Coroniazul', 'Tangara de Delattre', 'Tangara Dorsirroja',
    'Tangara Flamígera', 'Tangara Luctuosa', 'Tangara Montana', 'Tangara Negra', 'Tangara Pechicanela'
]

# Cargar modelo
import gdown


# ID de tu archivo de Google Drive
file_id = "1M_LF862ZvnFJeGDFr9egv0hiXBu_f3DE"

# Enlace directo (convertido)
url = f"https://drive.google.com/uc?id={file_id}"

# Nombre temporal del archivo descargado
model_path = "best_model.keras"

# Descargar el modelo desde Drive
gdown.download(url, model_path, quiet=False)

# Cargar el modelo descargado
model = load_model(model_path)
print("✅ Modelo cargado correctamente desde Google Drive")


# Función para cargar y preprocesar la imagen
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Interfaz
st.title("Clasificador de Aves")
st.write("Sube una imagen de un ave y te diremos a qué especie pertenece.")

# Subir imagen
uploaded_file = st.file_uploader(" Cargar Imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    temp_image_path = 'temp_image.jpg'
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    img_array = load_and_preprocess_image(temp_image_path)

    # Predicción
    predictions = model.predict(img_array)[0]
    pred_class = np.argmax(predictions)
    pred_prob = np.max(predictions)

    st.write(f"🔍 *Predicción:* {names[pred_class]}")
    st.write(f"📊 *Probabilidad:* {pred_prob * 100:.2f}%")

    # Mostrar imagen original
    img_cv = cv2.imread(temp_image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title(f"{names[pred_class]} ({pred_prob * 100:.2f}%)")
    ax.axis('off')
    st.pyplot(fig)

    # Leer base de datos
    df = pd.read_excel("aves_info_corregido.xlsx", header=0)
    df.columns = df.columns.str.strip()

    # Información principal
    info_ave = df[df["Nombre"] == names[pred_class]]
    if not info_ave.empty:
        st.subheader("📚 Información del ave:")
        st.write(f"**Nombre común:** {info_ave['Nombre_corto'].values[0]}")
        st.write(f"**Descripción:** {info_ave['Descripción'].values[0]}")
        st.write(f"**Hábitat:** {info_ave['Hábitat'].values[0]}")
        st.write(f"**Alimentación:** {info_ave['Alimentación'].values[0]}")
        st.write(f"**Estado de conservación:** {info_ave['Estado'].values[0]}")
        st.markdown(f"[🔎 Buscar más en Google]({info_ave['Busqueda_Google'].values[0]})", unsafe_allow_html=True)

        # Mostrar imágenes forzadas del mismo tamaño
        st.subheader("📸 Imágenes del ave:")
        img_urls = [
            info_ave['Imagen_1'].values[0],
            info_ave['Imagen_2'].values[0]
        ]

        col1, col2 = st.columns(2)

        def mostrar_img(url, columna, caption):
            if pd.notna(url) and url.strip() != "":
                try:
                    response = requests.get(url)
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    img = img.resize((300, 300))
                    columna.image(img, caption=caption)
                except Exception as e:
                    columna.warning(f"Error al cargar imagen: {e}")
            else:
                columna.warning("No se encontró imagen.")

        mostrar_img(img_urls[0], col1, "Imagen 1")
        mostrar_img(img_urls[1], col2, "Imagen 2")

    else:
        st.warning("No se encontró información sobre esta ave en la base de datos.")

    # Mostrar las 3 especies con mayor probabilidad (con imágenes)
    st.subheader("Resultados Busqueda:")
    top3_indices = predictions.argsort()[-3:][::-1]

    for i in top3_indices:
        especie = names[i]
        probabilidad = predictions[i] * 100
        st.markdown(f" {especie} — {probabilidad:.2f}%")

        info_opcional = df[df["Nombre"] == especie]
        if not info_opcional.empty:
            img_url = info_opcional['Imagen_1'].values[0]
            if pd.notna(img_url) and img_url.strip() != "":
                try:
                    response = requests.get(img_url)
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    img = img.resize((300, 300))
                    st.image(img, caption=f"{especie} ({probabilidad:.2f}%)")
                except Exception as e:
                    st.warning(f"No se pudo cargar la imagen para {especie}: {e}")
            else:
                st.warning(f"No hay imagen disponible para {especie}")
        else:
            st.warning(f"No hay información disponible para {especie}")

    # Borrar imagen temporal
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
