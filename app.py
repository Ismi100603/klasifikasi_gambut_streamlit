import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #003366;
    }
    .subtitle {
        font-size: 18px;
        color: #666;
    }
    .result {
        font-size: 22px;
        font-weight: bold;
        color: #009999;
    }
    </style>
""", unsafe_allow_html=True)

# Judul halaman
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">Klasifikasi Jenis Tanah Gambut</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Model CNN untuk mengenali jenis tanah gambut berdasarkan gambar</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

from utils import download_model
download_model()

# Load model
model = tf.keras.models.load_model('model/cnn_model.h5')

# Label kelas prediksi
label_kelas = ['Fibrik', 'Hemik', 'Nontanah', 'Saprik']

# Upload gambar
st.markdown("### 📤 Silakan upload gambar tanah:")
uploaded_file = st.file_uploader("Pilih gambar (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

# Layout kolom
col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    col1.image(image, caption='🖼️ Gambar yang Anda unggah', use_column_width=True)

    # Prediksi
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    pred = model.predict(img_array)
    pred_class = label_kelas[np.argmax(pred)]
    confidence = np.max(pred) * 100

    # Menampilkan hasil prediksi
    with col2:
        st.markdown("### 🔍 Hasil Klasifikasi")
        if pred_class == "Nontanah":
            st.error("⚠️ Ini **bukan** tanah gambut.")
        else:
            st.markdown(f"<div class='result'>Jenis tanah: <strong>{pred_class}</strong></div>", unsafe_allow_html=True)
            st.markdown(f"Tingkat kepercayaan: **{confidence:.2f}%**")

else:
    st.info("📎 Silakan unggah gambar terlebih dahulu untuk diprediksi.")

