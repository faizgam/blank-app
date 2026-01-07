import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Klasifikasi Kucing & Anjing",
    page_icon="🐶",
    layout="centered"
)

st.title("🐱🐶 Klasifikasi Gambar Kucing & Anjing")
st.write("""
Aplikasi ini menggunakan **Convolutional Neural Network (CNN)**
untuk mengklasifikasikan gambar **kucing** dan **anjing**.
""")

uploaded_file = st.file_uploader(
    "📤 Upload gambar (jpg/png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Gambar")
    if uploaded_file:
        st.image(image, use_column_width=True)

with col2:
    st.subheader("📊 Hasil Prediksi")

