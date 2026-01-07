import streamlit as st
import tensorflow as tf
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
