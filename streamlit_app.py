import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Klasifikasi Kucing & Anjing",
    page_icon="🐶",
    layout="centered"
)

st.title("Klasifikasi Gambar Kucing & Anjing")
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



#tombol prediksi

if uploaded_file:
    if st.button("🔍 Prediksi"):
        with st.spinner("Sedang memproses..."):
            img = image.resize((96, 96))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            model = tf.keras.models.load_model("model.h5")
            pred = model.predict(img)[0][0]

            label = "🐱 Kucing" if pred < 0.5 else "🐶 Anjing"
            confidence = (1 - pred) * 100 if pred < 0.5 else pred * 100

            st.success(f"Hasil: **{label}**")
            st.info(f"Tingkat Keyakinan: **{confidence:.2f}%**")


# ===== UPLOAD GAMBAR =====
uploaded_file = st.file_uploader(
    "Upload gambar daun mangga",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((96, 96))
    st.image(image, caption=uploaded_file.name, use_column_width=True)

    # ===== PREPROCESS =====
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ===== PREDIKSI =====
    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100
    class_index = np.argmax(prediction)
    result = class_names[class_index]

    # ===== OUTPUT =====
    st.markdown(
        f"<div class='result-box'>🌱 Penyakit: {result}</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='conf-box'>📊 Confidence: {confidence:.2f}%</div>",
        unsafe_allow_html=True
    )
