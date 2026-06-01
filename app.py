import streamlit as st
from PIL import Image, ImageOps

st.title("Car Viewer App 🚗")

uploaded_file = st.file_uploader("ارفع صورة السيارة...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة')

    # معالجة بسيطة للصورة للتأكد أن الكود يعمل
    st.write("تم معالجة الصورة بنجاح!")
    gray_image = ImageOps.grayscale(image)
    st.image(gray_image, caption='نسخة أبيض وأسود')
