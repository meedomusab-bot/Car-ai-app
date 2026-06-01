import streamlit as st
from PIL import Image

st.title("Car Detection App")
st.write("التطبيق قيد التحديث - جاري تهيئة الأنظمة...")

uploaded_file = st.file_uploader("ارفع صورة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_column_width=True)
    st.write("تم استلام الصورة بنجاح!")
