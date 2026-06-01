import streamlit as st
from ultralytics import YOLO

# استخدام نموذج خفيف جداً
model = YOLO('yolov8n.pt')

st.title("Car AI App")

uploaded_file = st.file_uploader("ارفع صورة السيارة...", type=["jpg", "png"])

if uploaded_file is not None:
    # الاكتشاف مباشرة
    results = model.predict(uploaded_file)
    # عرض النتيجة
    st.image(results[0].plot(), use_column_width=True)
