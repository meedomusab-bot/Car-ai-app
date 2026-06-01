import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# استخدام cache لتقليل استهلاك الذاكرة
@st.cache_resource
def get_model():
    return YOLO('yolov8n.pt')

model = get_model()

st.title("Car Intelligence System 🚗")

uploaded_file = st.file_uploader("ارفع صورة السيارة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة الأصلية', use_column_width=True)
    
    # تنفيذ الكشف
    results = model.predict(image)
    
    # عرض النتيجة
    res_plotted = results[0].plot()
    st.image(res_plotted, caption='النتائج', use_column_width=True)
