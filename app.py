import streamlit as st
from ultralytics import YOLO
from PIL import Image

# تحميل النموذج
model = YOLO('yolov8n.pt')

st.title("Car Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)
    
    # التحليل
    results = model.predict(image)
    
    # عرض النتائج
    st.image(results[0].plot(), caption='Detected Image', use_column_width=True)
