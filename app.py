import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO('yolov8n.pt')

st.title("Car Detection App - Pro")

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)
    
    results = model.predict(image, conf=confidence)
    
    res_plotted = results[0].plot()
    st.image(res_plotted, caption='Detected Image', use_column_width=True)
