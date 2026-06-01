import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# تحميل نموذج YOLO
model = YOLO('yolov8n.pt')

st.title("Car Detection App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # إجراء التحليل
    results = model(image)
    
    # عرض النتائج
    for result in results:
        res_plotted = result.plot()
        st.image(res_plotted, caption='Detected Image', use_column_width=True)
