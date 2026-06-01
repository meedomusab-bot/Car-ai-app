import streamlit as st
from ultralytics import YOLO
from PIL import Image
import easyocr
import numpy as np

# تحميل النموذج (يتم تحميله مرة واحدة)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

st.title("Car Detection System 🚗")

uploaded_file = st.file_uploader("ارفع صورة السيارة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة الأصلية', use_column_width=True)
    
    # الاكتشاف
    results = model.predict(image)
    
    # عرض النتائج
    st.image(results[0].plot(), caption='المركبات المكتشفة', use_column_width=True)
    
    # زر إضافي لقراءة اللوحة (اختياري للتقليل من استهلاك الذاكرة)
    if st.button("محاولة قراءة اللوحة"):
        reader = easyocr.Reader(['en'])
        for box in results[0].boxes:
            if int(box.cls) in [2, 5, 7]: # سيارة
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = image.crop((x1, y1, x2, y2))
                res = reader.readtext(np.array(crop))
                for (bbox, text, prob) in res:
                    st.success(f"الرقم المكتشف: {text}")
