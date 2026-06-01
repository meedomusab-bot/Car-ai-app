import streamlit as st
from ultralytics import YOLO
from PIL import Image

# تحميل النموذج
model = YOLO('yolov8n.pt')

st.title("Car Intelligence System 🚗")

uploaded_file = st.file_uploader("ارفع صورة السيارة هنا...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة الأصلية', use_column_width=True)
    
    # الاكتشاف
    results = model.predict(image)
    
    # عرض النتائج
    st.image(results[0].plot(), caption='تم اكتشاف المركبات بنجاح')
    
    st.success("تم تحليل الصورة بنجاح باستخدام YOLOv8")
    st.info("نصيحة: لقراءة اللوحات بدقة عالية، يُفضل استخدام واجهة API مثل (Google Vision) بدلاً من `easyocr` الثقيلة على السيرفر.")
