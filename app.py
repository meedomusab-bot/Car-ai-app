import streamlit as st
from PIL import Image, ImageOps

st.title("Car AI System 🚗")
uploaded_file = st.file_uploader("ارفع صورة...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة')
    
    # تحميل الذكاء الاصطناعي فقط عند الطلب
    if st.button("اكتشف السيارات في الصورة"):
        with st.spinner("جاري تحميل النموذج والتحليل..."):
            try:
                from ultralytics import YOLO
                model = YOLO('yolov8n.pt')
                results = model.predict(image)
                st.image(results[0].plot(), caption='النتيجة')
            except Exception as e:
                st.error(f"حدث خطأ أثناء التحميل: {e}")
