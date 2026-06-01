import streamlit as st
from ultralytics import YOLO
from PIL import Image
import easyocr
import numpy as np

# تحميل النماذج
@st.cache_resource
def load_models():
    model = YOLO('yolov8n.pt')
    # نقوم بتحميل القارئ مرة واحدة فقط
    reader = easyocr.Reader(['en'])
    return model, reader

model, reader = load_models()

st.title("Car Intelligence System 🚗")

uploaded_file = st.file_uploader("ارفع صورة السيارة هنا...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة الأصلية', use_column_width=True)
    
    # الاكتشاف
    results = model.predict(image)
    
    for box in results[0].boxes:
        if int(box.cls) in [2, 5, 7]: # سيارة، حافلة، شاحنة
            # 1. قص السيارة
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            car_crop = image.crop((x1, y1, x2, y2))
            
            # 2. تحويل الصورة المقصوصة إلى تنسيق يفهمه easyocr (numpy array)
            car_crop_np = np.array(car_crop)
            
            # 3. قراءة النص (OCR)
            text_results = reader.readtext(car_crop_np)
            
            st.write("---")
            st.image(car_crop, caption="المركبة المكتشفة")
            
            found_text = False
            for (bbox, text, prob) in text_results:
                if prob > 0.2:
                    st.success(f"رقم اللوحة المحتمل: {text}")
                    found_text = True
            
            if not found_text:
                st.info("لم يتم العثور على لوحة واضحة.")

    st.image(results[0].plot(), caption='النتائج')
