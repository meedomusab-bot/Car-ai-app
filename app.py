import streamlit as st
from ultralytics import YOLO
from PIL import Image
import easyocr

# تحميل النماذج
model = YOLO('yolov8n.pt')
reader = easyocr.Reader(['en']) 

st.title("Car Detection & Plate Reader")

uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)
    
    # تحديد موقع السيارة
    results = model.predict(image)
    
    # فحص المركبات
    for box in results[0].boxes:
        if int(box.cls) in [2, 5, 7]: # سيارة، حافلة، شاحنة
            # قص صورة السيارة للتركيز عليها
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image.crop((x1, y1, x2, y2))
            
            # قراءة النصوص في الصورة المقصوصة
            text_results = reader.readtext(crop)
            for (bbox, text, prob) in text_results:
                if prob > 0.2:
                    st.success(f"تم اكتشاف لوحة/رقم: {text}")

    st.image(results[0].plot(), caption='Detected Image', use_column_width=True)
