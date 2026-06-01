import streamlit as st
from ultralytics import YOLO
from PIL import Image

# تحميل النموذج
model = YOLO('yolov8n.pt')

st.title("Car Detection App - Pro")

# إضافة شريط للتحكم في الدقة
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)
    
    # إجراء التحليل
    results = model.predict(image, conf=confidence)
    
    # حساب عدد السيارات
    # نفلتر النتائج لنبحث عن فئة السيارات (class 2 في نموذج YOLO)
    car_count = 0
    for box in results[0].boxes:
        if int(box.cls) == 2:  # الرقم 2 يمثل "سيارة" في نموذج coco
            car_count += 1
            
    # عرض النتيجة
    st.success(f"تم اكتشاف {car_count} سيارة في الصورة!")
    
    # عرض الصورة بالنتائج
    res_plotted = results[0].plot()
    st.image(res_plotted, caption='Detected Image', use_column_width=True)
