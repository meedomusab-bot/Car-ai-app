import streamlit as st

st.title("Car AI App - Testing Mode")
st.write("إذا رأيت هذه الصفحة، فالسيرفر يعمل الآن بنجاح!")

if st.button("تحميل محرك الذكاء الاصطناعي"):
    try:
        from ultralytics import YOLO
        st.write("جاري تحميل النموذج، يرجى الانتظار...")
        model = YOLO('yolov8n.pt')
        st.success("تم تحميل النموذج بنجاح!")
    except Exception as e:
        st.error(f"حدث خطأ أثناء التحميل: {e}")
