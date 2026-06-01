    # حساب عدد المركبات (سيارة، حافلة، شاحنة)
    # أرقام الفئات في YOLO هي: 2: سيارة، 5: حافلة، 7: شاحنة
    vehicle_classes = [2, 5, 7] 
    car_count = 0
    for box in results[0].boxes:
        if int(box.cls) in vehicle_classes:
            car_count += 1
            
    st.success(f"تم اكتشاف {car_count} مركبة في الصورة!")
