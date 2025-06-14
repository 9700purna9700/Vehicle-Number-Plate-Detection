
import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np

st.title("🚗 Vehicle Number Plate Detector & Recognizer")

uploaded_file = st.file_uploader("Upload a car image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_column_width=True)

    # Load cascade
    plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, 1.1, 10)

    if len(plates) == 0:
        st.warning("No number plate detected!")
    else:
        for (x, y, w, h) in plates:
            plate = image[y:y + h, x:x + w]
            st.image(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), caption='Detected Plate', use_column_width=False)

            # OCR
            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            plate_text = pytesseract.image_to_string(plate_gray, config='--psm 8')
            st.success(f"Detected Number Plate Text: {plate_text.strip()}")
