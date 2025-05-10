# app.py
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tempfile
import os

# Title
st.title("Defective vs Proper Image Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

# Load model (assumes you’ve saved it as 'mobilenetv2_finetuned.h5')
@st.cache_resource
def load_mymodel():
    model = load_model("mobilenetv2_finetuned.h5")
    return model

model = load_mymodel()

# Preprocess function
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)

# Run prediction
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(img)
    prediction = model.predict(processed)[0][0]
    label = "Defective" if prediction >= 0.5 else "Proper"

    st.markdown(f"### Prediction: `{label}`")
    st.markdown(f"Confidence: `{prediction:.2f}`")

