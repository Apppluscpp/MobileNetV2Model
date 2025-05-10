import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

st.set_page_config(page_title="Defect Detector", layout="centered")
st.title("🛠️ Defect Detection with MobileNetV2")

# Load model from unzipped folder
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenetv2_model.h5")

model = load_model()

# Preprocessing function
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))            # Resize to model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Convert BGR to RGB
    img = img.astype(np.float32) / 255.0         # Normalize to [0, 1]
    return np.expand_dims(img, axis=0)           # Add batch dimension

# File uploader UI
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and decode image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # Load as BGR

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    # Predict
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)[0][0]  # Get sigmoid output

    label = "🟥 Defective" if prediction >= 0.5 else "🟩 Proper"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    st.markdown(f"### Prediction: {label}")
    st.markdown(f"**Confidence:** {confidence:.2%}")
