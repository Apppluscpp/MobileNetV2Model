import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

st.set_page_config(page_title="Defect Detection", layout="centered")
st.title("🛠️ MobileNetV2 Defect Detector")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenetv2_model.keras")

model = load_model()

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    label = "🟥 Defective" if prediction >= 0.5 else "🟩 Proper"
    st.markdown(f"### Prediction: {label}")
    st.markdown(f"**Confidence:** {prediction:.2%}")
