import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.models import Model

st.title("Defect Detection (MobileNetV2 Rebuilt)")

@st.cache_resource
def build_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    # Load weights from file if you have .weights.h5 (optional)
    model.load_weights("mobilenetv2_weights.h5")  # OPTIONAL: If you saved weights only
    return model

model = build_model()

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image")

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]
    label = "🟥 Defective" if prediction >= 0.5 else "🟩 Proper"
    st.markdown(f"### Prediction: {label}")
    st.markdown(f"Confidence: {prediction:.2%}")
