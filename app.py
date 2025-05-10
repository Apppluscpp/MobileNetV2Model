import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Defect Detection", layout="centered")
st.title("🛠️ MobileNetV2 Defect Detector")

# ---------------------- Sidebar UI Controls ----------------------
st.sidebar.header("🔧 Preprocessing Options")
clahe_enabled = st.sidebar.checkbox("Apply CLAHE", value=True)
gray_enabled = st.sidebar.checkbox("Convert to Grayscale before CLAHE", value=False)
clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0)
tile_size = st.sidebar.slider("CLAHE Tile Size", 4, 16, 8)

st.sidebar.header("🧬 Morphology Settings")
min_blob_area = st.sidebar.slider("Min Contour Area to Show", 0, 500, 10)

st.sidebar.header("🔍 Prediction Control")
threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5)

# ---------------------- Model Loader ----------------------
@st.cache_resource
def load_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.load_weights("mobilenetv2_weights.h5")
    return model

model = load_model()

# ---------------------- Preprocessing ----------------------
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    if gray_enabled:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if clahe_enabled:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        cl = clahe.apply(l)
        img = cv2.merge((cl, a, b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    return np.expand_dims(img_norm, axis=0), img_rgb

# ---------------------- Morphology Feature Extraction ----------------------
def extract_morphology_features(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blob_areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) >= min_blob_area]
    total_blob_area = sum(blob_areas)
    output = np.stack([gray, gray, gray], axis=-1)
    cv2.drawContours(output, [cnt for cnt in contours if cv2.contourArea(cnt) >= min_blob_area], -1, (0, 255, 0), 1)

    return {
        "num_blobs": len(blob_areas),
        "avg_blob_area": np.mean(blob_areas) if blob_areas else 0,
        "max_blob_area": max(blob_areas) if blob_areas else 0,
        "blob_area_ratio": total_blob_area / (gray.shape[0] * gray.shape[1])
    }, output

# ---------------------- File Upload and Inference ----------------------
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(original_rgb, caption="Original Image", use_column_width=True)

    processed, processed_rgb = preprocess_image(image_bgr)
    prediction = model.predict(processed)[0][0]

    label = "🔴 Defective" if prediction >= threshold else "🟢 Proper"
    confidence = prediction if prediction >= threshold else 1 - prediction

    st.markdown(f"### Prediction: {label}")
    st.metric("Confidence", f"{confidence:.2%}")
    st.progress(prediction)
    st.image(processed_rgb, caption="Preprocessed Image", use_column_width=True)

    if st.checkbox("Show Morphology Features"):
        features, morph_image = extract_morphology_features(processed[0])
        with st.expander("Morphology Feature Details"):
            st.json(features)
        st.image(morph_image, caption="Contours (Filtered by Min Area)", use_column_width=True)

# ---------------------- Realtime Camera Input ----------------------
st.markdown("---")
st.header("📷 Real-Time Camera Inference")
camera_img = st.camera_input("Take a photo")

if camera_img:
    file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(original_rgb, caption="Captured Image", use_column_width=True)

    processed, processed_rgb = preprocess_image(image_bgr)
    prediction = model.predict(processed)[0][0]

    label = "🔴 Defective" if prediction >= threshold else "🟢 Proper"
    confidence = prediction if prediction >= threshold else 1 - prediction

    st.markdown(f"### Camera Prediction: {label}")
    st.metric("Confidence", f"{confidence:.2%}")
    st.progress(prediction)
    st.image(processed_rgb, caption="Preprocessed Camera Image", use_column_width=True)

    if st.checkbox("Show Morphology Features (Camera)"):
        features, morph_image = extract_morphology_features(processed[0])
        with st.expander("Camera Morphology Feature Details"):
            st.json(features)
        st.image(morph_image, caption="Camera Contours (Filtered)", use_column_width=True)
