import time
import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🔍",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0D1117;
    color: #E6EDF3;
}

/* Title */
h1 {
    font-family: 'Space Mono', monospace !important;
    color: #58A6FF !important;
    font-size: 2rem !important;
    letter-spacing: -1px;
}

/* Subheaders */
h3 {
    font-family: 'Space Mono', monospace !important;
    color: #8B949E !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* Prediction card */
.pred-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-left: 4px solid #58A6FF;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}

.pred-label {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #58A6FF;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.pred-conf {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: #8B949E;
    margin-top: 0.2rem;
}

/* Timing row */
.timing-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}

.timing-box {
    background: #0D1117;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #3FB950;
    flex: 1;
    text-align: center;
}

/* Prob bar row */
.bar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #8B949E;
}

/* Info box */
.info-box {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #8B949E;
    margin-top: 1.5rem;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #161B22;
    border: 2px dashed #30363D;
    border-radius: 8px;
    padding: 1rem;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #8B949E !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    color: #58A6FF !important;
}

/* Divider */
hr {
    border-color: #21262D;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LABELS = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]

LABEL_EMOJI = {
    "airplane": "✈️", "automobile": "🚗", "bird": "🐦",
    "cat": "🐱", "deer": "🦌", "dog": "🐶", "frog": "🐸",
    "horse": "🐴", "ship": "🚢", "truck": "🚛"
}

# ── Model path: looks one folder up from app/ for the .h5 file ────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "cifar10_cnn_model.h5")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "cifar10_cnn_model.h5")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "cifar10_cnn_model.h5"


# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(pil_img: Image.Image) -> np.ndarray:
    """Resize to 32x32, normalize to [0,1], add batch dim."""
    img = pil_img.convert("RGB").resize((32, 32), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # shape: (1, 32, 32, 3)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(model, pil_img: Image.Image):
    t_pre0 = time.perf_counter()
    x = preprocess(pil_img)
    t_pre1 = time.perf_counter()

    logits = model.predict(x, verbose=0)
    t_inf1 = time.perf_counter()

    # Convert logits to probabilities with softmax
    probs = tf.nn.softmax(logits[0]).numpy()
    idx = int(np.argmax(probs))

    return {
        "label":         LABELS[idx],
        "emoji":         LABEL_EMOJI[LABELS[idx]],
        "confidence":    float(probs[idx]),
        "probs":         probs,
        "preprocess_ms": (t_pre1 - t_pre0) * 1000,
        "inference_ms":  (t_inf1 - t_pre1) * 1000,
        "total_ms":      (t_inf1 - t_pre0) * 1000,
    }


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 CIFAR-10 Classifier")
st.markdown("**AI Aggregator** — Upload an image and the CNN model will classify it into one of 10 categories.")
st.markdown("---")

# Load model
model = load_model(MODEL_PATH)

if model is None:
    st.error(
        "⚠️ Model file not found! Make sure `cifar10_cnn_model.h5` exists in your "
        "project root folder (one level above the `app/` folder). "
        "Run the training notebook first."
    )
    st.stop()

# Upload
st.markdown("### Upload an Image")
uploaded = st.file_uploader(
    "Drag and drop or browse — JPG, PNG, JPEG accepted",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.markdown("""
    <div class="info-box">
    📌 <strong>How to use:</strong> Upload any image above. The model will resize it to 32×32 pixels
    and predict which of the 10 CIFAR-10 categories it belongs to.<br><br>
    <strong>Categories:</strong> ✈️ Airplane &nbsp;|&nbsp; 🚗 Automobile &nbsp;|&nbsp; 🐦 Bird &nbsp;|&nbsp;
    🐱 Cat &nbsp;|&nbsp; 🦌 Deer &nbsp;|&nbsp; 🐶 Dog &nbsp;|&nbsp; 🐸 Frog &nbsp;|&nbsp;
    🐴 Horse &nbsp;|&nbsp; 🚢 Ship &nbsp;|&nbsp; 🚛 Truck
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Process image
image = Image.open(uploaded)
result = predict(model, image)

# Layout: image left, result right
st.markdown("---")
col1, col2 = st.columns([1, 1.4], gap="large")

with col1:
    st.markdown("### Your Image")
    st.image(image, use_container_width=True, caption=uploaded.name)
    st.markdown(f"<div class='bar-label'>Original size: {image.width} × {image.height} px → resized to 32 × 32</div>",
                unsafe_allow_html=True)

with col2:
    st.markdown("### Prediction")
    st.markdown(f"""
    <div class="pred-card">
        <div class="pred-label">{result['emoji']} {result['label']}</div>
        <div class="pred-conf">Confidence: <strong>{result['confidence']*100:.1f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Timing")
    st.markdown(f"""
    <div class="timing-row">
        <div class="timing-box">⚙️ Preprocess<br>{result['preprocess_ms']:.1f} ms</div>
        <div class="timing-box">🧠 Inference<br>{result['inference_ms']:.1f} ms</div>
        <div class="timing-box">⏱️ Total<br>{result['total_ms']:.1f} ms</div>
    </div>
    """, unsafe_allow_html=True)

# Probability bar chart
st.markdown("---")
st.markdown("### Confidence Across All 10 Classes")

import pandas as pd
prob_df = pd.DataFrame({
    "Class": [f"{LABEL_EMOJI[l]} {l}" for l in LABELS],
    "Confidence (%)": [round(float(p) * 100, 2) for p in result["probs"]]
}).set_index("Class")

st.bar_chart(prob_df, y="Confidence (%)", height=300)

# Footer comparison note
st.markdown("---")
st.markdown("""
<div class="info-box">
⚠️ <strong>Note on real-world accuracy:</strong> This model was trained on 32×32 pixel CIFAR-10 images.
Real photos are much larger and more complex, so predictions on user-uploaded photos may be less accurate
than the ~75% test accuracy achieved on the CIFAR-10 test set. This difference is called
<em>distribution shift</em> and is a key topic in the project analysis.
</div>
""", unsafe_allow_html=True)

st.markdown("<br><div style='text-align:center; color:#30363D; font-family:Space Mono; font-size:0.7rem'>FSE560 · AI Image Classifier · T3</div>", unsafe_allow_html=True)