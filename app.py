import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Page config
st.set_page_config(page_title="Smart Digit Recognizer", page_icon="🧠", layout="centered")

# Cache the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.keras")

model = load_model()

# Custom CSS (Keep only necessary styles)
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 20px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #6200ea;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #3700b3;
    }
    .prediction-box {
        background: linear-gradient(135deg, #bbdefb, #9575cd);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        color: #1a1a1a;
        font-size: 56px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-left: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title">🧠 Smart Digit Recognizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Draw a digit (0-9) on the canvas and click Predict to see the result!</div>', unsafe_allow_html=True)

# Use columns to put canvas and prediction side by side
col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.write("")  # empty placeholder for spacing
    if st.button("✨ Predict"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data.astype("uint8"))
            img = ImageOps.grayscale(img)
            img = img.resize((28, 28))
            img_array = np.array(img).astype("float32") / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            prediction = model.predict(img_array)
            pred_class = np.argmax(prediction, axis=1)[0]

            emoji_map = {i: f"{i}\u20E3" for i in range(10)}
            emoji = emoji_map.get(pred_class, "🔢")

            st.markdown(
                f'<div class="prediction-box">{emoji}</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("Please draw a digit before predicting.")