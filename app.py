import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Page config
st.set_page_config(page_title="Smart Digit Recognizer", page_icon="🧠", layout="centered")

# Cache the model to avoid reload on every rerun
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.keras")

model = load_model()

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Global styling */
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    /* Title styling */
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 10px;
    }
    /* Subtitle styling */
    .subtitle {
        font-size: 20px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    /* Canvas container styling */
    .canvas-container {
        border: 3px solid #ddd;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        background-color: #fff;
        padding: 10px;
        margin-bottom: 20px;
        display: inline-block; /* Ensure container wraps tightly around canvas */
    }
    /* Remove extra black box around canvas */
    .stCanvas {
        background-color: transparent !important; /* Ensure no extra background */
    }
    /* Predict button styling */
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
    /* Prediction output styling with gradient background */
    .prediction-box {
        background: linear-gradient(135deg, #bbdefb, #9575cd); /* Soft blue to purple gradient */
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        color: #1a1a1a;
        font-size: 56px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-top: 20px;
    }
    /* Warning message styling */
    .stAlert {
        background-color: #ffebee;
        color: #c62828;
        border-radius: 8px;
        padding: 15px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title">🧠 Smart Digit Recognizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Draw a digit (0-9) on the canvas and click Predict to see the result!</div>', unsafe_allow_html=True)

# Canvas with enhanced styling
st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
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
st.markdown('</div>', unsafe_allow_html=True)

# Predict button
if st.button("✨ Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas to grayscale PIL image
        img = Image.fromarray(canvas_result.image_data.astype("uint8"))
        img = ImageOps.grayscale(img)
        img = img.resize((28, 28))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array)
        pred_class = np.argmax(prediction, axis=1)[0]

        # Emoji map
        emoji_map = {
            0: "0️⃣",
            1: "1️⃣",
            2: "2️⃣",
            3: "3️⃣",
            4: "4️⃣",
            5: "5️⃣",
            6: "6️⃣",
            7: "7️⃣",
            8: "8️⃣",
            9: "9️⃣"
        }
        emoji = emoji_map.get(pred_class, "🔢")

        # Show prediction in a large, styled box
        st.markdown(
            f'<div class="prediction-box">Predicted Digit: {emoji}</div>',
            unsafe_allow_html=True
        )
    else:
        st.warning("Please draw a digit before predicting.")