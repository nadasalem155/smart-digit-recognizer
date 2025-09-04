import streamlit as st
# Page config
st.set_page_config(page_title="Smart Digit Recognizer", page_icon="🧠", layout="centered")
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Cache the model to avoid reload on every rerun
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.keras")

model = load_model()

st.title("🧠 Smart Digit Recognizer")
st.write("Draw a digit (0-9) and click Predict!")

# Canvas
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

        # Emoji map (replace number)
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

        # Show prediction in large styled box under canvas
        st.markdown(
            f"""
            <div style="
                display:inline-block;
                background-color:#fff176; /* light attractive yellow */
                border-radius:20px;
                padding:25px 40px;
                text-align:center;
                color:#1a1a1a; /* dark text color */
                font-size:40px;
                font-weight:bold;
                box-shadow: 3px 3px 15px rgba(0,0,0,0.3);
                margin-top:15px;
            ">
                The predicted digit is: {emoji}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Please draw a digit before predicting.")