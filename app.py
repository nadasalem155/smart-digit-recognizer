import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import joblib
import pandas as pd
import time

# Cache model and scaler loading for efficiency
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('mlp_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("Digit Draw & Predict with Interactive Questions and Challenge Timer")

# Create a canvas for the user to draw a digit
canvas_result = st_canvas(
    fill_color="#000000",          # Black background
    stroke_width=15,               # Brush thickness
    stroke_color="#FFFFFF",        # White brush color
    background_color="#000000",    # Black canvas background
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Challenge mode toggle and timer setup
challenge_mode = st.checkbox("Enable Challenge Mode (10 seconds timer)")
time_limit = 10

if challenge_mode:
    st.write(f"You have {time_limit} seconds to draw your digit!")

# Process the canvas image if drawing exists
if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 3] != 0):
    img = canvas_result.image_data

    # Convert RGBA image to grayscale
    img_gray = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGBA2GRAY)

    # Invert colors to match MNIST style (white digit on black background)
    img_invert = cv2.bitwise_not(img_gray)

    # Resize to 28x28 pixels
    img_resized = cv2.resize(img_invert, (28, 28), interpolation=cv2.INTER_AREA)

    # Flatten image to 1D array
    img_flatten = img_resized.flatten().reshape(1, -1)

    # Convert to DataFrame with the exact columns names as training data
    column_names = [f"pixel_{i}" for i in range(784)]
    img_df = pd.DataFrame(img_flatten, columns=column_names)

    # Apply scaling using the pre-fitted scaler
    img_scaled = scaler.transform(img_df)

    # Display the processed input image
    st.image(img_resized, width=100, caption="Processed Input (28x28)")

    # Predict probabilities for each digit class
    probs = model.predict_proba(img_scaled)[0]
    pred = np.argmax(probs)
    confidence = probs[pred]

    st.markdown(f"### Prediction: *{pred}* with confidence *{confidence:.2f}*")

    # If confidence is low, ask clarifying questions to improve prediction
    if confidence < 0.6:
        st.write("I'm not very confident. Can you answer some quick questions?")

        circle = st.radio("Does the digit contain a circle or round part?", ("Yes", "No"))
        straight = st.radio("Does the digit have straight lines?", ("Yes", "No"))

        # Set of all digits 0-9
        possible_digits = set(range(10))

        # Narrow possible digits based on circle presence
        if circle == "Yes":
            possible_digits -= {1, 7}
        else:
            possible_digits &= {1, 7}

        # Narrow possible digits based on straight lines presence
        if straight == "Yes":
            possible_digits &= {0, 1, 4, 7, 9}
        else:
            possible_digits -= {0, 1, 4, 7, 9}

        # Filter predictions probabilities by possible digits
        filtered_probs = [(digit, prob) for digit, prob in enumerate(probs) if digit in possible_digits]

        if filtered_probs:
            pred, confidence = max(filtered_probs, key=lambda x: x[1])
            st.markdown(f"### Updated Prediction: *{pred}* with confidence *{confidence:.2f}*")
        else:
            st.write("Sorry, your answers conflict with prediction probabilities.")

else:
    st.write("Please draw a digit on the canvas.")

# Challenge timer with progress bar
if challenge_mode:
    if st.button("Start Challenge"):
        progress = st.progress(0)
        for i in range(time_limit):
            time.sleep(1)
            progress.progress((i + 1) / time_limit)
        st.write("Time's up! Submit your drawing prediction now.")