import joblib
import streamlit as st
import numpy as np
import os
import cv2
from streamlit_drawable_canvas import st_canvas
from sklearn.datasets import fetch_openml

# Load trained model
model = joblib.load("mnist_ensemble_final.pkl")

# Load MNIST dataset
@st.cache_data
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"].astype(int)
    X = X / 255.0  # Normalize
    return X, y

X_test, y_test = load_mnist()

# Initialize session states
if "test_sample_idx" not in st.session_state:
    st.session_state.test_sample_idx = np.random.randint(len(X_test))
if "drawn_digit" not in st.session_state:
    st.session_state.drawn_digit = None

# UI
st.title("MNIST Digit Classifier")

st.write(
    "🖊️ Draw a digit (0-9) below or click **Test with Real MNIST Sample**."
)

# --- Test Button for Real MNIST Sample ---
if st.button("Test with Real MNIST Sample"):
    st.session_state.test_sample_idx = np.random.randint(len(X_test))  # Update sample

sample_idx = st.session_state.test_sample_idx
test_image = X_test[sample_idx].reshape(1, -1)
test_label = y_test[sample_idx]

# Predict and display test sample
predicted_label = model.predict(test_image)
st.image(test_image.reshape(28, 28), caption=f"MNIST Sample (True Label: {test_label})", width=150, clamp=True)
st.success(f"🧠 Prediction: **{predicted_label[0]}**")


# --- Drawing Canvas ---
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Process and classify drawn digit
if canvas_result.image_data is not None:
    img = canvas_result.image_data[..., :3]  # Drop alpha channel
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = 255 - img  # Invert colors
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    # Store drawn digit in session state
    st.session_state.drawn_digit = img / 255.0  # Normalize

    # Show processed image
    st.image(img, caption="Processed Image", width=150, clamp=True)

if st.session_state.drawn_digit is not None and st.button("Classify Drawn Digit"):
    drawn_img = st.session_state.drawn_digit.reshape(1, -1)
    drawn_prediction = model.predict(drawn_img)
    st.success(f"🧠 Prediction: **{drawn_prediction[0]}**")
