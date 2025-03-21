import joblib
import streamlit as st
import numpy as np
import os
import cv2
from streamlit_drawable_canvas import st_canvas
from sklearn.datasets import fetch_openml

# Load trained model
model = joblib.load("mnist_ensemble_final.pkl")

# Initialize session states
if "drawn_digit" not in st.session_state:
    st.session_state.drawn_digit = None

# UI
st.title("MNIST Digit Classifier")

st.write(
    "🖊️ Draw a digit (0-9) below or click **Test with Real MNIST Sample**."
)

# --- Function to Process Drawn Digits ---
def preprocess_drawn_digit(img):
    """ Process the drawn digit to match MNIST format """
    # Convert to grayscale
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    # Invert colors (white digit on black background)
    ## img = 255 - img  

    # Apply threshold to remove noise
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # Find bounding box of the digit
    coords = cv2.findNonZero(img)
    
    if coords is None or len(coords) < 10:  # Ensure enough pixels are detected
        return None  # Return None to indicate no digit found

    x, y, w, h = cv2.boundingRect(coords)
    
    # If bounding box is too small, return None
    if w < 5 or h < 5:
        st.warning("⚠️ Bounding box too small!")
        return None

    digit = img[y:y+h, x:x+w]  # Crop to bounding box

    # Resize the digit to 20x20 (maintaining aspect ratio)
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    # Center the digit in a 28x28 image
    centered_img = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    centered_img[y_offset:y_offset+20, x_offset:x_offset+20] = digit

    # Normalize pixel values to [0,1] like MNIST
    centered_img = centered_img.astype(np.float32) / 255.0
    
    return centered_img

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
    processed_img = preprocess_drawn_digit(img)  # Use improved function

    if processed_img is None:
        st.error("⚠️ No digit detected! Please draw more clearly.")
    else:
        # Show processed image
        st.image(processed_img, caption="Processed Image", width=150, clamp=True)

        st.session_state.drawn_digit = processed_img

if st.session_state.drawn_digit is not None and st.button("Classify Drawn Digit"):
    drawn_img = st.session_state.drawn_digit.reshape(1, -1)
    drawn_prediction = model.predict(drawn_img)
    st.success(f"🧠 Prediction: **{drawn_prediction[0]}**")
