import joblib
import streamlit as st
import numpy as np
from sklearn.datasets import fetch_openml

# Load the trained model
model = joblib.load("mnist_ensemble_final.pkl")

# Load MNIST test data only once (caching)
@st.cache_data
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"].astype(int)
    X = X / 255.0  # Normalize
    return X, y

X_test, y_test = load_mnist()

# Streamlit UI
st.title("MNIST Digit Classifier")

# Initialize session state for test sample index
if "test_sample_idx" not in st.session_state:
    st.session_state.test_sample_idx = np.random.randint(len(X_test))

if st.button("Test with Real MNIST Sample"):
    st.session_state.test_sample_idx = np.random.randint(len(X_test))  # Update to a new random sample

# Retrieve the currently selected random sample
sample_idx = st.session_state.test_sample_idx
test_image = X_test[sample_idx].reshape(1, -1)
test_label = y_test[sample_idx]

# Make prediction
predicted_label = model.predict(test_image)

# Show result
st.image(test_image.reshape(28, 28), caption=f"MNIST Sample (True Label: {test_label})", width=150, clamp=True)
st.success(f"🧠 Prediction: **{predicted_label[0]}**")
