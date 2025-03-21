# 1. Importing Necessary Modules
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.datasets import fetch_openml

# 2. Loading the Data
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
np.random.seed(42)


# 3. Load the trained MNIST model
model = joblib.load("mnist_model.pkl")

st.title("MNIST Digit Classifier")
st.write("Draw a number between 0 and 9 in the box below:")

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

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img.reshape(1, -1)

    if st.button("Klassificera"):
        prediction = model.predict(img)
        st.success(f"Modellen förutspår: **{prediction[0]}**")
