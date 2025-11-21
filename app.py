import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from homer_bart_prediction import predict_image,preprocess_image
import streamlit as st
import logging

st.set_page_config(page_title="Homer vs Bart Classifier", page_icon="🟡")

# Cache model for performance
@st.cache_resource
def load_my_model():
    return load_model("homer_bart_model.keras")

model = load_my_model()

st.title(" Homer vs Bart Classifier")
st.write("Upload an image and the model will predict whether it's **Homer** or **Bart**.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded:
    # Show image
    st.image(uploaded, caption="Uploaded Image", width=300, height=300)

    # Read image as bytes
    bytes_data = uploaded.read()

    # Predict
    label, prob = predict_image(model, bytes_data)
    logging.info(f"Predicted Label: {label}, Probability: {prob:.4f}")

    st.subheader(f"So the name of the character you uploaded is : **{label}**")

   
    st.write(f"Upload another image of your friend {label} ")


    st.write(f"Enjoyed the app? Share it with your friends! 🚀")

