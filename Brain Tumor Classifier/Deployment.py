# brain_app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image

st.title("Brain Tumor Classification")

@st.cache_resource
def load_my_model():
    return load_model("Brain Classifier.h5")

model = load_my_model()

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_container_width=True)

    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    predicted_class = class_names[np.argmax(pred)]

    st.write(f"Predicted Tumor Type: **{predicted_class}**")
    
# streamlit run "c:/Users/LOQ/Desktop/Brain Toumer Classifier/brain.py"
