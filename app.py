import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load trained model
model = load_model("breed_classifier.h5")

# Automatically read class names from train folder
train_folder = "data/train"
class_names = sorted(os.listdir(train_folder))  # sorted for consistency

st.title("🐄 Cattle & Buffalo Breed Classifier Prototype")
st.write(
    "Upload an image of a cattle/buffalo and the app will predict its breed.\n\n"
    "✅ Hackathon prototype demo."
)

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = 100 * np.max(tf.nn.softmax(prediction[0]))

    # Confidence threshold check
    threshold = 00.0  # percentage
    if confidence < threshold:
        st.warning("Predicted Breed: **Unknown Breed** ❌")
        
    else:
        breed = class_names[index]
        st.success(f"Predicted Breed: **{breed}** 🐂")
        

