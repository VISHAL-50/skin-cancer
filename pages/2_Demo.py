import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model_path = 'model/InceptionV3.h5'  # Replace with the path to your .h5 file
model = load_model(model_path)

# Define the class labels
class_labels = ["basal cell carcinoma", "non cancer", "melanoma"]

# Streamlit app

st.title("Skin Cancer Classifier")

# Slider

# Text input
name = st.text_input('Enter your name')

# Select box
gender = st.selectbox('Select a Gender', ["Male", "Female", "Other"])

age = st.slider('Select your age', 0, 100, 25)

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
if st.button("Classify"):
    # Preprocess the image for the model
    image = image.resize((224, 224))  # Adjust the size based on your model's input size
    image = np.asarray(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    prediction = model.predict(image)

   # Display the result
    st.header("Prediction:")
    if np.argmax(prediction) is not None:
        st.write(f"Name: {name}")
        st.write(f"Age: {age}")
        st.write(f"Gender: {gender}")
        st.write(f"Class: {class_labels[np.argmax(prediction)]}")
        st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
        n="For more information about "
        if class_labels[np.argmax(prediction)]==class_labels[0]:
            st.write(f"[{n+class_labels[np.argmax(prediction)]} ](https://www.ncbi.nlm.nih.gov/books/NBK482439/)")
        if class_labels[np.argmax(prediction)]==class_labels[2]:
            st.write(f"[{n+class_labels[np.argmax(prediction)]} ](https://www.cancer.gov/types/skin/patient/melanoma-treatment-pdq)")


st.write("Note :")
st.markdown('''
- Our model has 3000 images, 1/3rd are of non- cancerous skin and others are cancerous
- The image you're providing to this machine learning model should be clearly visible and the skin must be shave in the terms of avoiding any conflits
''')
