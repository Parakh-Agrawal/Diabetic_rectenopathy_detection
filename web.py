import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from tensorflow.keras.applications.efficientnet import preprocess_input

# ✅ Page config (must be first)
st.set_page_config(page_title="Diabetic Retinopathy Detector", layout="centered")

# ✅ Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("efficientnetb3_retinopathy.h5")

model = load_model()

# ✅ Class labels (order must match training)
class_labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# ✅ Preprocessing function (RGB, 512x512, EfficientNet-preprocessed)
def preprocess_image(file):
    img = Image.open(file).convert('RGB')
    img_resized = img.resize((512, 512))
    img_array = np.array(img_resized)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0), img_resized

# ✅ App Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Diabetic Retinopathy Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload a retinal image to analyze diabetic retinopathy using EfficientNetB3.</p>", unsafe_allow_html=True)
st.markdown("---")

# ✅ Sidebar
st.sidebar.header("*Options*")
page_selection = st.sidebar.radio(
    "Navigate",
    ("Home", "About Us", "Help", "Contact")
)

# ✅ Home Page
if page_selection == "Home":
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Upload a retinal image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    st.markdown("---")

    if uploaded_file is not None:
        # Display image
        st.subheader("Image Preview")
        st.image(uploaded_file, use_column_width=True)

        # Preprocessing
        input_img, resized_img = preprocess_image(uploaded_file)

        # Predict on button click
        st.subheader("Model Prediction")
        st.markdown("<p style='font-size: 16px;'>Click the button below to run the model and get predictions:</p>", unsafe_allow_html=True)

        if st.button("Run Model", key="run_model"):
            with st.spinner("Analyzing the image... Please wait."):
                prediction = model.predict(input_img)
                idx = int(np.argmax(prediction))
                label = class_labels[idx]

            st.success(f"*Diabetic Retinopathy Detected*: {label}")
            st.markdown("<h5 style='color: #4CAF50;'>Analysis Complete!</h5>", unsafe_allow_html=True)

            # Optional: Class probabilities
            st.subheader("Prediction Confidence")
            st.bar_chart(pd.Series(prediction[0], index=class_labels.values()))

            # Comparison view
            st.markdown("---")
            st.subheader("Image Preview")
            col1, col2 = st.columns(2)

            with col1:
                st.image(resized_img, caption="Resized Input (512x512)", use_column_width=True)
                st.markdown("<h5 style='text-align: center;'>Processed Image</h5>", unsafe_allow_html=True)

            with col2:
                st.image(uploaded_file, caption="Original Uploaded Image", use_column_width=True)
                st.markdown("<h5 style='text-align: center;'>Original Image</h5>", unsafe_allow_html=True)

# ✅ About Us Page
elif page_selection == "About Us":
    st.header("About Us")
    st.write("""
    This application uses a deep learning model (EfficientNetB3) trained on the APTOS 2019 dataset to assist in early detection of diabetic retinopathy.
    
    Our mission is to empower healthcare professionals with reliable AI-powered tools.
    """)

# ✅ Help Page
elif page_selection == "Help":
    st.header("Help")
    st.write("""
    - **Step 1:** Go to the 'Home' page.
    - **Step 2:** Upload a valid retinal image (.jpg, .png).
    - **Step 3:** Click "Run Model" to classify the stage of diabetic retinopathy.
    - The app will display both the result and prediction confidence.
    """)

# ✅ Contact Page
elif page_selection == "Contact":
    st.header("Contact")
    st.write("""
    Reach out to us with questions or feedback:
    
    - 📧 Email: support@drdetector.ai  
    - ☎️ Phone: +123-456-7890  
    - 🌐 Website: www.dr-detector.ai
    """)
