import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
model = YOLO("blood_cell_detection.pt")

# Streamlit app
st.title("Blood Cell Prediction")

# Image uploader
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Process the image when button is clicked
    if st.button("Predict"):
        st.write("Processing...")
        
        # Perform prediction
        results = model(image)
        
        # Get image with bounding boxes
        result_image = results[0].plot()

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.image(result_image, caption="Prediction Result", use_column_width=True)
