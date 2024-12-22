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
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image when button is clicked
    if st.button("Predict"):
        st.write("Processing...")
        
        # Perform prediction
        results = model(image)
        
        # Get image with bounding boxes
        result_image = results[0].plot()

        # Display the result
        st.image(result_image, caption="Prediction Result", use_column_width=True)