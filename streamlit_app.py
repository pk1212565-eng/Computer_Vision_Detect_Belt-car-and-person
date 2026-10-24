import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="AI Safety Detection Dashboard", layout="wide")

st.title("🪖 AI Safety Violation Detection Dashboard")
st.write("Detect **Helmets, License Plates, and Seatbelts** using YOLOv11n Backend API.")

# Backend API Endpoint
# Agar 127.0.0.1 kaam na kare, toh isko try karein:
API_URL = "http://localhost:8000/predict"


# Sidebar configurations
st.sidebar.header("Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

# Bounding box color codes
COLOR_MAP = {
    "helmet": (0, 255, 0),         # Green
    "license plate": (0, 255, 255), # Yellow
    "seatbelt": (255, 0, 0),       # Blue
    "person": (255, 128, 0),       # Orange
    "car": (128, 0, 128)           # Purple
}

# Image uploader widget
uploaded_file = st.file_uploader("Upload an Image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display original image
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        # FIXED: use_column_width ko use_container_width se badal diya
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Detection Result")
        
        if st.button("Run AI Detection 🚀"):
            with st.spinner("Processing image via FastAPI backend..."):
                try:
                    # Send image to FastAPI endpoint
                    files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        predictions = response.json().get("predictions", [])
                        
                        # Convert PIL image to OpenCV format for drawing
                        img_cv = np.array(image)
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                        
                        detections_count = 0
                        
                        # Draw bounding boxes based on API response
                        for pred in predictions:
                            if pred["confidence"] >= conf_threshold:
                                detections_count += 1
                                x1, y1, x2, y2 = pred["box"]
                                label = f"{pred['class']} {int(pred['confidence']*100)}%"
                                
                                # Choose color based on class name
                                color = COLOR_MAP.get(pred["class"].lower(), (255, 255, 255))
                                
                                # Draw Rectangle & Label Text
                                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
                                cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Convert back to RGB for Streamlit rendering
                        img_result = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        # FIXED: yahan bhi use_container_width kar diya
                        st.image(img_result, use_container_width=True)
                        
                        if detections_count > 0:
                            st.success(f"Successfully detected {detections_count} objects!")
                        else:
                            st.info("No objects detected above the confidence threshold.")
                            
                    else:
                        st.error(f"Backend API error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to FastAPI backend. Is your `app.py` running on port 8000?")
