import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Semen Straw Counting", page_icon="🔍")

def load_model():
    model = YOLO("./weight_file/yolo11_best_300.pt")
    return model

def detect_objects(model, image, conf_thresh):
    results = model(image, conf=conf_thresh)
    
    boxes = results[0].boxes  # Extract bounding box object
    img = np.array(image)
    
    straw_count = 0
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        
        if cls == 1: 
            straw_count += 1
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return img, straw_count

def main():
    st.title("Semen Straw Counting")
    
    model = load_model()
    
    conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Process Image"):
            processed_img, straw_count = detect_objects(model, image, conf_thresh)
            st.image(processed_img, caption="Predicted Image", use_column_width=True)
            st.write(f"### Total Straw Count: {straw_count}")
        
if __name__ == "__main__":
    main()
