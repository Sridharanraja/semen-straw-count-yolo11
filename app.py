import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Semen Straw Counting", page_icon="🔍")

@st.cache_resource()
def load_model():
    model = YOLO("./weight_file/yolo11_best_300.pt")
    return model

def detect_objects(model, image, conf_thresh,max_det=2500):
    results = model.predict(image, conf=conf_thresh, max_det=max_det)  #imgsz=imgsz, iou=iou, 

    if not results or len(results[0].boxes) == 0:
        return np.array(image), 0  # Return original image with zero count if no detections

    boxes = results[0].boxes  # Get bounding boxes
    img = np.array(image)  
    straw_count = 0  

    for box in boxes:
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class label
        
        if conf >= conf_thresh and cls == 1:  # Check confidence and class label
            straw_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert to int
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

    return img, straw_count

def main():
    st.title("Semen Straw Counting")
    
    model = load_model()
    
    conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.25)
    
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Process Image"):
            processed_img, straw_count = detect_objects(model, image, conf_thresh,max_det=2500)
            st.image(processed_img, caption="Predicted Image", use_column_width=True)
            st.write(f"### Total Straw Count: {straw_count}")
        
if __name__ == "__main__":
    main()
