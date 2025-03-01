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

# Function to check if a straw is inside an ROI
def is_inside_roi(straw_box, roi_boxes):
    x1_s, y1_s, x2_s, y2_s = straw_box  # Straw bounding box
    
    for roi_box in roi_boxes:
        x1_r, y1_r, x2_r, y2_r = roi_box  # ROI bounding box
        
        # Check if straw is completely inside the ROI
        if x1_r <= x1_s and y1_r <= y1_s and x2_r >= x2_s and y2_r >= y2_s:
            return True
    return False

# Object detection function with max_det applied per class
def detect_objects(model, image, straw_conf, roi_conf=0.7, iou=0.7):
    # Detect ROI first (max_det=1)
    roi_results = model.predict(image, iou=iou, conf=roi_conf, max_det=1)  

    # Detect straws (max_det=2500)
    straw_results = model.predict(image, iou=iou, conf=straw_conf, max_det=2500)

    img = np.array(image)  
    roi_boxes = []  # List to store ROI bounding boxes
    straw_count = 0  

    # Collect ROI bounding boxes
    for box in roi_results[0].boxes:
        roi_boxes.append(list(map(int, box.xyxy[0].tolist())))
        

    # Now, check for straw detections inside the ROI
    for box in straw_results[0].boxes:
        straw_box = list(map(int, box.xyxy[0].tolist()))
        if is_inside_roi(straw_box, roi_boxes):  # Only count if inside ROI
            straw_count += 1
            x1, y1, x2, y2 = straw_box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

    return img, straw_count

# Streamlit app layout
def main():
    st.title("Semen Straw Detection and Count")

    model = load_model()

    # Only allow adjusting the Straw confidence threshold
    straw_conf = st.slider("Straw Confidence Threshold", 0.1, 1.0, 0.25)

    # Image upload
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # Display uploaded image
        st.write("### Uploaded Image:")
        st.image(image, caption="Original Image", use_column_width=True)

        if st.button("Process Image"):
            processed_img, straw_count = detect_objects(model, image, straw_conf)

            
            st.image(processed_img, caption="Predicted Image")

            # Show total straw count inside ROI
            st.write(f"### Total Straw Count: {straw_count}")

if __name__ == "__main__":
    main()
