import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Semen Straw Counting", page_icon="🐮")


@st.cache_resource()
def load_models():

    roi_model = YOLO("./weight_file/best_roi_200.pt") # ROI Model
    straw_model = YOLO("./new_weight_file/best_roi_straw_600.pt")  # Combined Model
    
    # straw_model = YOLO("./weight_file/best_roi_straw_400.pt")  # Combined Model
    
    # roi_model = YOLO("./weight_file/roi_best_100.pt") # ROI Model
    # straw_model = YOLO("./weight_file/yolo11_best_300.pt")  # Combined Model
    return roi_model, straw_model

def detect_roi(roi_model, image, conf_thresh=0.5):
    """ Detects the Region of Interest (ROI) and returns the largest bounding box """
    results = roi_model.predict(image, conf=conf_thresh)

    if not results or len(results[0].boxes) == 0:
        return None  # No ROI detected

    max_area = 0
    best_box = None

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)

        if area > max_area:
            max_area = area
            best_box = (x1, y1, x2, y2)

    return best_box

def detect_straws(straw_model, image, roi_box, conf_thresh,max_det=2500):
    """ Detects only straws inside the ROI, ignoring ROI boxes from the second model """
    if not roi_box:
        return image, 0  # No ROI detected, return original image

    x1, y1, x2, y2 = roi_box

    # Create a mask to keep only the ROI
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    # Detect objects using the combined model
    results = straw_model.predict(image, conf=conf_thresh, max_det=max_det)

    if not results or len(results[0].boxes) == 0:
        return image, 0

    straw_count = 0
    for box in results[0].boxes:
        x1_s, y1_s, x2_s, y2_s = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0].item())  # Class ID (0 = ROI, 1 = Straw)

        if class_id == 1 and mask[y1_s:y2_s, x1_s:x2_s].sum() > 0:  # Count only straws inside ROI
            cv2.rectangle(image, (x1_s, y1_s), (x2_s, y2_s), (0, 255, 0), 2)  # Draw straw boxes
            straw_count += 1

    return image, straw_count

def main():
    st.title("Semen Straw Counting")

    roi_model, straw_model = load_models()

    roi_conf_thresh = 0.5  # Fixed confidence for ROI detection
    straw_conf_thresh = st.slider("Straw Detection Confidence Threshold", 0.1, 1.0, 0.25)

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Process Image"):
            # Step 1: Detect ROI using the ROI model
            roi_box = detect_roi(roi_model, image_np, roi_conf_thresh)

            # Step 2: Detect straws inside the ROI using the combined model
            processed_img, straw_count = detect_straws(straw_model, image_np, roi_box, straw_conf_thresh,max_det=2500)

            # Step 3: Display the results
            st.image(processed_img, caption="Detected Straws", use_container_width=True)
            st.write(f"### Total Straw Count: {straw_count}")

if __name__ == "__main__":
    main()
