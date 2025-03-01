import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch,torchvision

st.set_page_config(page_title="Semen Straw Counting", page_icon="🔍")

@st.cache_resource()
def load_model():
    model = YOLO("./weight_file/yolo11_best_300.pt")
    return model

def detect_objects(model, image, conf_thresh, iou_thresh=0.3, edge_margin_ratio=0.03, min_box_size=15,max_roi_ratio=0.85,max_det=2500):
    results = model.predict(image, conf=conf_thresh,max_det=max_det)

    if not results or len(results[0].boxes) == 0:
        return np.array(image), 0

    img = np.array(image)
    img_h, img_w = img.shape[:2]
    edge_margin = int(img_w * edge_margin_ratio)

    boxes = results[0].boxes
    filtered_boxes = []
    straw_count = 0

    for box in boxes:
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Box width and height
        box_w, box_h = x2 - x1, y2 - y1

        # **Skip large bounding box (ROI)**
        if box_w > img_w * max_roi_ratio or box_h > img_h * max_roi_ratio:
            continue  

        # **Apply edge filter only on left/right (not top/bottom)**
        if (x1 < edge_margin or x2 > img_w - edge_margin):
            continue  

        # **Keep small straw detections**
        if box_w < min_box_size or box_h < min_box_size:
            continue  

        filtered_boxes.append([x1, y1, x2, y2, conf])

    # **Apply NMS (higher IoU to avoid losing too many)**
    if filtered_boxes:
        filtered_boxes = torch.tensor(filtered_boxes)
        keep = torchvision.ops.nms(filtered_boxes[:, :4], filtered_boxes[:, 4], iou_thresh)
        filtered_boxes = filtered_boxes[keep]

        # **Draw final bounding boxes**
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            straw_count += 1

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
