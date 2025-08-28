import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np


# Paths to models and test images
dir_ir = 'ir_images'
dir_rgb = 'rgb_images'
model_ir_path = 'runs/train/ir/weights/best.pt'
model_rgb_path = 'runs/train/rgb/weights/best.pt'
model_irgb_path = 'runs/train/irgb/weights/best.pt'

# Load models
@st.cache_resource
def load_all_models():
    model_ir = YOLO(model_ir_path)
    model_rgb = YOLO(model_rgb_path)
    model_irgb = YOLO(model_irgb_path)
    return model_ir, model_rgb, model_irgb

model_ir, model_rgb, model_irgb = load_all_models()


# UI

# Model selection
model_mode = st.radio('Select model mode:', ['ir+rgb', 'irgb'])

st.title('YOLOv11 IR & RGB Detection Matcher')

# Confidence threshold slider
conf_threshold = st.slider('Detection confidence threshold', min_value=0.0, max_value=1.0, value=0.75, step=0.01)

# List images
ir_images = sorted([f for f in os.listdir(dir_ir) if f.lower().endswith('.jpg')])
rgb_images = sorted([f for f in os.listdir(dir_rgb) if f.lower().endswith('.jpg')])

col1, col2 = st.columns(2)


with col1:
    ir_choice = st.selectbox('Select IR image', ir_images)
    ir_img_path = os.path.join(dir_ir, ir_choice)
    ir_img = Image.open(ir_img_path)
    st.image(ir_img, caption='IR Image', use_container_width=True)


with col2:
    rgb_choice = st.selectbox('Select RGB image', rgb_images)
    rgb_img_path = os.path.join(dir_rgb, rgb_choice)
    rgb_img = Image.open(rgb_img_path)
    st.image(rgb_img, caption='RGB Image', use_container_width=True)

if st.button('Run Detection and Match'):

    # Run detection with user-selected confidence threshold
    if model_mode == 'irgb':
        ir_results = model_irgb(ir_img_path, conf=conf_threshold)[0]
        rgb_results = model_irgb(rgb_img_path, conf=conf_threshold)[0]
    else:
        ir_results = model_ir(ir_img_path, conf=conf_threshold)[0]
        rgb_results = model_rgb(rgb_img_path, conf=conf_threshold)[0]


    # Draw detections and convert BGR to RGB for correct display
    ir_annot = ir_results.plot()
    rgb_annot = rgb_results.plot()
    if isinstance(ir_annot, np.ndarray):
        ir_annot = ir_annot[..., ::-1]  # BGR to RGB
    if isinstance(rgb_annot, np.ndarray):
        rgb_annot = rgb_annot[..., ::-1]  # BGR to RGB
    st.subheader('Detections')
    det_col1, det_col2 = st.columns(2)
    with det_col1:
        st.image(ir_annot, caption='IR Detection', use_container_width=True)
    with det_col2:
        st.image(rgb_annot, caption='RGB Detection', use_container_width=True)


    # Get boxes: [x1, y1, x2, y2, conf, cls]
    ir_boxes = ir_results.boxes.xyxy.cpu().numpy() if ir_results.boxes is not None else np.array([])
    rgb_boxes = rgb_results.boxes.xyxy.cpu().numpy() if rgb_results.boxes is not None else np.array([])
    ir_classes = ir_results.boxes.cls.cpu().numpy() if ir_results.boxes is not None else np.array([])
    rgb_classes = rgb_results.boxes.cls.cpu().numpy() if rgb_results.boxes is not None else np.array([])

    # For irgb model, filter classes for each image type
    if model_mode == 'irgb':
        # irgb model: 0=IR_BIRD, 1=IR_DRONE, 2=RGB_BIRD, 3=RGB_DRONE
        # For IR image, only keep 0,1; for RGB image, only keep 2,3
        ir_mask = np.isin(ir_classes, [0, 1])
        rgb_mask = np.isin(rgb_classes, [2, 3])
        ir_boxes = ir_boxes[ir_mask]
        ir_classes = ir_classes[ir_mask]
        rgb_boxes = rgb_boxes[rgb_mask]
        rgb_classes = rgb_classes[rgb_mask]

    # Intersection check
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    matches = []
    for i, ibox in enumerate(ir_boxes):
        for j, rbox in enumerate(rgb_boxes):
            if iou(ibox, rbox) > 0.5:
                matches.append((i, j, iou(ibox, rbox)))

    st.subheader('Matching Results')
    if matches:
        for idx, (i, j, score) in enumerate(matches):
            st.write(f"IR box {i} matches RGB box {j} (IoU={score:.2f})")

            # Get intersection box
            ibox = ir_boxes[i]
            rbox = rgb_boxes[j]
            xA = max(ibox[0], rbox[0])
            yA = max(ibox[1], rbox[1])
            xB = min(ibox[2], rbox[2])
            yB = min(ibox[3], rbox[3])
            # Intersection box
            inter_w = xB - xA
            inter_h = yB - yA
            if inter_w <= 0 or inter_h <= 0:
                continue

            # Zoom intersection to 56% of crop (add padding)
            zoom_factor = 0.56
            pad_w = (inter_w / zoom_factor - inter_w) / 2
            pad_h = (inter_h / zoom_factor - inter_h) / 2
            crop_x1 = int(max(0, xA - pad_w))
            crop_y1 = int(max(0, yA - pad_h))
            crop_x2 = int(min(ir_annot.shape[1], xB + pad_w))
            crop_y2 = int(min(ir_annot.shape[0], yB + pad_h))

            # Crop both annotated images
            ir_crop = ir_annot[crop_y1:crop_y2, crop_x1:crop_x2]
            rgb_crop = rgb_annot[crop_y1:crop_y2, crop_x1:crop_x2]

            # Convert to PIL for overlay
            ir_pil = Image.fromarray(ir_crop)
            rgb_pil = Image.fromarray(rgb_crop)
            # Overlay with alpha
            overlay = Image.blend(ir_pil, rgb_pil, alpha=0.5)

            # Resize to 25% of original width
            width = overlay.width
            height = overlay.height
            new_width = max(1, int(width * 0.25))
            new_height = max(1, int(height * 0.25))
            overlay_small = overlay.resize((new_width, new_height))
            # Center using columns
            c1, c2, c3 = st.columns([1,2,1])
            with c2:
                st.image(overlay_small, caption=f'Zoomed Intersection {idx+1}', use_container_width=False)
    else:
        st.write('No matching boxes found (IoU > 0.5)')
