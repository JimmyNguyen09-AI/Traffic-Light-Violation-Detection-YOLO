import streamlit as st
import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO
import tempfile
from PIL import Image

FRAME_W, FRAME_H = 640, 480
COCO_WEIGHTS = "models/yolov8m.pt"
PLATE_WEIGHTS = "train/runs/license_plate_yolov8m_adamw/weights/best.pt"
STOPLINE_JSON = "configs/stop_line.json"
LIGHT_JSON = "configs/red_light_coords.json"
DEVICE = "cuda"
CONF_VEH = 0.25
IOU_VEH = 0.5
TRACKER = "bytetrack.yaml"
CONF_PLATE = 0.5
IOU_PLATE = 0.5
VEHICLE_CLS = {2, 5, 7}

from utils.traffic_light_detector.detect_light_color import load_coords, detect_light_color

def load_stop_line(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    a, b = data["stop_line"]
    return (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
def bottom_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, float(y2))
def side_of_line(p, a, b) -> int:
    """Tính phía của điểm p so với đường thẳng từ a đến b"""
    val = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if val > 0: return 1
    if val < 0: return -1
    return 0
def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2
def main():
    st.set_page_config(page_title="Red Light Violation Detection", layout="wide")
    st.title(" Red Light Violation Detection System")
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        else:
            video_path = "red_light_violation.mp4"
            st.info("Using default video")
        debug_mode = st.checkbox("Debug Mode", value=True)
        process_btn = st.button("Start Detection", type="primary")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Video Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("Violations Detected")
        violations_placeholder = st.empty()

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        frame_counter = st.empty()
    with stats_col2:
        light_status = st.empty()
    with stats_col3:
        violation_counter = st.empty()
    with stats_col4:
        tracked_vehicles = st.empty()
    if debug_mode:
        debug_log = st.expander("Debug Log", expanded=False)
    
    if process_btn:
        try:
            stop_a, stop_b = load_stop_line(STOPLINE_JSON)
            polygon, _bbox = load_coords(LIGHT_JSON)
            with st.spinner("Loading AI models..."):
                vehicle_model = YOLO(COCO_WEIGHTS)
                plate_model = YOLO(PLATE_WEIGHTS)           
            st.success("Models loaded successfully!")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error(f"Cannot open video: {video_path}")
                return
            
            prev_side = {}
            reported_ids = set()
            violations_list = []
            frame_idx = 0
            debug_messages = []
            
            stop_detection = st.button("Stop Detection")
            
            while cap.isOpened() and not stop_detection:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
            
                frame_idx += 1
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                light_state, counts = detect_light_color(frame, polygon)
                results = vehicle_model.track(
                    source=frame,
                    conf=CONF_VEH,
                    iou=IOU_VEH,
                    device=DEVICE,
                    tracker=TRACKER,
                    persist=True,
                    verbose=False
                )
                r = results[0]
                vis = frame.copy()
                cv2.polylines(vis, [polygon], True, (0, 255, 255), 2)
                cv2.line(vis, stop_a, stop_b, (255, 255, 0), 3)
                cv2.putText(vis, f"Frame: {frame_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis, f"Light: {light_state}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 0) if light_state == "GREEN" else (0, 0, 255), 2)
                
                current_vehicles = 0
                
                if r.boxes is not None and r.boxes.xyxy is not None and r.boxes.id is not None:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    cls = r.boxes.cls.cpu().numpy().astype(int)
                    ids = r.boxes.id.cpu().numpy().astype(int)
                    
                    for i, box in enumerate(xyxy):
                        c = int(cls[i])
                        if c not in VEHICLE_CLS:
                            continue
                        
                        current_vehicles += 1
                        x1, y1, x2, y2 = box
                        track_id = int(ids[i])
                        
                        x1i, y1i, x2i, y2i = clamp_box(x1, y1, x2, y2, FRAME_W, FRAME_H)
                        color = (0, 255, 0)
                        if track_id in reported_ids:
                            color = (0, 0, 255)
                        cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), color, 2)
                        p = bottom_center(x1i, y1i, x2i, y2i)
                        cv2.circle(vis, (int(p[0]), int(p[1])), 5, (255, 0, 255), -1)
                        cv2.putText(vis, f"ID:{track_id}", (x1i, y1i - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        curr_side = side_of_line(p, stop_a, stop_b)
                        if debug_mode:
                            cv2.putText(vis, f"Side:{curr_side}", (x1i, y2i + 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        crossed = False
                        if track_id in prev_side:
                            prev = prev_side[track_id]
                            # Crossing happens when: prev != 0, curr != 0, and they differ
                            if prev != 0 and curr_side != 0 and prev != curr_side:
                                crossed = True
                                if debug_mode:
                                    msg = f"Frame {frame_idx}: ID {track_id} CROSSED! prev={prev}, curr={curr_side}"
                                    debug_messages.append(msg)
                                    if len(debug_messages) > 20:
                                        debug_messages.pop(0)
                        prev_side[track_id] = curr_side
                        if light_state == "RED" and crossed and track_id not in reported_ids:
                            if debug_mode:
                                msg = f"Frame {frame_idx}: ID {track_id} VIOLATION DETECTED! Checking plate..."
                                debug_messages.append(msg)
                            
                            vehicle_crop = frame[y1i:y2i, x1i:x2i].copy()

                            pres = plate_model.predict(
                                source=vehicle_crop,
                                conf=CONF_PLATE,
                                iou=IOU_PLATE,
                                device=DEVICE,
                                verbose=False
                            )[0]
                            if pres.boxes is not None and len(pres.boxes) > 0:
                                reported_ids.add(track_id)
                                
                                confs = pres.boxes.conf.cpu().numpy()
                                best_idx = int(np.argmax(confs))
                                px1, py1, px2, py2 = pres.boxes.xyxy[best_idx].cpu().numpy()
                                vw, vh = vehicle_crop.shape[1], vehicle_crop.shape[0]
                                px1, py1, px2, py2 = clamp_box(px1, py1, px2, py2, vw, vh)
                                plate_crop = vehicle_crop[py1:py2, px1:px2].copy()
                                gx1, gy1 = x1i + px1, y1i + py1
                                gx2, gy2 = x1i + px2, y1i + py2
                                cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 0, 255), 3)
                                cv2.putText(vis, "VIOLATION!", (x1i, max(0, y1i - 30)),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                
                                vehicle_rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
                                plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                                
                                violations_list.append({
                                    'frame': frame_idx,
                                    'id': track_id,
                                    'vehicle': vehicle_rgb,
                                    'plate': plate_rgb
                                })
                                
                                if debug_mode:
                                    msg = f"Frame {frame_idx}: ID {track_id} SAVED with plate!"
                                    debug_messages.append(msg)
                            else:
                                if debug_mode:
                                    msg = f"Frame {frame_idx}: ID {track_id} NO PLATE detected, skipped"
                                    debug_messages.append(msg)
                
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                video_placeholder.image(vis_rgb, channels="RGB", use_container_width=True)
                frame_counter.metric("Frame", frame_idx)
                light_status.metric("Light Status", light_state, 
                                   delta="STOP" if light_state == "RED" else "GO")
                violation_counter.metric("Violations", len(violations_list))
                tracked_vehicles.metric("Tracked", current_vehicles)
                if debug_mode and debug_messages:
                    with debug_log:
                        for msg in reversed(debug_messages[-10:]):
                            st.text(msg)
                if violations_list:
                    with violations_placeholder.container():
                        for vio in reversed(violations_list[-5:]):  
                            st.divider()
                            st.write(f"**Frame {vio['frame']} - Vehicle ID: {vio['id']}**")
                            col_v, col_p = st.columns(2)
                            with col_v:
                                st.image(vio['vehicle'], caption="Vehicle", use_container_width=True)
                            with col_p:
                                st.image(vio['plate'], caption="License Plate", use_container_width=True)
            
            cap.release()
            st.success(f"Detection completed! Total violations: {len(violations_list)}")
            if violations_list:
                st.subheader("Final Results")
                st.write(f"Total **{len(violations_list)}** violations detected with license plates")
                st.write("### All Violations:")
                for idx, vio in enumerate(violations_list, 1):
                    with st.expander(f"Violation #{idx} - Frame {vio['frame']}, ID {vio['id']}"):
                        col_v, col_p = st.columns(2)
                        with col_v:
                            st.image(vio['vehicle'], caption="Vehicle", use_container_width=True)
                        with col_p:
                            st.image(vio['plate'], caption="License Plate", use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()