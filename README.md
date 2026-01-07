# Red Light Violation Detection System
**Real-time Traffic Violation Detection using YOLOv8, Computer Vision & Streamlit
## Overview

This project is an end-to-end computer vision system that automatically detects vehicles violating red traffic lights, captures their license plates, and visualizes violations in a real-time web interface.

The system integrates:

- Object detection & tracking

- Traffic light color recognition

- Stop-line crossing logic

- License plate detection

- Interactive Streamlit dashboard

- Dockerized deployment

## Key Features

🚗 Vehicle Detection & Tracking

- YOLOv8 + ByteTrack

- Stable vehicle IDs across frames

🚦 Traffic Light Color Detection

- HSV-based color segmentation (Red / Yellow / Green)

- ROI-based detection for robustness

🛑 Red Light Violation Detection

- Robust stop-line crossing logic using signed distance

- Red-light + crossing = violation event

🔍 License Plate Detection

- Fine-tuned YOLOv8 model for license plates

- Plate detection performed only on violating vehicles

🖥️ Interactive Streamlit App

- Real-time video visualization

- Violation gallery with cropped vehicle & plate images

- Debug mode with live logs & metrics

🐳 Docker Support
- Easy setup & reproducibility
- CPU & GPU compatible
## System Pipeline

```
Video Input
   ↓
Vehicle Detection (YOLOv8)
   ↓
Vehicle Tracking (ByteTrack)
   ↓
Traffic Light Detection (HSV)
   ↓
Stop Line Crossing Check
   ↓
Red Light Violation Event
   ↓
License Plate Detection (YOLOv8)
   ↓
Streamlit Visualization
```

## Project Structure
```
red_light_violation/
│
├── app.py                     # Streamlit application
├── Dockerfile                 # Docker build file
├── requirements.txt
│
├── models/                    # (ignored) YOLO pretrained weights
├── train/
│   ├── car_truck_train.py     # Vehicle training
│   ├── car_lisence_train.py   # License plate training
│
├── utils/
│   └── traffic_light_detector/
│       └── detect_light_color.py
│
├── configs/
│   ├── stop_line.json         # Stop line coordinates
│   └── red_light_coords.json  # Traffic light ROI
│
├── dataset/
│   └── data.yaml              # Dataset configuration (no images)
│
└── tools/
    ├── get_stop_line.py
    └── get_light_coordinates.py
```
## Installation (Local)
### Clone repository
```bash
git clone https://github.com/yourusername/red_light_violation.git
cd red_light_violation
```
### Install dependencies
```bash
pip install -r requirements.txt
```
### Run Streamlit app
```bash
streamlit run app.py
```
