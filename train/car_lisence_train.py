from ultralytics import YOLO
import torch
from pathlib import Path

device = 0 if torch.cuda.is_available() else "cpu"

def main():
    ROOT = Path(__file__).resolve().parent.parent
    DATA_YAML = ROOT / "dataset/data.yaml"
    PRETRAINED = ROOT / "models/yolov8m.pt" 

    model = YOLO(str(PRETRAINED))
    model.train(
        data=str(DATA_YAML),

        epochs=50,           
        batch=8,           
        imgsz=640,
        device=device,
        workers=8,
        patience=20,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # Augment
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.0,

        box=7.5,
        cls=0.5,
        dfl=1.5,            
        project="runs",
        name="license_plate_yolov8m_adamw",

        save=True,
        save_period=5,
        verbose=True
    )

if __name__ == "__main__":
    main()
