from ultralytics import YOLO
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
def main():
    print(f"Using: {device}")
    model = YOLO("yolov8s.pt")
    model.train(
        # ===== DATA =====
        data="../dataset_for_car_truck/vehicle.yaml",
        # ===== TRAINING =====
        epochs=20,                
        batch=16,              
        imgsz=640,
        device=device,                 
        workers=8,
        patience=15,            
        
        # ===== OPTIMIZER =====
        optimizer="AdamW",     
        lr0=0.001,               
        lrf=0.01,               
        momentum=0.937,
        weight_decay=0.0005,
        # ===== AUGMENTATION =====
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=3.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # ===== REGULARIZATION =====
        box=7.5,
        cls=0.5,
        dfl=1.5,
        freeze=10, 
        project="runs",
        name="car_truck_yolov8s_advanced",
        save=True,
        save_period=5,
        verbose=True
    )

if __name__ == "__main__":
    main()
