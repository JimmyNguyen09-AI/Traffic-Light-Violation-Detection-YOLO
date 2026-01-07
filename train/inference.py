# inference_license_plate_video.py
from ultralytics import YOLO
import cv2

WEIGHTS = "runs/license_plate_yolov8s_adamw/weights/best.pt" 
VIDEO_IN = "C:/Users/ngtru/Documents/CodeSpace/red_light_violation/red_light_violation.mp4"                                                # video mới, KHÔNG dùng train/val
CONF = 0.5
IMG_SIZE = 640
DEVICE = 0  

def main():
    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_IN}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = max(1, int(1000 / fps))

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame,(640,640))
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=CONF,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )

        vis = results[0].plot()
        cv2.imshow("License Plate Inference", vis)

        if cv2.waitKey(delay) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
