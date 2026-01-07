import json
import cv2
import numpy as np
from typing import Tuple,Dict
from pathlib import Path
import sys
from collections import deque
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from configs.config import LIGHT_JSON, VIDEO_IN


ROOT = Path(__file__).resolve().parent
json_path = ROOT.parent.parent / "configs" / LIGHT_JSON
video_path = ROOT.parent.parent / VIDEO_IN


color_buf = deque(maxlen=7)
def load_coords(json_path: str) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    with open(json_path,"r",encoding = "utf-8") as f:
        data = json.load(f)
    if "roi_polygon" not in data:
        raise ValueError("json must contain key 'roi_polygon'")
    pts = data["roi_polygon"]
    if not isinstance(pts, list) or len(pts) < 3:
        raise ValueError("'roi_polygon' must be a list of >= 3 points")
    polygon = np.array(pts, dtype=np.int32)
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return polygon, (x1, y1, x2, y2)
def stable_color(new_color:str)->str:
    color_buf.append(new_color)
    valid = [c for c in color_buf if c != "UNKNOWN"]
    if not valid:
        return "UNKNOWN"
    return max(set(valid), key=valid.count)
def detect_light_color(
        frame:np.ndarray,
        polygon: np.ndarray,
        min_pixels:int = 30,
        use_blur: bool = True
) -> Tuple[str,Dict[str,int]]:
    if use_blur:
        frame = cv2.GaussianBlur(frame,(5,5),0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [polygon], 255)
    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 60, 60), (90, 255, 255))
    red_count = cv2.countNonZero(cv2.bitwise_and(red_mask, red_mask, mask=roi_mask))
    yellow_count = cv2.countNonZero(cv2.bitwise_and(yellow_mask, yellow_mask, mask=roi_mask))
    green_count = cv2.countNonZero(cv2.bitwise_and(green_mask, green_mask, mask=roi_mask))
    counts = {"RED": int(red_count), "YELLOW": int(yellow_count), "GREEN": int(green_count)}
    color = max(counts, key=counts.get)
    if counts[color] < min_pixels:
        color = "UNKNOWN"
    color = stable_color(color)
    return color, counts
def draw_light_roi(frame: np.ndarray, polygon: np.ndarray,color: str,counts: Dict[str,int])-> np.ndarray:
    out = frame.copy()
    cv2.polylines(out, [polygon], True, (0, 255, 255), 2)
    cv2.putText(
        out,
        f"LIGHT: {color}  R:{counts['RED']} Y:{counts['YELLOW']} G:{counts['GREEN']}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) ,
        2
    )
    return out
if __name__ == "__main__":
    print("[INFO] json_path:", json_path)
    polygon, bbox = load_coords(json_path)
    print("[INFO] polygon:", polygon.tolist())
    print("[INFO] bbox:", bbox)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: red_light_violation.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    delay = max(1, int(1000 / fps))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        color, counts = detect_light_color(frame, polygon)
        vis = draw_light_roi(frame, polygon, color, counts)

        cv2.imshow("Light Detection", vis)
        key = cv2.waitKey(delay) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()