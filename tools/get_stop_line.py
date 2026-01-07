import cv2
from pathlib import Path
import sys
import json
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from configs.config import LIGHT_JSON, VIDEO_IN, FRAME_SIZE
ROOT = Path(__file__).resolve().parent
PROJECT_PATH = ROOT.parent
video_path = PROJECT_PATH / VIDEO_IN
  
SAVE_PATH = PROJECT_PATH / "configs/stop_line.json"
WINDOW_NAME = "Get Stop Line"


points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append([x, y])
            print(f"[CLICK] Point {len(points)}: ({x}, {y})")
        else:
            print("[INFO] Stop line already has 2 points")

def main():
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame,FRAME_SIZE)

        if not ret:
            break
        vis = frame.copy()
        for p in points:
            cv2.circle(vis, tuple(p), 6, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(vis, tuple(points[0]), tuple(points[1]), (0, 255, 255), 2)

        cv2.putText(
            vis,
            "Click 2 points for STOP LINE | S: Save | Q: Quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(WINDOW_NAME, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("s") and len(points) == 2:
            SAVE_PATH.parent.mkdir(exist_ok=True)
            with open(SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump({"stop_line": points}, f, indent=2)
            print(f"[SAVED] Stop line -> {SAVE_PATH.resolve()}")

        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
