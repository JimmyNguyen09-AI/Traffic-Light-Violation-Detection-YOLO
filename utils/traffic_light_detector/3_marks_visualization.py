import cv2
import json
import numpy as np

VIDEO_IN = "../red_light_violation.mp4"
LIGHT_JSON = "red_light_coords.json"
FRAME_SIZE = (640, 480)
with open(LIGHT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
polygon = np.array(data["roi_polygon"], dtype=np.int32)
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, FRAME_SIZE)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [polygon], 255)
    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    green_mask  = cv2.inRange(hsv, (40, 60, 60), (90, 255, 255))
    red_roi    = cv2.bitwise_and(red_mask, red_mask, mask=roi_mask)
    yellow_roi = cv2.bitwise_and(yellow_mask, yellow_mask, mask=roi_mask)
    green_roi  = cv2.bitwise_and(green_mask, green_mask, mask=roi_mask)
    red_vis = cv2.cvtColor(red_roi, cv2.COLOR_GRAY2BGR)
    red_vis[:, :, 2] = red_vis[:, :, 2]  

    yellow_vis = cv2.cvtColor(yellow_roi, cv2.COLOR_GRAY2BGR)
    yellow_vis[:, :, 1] = yellow_vis[:, :, 1]  
    yellow_vis[:, :, 2] = yellow_vis[:, :, 2]  

    green_vis = cv2.cvtColor(green_roi, cv2.COLOR_GRAY2BGR)
    green_vis[:, :, 1] = green_vis[:, :, 1]  
    frame_vis = frame.copy()
    cv2.polylines(frame_vis, [polygon], True, (0, 255, 255), 2)
    top = np.hstack([frame_vis, red_vis])
    bottom = np.hstack([yellow_vis, green_vis])
    combined = np.vstack([top, bottom])

    cv2.putText(combined, "ORIGINAL + ROI", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(combined, "RED MASK", (650, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(combined, "YELLOW MASK", (10, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(combined, "GREEN MASK", (650, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Traffic Light Mask Debug", combined)
    key = cv2.waitKey(30) & 0xFF
    if key == 27 or key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
