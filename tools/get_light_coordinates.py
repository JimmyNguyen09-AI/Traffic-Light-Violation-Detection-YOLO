import cv2
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import LIGHT_JSON, VIDEO_IN
ROOT = Path(__file__).resolve().parent
PROJECT_PATH = ROOT.parent
video_path = PROJECT_PATH / VIDEO_IN
coords = []
def mouse_callback(event, x,y,flags,param):
    """
    Get coords of objects by clicking on the screen    
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at x = {x}, y={y}")
        coords.append((x,y))
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_callback)
    display = frame.copy()
    for (x,y) in coords:
        cv2.circle(display,(x,y),4,(0,0,255),-1)
    cv2.imshow("Frame",display)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

