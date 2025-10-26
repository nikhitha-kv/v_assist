# camera.py
import cv2
import numpy as np
import threading
from config import DROIDCAM_INDEX, ROTATION_ANGLE, BLACKOUT_THRESHOLD

camera_lock = threading.Lock()
camera = None

def find_camera_index(max_indexes=5):
    for i in range(max_indexes):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return 0

def get_camera(index):
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(index)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                camera.set(cv2.CAP_PROP_FPS, 20)
                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        return camera

def rotate_frame(frame, angle):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def is_black_frame(frame, threshold=BLACKOUT_THRESHOLD):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < threshold
