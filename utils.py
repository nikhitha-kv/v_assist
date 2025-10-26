# utils.py
import requests
import numpy as np
import cv2

def get_location_ip():
    try:
        response = requests.get('https://ipinfo.io/json', timeout=5)
        data = response.json()
        return data.get('loc'), data.get('city'), data.get('region'), data.get('country')
    except:
        return None, None, None, None

def color_from_crop(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    try:
        crop = cv2.resize(crop_bgr, (60, 180), interpolation=cv2.INTER_AREA)
    except Exception:
        crop = crop_bgr
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 80])
    upper_red2 = np.array([180, 255, 255])
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    lower_green = np.array([36, 80, 60])
    upper_green = np.array([95, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    total_pixels = crop.shape[0] * crop.shape[1]
    red_ratio = np.count_nonzero(mask_red) / total_pixels
    yellow_ratio = np.count_nonzero(mask_yellow) / total_pixels
    green_ratio = np.count_nonzero(mask_green) / total_pixels

    ratios = {"red": red_ratio, "yellow": yellow_ratio, "green": green_ratio}
    color = max(ratios, key=ratios.get)
    if ratios[color] < 0.007:
        return None
    return color

def calculate_distance_category(box_area, frame_area):
    ratio = box_area / frame_area
    if ratio > 0.25:
        return "very close"
    elif ratio > 0.12:
        return "nearby"
    elif ratio > 0.04:
        return "at medium distance"
    else:
        return "far away"

def get_position(box, frame_width):
    x_center = (box[0] + box[2]) / 2
    if x_center < frame_width / 3:
        return "on your left"
    elif x_center > 2 * frame_width / 3:
        return "on your right"
    else:
        return "ahead of you"

def calculate_environmental_density(detections):
    count = len(detections)
    very_close_count = sum(1 for d in detections if d['distance'] == 'very close')
    return count + 2 * very_close_count

def estimate_speed(frame, previous_gray_frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if previous_gray_frame is None:
        return 0.0, gray
    flow = cv2.calcOpticalFlowFarneback(previous_gray_frame, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    average_speed = np.mean(magnitude)
    return average_speed, gray
