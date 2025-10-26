# detection.py
from ultralytics import YOLO
from utils import calculate_distance_category, get_position, color_from_crop
import cv2

class Detector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.6):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect_objects(self, frame, target_width, target_height):
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_w * frame_h

        small_frame = cv2.resize(frame, (target_width, target_height))
        results = self.model(small_frame, verbose=False, conf=self.conf_thresh)

        scale_x = frame_w / target_width
        scale_y = frame_h / target_height

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()

            xyxy_scaled = [
                xyxy[0] * scale_x,
                xyxy[1] * scale_y,
                xyxy[2] * scale_x,
                xyxy[3] * scale_y
            ]

            box_w = xyxy_scaled[2] - xyxy_scaled[0]
            box_h = xyxy_scaled[3] - xyxy_scaled[1]
            box_area = box_w * box_h

            class_name = results[0].names[cls_id]

            if class_name.lower() == "traffic light":
                x1, y1, x2, y2 = map(int, xyxy_scaled)
                crop = frame[max(0, y1):min(frame_h - 1, y2), max(0, x1):min(frame_w - 1, x2)]
                color = color_from_crop(crop)
                if color:
                    detections.append({
                        'class': f"traffic light is {color}",
                        'confidence': conf,
                        'area': box_area,
                        'distance': "nearby",
                        'position': get_position(xyxy_scaled, frame_w),
                        'box': xyxy_scaled
                    })
                    continue

            detections.append({
                'class': class_name,
                'confidence': conf,
                'area': box_area,
                'distance': calculate_distance_category(box_area, frame_area),
                'position': get_position(xyxy_scaled, frame_w),
                'box': xyxy_scaled
            })

        return detections
