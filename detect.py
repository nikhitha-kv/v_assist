from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")   # 'n' = nano version, fastest for real-time

# Access webcam (0 for default cam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 on the frame
    results = model(frame)

    # Visualize detections on frame
    annotated_frame = results[0].plot()

    # Show output window
    cv2.imshow("VeeAssist - Object Detection", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
