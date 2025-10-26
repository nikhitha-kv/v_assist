from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import threading
import time
import numpy as np
import smtplib
from email.mime.text import MIMEText
import requests
import pyttsx3  # For text-to-speech
import easyocr  # For OCR text reading

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize EasyOCR reader once (English)
reader = easyocr.Reader(['en'], gpu=False)

# DroidCam settings
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available")
        x=i
        cap.release()
    else:
        print(f"Camera index {i} not available")

DROIDCAM_INDEX = x 
ROTATION_ANGLE = 90

# OPTIMIZED PERFORMANCE SETTINGS FOR WIFI
PROCESS_EVERY_N_FRAMES = 4
OCR_PROCESS_EVERY_N_FRAMES = 8  # OCR every 8 frames for performance balance
DETECTION_CONFIDENCE = 0.6
TARGET_WIDTH = 320
TARGET_HEIGHT = 320
BASE_NARRATION_INTERVAL = 4.0  # Base narration interval seconds

# Blackout detection settings
BLACKOUT_THRESHOLD = 30
BLACKOUT_DURATION = 3

# Email configuration
GMAIL_USER = "smarthearner2115@gmail.com"
GMAIL_PASSWORD = "hmqevsyqoknthfwg"
EMERGENCY_CONTACTS = ["rgnikitha84@gmail.com"]

# Global variables
camera = None
camera_lock = threading.Lock()
current_narration = {"text": "Initializing...", "timestamp": time.time(), "count": 0}
latest_detections = []
latest_annotated_frame = None
frame_lock = threading.Lock()
should_speak = True
blackout_start_time = None
emergency_triggered = False

# Initialize TTS engine once for the whole app
tts = pyttsx3.init()
tts.setProperty("rate", 150)
tts.setProperty("volume", 1.0)

# Cooldown for narration speech
last_narration_speech_time = 0.0
NARRATION_SPEECH_COOLDOWN = 4.0

# Store previous gray frame for motion estimation
previous_gray_frame = None


def rotate_frame(frame, angle):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(DROIDCAM_INDEX)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                camera.set(cv2.CAP_PROP_FPS, 20)
                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        return camera


def is_black_frame(frame, threshold=BLACKOUT_THRESHOLD):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < threshold


def get_location_ip():
    try:
        response = requests.get('https://ipinfo.io/json', timeout=5)
        data = response.json()
        loc = data.get('loc')
        city = data.get('city')
        region = data.get('region')
        country = data.get('country')
        return loc, city, region, country
    except:
        return None, None, None, None


def send_email_alert():
    try:
        loc, city, region, country = get_location_ip()
        subject = "🚨 VeeAssist Emergency Alert"
        if loc:
            body = f"EMERGENCY DETECTED!\n\nUser may need immediate help.\n\nApproximate Location: Nagapatla \nCoordinates: [https://maps.app.goo.gl/XzTcAfiQxcvSVDST6](https://maps.app.goo.gl/XzTcAfiQxcvSVDST6) \n\nThis alert was triggered by VeeAssist navigation system."
        else:
            body = "EMERGENCY DETECTED!\n\nUser may need immediate help.\n\nLocation could not be determined.\n\nThis alert was triggered by VeeAssist navigation system."

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = GMAIL_USER

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASSWORD)

        for contact in EMERGENCY_CONTACTS:
            msg['To'] = contact
            server.sendmail(GMAIL_USER, contact, msg.as_string())
            print(f"✅ Emergency email sent to {contact}")

        server.quit()
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False


def trigger_emergency_alert():
    global emergency_triggered, current_narration
    if not emergency_triggered:
        emergency_triggered = True
        print("🚨 EMERGENCY ALERT TRIGGERED!")
        current_narration = {
            "text": "Emergency detected! Camera blackout. Sending alert to emergency contacts.",
            "timestamp": time.time(),
            "count": 0,
            "emergency": True
        }
        email_thread = threading.Thread(target=send_email_alert, daemon=True)
        email_thread.start()


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


def generate_narration(detections, ocr_text=None):
    global current_narration
    
    if emergency_triggered:
        return

    if not detections:
        narration = "Path is clear"
    else:
        traffic_lights = [d for d in detections if 'traffic light' in d['class']]
        other_dets = [d for d in detections if 'traffic light' not in d['class']]
        narration_parts = []

        # Priority list (higher index means higher priority)
        priority_order = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'obstacle', 'chair', 'table', 'bench']

        def get_priority(obj_name):
            for idx, name in enumerate(priority_order):
                if name in obj_name.lower():
                    return idx
            return len(priority_order)  # lowest priority if not found

        # Consider only very close and nearby objects
        filtered_objects = [d for d in other_dets if d['distance'] in ('very close', 'nearby')]

        # Sort by priority and area (larger area = closer objects)
        filtered_objects = sorted(filtered_objects, key=lambda x: (get_priority(x['class']), -x['area']))

        # Limit narration to top 3 important objects
        priority_objects = filtered_objects[:3]

        # Add traffic lights with high priority if present
        if traffic_lights:
            traffic_msgs = ". ".join([tl['class'] for tl in traffic_lights])
            narration_parts.append(f"Attention! {traffic_msgs}")

        # Build narration for high priority objects only
        for det in priority_objects:
            obj_name = det['class']
            distance = det['distance']
            position = det['position']
            narration_parts.append(f"{obj_name} {distance} {position}")

        narration = ". ".join(narration_parts)
        
        # Add warning if any very close object exists
        very_close = [d for d in filtered_objects if d['distance'] == 'very close']
        if very_close:
            narration = f"Warning! {narration}"

    # Append OCR detected text if available
    if ocr_text:
        ocr_text_clean = ocr_text.strip()
        if ocr_text_clean:
            narration += f". Reading text: {ocr_text_clean}"

    current_narration = {
        "text": narration,
        "timestamp": time.time(),
        "count": len(detections),
        "should_speak": should_speak,
        "emergency": False
    }

def speak_async(text):
    def run():
        try:
            tts.say(text)
            tts.runAndWait()
        except Exception as e:
            print("TTS error:", e)
    threading.Thread(target=run, daemon=True).start()


def ocr_on_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb_frame)
    meaningful_texts = []
    keywords = ['stop', 'exit', 'entrance', 'open', 'shop', 'welcome', 'parking', 'caution', 'danger', 'hospital']

    for (bbox, text, conf) in results:
        t_lower = text.lower().strip()
        # Len & keyword filter
        if len(t_lower) >= 3 or any(k in t_lower for k in keywords):
            # Optionally check confidence threshold if needed, e.g. conf > 0.3
            meaningful_texts.append(text.strip())

    # Join filtered meaningful texts, or return empty string if none
    return " ".join(meaningful_texts) if meaningful_texts else ""



def estimate_speed(frame):
    global previous_gray_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if previous_gray_frame is None:
        previous_gray_frame = gray
        return 0.0  # No speed info first frame
    flow = cv2.calcOpticalFlowFarneback(previous_gray_frame, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    average_speed = np.mean(magnitude)
    previous_gray_frame = gray
    return average_speed


def calculate_environmental_density(detections):
    count = len(detections)
    very_close_count = sum(1 for d in detections if d['distance'] == 'very close')
    density_score = count + 2 * very_close_count  # Weight close objects more
    return density_score


def process_detections():
    global latest_detections, latest_annotated_frame, blackout_start_time, emergency_triggered, last_narration_speech_time, previous_gray_frame

    frame_count = 0
    last_narration_time = time.time()

    while True:
        cap = get_camera()
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        frame_count += 1

        if is_black_frame(frame) and not emergency_triggered:
            if blackout_start_time is None:
                blackout_start_time = time.time()
                print("⚠️ Blackout detected, monitoring...")
            else:
                elapsed = time.time() - blackout_start_time
                if elapsed >= BLACKOUT_DURATION:
                    trigger_emergency_alert()
        else:
            blackout_start_time = None

        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            with frame_lock:
                if latest_annotated_frame is None:
                    latest_annotated_frame = rotate_frame(frame, ROTATION_ANGLE)
            continue

        frame = rotate_frame(frame, ROTATION_ANGLE)
        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_width * frame_height

        small_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        results = model(small_frame, verbose=False, conf=DETECTION_CONFIDENCE)

        scale_x = frame_width / TARGET_WIDTH
        scale_y = frame_height / TARGET_HEIGHT

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

            box_width = xyxy_scaled[2] - xyxy_scaled[0]
            box_height = xyxy_scaled[3] - xyxy_scaled[1]
            box_area = box_width * box_height

            class_name = results[0].names[cls_id]

            # Traffic light detection integrated as detection with color info
            if class_name.lower() == "traffic light":
                x1, y1, x2, y2 = map(int, xyxy_scaled)
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(frame_width - 1, x2), min(frame_height - 1, y2)
                crop = frame[y1c:y2c, x1c:x2c]
                color = color_from_crop(crop)
                if color:
                    detections.append({
                        'class': f"traffic light is {color}",
                        'confidence': conf,
                        'area': box_area,
                        'distance': "nearby",
                        'position': get_position(xyxy_scaled, frame_width),
                        'box': xyxy_scaled
                    })
                    continue  # Skip adding normal traffic light without color info

            detections.append({
                'class': class_name,
                'confidence': conf,
                'area': box_area,
                'distance': calculate_distance_category(box_area, frame_area),
                'position': get_position(xyxy_scaled, frame_width),
                'box': xyxy_scaled
            })

        # OCR processing every OCR_PROCESS_EVERY_N_FRAMES frames
        ocr_text = None
        if frame_count % OCR_PROCESS_EVERY_N_FRAMES == 0:
            ocr_text = ocr_on_frame(small_frame)

        # Estimate user speed & environmental density
        speed = estimate_speed(frame)
        density = calculate_environmental_density(detections)

        # Adaptive narration interval
        speed_factor = min(max(speed * 10, 0.5), 2.0)  # clamp 0.5 to 2
        density_factor = min(max(density / 5, 0.5), 2.0)

        adaptive_interval = BASE_NARRATION_INTERVAL / (speed_factor * density_factor)

        now = time.time()
        if now - last_narration_time > adaptive_interval:
            # Adjust voice rate according to speed (higher speed → faster speech)
            tts.setProperty('rate', int(150 * speed_factor))

            # Generate directional safety guidance for close persons
            guidance_msgs = []
            for det in detections:
                if det['class'] == 'person' and det['distance'] == 'very close':
                    pos = det['position']
                    if pos == 'on your left':
                        guidance_msgs.append("Person very close on your left, please move right.")
                    elif pos == 'on your right':
                        guidance_msgs.append("Person very close on your right, please move left.")
                    elif pos == 'ahead of you':
                        guidance_msgs.append("Person very close ahead, please be cautious.")

            guidance_text = " ".join(guidance_msgs)
            generate_narration(detections, ocr_text=ocr_text)
            # Append guidance messages
            if guidance_text:
                current_narration['text'] += " " + guidance_text

            last_narration_time = now

        # Speak narration if enabled and cooldown passed
        if should_speak and (now - last_narration_speech_time) > NARRATION_SPEECH_COOLDOWN:
            speak_async(current_narration['text'])
            last_narration_speech_time = now

        annotated_frame = frame.copy()

        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = map(int, box)

            if det['distance'] == 'very close':
                color = (0, 0, 255)
            elif det['distance'] == 'nearby':
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']}"
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        overlay_height = 90 if emergency_triggered or blackout_start_time else 70
        cv2.rectangle(annotated_frame, (0, 0), (frame_width, overlay_height), (0, 0, 0), -1)

        cv2.putText(annotated_frame, "VeeAssist", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Objects: {len(detections)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if emergency_triggered:
            cv2.putText(annotated_frame, "EMERGENCY ALERT SENT!", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif blackout_start_time is not None:
            elapsed = time.time() - blackout_start_time
            cv2.putText(annotated_frame, f"Blackout: {elapsed:.1f}s", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        narration_text = current_narration['text'][:40]
        cv2.rectangle(annotated_frame, (0, frame_height - 30), (frame_width, frame_height), (50, 50, 50), -1)
        cv2.putText(annotated_frame, narration_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        with frame_lock:
            latest_detections = detections
            latest_annotated_frame = annotated_frame


def generate_frames():
    frame_skip = 0
    while True:
        frame_skip += 1
        if frame_skip % 2 != 0:
            time.sleep(0.016)
            continue
        with frame_lock:
            if latest_annotated_frame is not None:
                frame = latest_annotated_frame.copy()
            else:
                frame = None
        if frame is None:
            time.sleep(0.1)
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/narration')
def get_narration():
    return jsonify(current_narration)


@app.route('/toggle_voice')
def toggle_voice():
    global should_speak
    should_speak = not should_speak
    return jsonify({"auto_speak": should_speak})


@app.route('/reset_emergency')
def reset_emergency():
    global emergency_triggered, blackout_start_time
    emergency_triggered = False
    blackout_start_time = None
    print("✅ Emergency state reset")
    return jsonify({"status": "reset"})


@app.route('/status')
def status():
    cap = get_camera()
    if cap and cap.isOpened():
        return jsonify({
            "status": "connected",
            "index": DROIDCAM_INDEX,
            "rotation": ROTATION_ANGLE,
            "model": "YOLOv8n (WiFi optimized)",
            "auto_speak": should_speak,
            "emergency_active": emergency_triggered,
            "blackout_monitoring": blackout_start_time is not None
        })
    else:
        return jsonify({
            "status": "disconnected",
            "error": "Camera not available"
        })


if __name__ == '__main__':
    print("=" * 60)
    print("🎯 VeeAssist - Complete System (WiFi Optimized with Adaptive Audio Feedback)")
    print("=" * 60)
    print("\n✅ Features Enabled:")
    print("  • Object Detection (YOLOv8n)")
    print("  • Traffic Light Color Detection Integrated into Narration")
    print("  • Auto Voice Narration (Adaptive Frequency & Tone)")
    print("  • Emergency Blackout Detection")
    print("  • Email Alerts with Location")
    print("  • Live English Text Reading (OCR via EasyOCR)")
    print("  • Directional Guidance for Nearby Persons")
    print("\n⚙️ Settings:")
    print(f"  • Camera Index: {DROIDCAM_INDEX}")
    print(f"  • Rotation: {ROTATION_ANGLE}°")
    print(f"  • Base Narration Interval: {BASE_NARRATION_INTERVAL}s")
    print(f"  • Process Rate: Every {PROCESS_EVERY_N_FRAMES} frames")
    print(f"  • OCR Process Rate: Every {OCR_PROCESS_EVERY_N_FRAMES} frames")
    print(f"  • Detection Size: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"  • Blackout Threshold: {BLACKOUT_THRESHOLD}")
    print(f"  • Alert Trigger: {BLACKOUT_DURATION}s")
    print(f"  • Emergency Contacts: {len(EMERGENCY_CONTACTS)}")
    print("\n🚀 Starting server...")
    print("📱 Access on phone: http://YOUR_PC_IP:5000")
    print("💻 Access on PC: http://localhost:5000")
    print("\nPress CTRL+C to stop")
    print("=" * 60)

    detection_thread = threading.Thread(target=process_detections, daemon=True)
    detection_thread.start()

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
