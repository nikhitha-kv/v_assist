# app.py
from flask import Flask, render_template, Response, jsonify
import threading
import time
import cv2
from config import *
from camera import get_camera, rotate_frame, is_black_frame, find_camera_index
from detection import Detector
from ocr_module import ocr_on_frame
from utils import estimate_speed, calculate_environmental_density
from tts_module import speak_async, tts
from narration import generate_narration, current_narration, emergency_triggered
from emergency import trigger_emergency_alert
import threading

app = Flask(__name__)

# Initialize global state
camera_index = 2
blackout_start_time = None
last_narration_time = 0
last_narration_speech_time = 0
previous_gray_frame = None
should_speak = True
latest_detections = []
latest_annotated_frame = None
frame_lock = threading.Lock()
emergency_flag = False

detector = Detector(conf_thresh=DETECTION_CONFIDENCE)

def set_emergency_flag(value=None):
    global emergency_flag
    if value is None:
        return emergency_flag
    emergency_flag = value

def set_narration(text):
    global current_narration
    current_narration["text"] = text
    current_narration["timestamp"] = time.time()
    current_narration["count"] = 0
    current_narration["emergency"] = True

def process_detections():
    global latest_detections, latest_annotated_frame, blackout_start_time, emergency_flag
    global last_narration_time, last_narration_speech_time, previous_gray_frame

    frame_count = 0

    while True:
        cap = get_camera(camera_index)
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        frame_count += 1

        if is_black_frame(frame) and not emergency_flag:
            if blackout_start_time is None:
                blackout_start_time = time.time()
                print("⚠️ Blackout detected, monitoring...")
            else:
                elapsed = time.time() - blackout_start_time
                if elapsed >= BLACKOUT_DURATION:
                    trigger_emergency_alert(set_emergency_flag, set_narration)
        else:
            blackout_start_time = None

        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            with frame_lock:
                if latest_annotated_frame is None:
                    latest_annotated_frame = rotate_frame(frame, ROTATION_ANGLE)
            continue

        frame = rotate_frame(frame, ROTATION_ANGLE)

        detections = detector.detect_objects(frame, TARGET_WIDTH, TARGET_HEIGHT)

        ocr_text = None
        if frame_count % OCR_PROCESS_EVERY_N_FRAMES == 0:
            ocr_text = ocr_on_frame(frame)

        speed, previous_gray_frame = estimate_speed(frame, previous_gray_frame)
        density = calculate_environmental_density(detections)

        speed_factor = min(max(speed * 10, 0.5), 2.0)
        density_factor = min(max(density / 5, 0.5), 2.0)

        adaptive_interval = BASE_NARRATION_INTERVAL / (speed_factor * density_factor)

        now = time.time()
        if now - last_narration_time > adaptive_interval:
            tts.setProperty('rate', int(TTS_RATE * speed_factor))
            generate_narration(detections, ocr_text=ocr_text)
            last_narration_time = now

        if should_speak and (now - last_narration_speech_time) > NARRATION_SPEECH_COOLDOWN and current_narration.get('should_speak', True):
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

        overlay_height = 90 if emergency_flag or blackout_start_time else 70
        cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)

        cv2.putText(annotated_frame, "VeeAssist", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Objects: {len(detections)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if emergency_flag:
            cv2.putText(annotated_frame, "EMERGENCY ALERT SENT!", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif blackout_start_time is not None:
            elapsed = time.time() - blackout_start_time
            cv2.putText(annotated_frame, f"Blackout: {elapsed:.1f}s", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        narration_text = current_narration['text'][:40]
        cv2.rectangle(annotated_frame, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
        cv2.putText(annotated_frame, narration_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        with frame_lock:
            latest_detections = detections
            latest_annotated_frame = annotated_frame

def generate_frames():
    frame_skip = 0
    import time
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
    current_narration['should_speak'] = should_speak
    return jsonify({"auto_speak": should_speak})

@app.route('/reset_emergency')
def reset_emergency():
    global emergency_flag, blackout_start_time
    emergency_flag = False
    blackout_start_time = None
    print("✅ Emergency state reset")
    return jsonify({"status": "reset"})

@app.route('/status')
def status():
    cap = get_camera(camera_index)
    if cap and cap.isOpened():
        return jsonify({
            "status": "connected",
            "index": camera_index,
            "rotation": ROTATION_ANGLE,
            "model": "YOLOv8n (WiFi optimized)",
            "auto_speak": should_speak,
            "emergency_active": emergency_flag,
            "blackout_monitoring": blackout_start_time is not None
        })
    else:
        return jsonify({
            "status": "disconnected",
            "error": "Camera not available"
        })

if __name__ == '__main__':
    import threading

    # Detect camera index dynamically on startup
    camera_index = find_camera_index()

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
    print(f"  • Camera Index: {camera_index}")
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
