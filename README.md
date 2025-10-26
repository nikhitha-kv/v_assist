# VeeAssist – Smart Navigation Assist

**VeeAssist** is a WiFi-optimized Flask application that uses your phone camera (via DroidCam) for real-time object detection (YOLOv8), English text reading (OCR), adaptive voice narration, blackout emergency alerts, and live video streaming UI — designed to assist safer navigation for visually impaired or general users.

---

## Features

- Real-time object detection with YOLOv8n
- Traffic light color detection integration
- Live English text reading via OCR (toggleable)
- Adaptive audio narration based on environmental density and user speed
- Emergency blackout detection with email alerts including user location
- Directional guidance for nearby persons
- Web interface with live video feed and narration text
- Enable/disable OCR reading with a button

---

## Installation

1. **Clone the repository**

git clone https://github.com/your-username/veeassist.git
cd veeassist


2. **Create a virtual environment and activate it**

python3 -m venv venv
source venv/bin/activate # macOS/Linux
venv\Scripts\activate # Windows


3. **Install dependencies**

pip install -r requirements.txt


Example `requirements.txt` includes:

flask
ultralytics
opencv-python
numpy
pyttsx3
easyocr
requests


4. **Connect DroidCam**

- Install and run DroidCam on your phone and PC.
- Use the DroidCam device index detected by the app (or set manually in the code).
- Ensure your phone and PC are on the same network.

5. **Run the application**

python app.py

Access the app from: `http://localhost:5000` or `http://<PC_IP>:5000`

---

## Usage

- The web UI shows real-time camera feed with detection boxes.
- Voice narration updates adaptively for detected objects and environmental context.
- Click the **Toggle OCR Reading** button to enable/disable reading detected text aloud.
- Emergency emails will be sent to configured contacts if blackout occurs.

---

## Project Structure

veeassist/
├── app.py # Main Flask app and detection thread
├── templates/
│ └── index.html # Web UI
├── static/
│ └── styles.css # Optional CSS styling
├── requirements.txt # Python dependencies
└── README.md # This file 

---

## Technologies Used

- Python 3.8+
- Flask web framework
- YOLOv8 (Ultralytics package) for object detection
- OpenCV for image processing
- pyttsx3 for text-to-speech
- EasyOCR for text recognition
- Requests for IP location fetching
- SMTP for email emergency alerts

---


## Author

Developed by Team Tech Busters  

---

## Acknowledgments

- Ultralytics YOLO for detection
- EasyOCR for text reading
- Flask for web UI framework
