# narration.py
import time

current_narration = {"text": "Initializing...", "timestamp": time.time(), "count": 0, "should_speak": True, "emergency": False}

emergency_triggered = False

def generate_narration(detections, ocr_text=None):
    global current_narration, emergency_triggered

    if emergency_triggered:
        return

    if not detections:
        narration = "Path is clear"
    else:
        traffic_lights = [d for d in detections if 'traffic light' in d['class']]
        other_dets = [d for d in detections if 'traffic light' not in d['class']]

        priority_order = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'obstacle', 'chair', 'table', 'bench']

        def get_priority(name):
            for idx, val in enumerate(priority_order):
                if val in name.lower():
                    return idx
            return len(priority_order)

        filtered = [d for d in other_dets if d['distance'] in ('very close', 'nearby')]
        filtered.sort(key=lambda x: (get_priority(x['class']), -x['area']))
        filtered = filtered[:3]

        narration_parts = []
        if traffic_lights:
            traffic_msgs = ". ".join([tl['class'] for tl in traffic_lights])
            narration_parts.append(f"Attention! {traffic_msgs}")

        for det in filtered:
            narration_parts.append(f"{det['class']} {det['distance']} {det['position']}")

        narration = ". ".join(narration_parts)
        if any(d['distance'] == 'very close' for d in filtered):
            narration = "Warning! " + narration

    if ocr_text:
        text = ocr_text.strip()
        if text:
            narration += f". Reading text: {text}"

    current_narration = {
        "text": narration,
        "timestamp": time.time(),
        "count": len(detections),
        "should_speak": current_narration.get("should_speak", True),
        "emergency": False
    }
