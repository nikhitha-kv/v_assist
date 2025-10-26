# ocr_module.py
import cv2
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

def ocr_on_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb_frame)
    meaningful_texts = []
    keywords = ['stop', 'exit', 'entrance', 'open', 'shop', 'welcome', 'parking', 'caution', 'danger', 'hospital']

    for (bbox, text, conf) in results:
        t_lower = text.lower().strip()
        if len(t_lower) >= 3 or any(k in t_lower for k in keywords):
            meaningful_texts.append(text.strip())

    return " ".join(meaningful_texts) if meaningful_texts else ""
