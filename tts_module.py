# tts_module.py
import pyttsx3
import threading
from config import TTS_RATE, TTS_VOLUME

tts = pyttsx3.init()
tts.setProperty("rate", TTS_RATE)
tts.setProperty("volume", TTS_VOLUME)

def speak_async(text):
    def run():
        try:
            tts.say(text)
            tts.runAndWait()
        except Exception as e:
            print("TTS error:", e)
    threading.Thread(target=run, daemon=True).start()
