# tts_engine.py

import pyttsx3
from config import TTS_ENABLED


class TTSEngine:
    def __init__(self):
        self.enabled = TTS_ENABLED
        if self.enabled:
            self.engine = pyttsx3.init()
        else:
            self.engine = None

    def speak(self, text: str):
        if not self.enabled:
            print(f"[TTS disabled] Would say: {text}")
            return

        text = text.strip()
        if not text:
            return

        self.engine.say(text)
        self.engine.runAndWait()
