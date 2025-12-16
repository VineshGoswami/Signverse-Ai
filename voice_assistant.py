import speech_recognition as sr

class VoiceAssistant:
    def __init__(self, mouse_controller):
        self.mouse = mouse_controller
        self.recognizer = sr.Recognizer()

    def listen(self):
        try:
            with sr.Microphone() as source:
                print("[VOICE] Listening...")
                audio = self.recognizer.listen(source, timeout=4)
            command = self.recognizer.recognize_google(audio).lower()
            print("[VOICE CMD]:", command)
            self.handle_command(command)
        except Exception:
            pass

    def handle_command(self, command: str):
        if "mouse on" in command or "enable mouse" in command:
            if not self.mouse.enabled:
                self.mouse.toggle()

        elif "mouse off" in command or "disable mouse" in command:
            if self.mouse.enabled:
                self.mouse.toggle()
