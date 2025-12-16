import pyautogui

class MouseController:
    def __init__(self):
        self.enabled = False
        self.dragging = False
        self.screen_w, self.screen_h = pyautogui.size()

    def toggle(self):
        self.enabled = not self.enabled
        print(f"[MOUSE] {'ON' if self.enabled else 'OFF'}")

    def move_by_landmark(self, x, y):
        if not self.enabled:
            return
        pyautogui.moveTo(
            int(x * self.screen_w),
            int(y * self.screen_h),
            duration=0.01
        )

    def left_click(self):
        if self.enabled:
            pyautogui.click()

    def right_click(self):
        if self.enabled:
            pyautogui.click(button="right")

    def scroll_up(self):
        if self.enabled:
            pyautogui.scroll(200)

    def scroll_down(self):
        if self.enabled:
            pyautogui.scroll(-200)

    def drag_toggle(self):
        if not self.enabled:
            return
        self.dragging = not self.dragging
        pyautogui.mouseDown() if self.dragging else pyautogui.mouseUp()
