import pyautogui
import numpy as np
import time
import math

# Configure PyAutoGUI for responsive desktop control
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0


class MouseController:
    """
    100% Hand Gesture Virtual Mouse & Desktop Controller.
    
    Gestures Supported:
    - Open Palm (5 fingers extended)  -> Enable Mouse Mode (Mouse ON)
    - Closed Fist (All fingers folded)-> Disable Mouse Mode (Mouse OFF)
    - Index Finger Move               -> Move Mouse Cursor smoothly across screen
    - Index + Thumb Pinch             -> Left Click (Single Pinch) / Double Click (Quick 2x Pinch)
    - Index + Thumb Hold (>0.4s)      -> Drag & Drop File
    - Index + Middle + Thumb Pinch    -> Double Click (Opens file/folder immediately)
    - Middle + Thumb Pinch            -> Right Click (Context Menu)
    - Index + Middle Finger Move UP/DN-> Scroll Up / Scroll Down
    """

    def __init__(self):
        self.enabled = True  # Start enabled for instant gesture use
        self.dragging = False
        self.screen_w, self.screen_h = pyautogui.size()

        # Smoothing and bounds mapping
        self.prev_x = self.screen_w / 2
        self.prev_y = self.screen_h / 2
        self.smooth_factor = 0.35  # Responsiveness vs smoothness
        self.margin = 0.15          # 15% frame margin to comfortably reach screen corners

        # Timers and state debouncers
        self.last_click_time = 0
        self.last_right_click_time = 0
        self.pinch_start_time = 0
        self.cooldown = 0.35

        self.palm_counter = 0
        self.fist_counter = 0

        self.index_thumb_pinched = False
        self.middle_thumb_pinched = False
        self.three_finger_pinched = False

        self.prev_scroll_y = None
        self.current_gesture_status = "Mouse Active (Move Index Finger)"

    def toggle(self):
        self.enabled = not self.enabled
        print(f"[MOUSE] {'ON' if self.enabled else 'OFF'}")

    def move_by_landmark(self, x: float, y: float):
        if not self.enabled:
            return

        x_mapped = np.clip((x - self.margin) / (1.0 - 2.0 * self.margin), 0.0, 1.0)
        y_mapped = np.clip((y - self.margin) / (1.0 - 2.0 * self.margin), 0.0, 1.0)

        target_x = x_mapped * self.screen_w
        target_y = y_mapped * self.screen_h

        curr_x = self.prev_x * (1 - self.smooth_factor) + target_x * self.smooth_factor
        curr_y = self.prev_y * (1 - self.smooth_factor) + target_y * self.smooth_factor

        self.prev_x = curr_x
        self.prev_y = curr_y

        pyautogui.moveTo(int(curr_x), int(curr_y))

    def process_hand_gestures(self, landmarks: list):
        """
        Process 21 MediaPipe hand landmarks to trigger full mouse desktop actions.
        """
        if not landmarks or len(landmarks) < 21:
            return

        now = time.time()
        wrist = landmarks[0]

        # Helper: check if finger is extended relative to wrist distance
        def is_ext(tip_idx, pip_idx):
            d_tip = math.hypot(landmarks[tip_idx][0] - wrist[0], landmarks[tip_idx][1] - wrist[1])
            d_pip = math.hypot(landmarks[pip_idx][0] - wrist[0], landmarks[pip_idx][1] - wrist[1])
            return d_tip > d_pip

        index_ext = is_ext(8, 6)
        middle_ext = is_ext(12, 10)
        ring_ext = is_ext(16, 14)
        pinky_ext = is_ext(20, 18)

        # 1. Mode Switch Gestures: Open Palm = Enable, Closed Fist = Disable
        if index_ext and middle_ext and ring_ext and pinky_ext:
            self.palm_counter += 1
            if self.palm_counter > 8:  # ~0.4s hold
                if not self.enabled:
                    self.enabled = True
                    print("[GESTURE] Open Palm -> Mouse Mode ENABLED")
                self.current_gesture_status = "Open Palm (Mouse ENABLED)"
        else:
            self.palm_counter = 0

        if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            self.fist_counter += 1
            if self.fist_counter > 10:  # ~0.5s hold
                if self.enabled:
                    self.enabled = False
                    if self.dragging:
                        pyautogui.mouseUp()
                        self.dragging = False
                    print("[GESTURE] Closed Fist -> Mouse Mode DISABLED")
                self.current_gesture_status = "Fist (Mouse DISABLED)"
        else:
            self.fist_counter = 0

        if not self.enabled:
            return

        # 2. Key Finger Landmark Distances
        thumb = landmarks[4]
        index = landmarks[8]
        middle = landmarks[12]

        d_index_thumb = math.hypot(index[0] - thumb[0], index[1] - thumb[1])
        d_middle_thumb = math.hypot(middle[0] - thumb[0], middle[1] - thumb[1])

        PINCH_THRESH = 0.05

        # 3. Double-Click Gesture (Index + Middle + Thumb Pinch together)
        if d_index_thumb < PINCH_THRESH and d_middle_thumb < PINCH_THRESH:
            if not self.three_finger_pinched and (now - self.last_click_time > self.cooldown):
                print("[GESTURE] 3-Finger Pinch -> Double Click (Opened File)")
                pyautogui.doubleClick()
                self.last_click_time = now
                self.three_finger_pinched = True
                self.current_gesture_status = "Double Click (Opened File)"
                return
        else:
            self.three_finger_pinched = False

        # 4. Left Click, Quick Double Pinch & Drag and Drop (Index + Thumb)
        if d_index_thumb < PINCH_THRESH:
            if not self.index_thumb_pinched:
                self.pinch_start_time = now
                if (now - self.last_click_time) < 0.45:
                    print("[GESTURE] Quick Double Pinch -> Double Click (Opened File)")
                    pyautogui.doubleClick()
                    self.current_gesture_status = "Double Click (Opened File)"
                else:
                    print("[GESTURE] Index Pinch -> Left Click")
                    pyautogui.click()
                    self.current_gesture_status = "Left Click"
                self.last_click_time = now
                self.index_thumb_pinched = True

            # Drag and drop if pinch held for > 0.4s
            elif not self.dragging and (now - self.pinch_start_time > 0.4):
                print("[GESTURE] Sustained Pinch -> Dragging Started")
                pyautogui.mouseDown()
                self.dragging = True
                self.current_gesture_status = "Dragging File..."
        else:
            if self.index_thumb_pinched:
                if self.dragging:
                    print("[GESTURE] Pinch Released -> Drag Dropped")
                    pyautogui.mouseUp()
                    self.dragging = False
                    self.current_gesture_status = "Dropped File"
                self.index_thumb_pinched = False

        # 5. Right Click Gesture (Middle + Thumb)
        if d_middle_thumb < PINCH_THRESH and d_index_thumb >= PINCH_THRESH:
            if not self.middle_thumb_pinched and (now - self.last_right_click_time > self.cooldown):
                print("[GESTURE] Middle Pinch -> Right Click")
                pyautogui.click(button="right")
                self.last_right_click_time = now
                self.middle_thumb_pinched = True
                self.current_gesture_status = "Right Click"
        else:
            self.middle_thumb_pinched = False

        # 6. Two-Finger Scroll Gesture (Index & Middle Extended, Ring & Pinky Folded)
        if index_ext and middle_ext and not ring_ext and not pinky_ext and d_index_thumb > 0.08:
            curr_y = (index[1] + middle[1]) / 2.0
            if self.prev_scroll_y is not None:
                dy = curr_y - self.prev_scroll_y
                if abs(dy) > 0.02:
                    if dy < 0:  # Moving Hand UP
                        pyautogui.scroll(250)
                        self.current_gesture_status = "Scrolling Up ▲"
                    else:       # Moving Hand DOWN
                        pyautogui.scroll(-250)
                        self.current_gesture_status = "Scrolling Down ▼"
            self.prev_scroll_y = curr_y
        else:
            self.prev_scroll_y = None

    def left_click(self):
        if self.enabled:
            pyautogui.click()

    def double_click(self):
        if self.enabled:
            pyautogui.doubleClick()

    def right_click(self):
        if self.enabled:
            pyautogui.click(button="right")

    def scroll_up(self):
        if self.enabled:
            pyautogui.scroll(300)

    def scroll_down(self):
        if self.enabled:
            pyautogui.scroll(-300)
