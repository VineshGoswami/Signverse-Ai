# live_interface.py
#
# Real-time sign inference + text buffer + TTS + virtual mouse
# Tries pygame.camera for webcam; if no camera found, falls back to
# imageio + imageio-ffmpeg.
#
# No real OpenCV is used; only your cv2.py stub exists for compatibility.

import time
from collections import deque
import os
import webbrowser
import pyautogui

import numpy as np
import torch
import mediapipe as mp
import pygame
import pygame.surfarray
import pygame.camera
import imageio  # for fallback webcam backend
from voice_assistant import VoiceAssistant
import threading

from config import (
    MODEL_PATH,
    FEATURE_DIM,
    SEQ_LEN,
    PREDICTION_SMOOTHING,
)
from models import BiLSTMGestureClassifier
from prepare_dataset import normalize_landmarks, landmarks_to_vector
from text_buffer import TextBuffer
from tts_engine import TTSEngine
from mouse_control import MouseController

mp_hands = mp.solutions.hands


# ------------------------- Webcam abstraction ------------------------- #

class WebcamSource:
    """
    Unified webcam source that tries, in order:
      1. pygame.camera
      2. imageio + ffmpeg ("0" device index)

    Provides: read() -> frame_rgb (H, W, 3) uint8, RGB
    """

    def __init__(self, size=(640, 480)):
        self.size = size
        self.mode = None
        self.cam = None
        self.reader = None

        # ---- Try pygame.camera backend first ----
        try:
            pygame.camera.init()
            cams = pygame.camera.list_cameras()
            print("[INFO] pygame.camera devices:", cams)

            if cams:
                dev = cams[0]
                print(f"[INFO] Using pygame.camera device: {dev}")
                self.cam = pygame.camera.Camera(dev, size, "RGB")
                self.cam.start()
                self.mode = "pygame"
                return
            else:
                print("[WARN] pygame.camera found no cameras. Falling back to imageio.")

        except Exception as e:
            print("[WARN] pygame.camera init failed:", e)

        # ---- Fallback: imageio + ffmpeg ----
        # ---- Fallback: imageio + ffmpeg via <video0> URI ----
        try:
            # <video0> is the standard imageio-ffmpeg syntax for the first webcam
            print("[INFO] Trying imageio webcam backend with <video0> (ffmpeg).")
            self.reader = imageio.get_reader("<video0>")
            self.mode = "imageio"
            return
        except Exception as e:
            print("[ERROR] imageio webcam init failed with <video0>:", e)

            # Optional: try <video1> as a second attempt
            try:
                print("[INFO] Trying imageio webcam backend with <video1> (ffmpeg).")
                self.reader = imageio.get_reader("<video1>")
                self.mode = "imageio"
                return
            except Exception as e2:
                print("[ERROR] imageio webcam init failed with <video1>:", e2)
                raise RuntimeError("No usable webcam backend found (pygame.camera and imageio failed).")

    def read(self):
        """
        Return a frame as (H, W, 3) RGB uint8, or raise RuntimeError on failure.
        """
        if self.mode == "pygame":
            surf = self.cam.get_image()
            if surf is None:
                raise RuntimeError("Empty frame from pygame.camera")
            arr = pygame.surfarray.array3d(surf)  # (W, H, 3)
            frame_rgb = np.transpose(arr, (1, 0, 2))  # (H, W, 3)
            return frame_rgb

        elif self.mode == "imageio":
            try:
                frame = self.reader.get_next_data()  # usually (H, W, 3) RGB
            except Exception as e:
                raise RuntimeError(f"imageio webcam read failed: {e}")

            # Ensure uint8 RGB
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            # If size differs, we can just use it directly; visualization adapts
            return frame

        else:
            raise RuntimeError("WebcamSource not initialized")

    def close(self):
        if self.mode == "pygame" and self.cam is not None:
            try:
                self.cam.stop()
            except Exception:
                pass
            pygame.camera.quit()

        if self.mode == "imageio" and self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass


# ------------------------- Model + label map ------------------------- #

def load_model_and_label_map(device):
    """
    Load trained BiLSTM model and label map from checkpoint saved by train.py.
    """
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    cfg = checkpoint["config"]
    label_map = checkpoint["label_map"]   # dict: str(idx) -> label string

    num_classes = cfg["num_classes"]

    model = BiLSTMGestureClassifier(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=num_classes,
        dropout=cfg["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    label_map_int = {int(k): v for k, v in label_map.items()}
    return model, label_map_int


# ------------------------- Pygame window ------------------------- #

def init_pygame_window(size=(960, 700)):
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("SignVerse AI - Live (no OpenCV)")
    font = pygame.font.SysFont("consolas", 20)
    return screen, font


# ------------------------- Token mapping (labels -> actions/words) ------------------------- #

TOKEN_MAP = {
    "ISL_Z": "[MOUSE_TOGGLE]",
    "ASL_Z": "[MOUSE_TOGGLE]",

    "ISL_T": "[LEFT_CLICK]",
    "ASL_T": "[LEFT_CLICK]",

    "ISL_R": "[RIGHT_CLICK]",
    "ASL_R": "[RIGHT_CLICK]",

    "ISL_A": "[SCROLL_UP]",
    "ASL_A": "[SCROLL_UP]",

    "ISL_Q": "[SCROLL_DOWN]",
    "ASL_Q": "[SCROLL_DOWN]",
}



def handle_token(text_buffer, tts, mouse, token):

    if token == "[MOUSE_TOGGLE]":
        mouse.toggle()

    elif token == "[LEFT_CLICK]":
        mouse.left_click()

    elif token == "[RIGHT_CLICK]":
        mouse.right_click()

    elif token == "[SCROLL_UP]":
        mouse.scroll_up()

    elif token == "[SCROLL_DOWN]":
        mouse.scroll_down()




# ------------------------- Main loop ------------------------- #

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Model + labels
    model, label_map = load_model_and_label_map(device)
    num_classes = len(label_map)
    print(f"[INFO] Loaded model with {num_classes} classes")

    # NLP / TTS / mouse
    text_buffer = TextBuffer()
    tts = TTSEngine()
    mouse = MouseController()

    # MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Pygame & webcam
    screen, font = init_pygame_window((960, 700))
    webcam = WebcamSource(size=(640, 480))

    clock = pygame.time.Clock()
    running = True

    seq_buffer = deque(maxlen=SEQ_LEN)
    last_prediction = None
    same_count = 0

    while running:
        # ---- Events ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_c:
                    text_buffer.clear()
                    print("[INFO] Text buffer cleared (key C).")
                elif event.key == pygame.K_b:
                    text_buffer.backspace()
                    print("[INFO] Backspace (key B).")
                elif event.key == pygame.K_m:
                    mouse.toggle()
                elif event.key == pygame.K_d:
                    mouse.double_click()
                    print("[INFO] Double Click (key D).")
                elif event.key == pygame.K_l:
                    mouse.left_click()
                    print("[INFO] Left Click (key L).")
                elif event.key == pygame.K_r:
                    mouse.right_click()
                    print("[INFO] Right Click (key R).")

        # ---- Webcam frame ----
        try:
            frame_rgb = webcam.read()
            if frame_rgb is None:
                raise RuntimeError("Empty frame from webcam")
        except Exception as e:
            print("[ERROR] Failed to read from webcam:", e)
            break

        # Flip horizontally for mirror effect
        frame_rgb = np.fliplr(frame_rgb)
        frame_rgb_u8 = frame_rgb.astype(np.uint8)

        # >>> ADDED (ONLY NEW LINE — REQUIRED FOR MEDIAPIPE STABILITY)
        frame_rgb_u8 = np.ascontiguousarray(frame_rgb_u8)

        # ---- MediaPipe Hands ----
        results = hands.process(frame_rgb_u8)

        feature_vec = None
        pred_label = None
        confidence = 0.0
        index_tip_x_norm = None
        index_tip_y_norm = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            lm_norm = normalize_landmarks(lm_list)
            feature_vec = landmarks_to_vector(lm_norm, include_z=False)

            # mouse: index fingertip
            index_tip = hand_landmarks.landmark[8]
            index_tip_x_norm = index_tip.x
            index_tip_y_norm = index_tip.y

            # Process real-time pinch gestures (Left click, Double click, Right click)
            mouse.process_hand_gestures(lm_list)

            if feature_vec.shape[0] != FEATURE_DIM:
                print(f"[WARN] Unexpected feature size: {feature_vec.shape}")
                feature_vec = None

        # ---- Mouse movement ----
        if index_tip_x_norm is not None and index_tip_y_norm is not None:
            mouse.move_by_landmark(index_tip_x_norm, index_tip_y_norm)

        # ---- Sequence + prediction ----
        if feature_vec is not None:
            seq_buffer.append(feature_vec)

            if len(seq_buffer) == SEQ_LEN:
                seq = np.stack(seq_buffer, axis=0)  # (T, F)
                seq = np.expand_dims(seq, axis=0)   # (1, T, F)
                seq_tensor = torch.from_numpy(seq).float().to(device)

                with torch.no_grad():
                    logits = model(seq_tensor)
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    pred_label = label_map[pred_idx]
                    confidence = probs[0, pred_idx].item()

                if pred_label == last_prediction:
                    same_count += 1
                else:
                    same_count = 1
                    last_prediction = pred_label

                if same_count >= PREDICTION_SMOOTHING:
                    token = TOKEN_MAP.get(pred_label, pred_label)
                    handle_token(text_buffer, tts, mouse, token)

                    print(f"[PRED] {pred_label} -> {token} (conf={confidence:.2f}) | "
                          f"Sentence: {text_buffer.get_sentence()}")

                    same_count = 0

        # ---- Draw window ----
        frame_display = np.transpose(frame_rgb, (1, 0, 2))
        surface = pygame.surfarray.make_surface(frame_display)

        screen.fill((20, 20, 20))
        screen.blit(surface, (10, 10))

        y = 500

        def draw_text(text, y_pos, color=(255, 255, 255)):
            surf = font.render(text, True, color)
            screen.blit(surf, (10, y_pos))
            return y_pos + 24

        y = draw_text(f"Last Sign Label: {pred_label if pred_label else '-'} (conf={confidence:.2f})", y)
        y = draw_text(f"Mouse Status: {'ACTIVE' if mouse.enabled else 'DISABLED'} | Action: {mouse.current_gesture_status}", y, (0, 255, 0) if mouse.enabled else (255, 100, 100))
        y = draw_text("Gestures: Palm=Mouse ON | Fist=Mouse OFF | Pinch=Left Click | 3-Pinch/2-Pinch=Double Click", y, (0, 220, 255))
        y = draw_text("Keys: M=Toggle Mouse, D=Double Click, C=Clear, Q=Quit", y, (200, 200, 200))

        pygame.display.flip()
        clock.tick(20)

    webcam.close()
    pygame.quit()


if __name__ == "__main__":
    main()
