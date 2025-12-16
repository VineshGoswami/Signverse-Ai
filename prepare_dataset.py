# prepare_dataset.py

import os
import json
import numpy as np
import mediapipe as mp
from PIL import Image

from config import (
    DATASET_CONFIG,
    PROCESSED_DATA_DIR,
    NPY_FEATURES,
    NPY_LABELS,
    LABEL_MAP_JSON,
    LABEL_DETAILS_JSON,
    SEQ_LEN,
    FEATURE_DIM,
)

mp_hands = mp.solutions.hands


def load_image_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def normalize_landmarks(landmarks):
    """
    landmarks: list of (x, y, z) for 21 hand points.
    Returns np.array shape (21, 3):
      - wrist at origin
      - scale normalized
    """
    lm = np.array(landmarks, dtype=np.float32)  # (21, 3)
    wrist = lm[0].copy()
    lm[:, :3] -= wrist

    dists = np.linalg.norm(lm[:, :3], axis=1)
    scale = np.max(dists)
    if scale < 1e-6:
        scale = 1.0
    lm[:, :3] /= scale
    return lm


def landmarks_to_vector(landmarks, include_z=False):
    """
    landmarks: np.array (21, 3)
    Return flat vector: (42,) or (63,)
    """
    if include_z:
        return landmarks.flatten()
    else:
        return landmarks[:, :2].flatten()


def extract_hand_landmarks_from_image(image_rgb):
    """
    image_rgb: numpy array (H, W, 3), uint8, RGB.
    Returns feature_vector (42,) or None.
    """
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        static_image_mode=True,
    ) as hands:
        results = hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        lm_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        lm_norm = normalize_landmarks(lm_list)
        feature_vec = landmarks_to_vector(lm_norm, include_z=False)
        return feature_vec


def prepare_dataset():
    all_sequences = []
    all_labels = []
    label_strings = []    # e.g. "ISL_A"
    label_details = []    # e.g. {"lang": "ISL", "class_name": "A"}
    class_string_to_index = {}

    for lang, root_dir in DATASET_CONFIG.items():
        if not os.path.isdir(root_dir):
            print(f"[WARN] Dataset root for {lang} does not exist: {root_dir}")
            continue

        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            full_label_str = f"{lang}_{class_name}"
            if full_label_str not in class_string_to_index:
                class_index = len(class_string_to_index)
                class_string_to_index[full_label_str] = class_index
                label_strings.append(full_label_str)
                label_details.append({
                    "lang": lang,
                    "class_name": class_name,
                })

            class_index = class_string_to_index[full_label_str]
            print(f"[INFO] Processing class '{full_label_str}' (index {class_index}) from {class_dir}")

            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if not fpath.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                try:
                    img_rgb = load_image_rgb(fpath)
                except Exception as e:
                    print(f"[WARN] Could not read image {fpath}: {e}")
                    continue

                feature_vec = extract_hand_landmarks_from_image(img_rgb)
                if feature_vec is None:
                    print(f"[WARN] No hand detected in {fpath}, skipping.")
                    continue

                if feature_vec.shape[0] != FEATURE_DIM:
                    print(f"[WARN] Unexpected feature size in {fpath}: {feature_vec.shape}")
                    continue

                # Static image -> fake sequence by repetition
                seq = np.tile(feature_vec, (SEQ_LEN, 1))  # (SEQ_LEN, FEATURE_DIM)
                all_sequences.append(seq)
                all_labels.append(class_index)

    if len(all_sequences) == 0:
        raise RuntimeError("No valid samples found in any dataset. Check paths and images.")

    X = np.stack(all_sequences, axis=0)  # (N, SEQ_LEN, FEATURE_DIM)
    y = np.array(all_labels, dtype=np.int64)

    print(f"[INFO] Total samples: {X.shape[0]}")
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[INFO] Number of classes (ASL+ISL): {len(label_strings)}")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    np.save(NPY_FEATURES, X)
    np.save(NPY_LABELS, y)

    label_map = {idx: s for idx, s in enumerate(label_strings)}
    with open(LABEL_MAP_JSON, "w") as f:
        json.dump(label_map, f, indent=2)

    detailed_map = {idx: d for idx, d in enumerate(label_details)}
    with open(LABEL_DETAILS_JSON, "w") as f:
        json.dump(detailed_map, f, indent=2)

    print(f"[INFO] Saved X to {NPY_FEATURES}")
    print(f"[INFO] Saved y to {NPY_LABELS}")
    print(f"[INFO] Saved label map to {LABEL_MAP_JSON}")
    print(f"[INFO] Saved label details to {LABEL_DETAILS_JSON}")


if __name__ == "__main__":
    prepare_dataset()
