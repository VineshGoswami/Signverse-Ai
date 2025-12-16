# config.py

import os

# === DATASET ROOTS ===
# Your datasets:
#   ISL: C:\Users\vines\Downloads\archive(7)\ISL_Dataset
#   ASL: C:\Users\vines\Downloads\archive(9)\asl_dataset

DATASET_CONFIG = {
    "ISL": r"C:\Users\vines\Downloads\archive(7)\ISL_Dataset",
    "ASL": r"C:\Users\vines\Downloads\archive(9)\asl_dataset",
}

# Folder for processed data & models
PROCESSED_DATA_DIR = os.path.join("data_processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Numpy data + label maps
NPY_FEATURES = os.path.join(PROCESSED_DATA_DIR, "X_sequences.npy")
NPY_LABELS = os.path.join(PROCESSED_DATA_DIR, "y_labels.npy")
LABEL_MAP_JSON = os.path.join(PROCESSED_DATA_DIR, "label_map.json")
LABEL_DETAILS_JSON = os.path.join(PROCESSED_DATA_DIR, "label_details.json")

# Sequence / feature parameters
SEQ_LEN = 20            # Frames per sequence
FEATURE_DIM = 42        # 21 landmarks * (x,y)

# Model choice (sequence-based, no CNN)
MODEL_NAME = "BiLSTMGestureClassifier"
MODEL_PATH = os.path.join(PROCESSED_DATA_DIR, "sign_bilstm_model.pt")

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3

# Live inference smoothing
PREDICTION_SMOOTHING = 5  # number of consecutive same predictions before accepting

# TTS
TTS_ENABLED = True
