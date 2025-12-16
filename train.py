# train.py
# Improved training script for BiLSTMGestureClassifier
# - class-weighted loss for imbalance
# - AdamW optimizer with weight decay
# - LR scheduler (ReduceLROnPlateau)
# - gradient clipping
# - small Gaussian noise augmentation on training sequences
#
# Checkpoint format:
#   {
#       "config": {...},
#       "model_state_dict": ...,
#       "label_map": { "0": "LABEL0", "1": "LABEL1", ... }
#   }

import os
import json
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import (
    NPY_FEATURES,
    NPY_LABELS,
    LABEL_MAP_JSON,
    MODEL_PATH,
    FEATURE_DIM,
    SEQ_LEN,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
)

from models import BiLSTMGestureClassifier


# ------------------------- Reproducibility helpers ------------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------- Dataset ------------------------- #

class SequenceDataset(Dataset):
    """
    Simple dataset for sequences of shape (T, F) and integer labels.
    Optionally applies small Gaussian noise to features for augmentation
    when is_train=True.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, is_train: bool = False):
        assert X.ndim == 3, f"Expected X to be (N, T, F), got {X.shape}"
        assert y.ndim == 1, f"Expected y to be (N,), got {y.shape}"

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.is_train = is_train

        # augmentation strength (tiny jitter)
        self.noise_std = 0.01 if is_train else 0.0

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        seq = self.X[idx]  # (T, F)
        label = self.y[idx]

        if self.is_train and self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, size=seq.shape).astype(np.float32)
            seq = seq + noise

        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)


# ------------------------- Data loading ------------------------- #

def load_dataset(test_size: float = 0.2, seed: int = 42) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict
]:
    """
    Load sequences and labels from NPY files, split into train/val,
    and return idx_to_label dict (index -> label_name) based on LABEL_MAP_JSON.

    Supports both possible JSON formats:
      1) {"ISL_A": 0, "ISL_B": 1, ...}   # label_name -> idx
      2) {"0": "ISL_A", "1": "ISL_B"}   # idx -> label_name
    """
    if not os.path.exists(NPY_FEATURES) or not os.path.exists(NPY_LABELS):
        raise FileNotFoundError(
            f"Could not find {NPY_FEATURES} or {NPY_LABELS}. "
            f"Run prepare_dataset.py first."
        )

    X = np.load(NPY_FEATURES)  # (N, T, F)
    y = np.load(NPY_LABELS)    # (N,)

    print(f"[INFO] Loaded dataset: X={X.shape}, y={y.shape}")

    n_samples, seq_len, feat_dim = X.shape
    if seq_len != SEQ_LEN:
        print(f"[WARN] SEQ_LEN in config ({SEQ_LEN}) != sequence length in data ({seq_len}). "
              f"Using {seq_len} from data.")
    if feat_dim != FEATURE_DIM:
        print(f"[WARN] FEATURE_DIM in config ({FEATURE_DIM}) != feature dim in data ({feat_dim}). "
              f"Using {feat_dim} from data.")

    num_classes = len(np.unique(y))
    print(f"[INFO] Number of classes: {num_classes}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    print(f"[INFO] Train set: {X_train.shape}, Val set: {X_val.shape}")

    # ----- Robust label_map loading -----
    if not os.path.exists(LABEL_MAP_JSON):
        raise FileNotFoundError(f"Could not find label map JSON at {LABEL_MAP_JSON}")

    with open(LABEL_MAP_JSON, "r") as f:
        label_map_raw = json.load(f)

    # Try to infer format
    # If values are ints → assume label_name -> idx
    # If values are non-ints → assume idx(str) -> label_name
    idx_to_label = {}

    if len(label_map_raw) == 0:
        raise ValueError("LABEL_MAP_JSON is empty.")

    sample_k, sample_v = next(iter(label_map_raw.items()))

    try:
        if isinstance(sample_v, int) or (isinstance(sample_v, str) and sample_v.isdigit()):
            # case 1: {"ISL_A": 0, "ISL_B": 1, ...}
            print("[INFO] Detected label_map format: label_name -> idx")
            for lbl_name, idx_val in label_map_raw.items():
                if isinstance(idx_val, str) and idx_val.isdigit():
                    idx = int(idx_val)
                elif isinstance(idx_val, int):
                    idx = idx_val
                else:
                    raise ValueError(f"Unexpected idx value in label_map: {idx_val!r}")
                idx_to_label[idx] = str(lbl_name)
        else:
            # case 2: {"0": "ISL_A", "1": "ISL_B", ...}
            print("[INFO] Detected label_map format: idx -> label_name")
            for idx_str, lbl_name in label_map_raw.items():
                try:
                    idx = int(idx_str)
                except ValueError:
                    raise ValueError(f"Key {idx_str!r} in label_map is not an integer index.")
                idx_to_label[idx] = str(lbl_name)
    except Exception as e:
        print("[ERROR] Failed to parse LABEL_MAP_JSON robustly:", e)
        print("[INFO] Falling back to simple sorted unique labels from y.")
        unique_indices = sorted(np.unique(y).tolist())
        idx_to_label = {idx: f"CLASS_{idx}" for idx in unique_indices}

    print(f"[INFO] Final idx_to_label mapping has {len(idx_to_label)} entries")

    return X_train, X_val, y_train, y_val, idx_to_label


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequencies.
    This helps when dataset is imbalanced.
    """
    num_classes = len(np.unique(y))
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)

    # Avoid division by zero
    counts[counts == 0] = 1.0

    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.mean()  # normalize around 1.0

    print("[INFO] Class counts:", counts.tolist())
    print("[INFO] Class weights:", weights.tolist())

    return torch.tensor(weights, dtype=torch.float32)


# ------------------------- Training / Evaluation ------------------------- #

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 5.0,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in dataloader:
        sequences = sequences.to(device)  # (B, T, F)
        labels = labels.to(device)        # (B,)

        optimizer.zero_grad()
        outputs = model(sequences)        # (B, C)
        loss = criterion(outputs, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def eval_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * sequences.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


# ------------------------- Main training routine ------------------------- #

def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. Load data
    X_train, X_val, y_train, y_val, idx_to_label = load_dataset()

    # 2. Build datasets and loaders
    train_dataset = SequenceDataset(X_train, y_train, is_train=True)
    val_dataset = SequenceDataset(X_val, y_val, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    num_classes = len(idx_to_label)

    # 3. Model
    model = BiLSTMGestureClassifier(
        input_dim=X_train.shape[2],    # feature_dim from data
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)

    # 4. Class-weighted loss + label smoothing  # <<< CHANGED
    class_weights = compute_class_weights(y_train).to(device)
    # label_smoothing helps generalization a bit
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # 5. Optimizer + scheduler (slightly smaller LR)  # <<< CHANGED
    effective_lr = LEARNING_RATE * 0.5  # half the original LR for more stable training
    print(f"[INFO] Using effective learning rate: {effective_lr}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=1e-2,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )

    # 6. Training loop with best checkpoint saving
    best_val_acc = 0.0
    best_epoch = -1

    # Train a bit longer than config if needed  # <<< CHANGED
    total_epochs = max(NUM_EPOCHS, 100)  # e.g. if NUM_EPOCHS=25, we use 35

    for epoch in range(1, total_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch}/{total_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            cfg = {
                "input_dim": X_train.shape[2],
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "num_classes": num_classes,
                "dropout": DROPOUT,
                "seq_len": X_train.shape[1],
            }

            # label_map: index -> label string (keys must be str)
            label_map = {str(idx): lbl for idx, lbl in idx_to_label.items()}

            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(
                {
                    "config": cfg,
                    "model_state_dict": model.state_dict(),
                    "label_map": label_map,
                },
                MODEL_PATH,
            )
            print(f"[INFO] Saved new best model to {MODEL_PATH}")

    print(f"[INFO] Training finished. Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
