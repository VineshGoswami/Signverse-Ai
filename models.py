# models.py

import torch
import torch.nn as nn


class BiLSTMGestureClassifier(nn.Module):
    """
    Bidirectional LSTM-based sequence classifier for sign gestures.

    Input:  (batch, seq_len, feature_dim)
    Output: (batch, num_classes) logits
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        batch_size = x.size(0)
        num_directions = 2

        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=x.device,
        )
        c0 = torch.zeros_like(h0)

        out, _ = self.lstm(x, (h0, c0))  # (batch, seq, hidden*2)
        last_out = out[:, -1, :]         # use last time step
        logits = self.fc(last_out)
        return logits
