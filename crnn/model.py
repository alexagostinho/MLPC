"""CRNN architecture for frame-wise multi-label sound event detection.

    [B, T, D] features
      -> Conv1d stack   (local temporal patterns: onsets/offsets)
      -> (Bi)GRU        (longer-range context across the recording)
      -> Linear -> C    (per-timestep multi-label logits)
"""
from __future__ import annotations

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CRNN(nn.Module):
    def __init__(self, in_dim, n_classes, conv_dim=128, conv_layers=2,
                 kernel_size=3, gru_dim=128, gru_layers=2, bidirectional=True,
                 dropout=0.3):
        super().__init__()
        convs = []
        prev = in_dim
        for _ in range(conv_layers):
            convs += [nn.Conv1d(prev, conv_dim, kernel_size=kernel_size,
                                padding=kernel_size // 2),
                      nn.BatchNorm1d(conv_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = conv_dim
        self.conv = nn.Sequential(*convs)
        # GRU dropout is only applied between stacked layers (>1).
        self.gru = nn.GRU(conv_dim, gru_dim, num_layers=gru_layers,
                          batch_first=True, bidirectional=bidirectional,
                          dropout=dropout if gru_layers > 1 else 0.0)
        out_dim = (2 if bidirectional else 1) * gru_dim
        self.head = nn.Linear(out_dim, n_classes)

    @classmethod
    def from_config(cls, cfg, in_dim, n_classes):
        return cls(in_dim, n_classes, conv_dim=cfg.conv_dim,
                   conv_layers=cfg.conv_layers, kernel_size=cfg.kernel_size,
                   gru_dim=cfg.gru_dim, gru_layers=cfg.gru_layers,
                   bidirectional=cfg.bidirectional, dropout=cfg.dropout)

    def forward(self, x, lengths):
        # x: [B, T, D]
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)          # [B, T, conv_dim]
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)        # [B, T, H]
        return self.head(out)                                      # [B, T, C] logits
