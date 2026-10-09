"""Sign classifier network: forward (B, 32, 192) -> logits (B, C).

Owner: A (Data/ML). See CONTRACT.md "Model artifacts".
"""
import torch
from torch import nn


class GRUNet(nn.Module):
    """2-layer bidirectional GRU (hidden 128, dropout 0.3), mean over time, linear."""

    def __init__(self, n_features, n_classes, hidden=128):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=2, bidirectional=True, dropout=0.3, batch_first=True)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        return self.head(self.gru(x)[0].mean(dim=1))


class TransformerNet(nn.Module):
    """Linear to 128, learned positions, 2 encoder layers (4 heads, ff 256, dropout 0.2), mean, linear."""

    def __init__(self, n_features, n_classes, T, dim=128):
        super().__init__()
        self.proj = nn.Linear(n_features, dim)
        self.pos = nn.Parameter(torch.zeros(1, T, dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.head(self.encoder(self.proj(x) + self.pos).mean(dim=1))


def build_model(config):
    """Return the torch.nn.Module described by config (model/artifacts/config.json)."""
    n_classes = len(config["classes"])
    if config["arch"] == "gru":
        return GRUNet(config["n_features"], n_classes)
    if config["arch"] == "transformer":
        return TransformerNet(config["n_features"], n_classes, config["T"])
    raise ValueError(f"unknown arch {config['arch']!r}")
