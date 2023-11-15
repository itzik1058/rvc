import torch
import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]
