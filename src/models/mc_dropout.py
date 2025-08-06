"""MC-dropout model."""
import torch
from torch import nn


class MCDropoutNet(nn.Module,):
    def __init__(self, dims: list[int], p: float = 0.2 ):
        super().__init__()

        self.fc = nn.Sequential(
            *[
                layer
                for in_dim, out_dim in zip(dims[:-2], dims[1:-1])
                for layer in (nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(p=p))
            ],
            nn.Linear(dims[-2], dims[-1])
        )
    def forward(self, x: torch.tensor):
        return self.fc(x)