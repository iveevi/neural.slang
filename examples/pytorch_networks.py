import torch
import torch.nn as nn
import torch.nn.functional as F


def frequency_encode(x: torch.Tensor, levels: int) -> torch.Tensor:
    if levels == 0:
        return x

    X = []
    for i in range(levels):
        X.append(torch.sin(2 ** i * torch.pi * x))
        X.append(torch.cos(2 ** i * torch.pi * x))
    return torch.cat(X, dim=1)


class PyTorchNetwork(nn.Module):
    def __init__(self, hidden: int, levels: int, input: int, output: int, hidden_layers: int = 0):
        super().__init__()
        encoded_size = 2 * levels * input if levels > 0 else input
        self.levels = levels
        self.hidden_layers = hidden_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(encoded_size, hidden))
        for i in range(hidden_layers):
            self.layers.append(nn.Linear(hidden, hidden))
        self.layers.append(nn.Linear(hidden, output))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = frequency_encode(x, self.levels)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
