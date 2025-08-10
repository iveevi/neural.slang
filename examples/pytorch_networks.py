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
    def __init__(self, hidden: int, levels: int, input: int, output: int):
        super().__init__()
        encoded_size = 2 * levels * input if levels > 0 else input
        self.levels = levels
        self.layer1 = nn.Linear(encoded_size, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, hidden)
        self.layer4 = nn.Linear(hidden, output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = frequency_encode(x, self.levels)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x
