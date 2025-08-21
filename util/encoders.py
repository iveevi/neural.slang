import torch
import torch.nn as nn


class FourierEncoder(nn.Module):
    def __init__(self, input_dim: int, levels: int):
        super().__init__()
        self.input_dim = input_dim
        self.levels = levels
        self.output_dim = 2 * levels * input_dim if levels > 0 else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.levels == 0:
            return x
        
        encoded = []
        for i in range(self.levels):
            freq = 2 ** i * torch.pi
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=1)
