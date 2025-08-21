from dataclasses import dataclass
from .objects import MLP, Optimizable
import slangpy as spy
import numpy as np
import torch.nn as nn
from util import *


@dataclass
class AddressBasedMLP(MLP):
    parameter_buffer: spy.Buffer
    gradient_buffer: spy.Buffer

    def dict(self):
        return {
            "parameterBuffer": self.parameter_buffer,
            "gradientBuffer": self.gradient_buffer,
        }

    def slang_type(self) -> str:
        return f"AddressBasedMLP<{self.input}, {self.output}, {self.hidden}, {self.hidden_layers}, ReLU<float>, Identity<float>>"

    @property
    def parameter_count(self):
        layer_shapes = [
            (self.input, self.hidden),
            *[(self.hidden, self.hidden) for _ in range(self.hidden_layers)],
            (self.hidden, self.output),
        ]

        total_params = 0
        for in_size, out_size in layer_shapes:
            # Weight matrix: in_size * out_size
            # Bias vector: out_size
            total_params += in_size * out_size + out_size

        return total_params

    @staticmethod
    def new(device: spy.Device, hidden: int, hidden_layers: int, input: int, output: int):
        layer_shapes = [
            (input, hidden),
            *[(hidden, hidden) for _ in range(hidden_layers)],
            (hidden, output),
        ]

        layers = [ linear_to_numpy(nn.Linear(s[0], s[1])).flatten() for s in layer_shapes ]
        parameters = np.ascontiguousarray(np.concatenate(layers, axis=0))
        gradients = np.zeros_like(parameters)

        parameter_buffer = create_buffer_32b(device, parameters)
        gradient_buffer = create_buffer_32b(device, gradients)

        return AddressBasedMLP(
            device=device,
            parameter_buffer=parameter_buffer,
            gradient_buffer=gradient_buffer,
            hidden=hidden,
            hidden_layers=hidden_layers,
            input=input,
            output=output,
        )

    def alloc_optimizer_states(self, device: spy.Device, optimizer):
        from .optimizers import Adam
        assert isinstance(optimizer, Adam)
        return create_buffer_32b(device, np.zeros((3 * self.parameter_count,), dtype=np.float32), 3)