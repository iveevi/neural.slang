from .objects import Optimizable
from dataclasses import dataclass
from util import create_buffer_32b
import numpy as np
import slangpy as spy


@dataclass
class RandomFourierFeatures(Optimizable):
    parameter_buffer: spy.Buffer
    gradient_buffer: spy.Buffer
    input: int
    features: int

    def dict(self):
        return {
            "parameterBuffer": self.parameter_buffer,
            "gradientBuffer": self.gradient_buffer,
        }

    def slang_type(self) -> str:
        return f"RandomFourierFeatures<{self.input}, {self.features}>"

    @property
    def parameter_count(self):
        return self.features * self.input

    @staticmethod
    def new(device: spy.Device, input: int, features: int, sigma: float):
        parameters = np.random.randn(features * input) * sigma
        parameter_buffer = create_buffer_32b(device, parameters)
        gradient_buffer = create_buffer_32b(device, np.zeros_like(parameters))
        return RandomFourierFeatures(
            device=device,
            parameter_buffer=parameter_buffer,
            gradient_buffer=gradient_buffer,
            input=input,
            features=features,
        )

    def alloc_optimizer_states(self, device: spy.Device, optimizer):
        from .optimizers import Adam
        assert isinstance(optimizer, Adam)
        return create_buffer_32b(device, np.zeros((3 * self.parameter_count,), dtype=np.float32), 3)