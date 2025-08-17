from common import *
from dataclasses import dataclass
from .objects import Optimizer, Grid
from .optimizers import Adam
import slangpy as spy


@dataclass
class FeatureGrid(Grid):
    parameter_buffer: spy.Buffer
    gradient_buffer: spy.Buffer
    offset: int
    resolution: int

    def dict(self):
        return {
            "parameterBuffer": self.parameter_buffer,
            "gradientBuffer": self.gradient_buffer,
            "offset": self.offset,
            "resolution": self.resolution,
        }

    @property
    def parameter_count(self):
        return self.resolution ** self.dimension * self.features

    def alloc_optimizer_states(self, device: spy.Device, optimizer: Optimizer):
        assert isinstance(optimizer, Adam)
        return create_buffer_32b(device, np.zeros((3 * self.parameter_count,), dtype=np.float32), 3)

    @staticmethod
    def new(device: spy.Device, dimension: int, features: int, resolution: int):
        p = 2 * np.random.rand(resolution ** dimension * features).astype(np.float32) - 1
        g = np.zeros((resolution ** dimension * features,), dtype=np.float32)
        parameter_buffer = create_buffer_32b(device, p, features)
        gradient_buffer = create_buffer_32b(device, g, features)
        return FeatureGrid(
            parameter_buffer=parameter_buffer,
            gradient_buffer=gradient_buffer,
            offset=0,
            resolution=resolution,
            dimension=dimension,
            features=features,
        )