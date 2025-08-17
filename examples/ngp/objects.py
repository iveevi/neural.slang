from __future__ import annotations
from typing import Any
import slangpy as spy
import numpy as np
from dataclasses import dataclass
from common import *


class Object:
    def dict(self):
        raise NotImplementedError("Object must implement dict")


class Optimizer(Object):
    pass


class MLP(Object):
    pass


@dataclass
class Adam(Optimizer):
    alpha: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def dict(self):
        return {
            "alpha": self.alpha,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
        }


@dataclass
class FeatureGrid(Object):
    parameter_buffer: spy.Buffer
    gradient_buffer: spy.Buffer
    offset: int
    resolution: int
    features: int
    dimension: int

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
            parameter_buffer,
            gradient_buffer,
            0,
            resolution,
            features,
            dimension,
        )
