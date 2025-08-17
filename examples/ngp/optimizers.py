from dataclasses import dataclass
from .objects import Optimizer


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