from dataclasses import dataclass
from .objects import Optimizable


@dataclass
class RandomFourierEncoder(Optimizable):
    input: int
    features: int