from dataclasses import dataclass

class Object:
    def dict(self):
        raise NotImplementedError("Object must implement dict")


class Optimizer(Object):
    pass


@dataclass
class MLP(Object):
    input: int
    output: int
    hidden: int
    hidden_layers: int


@dataclass
class Grid(Object):
    dimension: int
    features: int