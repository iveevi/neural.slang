from dataclasses import dataclass
import slangpy as spy

class Object:
    def dict(self):
        raise NotImplementedError("Object must implement dict")


class Optimizer(Object):
    pass


class Optimizable(Object):
    @property
    def parameter_count(self):
        raise NotImplementedError("Optimizable must implement parameter_count")
    
    def alloc_optimizer_states(self, device: spy.Device, optimizer: Optimizer):
        raise NotImplementedError("Optimizable must implement alloc_optimizer_states")

    # TODO: update method with shader cursor


@dataclass
class MLP(Optimizable):
    input: int
    output: int
    hidden: int
    hidden_layers: int


@dataclass
class Grid(Optimizable):
    dimension: int
    features: int