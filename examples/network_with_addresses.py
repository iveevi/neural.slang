import pathlib
import slangpy as spy
import numpy as np
import torch.nn as nn
from .util import create_buffer, linear_to_numpy


ROOT = pathlib.Path(__file__).parent.parent.absolute()


# TODO: base class for all networks
class Network:
    def __init__(self, device: spy.Device, hidden: int, hidden_layers: int, levels: int, input: int, output: int):
        self.device = device
        self.hidden = hidden
        self.hidden_layers = hidden_layers
        self.levels = levels
        self.input = input
        self.output = output
        
        input_size = self.input if levels == 0 else 2 * levels * self.input

        self.layer_shapes = [
            (input_size, hidden),
            *[(hidden, hidden) for _ in range(hidden_layers)],
            (hidden, output),
        ]

        layers = [ linear_to_numpy(nn.Linear(s[0], s[1])).flatten() for s in self.layer_shapes ]
        sizes = [ layer.size for layer in layers ]

        self.layer_addresses_host = np.cumsum([0, *sizes])[:-1].astype(np.int32)

        layers = np.ascontiguousarray(np.concatenate(layers, axis=0))

        self.parameters = create_buffer(device, layers)
        self.gradients = create_buffer(device, np.zeros_like(layers))
        self.optimizer_states = create_buffer(device, np.zeros_like(layers).repeat(3, axis=0), 3 * 4)
        self.layer_addresses = create_buffer(device, self.layer_addresses_host)
        self.parameter_count = layers.size

    def input_vec(self, input: np.ndarray) -> spy.Buffer:
        assert input.ndim > 1
        assert input.shape[-1] == self.input
        return self.device.create_buffer(
            size=input.nbytes,
            struct_size=self.input * 4,
            usage=spy.BufferUsage.shader_resource,
            data=input,
        )

    def output_vec(self, output: np.ndarray) -> spy.Buffer:
        assert output.ndim > 1
        assert output.shape[-1] == self.output
        return self.device.create_buffer(
            size=output.nbytes,
            struct_size=self.output * 4,
            usage=spy.BufferUsage.shader_resource,
            data=output,
        )

    def copy_weights(self, layer_index: int, layer: nn.Linear):
        begin = self.layer_addresses_host[layer_index]
        if layer_index + 1 >= len(self.layer_addresses_host):
            end = self.parameter_count
        else:
            end = self.layer_addresses_host[layer_index + 1]
        base = self.parameters.to_numpy().view(np.float32)
        base[begin:end] = linear_to_numpy(layer).flatten()
        self.parameters = create_buffer(self.device, base)

    def layer_to_numpy(self, layer_index: int) -> np.ndarray:
        shape = self.layer_shapes[layer_index]
        begin = self.layer_addresses_host[layer_index]
        if layer_index + 1 >= len(self.layer_addresses_host):
            end = self.parameter_count
        else:
            end = self.layer_addresses_host[layer_index + 1]
        base = self.parameters.to_numpy().view(np.float32)
        return base[begin:end].reshape(shape[0] + 1, shape[1])

    def layer_gradients_to_numpy(self, layer_index: int) -> np.ndarray:
        shape = self.layer_shapes[layer_index]
        begin = self.layer_addresses_host[layer_index]
        if layer_index + 1 >= len(self.layer_addresses_host):
            end = self.parameter_count
        else:
            end = self.layer_addresses_host[layer_index + 1]
        base = self.gradients.to_numpy().view(np.float32)
        return base[begin:end].reshape(shape[0] + 1, shape[1])

    def dict(self):
        return {
            "parameters": self.parameters,
            "gradients": self.gradients,
            "states": self.optimizer_states,
            "layerAddresses": self.layer_addresses,
            "parameterCount": self.parameter_count,
        }

class Pipeline:
    @staticmethod
    def compile_specialization_module(
        device: spy.Device,
        hidden: int,
        hidden_layers: int,
        levels: int,
        input: int,
        output: int,
    ) -> spy.SlangModule:
        source = f"""
        export static const int In = {input};
        export static const int Out = {output};
        export static const int Hidden = {hidden};
        export static const int HiddenLayers = {hidden_layers};
        export static const int Levels = {levels};
        """
        return device.load_module_from_source("specialization", source)

    def __init__(self, device: spy.Device, network: Network):
        SOURCE = ROOT / "examples" / "slang" / "network_with_addresses_kernels.slang"
        self.device = device
        self.signal_module = device.load_module(str(SOURCE))
        self.specialization_module = self.compile_specialization_module(
            device,
            network.hidden,
            network.hidden_layers,
            network.levels,
            network.input,
            network.output,
        )
        self.forward_kernel = device.create_compute_kernel(device.link_program(
            modules=[self.signal_module, self.specialization_module],
            entry_points=[self.signal_module.entry_point("forward")],
        ))
        self.backward_kernel = device.create_compute_kernel(device.link_program(
            modules=[self.signal_module, self.specialization_module],
            entry_points=[self.signal_module.entry_point("backward")],
        ))
        self.optimize_kernel = device.create_compute_kernel(device.link_program(
            modules=[self.signal_module, self.specialization_module],
            entry_points=[self.signal_module.entry_point("optimize")],
        ))

    def forward(self, network: Network, input: spy.Buffer, output: spy.Buffer):
        elements = input.size // (4 * network.input)
        
        self.forward_kernel.dispatch(
            thread_count=(elements, 1, 1),
            network=network.dict(),
            inputBuffer=input,
            outputBuffer=output,
        )

    def backward(self, network: Network, input: spy.Buffer, expected: spy.Buffer):
        elements = input.size // (4 * network.input)

        self.backward_kernel.dispatch(
            thread_count=(elements, 1, 1),
            network=network.dict(),
            inputBuffer=input,
            expectedBuffer=expected,
            boost=1.0/elements,
        )

    def optimize(self, network: Network, dispatch_size: int = 1024):
        self.optimize_kernel.dispatch(
            thread_count=(dispatch_size, 1, 1),
            network=network.dict(),
            count=network.parameter_count,
            dispatchSize=dispatch_size,
        )