import pathlib
import slangpy as spy
import numpy as np
import torch.nn as nn
from .util import create_buffer, linear_to_numpy


ROOT = pathlib.Path(__file__).parent.parent.absolute()


class Layer:
    def __init__(self, device: spy.Device, in_size: int, out_size: int):
        self.device = device
        self.in_size = in_size
        self.out_size = out_size

        np_params = np.zeros((in_size + 1, out_size), dtype=np.float32)
        np_adam_states = np.zeros(np_params.size * 3, dtype=np.float32)
        # np_sgd_states = np.zeros(np_params.size, dtype=np.float32)

        self.parameters = create_buffer(device, np_params)
        self.gradients = create_buffer(device, np.zeros_like(np_params))
        self.optimizer_states = create_buffer(device, np_adam_states, 3 * 4)
        # self.optimizer_states = create_buffer(device, np_sgd_states, 4)

        self.copy_weights(nn.Linear(in_size, out_size))

    def copy_weights(self, linear: nn.Linear):
        self.parameters = create_buffer(self.device, linear_to_numpy(linear))

    def parameters_to_numpy(self) -> np.ndarray:
        return self.parameters.to_numpy().view(np.float32).reshape(self.in_size + 1, self.out_size)

    def gradients_to_numpy(self) -> np.ndarray:
        return self.gradients.to_numpy().view(np.float32).reshape(self.in_size + 1, self.out_size)


# TODO: later generalize this to any size, depth, encoding, etc.
class Network:
    def __init__(self, device: spy.Device, hidden: int, hidden_layers: int, levels: int, input: int, output: int):
        assert hidden_layers == 2

        self.device = device
        self.hidden = hidden
        self.levels = levels
        self.input = input
        self.output = output
        self.input_size = self.input if levels == 0 else 2 * levels * self.input

        self.layers = [
            Layer(device, self.input_size, hidden),
            Layer(device, hidden, hidden),
            Layer(device, hidden, hidden),
            Layer(device, hidden, self.output),
        ]

    def parameters(self):
        return {
            "parameters": {
                "layer1": self.layers[0].parameters,
                "layer2": self.layers[1].parameters,
                "layer3": self.layers[2].parameters,
                "layer4": self.layers[3].parameters,
            }
        }

    def gradients(self):
        return {
            "gradients": {
                "layer1": self.layers[0].gradients,
                "layer2": self.layers[1].gradients,
                "layer3": self.layers[2].gradients,
                "layer4": self.layers[3].gradients,
            }
        }

    def counts(self) -> dict[str, int]:
        return {
            "layer1Count": self.hidden * (self.input_size + 1),
            "layer2Count": self.hidden * (self.hidden + 1),
            "layer3Count": self.hidden * (self.hidden + 1),
            "layer4Count": self.output * (self.hidden + 1),
        }

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

    def layer_to_numpy(self, layer_index: int) -> np.ndarray:
        return self.layers[layer_index].parameters_to_numpy()

    def layer_gradients_to_numpy(self, layer_index: int) -> np.ndarray:
        return self.layers[layer_index].gradients_to_numpy()


class TrainingPipeline:
    @staticmethod
    def compile_specialization_module(
        device: spy.Device,
        hidden: int,
        levels: int,
        input: int,
        output: int,
    ) -> spy.SlangModule:
        source = f"""
        export static const int In = {input};
        export static const int Out = {output};
        export static const int Hidden = {hidden};
        export static const int Levels = {levels};
        """
        return device.load_module_from_source("specialization", source)
    
    def __init__(self, device: spy.Device, network: Network):
        SOURCE = ROOT / "examples" / "slang" / "network_with_separate_buffers_kernels.slang"
        self.device = device
        self.signal_module = device.load_module(str(SOURCE))
        self.specialization_module = self.compile_specialization_module(
            device,
            network.hidden,
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
        globals = {}
        globals.update(network.parameters())

        elements = input.size // (4 * network.input)
        
        self.forward_kernel.dispatch(
            thread_count=(elements, 1, 1),
            vars=globals,
            inputBuffer=input,
            outputBuffer=output,
        )

    def backward(self, network: Network, input: spy.Buffer, expected: spy.Buffer):
        globals = {}
        globals.update(network.parameters())
        globals.update(network.gradients())

        elements = input.size // (4 * network.input)

        self.backward_kernel.dispatch(
            thread_count=(elements, 1, 1),
            vars=globals,
            inputBuffer=input,
            expectedBuffer=expected,
            boost=1.0/elements,
        )
    
    def optimize(self, network: Network, dispatch_size: int = 1024):        
        self.optimize_kernel.dispatch(
            thread_count=(dispatch_size, 1, 1),
            layer1Params=network.layers[0].parameters,
            layer1Grads=network.layers[0].gradients,
            layer1States=network.layers[0].optimizer_states,
            layer2Params=network.layers[1].parameters,
            layer2Grads=network.layers[1].gradients,
            layer2States=network.layers[1].optimizer_states,
            layer3Params=network.layers[2].parameters,
            layer3Grads=network.layers[2].gradients,
            layer3States=network.layers[2].optimizer_states,
            layer4Params=network.layers[3].parameters,
            layer4Grads=network.layers[3].gradients,
            layer4States=network.layers[3].optimizer_states,
            dispatchSize=dispatch_size,
            **network.counts(),
        )