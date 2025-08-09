import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import slangpy as spy
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d


HERE = pathlib.Path(__file__).parent.parent.absolute()


def generate_random_signal(length: int) -> np.ndarray:
    signal = 2 * np.random.rand(length) - 1
    signal = gaussian_filter1d(signal, sigma=2)
    return signal


def compile_specialization_module(device: spy.Device, hidden: int, levels: int) -> spy.SlangModule:
    source = f"""
    export static const int Hidden = {hidden};
    export static const int Levels = {levels};
    """
    return device.load_module_from_source("specialization", source)


def create_buffer(
    device: spy.Device,
    data: np.ndarray,
    struct_size: int = 4,
    usage: spy.BufferUsage = spy.BufferUsage.shader_resource,
) -> spy.Buffer:
    return device.create_buffer(
        size=data.nbytes,
        struct_size=struct_size,
        usage=usage,
        data=data,
    )


class Layer:
    @staticmethod
    def new_params(in_size: int, out_size: int) -> np.ndarray:
        # PyTorch initialization
        # TODO: manual initialization
        linear = nn.Linear(in_size, out_size)
        weights = linear.weight.detach().numpy().T
        bias = linear.bias.detach().numpy().reshape(1, -1)
        return np.ascontiguousarray(np.concatenate((weights, bias), axis=0).astype(np.float32))

    def __init__(self, device: spy.Device, in_size: int, out_size: int):
        self.device = device
        self.in_size = in_size
        self.out_size = out_size

        np_params = self.new_params(in_size, out_size)
        np_adam_states = np.zeros(np_params.shape[0] * 3, dtype=np.float32)

        self.parameters = create_buffer(device, np_params)
        self.gradients = create_buffer(device, np.zeros_like(np_params))
        self.adam_states = create_buffer(device, np_adam_states, 3 * 4)


# TODO: later generalize this to any size, depth, encoding, etc.
class Network:
    def __init__(self, device: spy.Device, hidden: int, levels: int):
        self.device = device
        self.hidden = hidden
        self.levels = levels
        self.input_size = 1 if levels == 0 else 2 * levels

        self.layers = [
            Layer(device, self.input_size, hidden),
            Layer(device, hidden, hidden),
            Layer(device, hidden, hidden),
            Layer(device, hidden, 1),
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

    def states(self):
        return {
            "states": {
                "layer1": self.layers[0].adam_states,
                "layer2": self.layers[1].adam_states,
                "layer3": self.layers[2].adam_states,
                "layer4": self.layers[3].adam_states,
            },
        }

    def counts(self) -> dict[str, int]:
        return {
            "layer1Count": self.hidden * (self.input_size + 1),
            "layer2Count": self.hidden * (self.hidden + 1),
            "layer3Count": self.hidden * (self.hidden + 1),
            "layer4Count": 1 * (self.hidden + 1),
        }


class Pipeline:
    def __init__(self, device: spy.Device, network: Network):
        self.device = device
        self.signal_module = device.load_module(str(HERE / "slang" / "examples" / "signal.slang"))
        self.specialization_module = compile_specialization_module(device, network.hidden, network.levels)
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

        elements = input.size // 4
        
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

        elements = input.size // 4

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
            layer1States=network.layers[0].adam_states,
            layer2Params=network.layers[1].parameters,
            layer2Grads=network.layers[1].gradients,
            layer2States=network.layers[1].adam_states,
            layer3Params=network.layers[2].parameters,
            layer3Grads=network.layers[2].gradients,
            layer3States=network.layers[2].adam_states,
            layer4Params=network.layers[3].parameters,
            layer4Grads=network.layers[3].gradients,
            layer4States=network.layers[3].adam_states,
            dispatchSize=dispatch_size,
            **network.counts(),
        )


# TODO: check against pytorch...
def main():
    length = 1024
    time = np.linspace(0, 1, length)
    signal = generate_random_signal(length)

    device = spy.create_device(
        spy.DeviceType.vulkan,
        enable_debug_layers=True,
        include_paths=[
            HERE / "slang" / "neural",
        ],
    )

    network = Network(device, 64, 8)

    pipeline = Pipeline(device, network)

    input = np.array(time, dtype=np.float32)
    signal = np.array(signal, dtype=np.float32)

    input_buffer = create_buffer(device, input)
    signal_buffer = create_buffer(device, signal)

    output_buffer = create_buffer(device, np.zeros_like(signal))

    # Training loop
    history = []
    for _ in tqdm(range(1000)):
        pipeline.forward(network, input_buffer, output_buffer)
        output = output_buffer.to_numpy().view(np.float32)
        loss = np.mean(np.square(output - signal))
        history.append(loss)

        pipeline.backward(network, input_buffer, signal_buffer)
        pipeline.optimize(network)

    pipeline.forward(network, input_buffer, output_buffer)
    output = output_buffer.to_numpy().view(np.float32)

    _, ax = plt.subplots(2, 1)
    ax[0].plot(time, output)
    ax[0].plot(time, signal)
    ax[1].plot(history)
    ax[1].set_yscale('log')
    plt.show()

if __name__ == "__main__":
    sns.set_theme()
    sns.set_palette("pastel")

    main()
