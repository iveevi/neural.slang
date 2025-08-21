import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import slangpy as spy
import torch
import torch.nn as nn
import torch.nn.functional as F
from alive_progress import alive_bar
from scipy.ndimage import gaussian_filter1d
from util import *
from ngp import AddressBasedMLP, Adam
from util.encoders import FourierEncoder


HERE = ROOT / "examples" / "signal"


# Utility functions for MLP operations
def get_layer_shapes(mlp: AddressBasedMLP):
    return [
        (mlp.input, mlp.hidden),
        *[(mlp.hidden, mlp.hidden) for _ in range(mlp.hidden_layers)],
        (mlp.hidden, mlp.output),
    ]


def get_layer_addresses(mlp: AddressBasedMLP):
    layer_shapes = get_layer_shapes(mlp)
    sizes = [(s[0] + 1) * s[1] for s in layer_shapes]  # +1 for bias
    return np.cumsum([0, *sizes])[:-1].astype(np.uint32)


def copy_from_pytorch(mlp: AddressBasedMLP, pytorch_model: nn.Sequential, encoded_size: int):
    # Check that the structures match
    layer_shapes = get_layer_shapes(mlp)
    assert mlp.input == encoded_size, f"Input size mismatch: {mlp.input} != {encoded_size}"
    
    # Calculate layer addresses
    layer_addresses = get_layer_addresses(mlp)
    
    # Copy each layer
    parameters = mlp.parameter_buffer.to_numpy().view(np.float32)
    layer_idx = 0
    for module in pytorch_model:
        if isinstance(module, nn.Linear):
            layer_params = linear_to_numpy(module).flatten()
            address = layer_addresses[layer_idx]
            shape = layer_shapes[layer_idx]
            size = (shape[0] + 1) * shape[1]
            parameters[address:address + size] = layer_params
            layer_idx += 1
    
    mlp.parameter_buffer.copy_from_numpy(parameters)


def layer_to_numpy(mlp: AddressBasedMLP, layer_index: int) -> np.ndarray:
    layer_shapes = get_layer_shapes(mlp)
    layer_addresses = get_layer_addresses(mlp)
    
    shape = layer_shapes[layer_index]
    address = layer_addresses[layer_index]
    sizes = [(s[0] + 1) * s[1] for s in layer_shapes]
    size = sizes[layer_index]
    
    parameters = mlp.parameter_buffer.to_numpy().view(np.float32)
    return parameters[address:address + size].reshape(shape[0] + 1, shape[1])


def layer_gradients_to_numpy(mlp: AddressBasedMLP, layer_index: int) -> np.ndarray:
    layer_shapes = get_layer_shapes(mlp)
    layer_addresses = get_layer_addresses(mlp)
    
    shape = layer_shapes[layer_index]
    address = layer_addresses[layer_index]
    sizes = [(s[0] + 1) * s[1] for s in layer_shapes]
    size = sizes[layer_index]
    
    gradients = mlp.gradient_buffer.to_numpy().view(np.float32)
    return gradients[address:address + size].reshape(shape[0] + 1, shape[1])


class Pipeline:
    @staticmethod
    def load_specialization_module(device: spy.Device, mlp: AddressBasedMLP, levels: int):
        source = f"""
        export static const int Hidden = {mlp.hidden};
        export static const int HiddenLayers = {mlp.hidden_layers};
        export static const int Levels = {levels};
        """
        return device.load_module_from_source("specialization", source)

    def __init__(self, device: spy.Device, mlp: AddressBasedMLP, levels: int):
        SOURCE = HERE / "slang" / "main.slang"
        
        self.device = device
        self.module = device.load_module(str(SOURCE))
        self.specialization_module = self.load_specialization_module(device, mlp, levels)

        self.forward_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "forward")
        self.backward_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "backward")

    def forward(self, mlp: AddressBasedMLP, input_buffer: spy.Buffer, output_buffer: spy.Buffer, sample_count: int):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.forward_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mlp = mlp.dict()
            cursor.inputBuffer = input_buffer
            cursor.outputBuffer = output_buffer
            cmd.dispatch(thread_count=(sample_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    def backward(self, mlp: AddressBasedMLP, input_buffer: spy.Buffer, expected_buffer: spy.Buffer, sample_count: int):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.backward_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mlp = mlp.dict()
            cursor.inputBuffer = input_buffer
            cursor.expectedBuffer = expected_buffer
            cursor.boost = 1.0 / sample_count
            cmd.dispatch(thread_count=(sample_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())


def generate_random_signal(length: int) -> np.ndarray:
    signal = 2 * np.random.rand(length) - 1
    signal = gaussian_filter1d(signal, sigma=2)
    return signal


def main():
    # Prepare data
    length = 1024
    time = np.linspace(0, 1, length)
    signal = generate_random_signal(length)
    time = np.array(time, dtype=np.float32).reshape(-1, 1)
    signal = np.array(signal, dtype=np.float32).reshape(-1, 1)

    # Configuration
    hidden = 32
    levels = 8
    hidden_layers = 3

    # Prepare PyTorch
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create PyTorch model as Sequential
    encoder = FourierEncoder(input_dim=1, levels=levels)
    encoded_size = encoder.output_dim
    
    layers = []
    layers.append(nn.Linear(encoded_size, hidden))
    layers.append(nn.ReLU())
    for _ in range(hidden_layers):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden, 1))
    
    torch_network = nn.Sequential(encoder, *layers).to(torch_device)
    torch_optimizer = torch.optim.Adam(torch_network.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    torch_input = torch.from_numpy(time).to(torch_device)
    torch_signal = torch.from_numpy(signal).to(torch_device)

    # Prepare SlangPy
    device = create_device()
    
    # Create MLP and optimizer
    mlp = AddressBasedMLP.new(device, hidden=hidden, hidden_layers=hidden_layers, input=encoded_size, output=1)
    optimizer = Adam(alpha=1e-3)
    mlp_optimizer_states = mlp.alloc_optimizer_states(device, optimizer)
    
    # Create pipeline
    pipeline = Pipeline(device, mlp, levels)
    
    # Copy weights from PyTorch to Slang
    copy_from_pytorch(mlp, torch_network, encoded_size)
    
    # Create buffers
    input_buffer = create_buffer_32b(device, time)
    signal_buffer = create_buffer_32b(device, signal)
    output_buffer = create_buffer_32b(device, np.zeros_like(signal))

    # Extract linear layers from the sequential model for comparison
    torch_linear_layers = [module for module in torch_network if isinstance(module, nn.Linear)]
    num_layers = len(torch_linear_layers)
    
    # Training loop
    torch_history = []
    slangpy_history = []
    output_delta = []
    layer_deltas = {i: [] for i in range(num_layers)}
    layer_gradient_deltas = {i: [] for i in range(num_layers)}

    iterations = 1000
    with alive_bar(iterations) as bar:
        for _ in range(iterations):
            # PyTorch side
            torch_network_output = torch_network(torch_input)
            loss = F.mse_loss(torch_network_output, torch_signal)
            torch_history.append(loss.item())

            torch_network.zero_grad()
            loss.backward()

            # SlangPy side
            pipeline.forward(mlp, input_buffer, output_buffer, length)
            slangpy_network_output = output_buffer.to_numpy().view(np.float32).reshape(-1, 1)
            loss = np.mean(np.square(slangpy_network_output - signal))
            slangpy_history.append(loss)

            pipeline.backward(mlp, input_buffer, signal_buffer, length)
            
            # Calculate output deltas
            torch_network_output_np = torch_network_output.detach().cpu().numpy().reshape(-1)
            slangpy_network_output_flat = slangpy_network_output.reshape(-1)
            output_delta.append(np.mean(np.abs(torch_network_output_np - slangpy_network_output_flat)))

            # Calculate layer deltas
            torch_layers = [linear_to_numpy(layer) for layer in torch_linear_layers]
            slangpy_layers = [layer_to_numpy(mlp, i) for i in range(len(torch_layers))]
            
            for i, (torch_layer, slangpy_layer) in enumerate(zip(torch_layers, slangpy_layers)):
                layer_deltas[i].append(np.mean(np.abs(torch_layer - slangpy_layer)))

            # Calculate gradient deltas
            torch_layer_gradients = [linear_gradients_to_numpy(layer) for layer in torch_linear_layers]
            slangpy_layer_gradients = [layer_gradients_to_numpy(mlp, i) for i in range(len(torch_layer_gradients))]
            
            for i, (torch_layer_gradient, slangpy_layer_gradient) in enumerate(zip(torch_layer_gradients, slangpy_layer_gradients)):
                layer_gradient_deltas[i].append(np.mean(np.abs(torch_layer_gradient - slangpy_layer_gradient)))
            
            # Optimize
            mlp.update(optimizer, mlp_optimizer_states)
            torch_optimizer.step()

            # Check that slangpy resets gradients
            slangpy_layer_gradients_after = [layer_gradients_to_numpy(mlp, i) for i in range(len(torch_layer_gradients))]
            for i, slangpy_layer_gradient in enumerate(slangpy_layer_gradients_after):
                assert slangpy_layer_gradient.sum() == 0, f"Gradients not reset for layer {i}"
            
            bar()

    # Extract final output
    pipeline.forward(mlp, input_buffer, output_buffer, length)
    slangpy_output = output_buffer.to_numpy().view(np.float32).reshape(-1)

    with torch.no_grad():
        output = torch_network(torch_input)
        torch_output = output.cpu().numpy().reshape(-1)

    # Plot results
    _, ax = plt.subplots(3, 2, figsize=(12, 10))
    ax = ax.flatten()
    
    ax[0].set_title("Reconstruction")
    ax[0].plot(time, torch_output, label="PyTorch", alpha=0.8)
    ax[0].plot(time, slangpy_output, label="SlangPy", alpha=0.8, linestyle='--')
    ax[0].plot(time, signal, label="Signal", alpha=0.6, linewidth=2)
    ax[0].legend()
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Amplitude")
    
    ax[1].set_title("Loss History")
    ax[1].plot(torch_history, label="PyTorch", alpha=0.8)
    ax[1].plot(slangpy_history, label="SlangPy", alpha=0.8, linestyle='--')
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("MSE Loss")

    ax[2].set_title("Output Deltas")
    ax[2].plot(output_delta, label="Output", color='red', alpha=0.8)
    ax[2].set_yscale('log')
    ax[2].legend()
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("Mean Absolute Difference")

    ax[3].set_title("Layer Weight Deltas")
    for i in range(num_layers):
        ax[3].plot(layer_deltas[i], label=f"Layer {i}", alpha=0.8)
    ax[3].set_yscale('log')
    ax[3].legend()
    ax[3].set_xlabel("Iteration")
    ax[3].set_ylabel("Mean Absolute Difference")

    ax[4].set_title("Layer Gradient Deltas")
    for i in range(num_layers):
        ax[4].plot(layer_gradient_deltas[i], label=f"Layer {i}", alpha=0.8)
    ax[4].set_yscale('log')
    ax[4].legend()
    ax[4].set_xlabel("Iteration")
    ax[4].set_ylabel("Mean Absolute Difference")

    # Final comparison
    ax[5].set_title("Final Output Comparison")
    ax[5].plot(time, signal, label="Ground Truth", linewidth=2, alpha=0.8)
    ax[5].plot(time, torch_output, label="PyTorch", alpha=0.7)
    ax[5].plot(time, slangpy_output, label="SlangPy", alpha=0.7, linestyle='--')
    ax[5].legend()
    ax[5].set_xlabel("Time")
    ax[5].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sns.set_theme()
    sns.set_palette("pastel")
    main()
