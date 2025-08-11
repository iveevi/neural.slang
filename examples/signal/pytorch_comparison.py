import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import slangpy as spy
import pathlib
import torch
import torch.nn.functional as F
import argparse

from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

from ..util import linear_to_numpy, linear_gradients_to_numpy
from ..pytorch_networks import PyTorchNetwork


ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()


def generate_random_signal(length: int) -> np.ndarray:
    signal = 2 * np.random.rand(length) - 1
    signal = gaussian_filter1d(signal, sigma=2)
    return signal


def main(address_mode: bool = True):
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
    torch_network = PyTorchNetwork(hidden=hidden, levels=levels, input=1, output=1, hidden_layers=hidden_layers).to(torch_device)
    torch_optimizer = torch.optim.Adam(torch_network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    # torch_optimizer = torch.optim.SGD(torch_network.parameters(), lr=0.001, momentum=0.9)
    torch_input = torch.from_numpy(time).to(torch_device)
    torch_signal = torch.from_numpy(signal).to(torch_device)

    # Prepare SlangPy
    slangpy_device = spy.create_device(
        spy.DeviceType.vulkan,
        enable_debug_layers=True,
        include_paths=[
            ROOT / "neural",
        ],
    )

    if address_mode:
        from ..network_with_addresses import Network, Pipeline
        slangpy_network = Network(slangpy_device, hidden=hidden, hidden_layers=hidden_layers, levels=levels, input=1, output=1)
        slangpy_pipeline = Pipeline(slangpy_device, slangpy_network)

        for i, layer in enumerate(torch_network.layers):
            slangpy_network.copy_weights(i, layer)
    else:
        from ..network_with_separate_buffers import Network, Pipeline
        slangpy_network = Network(slangpy_device, hidden=hidden, hidden_layers=hidden_layers, levels=levels, input=1, output=1)
        slangpy_pipeline = Pipeline(slangpy_device, slangpy_network)
        
        for i, layer in enumerate(torch_network.layers):
            slangpy_network.layers[i].copy_weights(layer)

    slangpy_input = slangpy_network.input_vec(time)
    slangpy_signal = slangpy_network.output_vec(signal)
    slangpy_output = slangpy_network.output_vec(np.zeros_like(signal))

    # Training loop
    torch_history = []
    slangpy_history = []

    output_delta = []

    layer_deltas = {i: [] for i in range(len(torch_network.layers))}

    layer_gradient_deltas = {i: [] for i in range(len(torch_network.layers))}

    for _ in tqdm(range(1000)):
        # PyTorch side
        torch_network_output = torch_network(torch_input)
        loss = F.mse_loss(torch_network_output, torch_signal)
        torch_history.append(loss.item())

        torch_network.zero_grad()
        loss.backward()

        # SlangPy side
        slangpy_pipeline.forward(slangpy_network, slangpy_input, slangpy_output)
        slangpy_network_output = slangpy_output.to_numpy().view(np.float32).reshape(-1, 1)
        loss = np.mean(np.square(slangpy_network_output - signal))
        slangpy_history.append(loss)

        slangpy_pipeline.backward(slangpy_network, slangpy_input, slangpy_signal)
        
        # Calculate output deltas
        torch_network_output = torch_network_output.detach().cpu().numpy().reshape(-1)
        slangpy_network_output = slangpy_network_output.reshape(-1)
        output_delta.append(np.mean(np.abs(torch_network_output - slangpy_network_output)))

        # Calculate layer deltas
        torch_layers = [linear_to_numpy(layer) for layer in torch_network.layers]
        slangpy_layers = [slangpy_network.layer_to_numpy(i) for i in range(len(torch_layers))]
        
        for i, (torch_layer, slangpy_layer) in enumerate(zip(torch_layers, slangpy_layers)):
            layer_deltas[i].append(np.mean(np.abs(torch_layer - slangpy_layer)))

        # Calculate gradient deltas
        torch_layer_gradients = [linear_gradients_to_numpy(layer) for layer in torch_network.layers]
        slangpy_layer_gradients = [slangpy_network.layer_gradients_to_numpy(i) for i in range(len(torch_layer_gradients))]
        
        for i, (torch_layer_gradient, slangpy_layer_gradient) in enumerate(zip(torch_layer_gradients, slangpy_layer_gradients)):
            layer_gradient_deltas[i].append(np.mean(np.abs(torch_layer_gradient - slangpy_layer_gradient)))
        
        # Optimize
        slangpy_pipeline.optimize(slangpy_network)
        torch_optimizer.step()

        # Check that slangpy resets gradients
        slangpy_layer_gradients = [slangpy_network.layer_gradients_to_numpy(i) for i in range(len(torch_layer_gradients))]

        for i, (slangpy_layer_gradient, torch_layer_gradient) in enumerate(zip(slangpy_layer_gradients, torch_layer_gradients)):
            assert slangpy_layer_gradient.sum() == 0, slangpy_layer_gradient

    # Extract final output
    slangpy_pipeline.forward(slangpy_network, slangpy_input, slangpy_output)
    slangpy_output = slangpy_output.to_numpy().view(np.float32).reshape(-1, 1)

    with torch.no_grad():
        output = torch_network(torch_input)
        torch_output = output.cpu().numpy().reshape(-1)

    # Plot results
    _, ax = plt.subplots(3, 2)
    ax = ax.flatten()
    
    ax[0].set_title("Reconstruction")
    ax[0].plot(time, torch_output, label="PyTorch")
    ax[0].plot(time, slangpy_output, label="SlangPy")
    ax[0].plot(time, signal, label="Signal")
    ax[0].legend()
    
    ax[1].set_title("Loss")
    ax[1].plot(torch_history, label="PyTorch")
    ax[1].plot(slangpy_history, label="SlangPy")
    ax[1].set_yscale('log')
    ax[1].legend()

    ax[2].set_title("Output Deltas")
    ax[2].plot(output_delta, label="Output")
    ax[2].set_yscale('log')
    ax[2].legend()

    ax[3].set_title("Layer Deltas")
    for i in range(len(torch_network.layers)):
        ax[3].plot(layer_deltas[i], label=f"Layer {i}")
    ax[3].set_yscale('log')
    ax[3].legend()

    ax[4].set_title("Gradient Deltas")
    for i in range(len(torch_network.layers)):
        ax[4].plot(layer_gradient_deltas[i], label=f"Layer {i}")
    ax[4].set_yscale('log')
    ax[4].legend()

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address-mode", action="store_true")
    args = parser.parse_args()

    sns.set_theme()
    sns.set_palette("pastel")

    main(args.address_mode)
