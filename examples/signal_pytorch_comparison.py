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

from .util import linear_to_numpy, linear_gradients_to_numpy
from .pytorch_networks import PyTorchNetwork


ROOT = pathlib.Path(__file__).parent.parent.absolute()


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
    hidden = 64
    levels = 8

    # Prepare PyTorch
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_network = PyTorchNetwork(hidden=hidden, levels=levels, input=1, output=1).to(torch_device)
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
        from .network_with_addresses import Network, Pipeline
        slangpy_network = Network(slangpy_device, hidden=hidden, hidden_layers=2, levels=levels, input=1, output=1)
        slangpy_pipeline = Pipeline(slangpy_device, slangpy_network)

        slangpy_network.copy_weights(0, torch_network.layer1)
        slangpy_network.copy_weights(1, torch_network.layer2)
        slangpy_network.copy_weights(2, torch_network.layer3)
        slangpy_network.copy_weights(3, torch_network.layer4)
    else:
        from .network_with_separate_buffers import Network, Pipeline
        slangpy_network = Network(slangpy_device, hidden=hidden, hidden_layers=2, levels=levels, input=1, output=1)
        slangpy_pipeline = Pipeline(slangpy_device, slangpy_network)
        
        slangpy_network.layers[0].copy_weights(torch_network.layer1)
        slangpy_network.layers[1].copy_weights(torch_network.layer2)
        slangpy_network.layers[2].copy_weights(torch_network.layer3)
        slangpy_network.layers[3].copy_weights(torch_network.layer4)

    slangpy_input = slangpy_network.input_vec(time)
    slangpy_signal = slangpy_network.output_vec(signal)
    slangpy_output = slangpy_network.output_vec(np.zeros_like(signal))

    # Training loop
    torch_history = []
    slangpy_history = []

    output_delta = []

    layer1_delta = []
    layer2_delta = []
    layer3_delta = []
    layer4_delta = []

    layer1_gradient_delta = []
    layer2_gradient_delta = []
    layer3_gradient_delta = []
    layer4_gradient_delta = []

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
        torch_layer1 = linear_to_numpy(torch_network.layer1)
        torch_layer2 = linear_to_numpy(torch_network.layer2)
        torch_layer3 = linear_to_numpy(torch_network.layer3)
        torch_layer4 = linear_to_numpy(torch_network.layer4)

        slangpy_layer1 = slangpy_network.layer_to_numpy(0)
        slangpy_layer2 = slangpy_network.layer_to_numpy(1)
        slangpy_layer3 = slangpy_network.layer_to_numpy(2)
        slangpy_layer4 = slangpy_network.layer_to_numpy(3)
        
        layer1_delta.append(np.mean(np.abs(torch_layer1 - slangpy_layer1)))
        layer2_delta.append(np.mean(np.abs(torch_layer2 - slangpy_layer2)))
        layer3_delta.append(np.mean(np.abs(torch_layer3 - slangpy_layer3)))
        layer4_delta.append(np.mean(np.abs(torch_layer4 - slangpy_layer4)))

        # Calculate gradient deltas
        torch_layer1_gradient = linear_gradients_to_numpy(torch_network.layer1)
        torch_layer2_gradient = linear_gradients_to_numpy(torch_network.layer2)
        torch_layer3_gradient = linear_gradients_to_numpy(torch_network.layer3)
        torch_layer4_gradient = linear_gradients_to_numpy(torch_network.layer4)
        
        slangpy_layer1_gradient = slangpy_network.layer_gradients_to_numpy(0)
        slangpy_layer2_gradient = slangpy_network.layer_gradients_to_numpy(1)
        slangpy_layer3_gradient = slangpy_network.layer_gradients_to_numpy(2)
        slangpy_layer4_gradient = slangpy_network.layer_gradients_to_numpy(3)
        
        layer1_gradient_delta.append(np.mean(np.abs(torch_layer1_gradient - slangpy_layer1_gradient)))
        layer2_gradient_delta.append(np.mean(np.abs(torch_layer2_gradient - slangpy_layer2_gradient)))
        layer3_gradient_delta.append(np.mean(np.abs(torch_layer3_gradient - slangpy_layer3_gradient)))
        layer4_gradient_delta.append(np.mean(np.abs(torch_layer4_gradient - slangpy_layer4_gradient)))
        
        # Optimize
        slangpy_pipeline.optimize(slangpy_network)
        torch_optimizer.step()

        # Check that slangpy resets gradients
        slangpy_layer1_gradient = slangpy_network.layer_gradients_to_numpy(0)
        slangpy_layer2_gradient = slangpy_network.layer_gradients_to_numpy(1)
        slangpy_layer3_gradient = slangpy_network.layer_gradients_to_numpy(2)
        slangpy_layer4_gradient = slangpy_network.layer_gradients_to_numpy(3)

        assert slangpy_layer1_gradient.sum() == 0, slangpy_layer1_gradient
        assert slangpy_layer2_gradient.sum() == 0, slangpy_layer2_gradient
        assert slangpy_layer3_gradient.sum() == 0, slangpy_layer3_gradient
        assert slangpy_layer4_gradient.sum() == 0, slangpy_layer4_gradient

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
    ax[3].plot(layer1_delta, label="Layer 1")
    ax[3].plot(layer2_delta, label="Layer 2")
    ax[3].plot(layer3_delta, label="Layer 3")
    ax[3].plot(layer4_delta, label="Layer 4")
    ax[3].set_yscale('log')
    ax[3].legend()

    ax[4].set_title("Gradient Deltas")
    ax[4].plot(layer1_gradient_delta, label="Layer 1")
    ax[4].plot(layer2_gradient_delta, label="Layer 2")
    ax[4].plot(layer3_gradient_delta, label="Layer 3")
    ax[4].plot(layer4_gradient_delta, label="Layer 4")
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
