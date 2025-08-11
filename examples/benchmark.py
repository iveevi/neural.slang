import numpy as np
import seaborn as sns
import slangpy as spy
import pathlib
import torch
import torch.nn.functional as F
import argparse

from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

from .network_with_separate_buffers import Network, Pipeline
from .pytorch_networks import PyTorchNetwork
from .profiling import profile, plot_profiling_results, reset_profiler, set_iteration_count


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
    hidden = 8
    levels = 0

    # Prepare SlangPy
    slangpy_device = spy.create_device(
        spy.DeviceType.vulkan,
        enable_debug_layers=False,
        include_paths=[
            ROOT / "neural",
        ],
    )

    if address_mode:
        from .network_with_addresses import Network, Pipeline
    else:
        from .network_with_separate_buffers import Network, Pipeline

    slangpy_network = Network(slangpy_device, hidden=hidden, hidden_layers=2, levels=levels, input=1, output=1)
    slangpy_pipeline = Pipeline(slangpy_device, slangpy_network)
    slangpy_input = slangpy_network.input_vec(time)
    slangpy_signal = slangpy_network.output_vec(signal)
    slangpy_output = slangpy_network.output_vec(np.zeros_like(signal))

    # Prepare PyTorch
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_network = PyTorchNetwork(hidden=hidden, levels=levels, input=1, output=1).to(torch_device)
    torch_optimizer = torch.optim.Adam(torch_network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    torch_input = torch.from_numpy(time).to(torch_device)
    torch_signal = torch.from_numpy(signal).to(torch_device)

    # Copy weights from PyTorch to SlangPy
    slangpy_network.layers[0].copy_weights(torch_network.layer1)
    slangpy_network.layers[1].copy_weights(torch_network.layer2)
    slangpy_network.layers[2].copy_weights(torch_network.layer3)
    slangpy_network.layers[3].copy_weights(torch_network.layer4)

    # Reset profiler for clean start
    reset_profiler()

    # Phases to profile
    @profile("pytorch_forward")
    def pytorch_forward():
        torch_network_output = torch_network(torch_input)
        loss = F.mse_loss(torch_network_output, torch_signal)
        torch.cuda.synchronize()
        return loss
    
    @profile("slangpy_forward")
    def slangpy_forward():
        slangpy_pipeline.forward(slangpy_network, slangpy_input, slangpy_output)
        slangpy_device.wait_for_idle()
        return slangpy_output
    
    @profile("pytorch_backward")
    def pytorch_backward(loss):
        torch_network.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
    
    @profile("slangpy_backward")
    def slangpy_backward():
        slangpy_pipeline.backward(slangpy_network, slangpy_input, slangpy_signal)
        slangpy_device.wait_for_idle()

    @profile("pytorch_optimize")
    def pytorch_optimize():
        torch_optimizer.step()
        torch.cuda.synchronize()

    @profile("slangpy_optimize")
    def slangpy_optimize():
        slangpy_pipeline.optimize(slangpy_network)
        slangpy_device.wait_for_idle()

    @profile("pytorch_inference")
    def pytorch_inference():
        with torch.no_grad():
            torch_network_output = torch_network(torch_input)
        torch.cuda.synchronize()
        return torch_network_output

    @profile("slangpy_inference")
    def slangpy_inference():
        slangpy_pipeline.forward(slangpy_network, slangpy_input, slangpy_output)
        slangpy_device.wait_for_idle()
        return slangpy_output

    # Training loops - separate SlangPy and PyTorch
    print("Running SlangPy training...")
    for i in tqdm(range(1000), desc="SlangPy"):
        set_iteration_count(i)
        
        slangpy_forward()
        slangpy_backward()
        slangpy_optimize()
    
    # Reset iteration count for PyTorch
    set_iteration_count(0)
    
    print("Running PyTorch training...")
    for i in tqdm(range(1000), desc="PyTorch"):
        set_iteration_count(i)
        
        pytorch_loss = pytorch_forward()
        pytorch_backward(pytorch_loss)
        pytorch_optimize()

    # Set PyTorch to eval mode for inference
    torch_network.eval()
    
    print("Running SlangPy inference...")
    for i in tqdm(range(1000), desc="SlangPy"):
        set_iteration_count(i)
        slangpy_inference()

    print("Running PyTorch inference...")
    for i in tqdm(range(1000), desc="PyTorch"):
        set_iteration_count(i)
        pytorch_inference()

    # Generate plots
    plot_profiling_results(title_suffix=" - Signal Learning")

if __name__ == "__main__":
    # TODO: move to util
    parser = argparse.ArgumentParser()
    parser.add_argument("--address-mode", action="store_true")
    args = parser.parse_args()

    sns.set_theme()
    sns.set_palette("pastel")

    main(address_mode=args.address_mode)
