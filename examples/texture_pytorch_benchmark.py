import numpy as np
import seaborn as sns
import slangpy as spy
import pathlib
import torch
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm

from .network_with_separate_buffers import Network, Pipeline
from .pytorch_networks import PyTorchNetwork
from .profiling import profile, plot_profiling_results, reset_profiler, set_iteration_count


ROOT = pathlib.Path(__file__).parent.parent.absolute()


def load_texture_data():
    """Load texture data for training"""
    image = Image.open(ROOT / "examples" / "media" / "texture-128.png")
    image = np.array(image)
    image = image[..., :3].astype(np.float32) / 255.0
    
    # Create UV coordinates
    uv = np.linspace(0, 1, image.shape[0])
    uv = np.stack(np.meshgrid(uv, uv, indexing='xy'), axis=-1)
    uv = uv.reshape(-1, 2)
    uv = uv.astype(np.float32)
    
    # Flatten image to match UV coordinates
    image_flat = image.reshape(-1, 3)
    
    return uv, image_flat, image.shape


def main():
    # Prepare data
    uv, image_flat, image_shape = load_texture_data()
    print(f"UV shape: {uv.shape}, Image shape: {image_flat.shape}")

    # Configuration
    hidden = 32
    levels = 8

    # Prepare SlangPy
    slangpy_device = spy.create_device(
        spy.DeviceType.vulkan,
        enable_debug_layers=True,
        include_paths=[
            ROOT / "neural",
        ],
    )

    slangpy_network = Network(slangpy_device, hidden=hidden, levels=levels, input=2, output=3)
    slangpy_pipeline = Pipeline(slangpy_device, slangpy_network)
    slangpy_input = slangpy_network.input_vec(uv)
    slangpy_target = slangpy_network.output_vec(image_flat)
    slangpy_output = slangpy_network.output_vec(np.zeros_like(image_flat))

    # Prepare PyTorch
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_network = PyTorchNetwork(hidden=hidden, levels=levels, input=2, output=3).to(torch_device)
    torch_optimizer = torch.optim.Adam(torch_network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    torch_input = torch.from_numpy(uv).to(torch_device)
    torch_target = torch.from_numpy(image_flat).to(torch_device)

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
        loss = F.mse_loss(torch_network_output, torch_target)
        torch.cuda.synchronize()
        return loss
    
    @profile("slangpy_forward")
    def slangpy_forward():
        slangpy_pipeline.forward(slangpy_network, slangpy_input, slangpy_output)
        slangpy_network_output = slangpy_output.to_numpy().view(np.float32).reshape(-1, 3)
        loss = np.mean(np.square(slangpy_network_output - image_flat))
        slangpy_device.wait_for_idle()
        return loss
    
    @profile("pytorch_backward")
    def pytorch_backward(loss):
        torch_network.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
    
    @profile("slangpy_backward")
    def slangpy_backward(_):
        slangpy_pipeline.backward(slangpy_network, slangpy_input, slangpy_target)
        slangpy_device.wait_for_idle()

    @profile("pytorch_optimize")
    def pytorch_optimize():
        torch_optimizer.step()
        torch.cuda.synchronize()

    @profile("slangpy_optimize")
    def slangpy_optimize():
        slangpy_pipeline.optimize(slangpy_network)
        slangpy_device.wait_for_idle()

    # Training loops - separate SlangPy and PyTorch
    print("Running SlangPy training...")
    for i in tqdm(range(100), desc="SlangPy"):
        set_iteration_count(i)
        
        slangpy_loss = slangpy_forward()
        slangpy_backward(slangpy_loss)
        slangpy_optimize()
    
    # Reset iteration count for PyTorch
    set_iteration_count(0)
    
    print("Running PyTorch training...")
    for i in tqdm(range(100), desc="PyTorch"):
        set_iteration_count(i)
        
        pytorch_loss = pytorch_forward()
        pytorch_backward(pytorch_loss)
        pytorch_optimize()

    # Generate plots
    plot_profiling_results(title_suffix=" - Texture Learning")


if __name__ == "__main__":
    sns.set_theme()
    sns.set_palette("pastel")

    main()
