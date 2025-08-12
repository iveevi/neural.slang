import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import slangpy as spy
import pathlib
import torch
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm

from ..util import linear_to_numpy, linear_gradients_to_numpy
from ..pytorch_networks import PyTorchNetwork


ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()


def load_texture_data():
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


def main(address_mode: bool = True):
    # Prepare data
    uv, image_flat, image_shape = load_texture_data()
    print(f"UV shape: {uv.shape}, Image shape: {image_flat.shape}")

    # Configuration
    hidden = 32
    levels = 4
    hidden_layers = 2

    # Prepare SlangPy
    slangpy_device = spy.create_device(
        spy.DeviceType.vulkan,
        enable_debug_layers=True,
        include_paths=[
            ROOT / "neural",
        ],
    )

    if address_mode:
        from ..network_with_addresses import Network, TrainingPipeline
        slangpy_network = Network(slangpy_device, hidden=hidden, hidden_layers=hidden_layers, levels=levels, input=2, output=3)
    else:
        from ..network_with_separate_buffers import Network, TrainingPipeline
        slangpy_network = Network(slangpy_device, hidden=hidden, hidden_layers=hidden_layers, levels=levels, input=2, output=3)

    slangpy_pipeline = TrainingPipeline(slangpy_device, slangpy_network)
    slangpy_input = slangpy_network.input_vec(uv)
    slangpy_target = slangpy_network.output_vec(image_flat)
    slangpy_output = slangpy_network.output_vec(np.zeros_like(image_flat))

    # Prepare PyTorch
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_network = PyTorchNetwork(hidden=hidden, levels=levels, input=2, output=3, hidden_layers=hidden_layers).to(torch_device)
    torch_optimizer = torch.optim.Adam(torch_network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    # torch_optimizer = torch.optim.SGD(torch_network.parameters(), lr=0.001, momentum=0.0)
    torch_input = torch.from_numpy(uv).to(torch_device)
    torch_target = torch.from_numpy(image_flat).to(torch_device)

    # Copy weights from PyTorch to SlangPy (different methods for different network types)
    if address_mode:
        # Address-based network uses layer index
        for i, layer in enumerate(torch_network.layers):
            slangpy_network.copy_weights(i, layer)
    else:
        # Separate buffers network has individual layer copy methods
        for i, layer in enumerate(torch_network.layers):
            slangpy_network.layers[i].copy_weights(layer)

    # Training loop
    torch_history = []
    slangpy_history = []

    output_delta = []

    layer_deltas = {i: [] for i in range(len(torch_network.layers))}
    layer_gradient_deltas = {i: [] for i in range(len(torch_network.layers))}

    for _ in tqdm(range(1000)):
        # PyTorch side
        torch_network_output = torch_network(torch_input)
        loss = F.mse_loss(torch_network_output, torch_target)
        torch_history.append(loss.item())

        torch_network.zero_grad()
        loss.backward()

        # SlangPy side
        slangpy_pipeline.forward(slangpy_network, slangpy_input, slangpy_output)
        slangpy_network_output = slangpy_output.to_numpy().view(np.float32).reshape(-1, 3)
        loss = np.mean(np.square(slangpy_network_output - image_flat))
        slangpy_history.append(loss)

        slangpy_pipeline.backward(slangpy_network, slangpy_input, slangpy_target)
        
        # Calculate output deltas
        torch_network_output_np = torch_network_output.detach().cpu().numpy()
        output_delta.append(np.mean(np.abs(torch_network_output_np - slangpy_network_output)))

        # Calculate layer deltas
        torch_layers = [linear_to_numpy(layer) for layer in torch_network.layers]
        if address_mode:
            slangpy_layers = [slangpy_network.layer_to_numpy(i) for i in range(len(torch_layers))]
        else:
            slangpy_layers = [slangpy_network.layers[i].parameters_to_numpy() for i in range(len(torch_layers))]
        
        for i, (torch_layer, slangpy_layer) in enumerate(zip(torch_layers, slangpy_layers)):
            layer_deltas[i].append(np.mean(np.abs(torch_layer - slangpy_layer)))

        # Calculate gradient deltas
        torch_layer_gradients = [linear_gradients_to_numpy(layer) for layer in torch_network.layers]
        if address_mode:
            slangpy_layer_gradients = [slangpy_network.layer_gradients_to_numpy(i) for i in range(len(torch_layer_gradients))]
        else:
            slangpy_layer_gradients = [slangpy_network.layers[i].gradients_to_numpy() for i in range(len(torch_layer_gradients))]
        
        for i, (torch_layer_gradient, slangpy_layer_gradient) in enumerate(zip(torch_layer_gradients, slangpy_layer_gradients)):
            layer_gradient_deltas[i].append(np.mean(np.abs(torch_layer_gradient - slangpy_layer_gradient)))
        
        # Optimize
        slangpy_pipeline.optimize(slangpy_network)
        torch_optimizer.step()

        # Check that slangpy resets gradients
        if address_mode:
            slangpy_layer_gradients = [slangpy_network.layer_gradients_to_numpy(i) for i in range(len(torch_layer_gradients))]
        else:
            slangpy_layer_gradients = [slangpy_network.layers[i].gradients_to_numpy() for i in range(len(torch_layer_gradients))]

        for i, (slangpy_layer_gradient, torch_layer_gradient) in enumerate(zip(slangpy_layer_gradients, torch_layer_gradients)):
            assert slangpy_layer_gradient.sum() == 0, slangpy_layer_gradient

    # Extract final output
    slangpy_pipeline.forward(slangpy_network, slangpy_input, slangpy_output)
    slangpy_output_final = slangpy_output.to_numpy().view(np.float32).reshape(image_shape)

    with torch.no_grad():
        output = torch_network(torch_input)
        torch_output_final = output.cpu().numpy().reshape(image_shape)

    # Plot results
    _, ax = plt.subplots(3, 3, figsize=(15, 12))
    ax = ax.flatten()
    
    # Show original texture
    ax[0].set_title("Original Texture")
    ax[0].imshow(image_flat.reshape(image_shape))
    ax[0].axis('off')
    
    # Show PyTorch reconstruction
    ax[1].set_title("PyTorch Reconstruction")
    ax[1].imshow(np.clip(torch_output_final, 0, 1))
    ax[1].axis('off')
    
    # Show SlangPy reconstruction
    ax[2].set_title("SlangPy Reconstruction")
    ax[2].imshow(np.clip(slangpy_output_final, 0, 1))
    ax[2].axis('off')
    
    ax[3].set_title("Loss")
    ax[3].plot(torch_history, label="PyTorch")
    ax[3].plot(slangpy_history, label="SlangPy")
    ax[3].set_yscale('log')
    ax[3].legend()

    ax[4].set_title("Output Deltas")
    ax[4].plot(output_delta, label="Output")
    ax[4].set_yscale('log')
    ax[4].legend()

    ax[5].set_title("Layer Deltas")
    for i in range(len(torch_network.layers)):
        ax[5].plot(layer_deltas[i], label=f"Layer {i}")
    ax[5].set_yscale('log')
    ax[5].legend()

    ax[6].set_title("Gradient Deltas")
    for i in range(len(torch_network.layers)):
        ax[6].plot(layer_gradient_deltas[i], label=f"Layer {i}")
    ax[6].set_yscale('log')
    ax[6].legend()

    # Show difference images
    ax[7].set_title("PyTorch Difference")
    diff_torch = torch_output_final - image_flat.reshape(image_shape)
    image = ax[7].imshow(np.sum(np.abs(diff_torch), axis=-1), cmap='RdBu')
    plt.colorbar(image, ax=ax[7])
    ax[7].axis('off')

    ax[8].set_title("SlangPy Difference")
    diff_slangpy = slangpy_output_final - image_flat.reshape(image_shape)
    image = ax[8].imshow(np.sum(np.abs(diff_slangpy), axis=-1), cmap='RdBu')
    plt.colorbar(image, ax=ax[8])
    ax[8].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--address-mode", action="store_true", default=False)
    args = parser.parse_args()
    
    sns.set_theme()
    sns.set_palette("pastel")

    main(address_mode=args.address_mode)
