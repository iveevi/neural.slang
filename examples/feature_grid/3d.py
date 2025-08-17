import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from common import *
from examples.ngp import Adam, FeatureGrid
from examples.util import *
from examples.feature_grid.pipeline import FeatureGridPipeline, plot_feature_grid_results


def generate_random_volume(width: int, height: int, depth: int) -> np.ndarray:
    volume = np.random.rand(depth, height, width, 3).astype(np.float32)
    # Apply some smoothing to make it more interesting
    for c in range(3):
        volume[:, :, :, c] = gaussian_filter(volume[:, :, :, c], sigma=1.5)
    return volume


class PyTorchFeatureGrid3D(nn.Module):
    def __init__(self, resolution: int):
        super().__init__()
        self.resolution = resolution
        # 3D grid with 3 features (RGB)
        self.feature_params = nn.Parameter(torch.zeros((resolution, resolution, resolution, 3), dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 2, "Input should be 2D (N, 3)"
        assert input.min() >= 0 and input.max() <= 1, "Input should be in [0, 1]"

        batch_size = input.shape[0]
        
        # Vectorized trilinear interpolation
        scaled = input * (self.resolution - 1)
        lower_idx = torch.floor(scaled).long().clamp(0, self.resolution - 2)
        upper_idx = (lower_idx + 1).clamp(0, self.resolution - 1)
        t = scaled - lower_idx.float()
        
        # Extract coordinates
        x_low, y_low, z_low = lower_idx[:, 0], lower_idx[:, 1], lower_idx[:, 2]
        x_up, y_up, z_up = upper_idx[:, 0], upper_idx[:, 1], upper_idx[:, 2]
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
        
        # Gather the eight corner values for each sample
        c000 = self.feature_params[z_low, y_low, x_low]  # (batch_size, 3)
        c001 = self.feature_params[z_low, y_low, x_up]   # (batch_size, 3)
        c010 = self.feature_params[z_low, y_up, x_low]   # (batch_size, 3)
        c011 = self.feature_params[z_low, y_up, x_up]    # (batch_size, 3)
        c100 = self.feature_params[z_up, y_low, x_low]   # (batch_size, 3)
        c101 = self.feature_params[z_up, y_low, x_up]    # (batch_size, 3)
        c110 = self.feature_params[z_up, y_up, x_low]    # (batch_size, 3)
        c111 = self.feature_params[z_up, y_up, x_up]     # (batch_size, 3)
        
        # Interpolate in x direction
        c00 = c000 * (1 - tx.unsqueeze(1)) + c001 * tx.unsqueeze(1)
        c01 = c010 * (1 - tx.unsqueeze(1)) + c011 * tx.unsqueeze(1)
        c10 = c100 * (1 - tx.unsqueeze(1)) + c101 * tx.unsqueeze(1)
        c11 = c110 * (1 - tx.unsqueeze(1)) + c111 * tx.unsqueeze(1)
        
        # Interpolate in y direction
        c0 = c00 * (1 - ty.unsqueeze(1)) + c01 * ty.unsqueeze(1)
        c1 = c10 * (1 - ty.unsqueeze(1)) + c11 * ty.unsqueeze(1)
        
        # Interpolate in z direction
        output = c0 * (1 - tz.unsqueeze(1)) + c1 * tz.unsqueeze(1)
        
        return output


def main():
    print("Running 3D Feature Grid Example")
    
    # Generate volume data
    width, height, depth = 64, 64, 64
    volume = generate_random_volume(width, height, depth)
    
    # Create 3D coordinates
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    z = np.linspace(0, 1, depth)
    xyz = np.stack(np.meshgrid(x, y, z, indexing='xy'), axis=-1)
    xyz = xyz.reshape(-1, 3).astype(np.float32)
    
    # Flatten volume to match xyz coordinates
    volume_flat = volume.reshape(-1, 3).astype(np.float32)
    
    resolution = 16
    sample_count = xyz.shape[0]

    # Generate the same initial parameters for both implementations
    initial_params = 2 * np.random.rand(resolution * resolution * resolution * 3).astype(np.float32) - 1

    # PyTorch implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_grid = PyTorchFeatureGrid3D(resolution=resolution).to(device)
    
    # Set the same initial parameters
    with torch.no_grad():
        pytorch_grid.feature_params.copy_(torch.tensor(initial_params.reshape(resolution, resolution, resolution, 3), device=device))
    
    xyz_tensor = torch.tensor(xyz, dtype=torch.float32, device=device)
    target = torch.tensor(volume_flat, dtype=torch.float32, device=device)

    # Get initial outputs BEFORE any training
    with torch.no_grad():
        pytorch_initial_output = pytorch_grid(xyz_tensor).cpu().numpy()

    optimizer = torch.optim.Adam(pytorch_grid.parameters(), lr=0.01)
    pytorch_losses = []
    pytorch_params_history = []

    # Run training iterations
    num_iterations = 500
    
    for epoch in tqdm(range(num_iterations), desc="PyTorch Training"):
        optimizer.zero_grad()
        output = pytorch_grid(xyz_tensor)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        pytorch_losses.append(loss.item())
        
        # Store parameters
        with torch.no_grad():
            pytorch_params_history.append(pytorch_grid.feature_params.cpu().numpy().copy())

    with torch.no_grad():
        pytorch_output = pytorch_grid(xyz_tensor).cpu().numpy()

    # SlangPy implementation
    slang_device = create_device()
    slang_grid = FeatureGrid.new(slang_device, dimension=3, features=3, resolution=resolution)
    
    # Set the same initial parameters
    slang_grid.parameter_buffer.copy_from_numpy(initial_params.astype(np.float32))
    
    slang_optimizer = Adam(alpha=0.01)
    slang_optimizer_states = slang_grid.alloc_optimizer_states(slang_device, slang_optimizer)
    slang_pipeline = FeatureGridPipeline(slang_device, dimension=3, features=3)

    # Allocate buffers
    input_buffer = create_buffer_32b(slang_device, xyz.astype(np.float32), 3)
    target_buffer = create_buffer_32b(slang_device, volume_flat.astype(np.float32), 3)
    output_buffer = create_buffer_32b(slang_device, np.zeros((sample_count, 3), dtype=np.float32), 3)
    loss_buffer = create_buffer_32b(slang_device, np.zeros(sample_count, dtype=np.float32), 1)

    # Get initial outputs BEFORE any training
    slang_pipeline.forward(slang_grid, input_buffer, output_buffer, sample_count)
    slang_initial_output = slang_pipeline.get_output(output_buffer).copy()

    slang_losses = []
    slang_params_history = []
    
    for epoch in tqdm(range(num_iterations), desc="SlangPy Training"):
        # Clear gradients
        slang_grid.gradient_buffer.copy_from_numpy(np.zeros(resolution * resolution * resolution * 3, dtype=np.float32))
        
        # Forward pass
        slang_pipeline.forward(slang_grid, input_buffer, output_buffer, sample_count)
        current_slang_output = slang_pipeline.get_output(output_buffer).copy()
        
        # Calculate loss manually for comparison
        slang_loss = np.mean((current_slang_output - volume_flat) ** 2)
        slang_losses.append(slang_loss)
        
        # Backward pass
        slang_pipeline.backward(slang_grid, input_buffer, target_buffer, loss_buffer, sample_count)
        
        # Update parameters
        slang_pipeline.update(slang_grid, slang_optimizer, slang_optimizer_states)
        
        # Store parameters
        slang_params_history.append(slang_grid.parameter_buffer.to_numpy().view(np.float32).copy())

    # Get final output
    slang_pipeline.forward(slang_grid, input_buffer, output_buffer, sample_count)
    slang_output = slang_pipeline.get_output(output_buffer)
    
    # Use common plotting function with 3D flag
    output_delta = plot_feature_grid_results(
        original_data=volume,
        pytorch_initial_output=pytorch_initial_output,
        slang_initial_output=slang_initial_output,
        pytorch_output=pytorch_output,
        slang_output=slang_output,
        pytorch_losses=pytorch_losses,
        slang_losses=slang_losses,
        pytorch_params_history=pytorch_params_history,
        slang_params_history=slang_params_history,
        initial_params=initial_params,
        resolution=resolution,
        data_shape=(depth, height, width)
    )


if __name__ == "__main__":
    main()
