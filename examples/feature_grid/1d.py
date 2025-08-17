import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from common import *
from examples.ngp import Adam, FeatureGrid
from examples.util import *
from examples.feature_grid.pipeline import FeatureGridPipeline, plot_feature_grid_results


def generate_random_signal(length: int) -> np.ndarray:
    signal = 2 * np.random.rand(length) - 1
    signal = gaussian_filter1d(signal, sigma=2)
    return signal


class PyTorchFeatureGrid1D(nn.Module):
    def __init__(self, resolution: int):
        super().__init__()
        self.resolution = resolution
        self.feature_params = nn.Parameter(torch.zeros((resolution,), dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 1, "Input should be 1D"
        assert input.min() >= 0 and input.max() <= 1, "Input should be in [0, 1]"

        # Vectorized linear interpolation
        scaled = input * (self.resolution - 1)
        lower_idx = torch.floor(scaled).long().clamp(0, self.resolution - 2)
        upper_idx = (lower_idx + 1).clamp(0, self.resolution - 1)
        t = scaled - lower_idx.float()

        # Gather values using advanced indexing
        lower_val = self.feature_params[lower_idx]
        upper_val = self.feature_params[upper_idx]

        # Linear interpolation
        output = lower_val * (1 - t) + upper_val * t

        return output


def main():
    print("Running 1D Feature Grid Example")
    
    length = 1024
    time = np.linspace(0, 1, length)
    signal = generate_random_signal(length)
    resolution = 64

    # Generate the same initial parameters for both implementations
    initial_params = 2 * np.random.rand(resolution).astype(np.float32) - 1

    # PyTorch implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_grid = PyTorchFeatureGrid1D(resolution=resolution).to(device)
    
    # Set the same initial parameters
    with torch.no_grad():
        pytorch_grid.feature_params.copy_(torch.tensor(initial_params, device=device))
    
    time_tensor = torch.tensor(time, dtype=torch.float32, device=device)
    target = torch.tensor(signal, dtype=torch.float32, device=device)

    # Get initial outputs BEFORE any training
    with torch.no_grad():
        pytorch_initial_output = pytorch_grid(time_tensor).cpu().numpy()

    optimizer = torch.optim.Adam(pytorch_grid.parameters(), lr=0.01)
    pytorch_losses = []
    pytorch_params_history = []

    # Run training iterations
    num_iterations = 100
    
    for epoch in tqdm(range(num_iterations), desc="PyTorch Training"):
        optimizer.zero_grad()
        output = pytorch_grid(time_tensor)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        pytorch_losses.append(loss.item())
        
        # Store parameters
        with torch.no_grad():
            pytorch_params_history.append(pytorch_grid.feature_params.cpu().numpy().copy())

    with torch.no_grad():
        pytorch_output = pytorch_grid(time_tensor).cpu().numpy()

    # SlangPy implementation
    slang_device = create_device()
    slang_grid = FeatureGrid.new(slang_device, dimension=1, features=1, resolution=resolution)
    
    # Set the same initial parameters
    slang_grid.parameter_buffer.copy_from_numpy(initial_params.astype(np.float32))
    
    slang_optimizer = Adam(alpha=0.01)
    slang_optimizer_states = slang_grid.alloc_optimizer_states(slang_device, slang_optimizer)
    slang_pipeline = FeatureGridPipeline(slang_device, dimension=1, features=1)

    # Convert 1D data to proper format for unified shader
    time_1d = time.astype(np.float32)
    signal_1d = signal.astype(np.float32)
    
    # Allocate buffers
    input_buffer = create_buffer_32b(slang_device, time_1d, 1)
    target_buffer = create_buffer_32b(slang_device, signal_1d, 1)
    output_buffer = create_buffer_32b(slang_device, np.zeros_like(signal_1d), 1)
    loss_buffer = create_buffer_32b(slang_device, np.zeros(length, dtype=np.float32), 1)

    # Get initial outputs BEFORE any training
    slang_pipeline.forward(slang_grid, input_buffer, output_buffer, length)
    slang_initial_output = slang_pipeline.get_output(output_buffer).copy()

    slang_losses = []
    slang_params_history = []
    
    for epoch in tqdm(range(num_iterations), desc="SlangPy Training"):
        # Clear gradients
        slang_grid.gradient_buffer.copy_from_numpy(np.zeros(resolution, dtype=np.float32))
        
        # Forward pass
        slang_pipeline.forward(slang_grid, input_buffer, output_buffer, length)
        current_slang_output = slang_pipeline.get_output(output_buffer).copy()
        
        # Calculate loss manually for comparison
        slang_loss = np.mean((current_slang_output - signal) ** 2)
        slang_losses.append(slang_loss)
        
        # Backward pass
        slang_pipeline.backward(slang_grid, input_buffer, target_buffer, loss_buffer, length)
        
        # Update parameters
        slang_pipeline.update(slang_grid, slang_optimizer, slang_optimizer_states)
        
        # Store parameters
        slang_params_history.append(slang_grid.parameter_buffer.to_numpy().view(np.float32).copy())

    # Get final output
    slang_pipeline.forward(slang_grid, input_buffer, output_buffer, length)
    slang_output = slang_pipeline.get_output(output_buffer)
    
    # Use common plotting function
    plot_feature_grid_results(
        original_data=signal,
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
        data_shape=(length,)
    )


if __name__ == "__main__":
    main()
