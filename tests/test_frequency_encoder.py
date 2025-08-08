import numpy as np
import pytest
import torch
import slangpy as spy
from .conftest import assert_close, RANDOM_SEEDS


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_frequency_encoder(device, make_kernel, random_seed):
    """Test FrequencyEncoder with 3D input and 4 levels."""
    kernel = make_kernel("frequency_encoder")
    np.random.seed(random_seed)
    
    # Create input data: 10 samples of 3D vectors including edge cases
    input_data = np.array([
        [0.0, 0.0, 0.0],      # All zeros
        [1.0, 1.0, 1.0],      # All ones
        [0.5, -0.5, 0.25],    # Mixed values
        *np.random.rand(7, 3).tolist()  # Random samples
    ], dtype=np.float32)
    
    # Create buffers
    input_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=3 * 4,  # 3 floats * 4 bytes each
        usage=spy.BufferUsage.shader_resource,
        data=input_data,
    )
    
    # Output size: 2 * Levels * Dim = 2 * 4 * 3 = 24 floats per sample
    output_size = 10 * 24 * 4  # 10 samples * 24 floats * 4 bytes
    output_buffer = device.create_buffer(
        size=output_size,
        struct_size=24 * 4,  # 24 floats * 4 bytes each
        usage=spy.BufferUsage.shader_resource,
    )
    
    # Dispatch kernel
    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
            }
        },
    )
    
    # Get results
    output = output_buffer.to_numpy().view(np.float32).reshape(10, 24)
    
    # Compute expected result manually
    # FrequencyEncoder<float, 3, 4> means:
    # - 3D input
    # - 4 frequency levels
    # - Output: 2 * 4 * 3 = 24 dimensions (sin and cos for each level and dimension)
    
    expected_parts = []
    
    for level in range(4):  # For each frequency level
        k = 2.0 ** level  # k = 1, 2, 4, 8
        frequency = k * np.pi
        
        # Compute sin and cos for all dimensions at this level
        sin_vals = np.sin(frequency * input_data)  # Shape: (10, 3)
        cos_vals = np.cos(frequency * input_data)  # Shape: (10, 3)
        
        # Concatenate sin and cos for this level
        level_encoding = np.concatenate([sin_vals, cos_vals], axis=1)  # Shape: (10, 6)
        expected_parts.append(level_encoding)
    
    # Concatenate all levels
    expected = np.concatenate(expected_parts, axis=1)  # Shape: (10, 24)
    
    assert_close(output, expected, rtol=1e-4, atol=1e-4)
    
    # Verify specific known values for zero input
    zero_output = output[0]
    for level in range(4):
        for dim in range(3):
            sin_idx = 2 * level * 3 + dim
            cos_idx = 2 * level * 3 + dim + 3
            assert abs(zero_output[sin_idx]) < 1e-6, f"Sin should be ~0 for zero input at level {level}, dim {dim}"
            assert abs(zero_output[cos_idx] - 1.0) < 1e-6, f"Cos should be ~1 for zero input at level {level}, dim {dim}"


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_frequency_encoder_derivative(device, make_kernel, random_seed):
    """Test FrequencyEncoder derivatives against PyTorch autograd."""
    kernel = make_kernel("frequency_encoder_derivative")
    np.random.seed(random_seed)
    
    # Create input data: 10 samples of 3D vectors
    input_data = np.array([
        [0.0, 0.0, 0.0],      # All zeros
        [1.0, 1.0, 1.0],      # All ones
        [0.5, -0.5, 0.25],    # Mixed values
        *np.random.rand(7, 3).tolist()  # Random samples
    ], dtype=np.float32)
    
    # Create buffers
    input_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=3 * 4,  # 3 floats * 4 bytes each
        usage=spy.BufferUsage.shader_resource,
        data=input_data,
    )
    
    # Output buffer for gradients (same size as input)
    output_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=3 * 4,
        usage=spy.BufferUsage.shader_resource,
    )
    
    # Dispatch kernel
    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
            }
        },
    )
    
    # Get derivative results
    derivatives = output_buffer.to_numpy().view(np.float32).reshape(10, 3)
    
    # Compute expected derivatives using PyTorch autograd
    input_torch = torch.tensor(input_data, requires_grad=True)
    
    # Implement frequency encoder in PyTorch
    def pytorch_frequency_encoder(input_tensor):
        """PyTorch implementation of FrequencyEncoder<float, 3, 4>."""
        batch_size = input_tensor.shape[0]
        encoded_parts = []
        
        for level in range(4):  # For each frequency level
            k = 2.0 ** level  # k = 1, 2, 4, 8
            frequency = k * torch.pi
            
            # Compute sin and cos for all dimensions at this level
            sin_vals = torch.sin(frequency * input_tensor)  # Shape: (batch_size, 3)
            cos_vals = torch.cos(frequency * input_tensor)  # Shape: (batch_size, 3)
            
            # Concatenate sin and cos for this level
            level_encoding = torch.cat([sin_vals, cos_vals], dim=1)  # Shape: (batch_size, 6)
            encoded_parts.append(level_encoding)
        
        # Concatenate all levels
        output = torch.cat(encoded_parts, dim=1)  # Shape: (batch_size, 24)
        return output
    
    # Compute forward pass and gradients
    encoded_output = pytorch_frequency_encoder(input_torch)
    
    # Create gradient tensor of all ones (like Vec24(1.0) in Slang)
    grad_output = torch.ones_like(encoded_output)
    
    # Compute gradients
    encoded_output.backward(grad_output)
    
    # Ensure gradients were computed
    assert input_torch.grad is not None, "Gradients were not computed"
    expected_derivatives = input_torch.grad.detach().numpy()
    
    # Compare results
    assert_close(derivatives, expected_derivatives, rtol=1e-4, atol=1e-4)