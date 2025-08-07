
import numpy as np
import pytest
import torch
import slangpy as spy
from .conftest import assert_close, RANDOM_SEEDS


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_mse_basic(device, make_kernel, random_seed):
    kernel = make_kernel("mse")
    np.random.seed(random_seed)
    input_data = 2 * np.random.rand(10, 16).astype(np.float32) - 1
    target_data = 2 * np.random.rand(10, 16).astype(np.float32) - 1
    
    input_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=16 * 4,
        usage=spy.BufferUsage.shader_resource,
        data=input_data,
    )
    
    target_buffer = device.create_buffer(
        size=target_data.nbytes,
        struct_size=16 * 4,
        usage=spy.BufferUsage.shader_resource,
        data=target_data,
    )
    
    output_buffer = device.create_buffer(
        size=10 * 4,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource,
    )
    
    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "target": target_buffer,
                "output": output_buffer,
            }
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32)
    expected = np.mean(np.square(input_data - target_data), axis=1)
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_mse_derivative(device, make_kernel, random_seed):
    kernel = make_kernel("mse_derivative")
    np.random.seed(random_seed)
    
    # Generate random predicted and expected values
    predicted_data = 2 * np.random.rand(10, 16).astype(np.float32) - 1
    expected_data = 2 * np.random.rand(10, 16).astype(np.float32) - 1
    
    predicted_buffer = device.create_buffer(
        size=predicted_data.nbytes,
        struct_size=16 * 4,
        usage=spy.BufferUsage.shader_resource,
        data=predicted_data,
    )
    
    expected_buffer = device.create_buffer(
        size=expected_data.nbytes,
        struct_size=16 * 4,
        usage=spy.BufferUsage.shader_resource,
        data=expected_data,
    )
    
    output_buffer = device.create_buffer(
        size=predicted_data.nbytes,
        struct_size=16 * 4,
        usage=spy.BufferUsage.shader_resource,
    )
    
    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "globals": {
                "predicted": predicted_buffer,
                "expected": expected_buffer,
                "output": output_buffer,
            }
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(10, 16)
    
    # Use PyTorch autograd to compute MSE derivative
    predicted_torch = torch.tensor(predicted_data, requires_grad=True)
    expected_torch = torch.tensor(expected_data)
    
    # Compute MSE loss
    mse_loss = torch.mean((predicted_torch - expected_torch) ** 2, dim=1)
    # Sum over batch dimension to get scalar loss for backprop
    total_loss = torch.sum(mse_loss)
    
    # Compute gradients
    total_loss.backward()
    expected_derivative = predicted_torch.grad.numpy()
    
    assert_close(output, expected_derivative)