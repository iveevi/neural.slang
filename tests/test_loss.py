import numpy as np
import pytest
import slangpy as spy
import torch
from .conftest import assert_close
from .test_utils import create_buffer_for_data, create_output_buffer


def create_specialization_module(device, in_size):
    source = f"""
    export static const int In = {in_size};
    """
    return device.load_module_from_source("specialization", source)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64, 128])
def test_mse_basic(device, make_kernel, random_seed, in_size):
    np.random.seed(random_seed)
    
    batch_size = 16
    input_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    target_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    
    specialization_module = create_specialization_module(device, in_size)
    kernel = make_kernel("mse", link_modules=[specialization_module])
    
    input_buffer = create_buffer_for_data(device, input_data, in_size * 4)
    target_buffer = create_buffer_for_data(device, target_data, in_size * 4)
    output_buffer = create_output_buffer(device, batch_size, 1)
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
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


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64, 128])
def test_mse_derivative(device, make_kernel, random_seed, in_size):
    np.random.seed(random_seed)
    
    # Generate random predicted and expected values
    batch_size = 16
    predicted_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    expected_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    
    specialization_module = create_specialization_module(device, in_size)
    kernel = make_kernel("mse_derivative", link_modules=[specialization_module])
    
    predicted_buffer = device.create_buffer(
        size=predicted_data.nbytes,
        struct_size=in_size * 4,
        usage=spy.BufferUsage.shader_resource,
        data=predicted_data,
    )
    
    expected_buffer = device.create_buffer(
        size=expected_data.nbytes,
        struct_size=in_size * 4,
        usage=spy.BufferUsage.shader_resource,
        data=expected_data,
    )
    
    output_buffer = device.create_buffer(
        size=predicted_data.nbytes,
        struct_size=in_size * 4,
        usage=spy.BufferUsage.shader_resource,
    )
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "predicted": predicted_buffer,
                "expected": expected_buffer,
                "output": output_buffer,
            }
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    
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