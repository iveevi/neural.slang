
import numpy as np
import pytest
import torch
import slangpy as spy
from .conftest import assert_close, RANDOM_SEEDS
from .test_utils import create_buffer_for_data, create_output_buffer


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_vector_relu_2d(device, make_kernel, random_seed):
    kernel = make_kernel("relu_vector")

    np.random.seed(random_seed)
    data = 2 * np.random.rand(10, 16).astype(np.float32) - 1
    
    input_buffer = create_buffer_for_data(device, data, 64)
    output_buffer = create_output_buffer(device, 10, 16)
    
    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
            }
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(10, 16)
    expected = np.where(data > 0, data, 0)
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_vector_relu_derivative(device, make_kernel, random_seed):
    kernel = make_kernel("relu_derivative")
    np.random.seed(random_seed)
    data = 2 * np.random.rand(10, 2).astype(np.float32) - 1
    
    input_buffer = device.create_buffer(
        size=data.nbytes,
        struct_size=8,
        usage=spy.BufferUsage.shader_resource,
        data=data,
    )
    
    output_buffer = device.create_buffer(
        size=data.nbytes,
        struct_size=8,
        usage=spy.BufferUsage.shader_resource,
    )
    
    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
            }
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(10, 2)
    
    # Use PyTorch backward to compute ReLU derivative
    input_torch = torch.tensor(data, requires_grad=True)
    
    # Apply ReLU activation
    relu_output = torch.relu(input_torch)
    
    # For vector output, we need to provide gradient tensor for backward()
    # Use ones to get the derivative of each element w.r.t. its input
    gradient_tensor = torch.ones_like(relu_output)
    relu_output.backward(gradient_tensor)
    
    expected = input_torch.grad.numpy()
    
    assert_close(output, expected)