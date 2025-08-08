
import numpy as np
import pytest
import torch
import slangpy as spy
from .conftest import assert_close, RANDOM_SEEDS
from .test_utils import create_buffer_for_data, create_output_buffer


def create_specialization_module(device, in_size):
    source = f"""
    import neural;
    export static const int In = {in_size};
    """
    return device.load_module_from_source("specialization", source)

@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64, 128])
def test_relu(device, make_kernel, random_seed, in_size):
    np.random.seed(random_seed)
    batch_size = 16
    data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1

    specialization_module = create_specialization_module(device, in_size)
    kernel = make_kernel("relu_vector", link_modules=[specialization_module])
    
    input_buffer = create_buffer_for_data(device, data, 4 * in_size)
    output_buffer = create_output_buffer(device, batch_size, in_size)
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
            }
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    expected = np.where(data > 0, data, 0)
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64, 128])
def test_relu_derivative(device, make_kernel, random_seed, in_size):
    np.random.seed(random_seed)
    batch_size = 16
    data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1

    specialization_module = create_specialization_module(device, in_size)
    kernel = make_kernel("relu_derivative", link_modules=[specialization_module])
    
    input_buffer = create_buffer_for_data(device, data, 4 * in_size)
    output_buffer = create_output_buffer(device, batch_size, in_size)
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
            }
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    
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