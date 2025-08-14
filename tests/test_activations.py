
import numpy as np
import pytest
import torch
from .conftest import assert_close
from common import *


def create_specialization_module(device, in_size):
    source = f"""
    import neural;
    export static const int In = {in_size};
    """
    return device.load_module_from_source("specialization", source)


# TODO: parameterize with types of vectors
@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64, 128])
def test_relu(device, make_kernel, random_seed, in_size):
    np.random.seed(random_seed)
    batch_size = 16
    data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1

    specialization_module = create_specialization_module(device, in_size)
    kernel = make_kernel("relu", link_modules=[specialization_module])
    
    input_buffer = create_buffer_32b(device, data, in_size)
    output_buffer = create_batched_buffer_32b(device, batch_size, in_size)
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
                "count": batch_size,
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
    
    input_buffer = create_buffer_32b(device, data, in_size)
    output_buffer = create_batched_buffer_32b(device, batch_size, in_size)
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
                "count": batch_size,
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


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64, 128])
@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.2])
def test_leaky_relu(device, make_kernel, random_seed, in_size, alpha):
    np.random.seed(random_seed)
    batch_size = 16
    data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1

    specialization_module = create_specialization_module(device, in_size)
    kernel = make_kernel("leaky_relu", link_modules=[specialization_module])
    
    input_buffer = create_buffer_32b(device, data, in_size)
    output_buffer = create_batched_buffer_32b(device, batch_size, in_size)
    alpha_buffer = create_buffer_32b(device, np.array([alpha], dtype=np.float32))
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
                "alpha": alpha_buffer,
                "count": batch_size,
            }
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    expected = np.where(data > 0, data, alpha * data)
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64, 128])
@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.2])
def test_leaky_relu_derivative(device, make_kernel, random_seed, in_size, alpha):
    np.random.seed(random_seed)
    batch_size = 16
    data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1

    specialization_module = create_specialization_module(device, in_size)
    kernel = make_kernel("leaky_relu_derivative", link_modules=[specialization_module])
    
    input_buffer = create_buffer_32b(device, data, in_size)
    output_buffer = create_batched_buffer_32b(device, batch_size, in_size)
    alpha_buffer = create_buffer_32b(device, np.array([alpha], dtype=np.float32))
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
                "alpha": alpha_buffer,
                "count": batch_size,
            }
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    
    # Use PyTorch backward to compute Leaky ReLU derivative
    input_torch = torch.tensor(data, requires_grad=True)
    
    # Apply Leaky ReLU activation
    leaky_relu_output = torch.nn.functional.leaky_relu(input_torch, negative_slope=alpha)
    
    # For vector output, we need to provide gradient tensor for backward()
    # Use ones to get the derivative of each element w.r.t. its input
    gradient_tensor = torch.ones_like(leaky_relu_output)
    leaky_relu_output.backward(gradient_tensor)
    
    expected = input_torch.grad.numpy()
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64, 128])
def test_sine(device, make_kernel, random_seed, in_size):
    np.random.seed(random_seed)
    batch_size = 16
    data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1

    specialization_module = create_specialization_module(device, in_size)
    kernel = make_kernel("sine", link_modules=[specialization_module])
    
    input_buffer = create_buffer_32b(device, data, in_size)
    output_buffer = create_batched_buffer_32b(device, batch_size, in_size)
    
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
    expected = np.sin(data)
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64, 128])
def test_sine_derivative(device, make_kernel, random_seed, in_size):
    np.random.seed(random_seed)
    batch_size = 16
    data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1

    specialization_module = create_specialization_module(device, in_size)
    kernel = make_kernel("sine_derivative", link_modules=[specialization_module])
    
    input_buffer = create_buffer_32b(device, data, in_size)
    output_buffer = create_batched_buffer_32b(device, batch_size, in_size)
    
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
    
    # Use PyTorch backward to compute Sine derivative
    input_torch = torch.tensor(data, requires_grad=True)
    
    # Apply Sine activation
    sine_output = torch.sin(input_torch)
    
    # For vector output, we need to provide gradient tensor for backward()
    # Use ones to get the derivative of each element w.r.t. its input
    gradient_tensor = torch.ones_like(sine_output)
    sine_output.backward(gradient_tensor)
    
    expected = input_torch.grad.numpy()
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64, 128])
def test_exp(device, make_kernel, random_seed, in_size):
    np.random.seed(random_seed)
    batch_size = 16
    # Use smaller range for exp to avoid overflow
    data = np.random.rand(batch_size, in_size).astype(np.float32) * 2 - 1

    specialization_module = create_specialization_module(device, in_size)
    kernel = make_kernel("exp", link_modules=[specialization_module])
    
    input_buffer = create_buffer_32b(device, data, in_size)
    output_buffer = create_batched_buffer_32b(device, batch_size, in_size)
    
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
    expected = np.exp(data)
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64, 128])
def test_exp_derivative(device, make_kernel, random_seed, in_size):
    np.random.seed(random_seed)
    batch_size = 16
    # Use smaller range for exp to avoid overflow
    data = np.random.rand(batch_size, in_size).astype(np.float32) * 2 - 1

    specialization_module = create_specialization_module(device, in_size)
    kernel = make_kernel("exp_derivative", link_modules=[specialization_module])
    
    input_buffer = create_buffer_32b(device, data, in_size)
    output_buffer = create_batched_buffer_32b(device, batch_size, in_size)
    
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
    
    # Use PyTorch backward to compute Exp derivative
    input_torch = torch.tensor(data, requires_grad=True)
    
    # Apply Exp activation
    exp_output = torch.exp(input_torch)
    
    # For vector output, we need to provide gradient tensor for backward()
    # Use ones to get the derivative of each element w.r.t. its input
    gradient_tensor = torch.ones_like(exp_output)
    exp_output.backward(gradient_tensor)
    
    expected = input_torch.grad.numpy()
    
    assert_close(output, expected)