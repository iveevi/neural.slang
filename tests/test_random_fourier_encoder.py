import numpy as np
import pytest
import torch
from .conftest import assert_close
from util import *


def create_specialization_module(device, dim, features):
    source = f"""
    export static const int DIM = {dim};
    export static const int FEATURES = {features};
    """
    return device.load_module_from_source("specialization", source)


def random_fourier_encoder(input_data: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """Reference implementation of the random fourier encoder.
    
    The encoder applies a linear transformation followed by sine activation:
    output = sin(input @ parameters)
    
    Args:
        input_data: Input tensor of shape (batch_size, dim)
        parameters: Parameter tensor of shape (dim, features)
    
    Returns:
        Output tensor of shape (batch_size, features)
    """
    # Matrix multiplication: (batch_size, dim) @ (dim, features) -> (batch_size, features)
    linear_output = torch.matmul(input_data, parameters)
    # Apply sine activation
    return torch.sin(linear_output)


@pytest.mark.parametrize("random_seed", [0, 42])
@pytest.mark.parametrize("dim", range(1, 5))
@pytest.mark.parametrize("features", [2, 4, 8, 16])
def test_random_fourier_encoder(device, make_kernel, random_seed, dim, features):
    batch_size = 16
    np.random.seed(random_seed)

    specialization_module = create_specialization_module(device, dim, features)
    kernel = make_kernel("random_fourier_encoder", link_modules=[specialization_module])
    
    # Input data - random values
    input_data = np.random.randn(batch_size, dim).astype(np.float32)
    
    # Parameters - random initialization (typically scaled by some sigma)
    sigma = 1.0
    parameters = np.random.randn(dim, features).astype(np.float32) * sigma
    
    input_buffer = create_buffer_32b(device, input_data, dim)
    output_buffer = create_batched_buffer_32b(device, batch_size, features)
    parameter_buffer = create_buffer_32b(device, parameters.flatten())
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
                "parameters": parameter_buffer,
                "count": batch_size,
            }
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, features)
    expected = random_fourier_encoder(torch.tensor(input_data), torch.tensor(parameters)).numpy()
    assert_close(output, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("random_seed", [0, 42])
@pytest.mark.parametrize("dim", range(1, 5))
@pytest.mark.parametrize("features", [2, 4, 8, 16])
def test_random_fourier_encoder_derivative(device, make_kernel, random_seed, dim, features):
    batch_size = 16
    np.random.seed(random_seed)

    specialization_module = create_specialization_module(device, dim, features)
    kernel = make_kernel("random_fourier_encoder_derivative", link_modules=[specialization_module])
    
    # Input data - random values
    input_data = np.random.randn(batch_size, dim).astype(np.float32)
    
    # Parameters - random initialization
    sigma = 1.0
    parameters = np.random.randn(dim, features).astype(np.float32) * sigma
    
    input_buffer = create_buffer_32b(device, input_data, dim)
    dinput_buffer = create_batched_buffer_32b(device, batch_size, dim)
    parameter_buffer = create_buffer_32b(device, parameters.flatten())
    dparameter_buffer = create_buffer_32b(device, np.zeros(dim * features, dtype=np.float32))
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "dinput": dinput_buffer,
                "parameters": parameter_buffer,
                "dparameters": dparameter_buffer,
                "count": batch_size,
            }
        },
    )
    
    dinput = dinput_buffer.to_numpy().view(np.float32).reshape(batch_size, dim)
    dparameters = dparameter_buffer.to_numpy().view(np.float32).reshape(dim, features)
    
    # Compute expected derivatives using PyTorch autograd
    input_torch = torch.tensor(input_data, requires_grad=True)
    parameters_torch = torch.tensor(parameters, requires_grad=True)
    
    output = random_fourier_encoder(input_torch, parameters_torch)
    grad_output = torch.ones_like(output)
    output.backward(grad_output)
    
    expected_dinput = input_torch.grad.detach().numpy()
    expected_dparameters = parameters_torch.grad.detach().numpy()
    
    assert_close(dinput, expected_dinput, rtol=1e-3, atol=1e-5)
    assert_close(dparameters, expected_dparameters, rtol=1e-3, atol=1e-5)



