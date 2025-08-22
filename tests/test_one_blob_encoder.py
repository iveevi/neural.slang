import numpy as np
import pytest
import torch
from .conftest import assert_close
from util import *


def create_specialization_module(device, n, k):
    source = f"""
    export static const int N = {n};
    export static const int K = {k};
    """
    return device.load_module_from_source("specialization", source)


def one_blob_encoder(input_data: torch.Tensor, k: int) -> torch.Tensor:
    """Reference implementation of the one-blob encoder using Gaussian kernels."""
    batch_size, n = input_data.shape
    output = torch.zeros(batch_size, n * k)
    
    for i in range(n):
        s = input_data[:, i]
        for j in range(k):
            x = j / (k - 1) if k > 1 else 0.0
            diff = s - x
            kernel_val = torch.exp(-diff * diff / (k * k))
            output[:, i * k + j] = kernel_val
    
    return output


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("k", range(2, 9))
def test_one_blob_encoder(device, make_kernel, random_seed, n, k):
    batch_size = 16
    np.random.seed(random_seed)

    specialization_module = create_specialization_module(device, n, k)
    kernel = make_kernel("one_blob_encoder", link_modules=[specialization_module])
    
    # Input data in range [0, 1] to match the encoder's expected range
    input_data = np.random.rand(batch_size, n).astype(np.float32)
    
    input_buffer = create_buffer_32b(device, input_data, n)
    output_buffer = create_batched_buffer_32b(device, batch_size, n * k)
    
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
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, n * k)
    expected = one_blob_encoder(torch.tensor(input_data), k).numpy()
    assert_close(output, expected, rtol=1e-4, atol=1e-4)
    

@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("k", range(2, 9))
def test_one_blob_encoder_derivative(device, make_kernel, random_seed, n, k):
    batch_size = 16
    np.random.seed(random_seed)

    specialization_module = create_specialization_module(device, n, k)
    kernel = make_kernel("one_blob_encoder_derivative", link_modules=[specialization_module])
    
    # Input data in range [0, 1] to match the encoder's expected range
    input_data = np.random.rand(batch_size, n).astype(np.float32)
    
    input_buffer = create_buffer_32b(device, input_data, n)
    dinput_buffer = create_batched_buffer_32b(device, batch_size, n)
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "dinput": dinput_buffer,
                "count": batch_size,
            }
        },
    )
    
    derivatives = dinput_buffer.to_numpy().view(np.float32).reshape(batch_size, n)
    
    # Compute expected derivatives using PyTorch autograd
    input_torch = torch.tensor(input_data, requires_grad=True)
    encoded_output = one_blob_encoder(input_torch, k)
    grad_output = torch.ones_like(encoded_output)
    encoded_output.backward(grad_output)
    
    expected_derivatives = input_torch.grad.detach().numpy()
    assert_close(derivatives, expected_derivatives, rtol=1e-3, atol=1e-3)
