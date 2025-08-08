import numpy as np
import pytest
import torch
import slangpy as spy
from .conftest import assert_close, RANDOM_SEEDS
from .test_utils import create_buffer_for_data, create_output_buffer


def create_specialization_module(device, in_size, levels):
    source = f"""
    export static const int In = {in_size};
    export static const int Levels = {levels};
    """
    return device.load_module_from_source("specialization", source)


def frequency_encoder(input_data: torch.Tensor, levels: int) -> torch.Tensor:
    encoded_parts = []
    for level in range(levels):
        k = 2.0 ** level
        frequency = k * torch.pi
        sin_vals = torch.sin(frequency * input_data)
        cos_vals = torch.cos(frequency * input_data)
        encoded_parts.append(torch.cat([sin_vals, cos_vals], dim=1))
    return torch.cat(encoded_parts, dim=1)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", range(1, 5))
@pytest.mark.parametrize("levels", range(1, 9))
def test_frequency_encoder(device, make_kernel, random_seed, in_size, levels):
    batch_size = 16
    np.random.seed(random_seed)

    specialization_module = create_specialization_module(device, in_size, levels)
    kernel = make_kernel("frequency_encoder", link_modules=[specialization_module])
    
    input_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    
    input_buffer = create_buffer_for_data(device, input_data, in_size * 4)
    output_buffer = create_output_buffer(device, batch_size, 2 * levels * in_size)
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
            }
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, 2 * levels * in_size)
    expected = frequency_encoder(torch.tensor(input_data), levels).numpy()
    assert_close(output, expected, rtol=1e-4, atol=1e-4)
    

@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", range(1, 5))
@pytest.mark.parametrize("levels", range(1, 9))
def test_frequency_encoder_derivative(device, make_kernel, random_seed, in_size, levels):
    batch_size = 16
    np.random.seed(random_seed)

    specialization_module = create_specialization_module(device, in_size, levels)
    kernel = make_kernel("frequency_encoder_derivative", link_modules=[specialization_module])
    
    input_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    
    input_buffer = create_buffer_for_data(device, input_data, in_size * 4)
    dinput_buffer = create_output_buffer(device, batch_size, in_size)
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "dinput": dinput_buffer,
            }
        },
    )
    
    derivatives = dinput_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    
    input_torch = torch.tensor(input_data, requires_grad=True)
    encoded_output = frequency_encoder(input_torch, levels)
    grad_output = torch.ones_like(encoded_output)
    encoded_output.backward(grad_output)
    
    expected_derivatives = input_torch.grad.detach().numpy()
    assert_close(derivatives, expected_derivatives, rtol=1e-3, atol=1e-3)