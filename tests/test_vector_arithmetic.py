import numpy as np
import pytest
import torch
from .conftest import assert_close
from common.util import *


def create_specialization_module(device, in_size):
    source = f"""
    export static const int In = {in_size};
    """
    return device.load_module_from_source("specialization", source)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64, 128])
def test_vector_arithmetic_basic(device, make_kernel, random_seed, in_size):
    batch_size = 16
    np.random.seed(random_seed)

    specialization_module = create_specialization_module(device, in_size)
    kernel = make_kernel("vector_arithmetic", link_modules=[specialization_module])
    
    input_a = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    input_b = np.random.rand(batch_size, in_size).astype(np.float32) + 0.1
    input_b[::2] = -input_b[::2]
    
    input_a_buffer = create_buffer_for_data(device, input_a, in_size * 4)
    input_b_buffer = create_buffer_for_data(device, input_b, in_size * 4)
    
    output_add_buffer = create_output_buffer(device, batch_size, in_size)
    output_sub_buffer = create_output_buffer(device, batch_size, in_size)
    output_mul_buffer = create_output_buffer(device, batch_size, in_size)
    output_div_buffer = create_output_buffer(device, batch_size, in_size)
    
    # Dispatch kernel
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input_a": input_a_buffer,
                "input_b": input_b_buffer,
                "output_add": output_add_buffer,
                "output_sub": output_sub_buffer,
                "output_mul": output_mul_buffer,
                "output_div": output_div_buffer,
            }
        },
    )
    
    # Get results
    output_add = output_add_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    output_sub = output_sub_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    output_mul = output_mul_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    output_div = output_div_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    
    # Compute expected results using numpy
    expected_add = input_a + input_b
    expected_sub = input_a - input_b
    expected_mul = input_a * input_b
    expected_div = input_a / input_b
    
    # Assert results
    assert_close(output_add, expected_add)
    assert_close(output_sub, expected_sub)
    assert_close(output_mul, expected_mul)
    assert_close(output_div, expected_div)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64, 128])
def test_vector_arithmetic_derivatives(device, make_kernel, random_seed, in_size):
    batch_size = 16
    np.random.seed(random_seed)

    specialization_module = create_specialization_module(device, in_size)
    kernel = make_kernel("vector_arithmetic_derivatives", link_modules=[specialization_module])
    
    batch_size = 16
    input_a = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    input_b = np.random.rand(batch_size, in_size).astype(np.float32) + 0.1
    input_b[::2] = -input_b[::2]
    
    input_a_buffer = create_buffer_for_data(device, input_a, in_size * 4)
    input_b_buffer = create_buffer_for_data(device, input_b, in_size * 4)
    
    output_buffers = {}
    for op in ['add', 'sub', 'mul', 'div']:
        for var in ['a', 'b']:
            output_buffers[f'output_{op}_{var}'] = create_output_buffer(device, batch_size, in_size)
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input_a": input_a_buffer,
                "input_b": input_b_buffer,
                **output_buffers
            }
        },
    )
    
    derivatives = {}
    for op in ['add', 'sub', 'mul', 'div']:
        for var in ['a', 'b']:
            key = f'{op}_{var}'
            derivatives[key] = output_buffers[f'output_{key}'].to_numpy().view(np.float32).reshape(batch_size, in_size)
    
    a_torch = torch.tensor(input_a, requires_grad=True)
    b_torch = torch.tensor(input_b, requires_grad=True)
    
    add_result = a_torch + b_torch
    gradient_tensor = torch.ones_like(add_result)
    add_result.backward(gradient_tensor, retain_graph=True)
    expected_add_a = a_torch.grad.numpy()
    expected_add_b = b_torch.grad.numpy()
    assert_close(derivatives['add_a'], expected_add_a)
    assert_close(derivatives['add_b'], expected_add_b)
    
    a_torch.grad.zero_()
    b_torch.grad.zero_()
    
    sub_result = a_torch - b_torch
    sub_result.backward(gradient_tensor, retain_graph=True)
    expected_sub_a = a_torch.grad.numpy()
    expected_sub_b = b_torch.grad.numpy()
    assert_close(derivatives['sub_a'], expected_sub_a)
    assert_close(derivatives['sub_b'], expected_sub_b)
    
    a_torch.grad.zero_()
    b_torch.grad.zero_()
    
    mul_result = a_torch * b_torch
    mul_result.backward(gradient_tensor, retain_graph=True)
    expected_mul_a = a_torch.grad.numpy()
    expected_mul_b = b_torch.grad.numpy()
    assert_close(derivatives['mul_a'], expected_mul_a)
    assert_close(derivatives['mul_b'], expected_mul_b)
    
    a_torch.grad.zero_()
    b_torch.grad.zero_()
    
    div_result = a_torch / b_torch
    div_result.backward(gradient_tensor)
    expected_div_a = a_torch.grad.numpy()
    expected_div_b = b_torch.grad.numpy()
    assert_close(derivatives['div_a'], expected_div_a)
    assert_close(derivatives['div_b'], expected_div_b)