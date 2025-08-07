import numpy as np
import pytest
import torch
import slangpy as spy
from .conftest import assert_close, RANDOM_SEEDS


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_vector_arithmetic_basic(device, make_kernel, random_seed):
    """Test basic vector arithmetic operations (add, sub, mul, div)."""
    kernel = make_kernel("vector_arithmetic")
    np.random.seed(random_seed)
    
    # Create test data: 10 samples of 4D vectors
    # Avoid division by zero by ensuring no values too close to zero for input_b
    input_a = 2 * np.random.rand(10, 4).astype(np.float32) - 1  # Range [-1, 1]
    input_b = np.random.rand(10, 4).astype(np.float32) + 0.1   # Range [0.1, 1.1] to avoid div by zero
    
    # Make some values negative for input_b to test different scenarios
    input_b[::2] = -input_b[::2]  # Make every other sample negative
    
    # Create buffers
    input_a_buffer = device.create_buffer(
        size=input_a.nbytes,
        struct_size=4 * 4,  # 4 floats * 4 bytes each
        usage=spy.BufferUsage.shader_resource,
        data=input_a,
    )
    
    input_b_buffer = device.create_buffer(
        size=input_b.nbytes,
        struct_size=4 * 4,
        usage=spy.BufferUsage.shader_resource,
        data=input_b,
    )
    
    # Output buffers for each operation
    output_add_buffer = device.create_buffer(
        size=input_a.nbytes,
        struct_size=4 * 4,
        usage=spy.BufferUsage.shader_resource,
    )
    
    output_sub_buffer = device.create_buffer(
        size=input_a.nbytes,
        struct_size=4 * 4,
        usage=spy.BufferUsage.shader_resource,
    )
    
    output_mul_buffer = device.create_buffer(
        size=input_a.nbytes,
        struct_size=4 * 4,
        usage=spy.BufferUsage.shader_resource,
    )
    
    output_div_buffer = device.create_buffer(
        size=input_a.nbytes,
        struct_size=4 * 4,
        usage=spy.BufferUsage.shader_resource,
    )
    
    # Dispatch kernel
    kernel.dispatch(
        thread_count=(10, 1, 1),
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
    output_add = output_add_buffer.to_numpy().view(np.float32).reshape(10, 4)
    output_sub = output_sub_buffer.to_numpy().view(np.float32).reshape(10, 4)
    output_mul = output_mul_buffer.to_numpy().view(np.float32).reshape(10, 4)
    output_div = output_div_buffer.to_numpy().view(np.float32).reshape(10, 4)
    
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


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_vector_arithmetic_derivatives(device, make_kernel, random_seed):
    """Test derivatives of vector arithmetic operations against PyTorch autograd."""
    kernel = make_kernel("vector_arithmetic_derivatives")
    np.random.seed(random_seed)
    
    # Create test data
    input_a = 2 * np.random.rand(10, 4).astype(np.float32) - 1  # Range [-1, 1]
    input_b = np.random.rand(10, 4).astype(np.float32) + 0.1   # Range [0.1, 1.1] to avoid div by zero
    
    # Make some values negative for input_b
    input_b[::2] = -input_b[::2]
    
    # Create buffers
    input_a_buffer = device.create_buffer(
        size=input_a.nbytes,
        struct_size=4 * 4,
        usage=spy.BufferUsage.shader_resource,
        data=input_a,
    )
    
    input_b_buffer = device.create_buffer(
        size=input_b.nbytes,
        struct_size=4 * 4,
        usage=spy.BufferUsage.shader_resource,
        data=input_b,
    )
    
    # Output buffers for derivatives
    output_buffers = {}
    for op in ['add', 'sub', 'mul', 'div']:
        for var in ['a', 'b']:
            output_buffers[f'output_{op}_{var}'] = device.create_buffer(
                size=input_a.nbytes,
                struct_size=4 * 4,
                usage=spy.BufferUsage.shader_resource,
            )
    
    # Dispatch kernel
    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "globals": {
                "input_a": input_a_buffer,
                "input_b": input_b_buffer,
                **output_buffers
            }
        },
    )
    
    # Get results
    derivatives = {}
    for op in ['add', 'sub', 'mul', 'div']:
        for var in ['a', 'b']:
            key = f'{op}_{var}'
            derivatives[key] = output_buffers[f'output_{key}'].to_numpy().view(np.float32).reshape(10, 4)
    
    # Compute expected derivatives using PyTorch with backward()
    a_torch = torch.tensor(input_a, requires_grad=True)
    b_torch = torch.tensor(input_b, requires_grad=True)
    
    # Test addition derivatives
    add_result = a_torch + b_torch
    gradient_tensor = torch.ones_like(add_result)
    add_result.backward(gradient_tensor, retain_graph=True)
    expected_add_a = a_torch.grad.numpy()
    expected_add_b = b_torch.grad.numpy()
    assert_close(derivatives['add_a'], expected_add_a)
    assert_close(derivatives['add_b'], expected_add_b)
    
    # Reset gradients for next operation
    a_torch.grad.zero_()
    b_torch.grad.zero_()
    
    # Test subtraction derivatives
    sub_result = a_torch - b_torch
    sub_result.backward(gradient_tensor, retain_graph=True)
    expected_sub_a = a_torch.grad.numpy()
    expected_sub_b = b_torch.grad.numpy()
    assert_close(derivatives['sub_a'], expected_sub_a)
    assert_close(derivatives['sub_b'], expected_sub_b)
    
    # Reset gradients for next operation
    a_torch.grad.zero_()
    b_torch.grad.zero_()
    
    # Test multiplication derivatives
    mul_result = a_torch * b_torch
    mul_result.backward(gradient_tensor, retain_graph=True)
    expected_mul_a = a_torch.grad.numpy()
    expected_mul_b = b_torch.grad.numpy()
    assert_close(derivatives['mul_a'], expected_mul_a)
    assert_close(derivatives['mul_b'], expected_mul_b)
    
    # Reset gradients for next operation
    a_torch.grad.zero_()
    b_torch.grad.zero_()
    
    # Test division derivatives
    div_result = a_torch / b_torch
    div_result.backward(gradient_tensor)
    expected_div_a = a_torch.grad.numpy()
    expected_div_b = b_torch.grad.numpy()
    assert_close(derivatives['div_a'], expected_div_a)
    assert_close(derivatives['div_b'], expected_div_b)