import numpy as np
import pytest
import slangpy as spy
import torch
from .conftest import assert_close


# TODO: also parameterize activation functions with link-time specialization


def create_specialization_module(device, in_size, out_size):
    source = f"""
    export static const int In = {in_size};
    export static const int Out = {out_size};
    """
    return device.load_module_from_source("specialization", source)


def create_linear_layer_data(input_size, output_size, random_seed):
    torch.manual_seed(random_seed)
    
    linear_layer = torch.nn.Linear(input_size, output_size)
    
    # Extract parameters in the format expected by the kernel
    weights_data = linear_layer.weight.detach().numpy().T  # Transpose back for kernel
    bias_data = linear_layer.bias.detach().numpy().reshape(1, -1)
    parameters_data = np.ascontiguousarray(np.concatenate((weights_data, bias_data), axis=0).astype(np.float32))
    
    return linear_layer, parameters_data


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [32, 64, 128])
@pytest.mark.parametrize("out_size", [32, 64, 128])
def test_feed_forward(device, make_kernel, random_seed, in_size, out_size):
    np.random.seed(random_seed)
    
    # Create linear layer and parameters
    batch_size = 16
    linear_layer, parameters_data = create_linear_layer_data(in_size, out_size, random_seed)
    
    specialization_module = create_specialization_module(device, in_size, out_size)
    kernel = make_kernel("feed_forward", link_modules=[specialization_module])
    
    # Create input data (10 samples, in_size features each)
    input_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    
    # Create buffers
    input_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=in_size * 4,
        usage=spy.BufferUsage.shader_resource,
        data=input_data,
    )
    
    output_buffer = device.create_buffer(
        size=batch_size * out_size * 4,
        struct_size=out_size * 4,
        usage=spy.BufferUsage.shader_resource,
    )
    
    parameters_buffer = device.create_buffer(
        size=parameters_data.nbytes,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource,
        data=parameters_data,
    )
    
    # Dispatch kernel
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "parameters": parameters_buffer,
                "input": input_buffer,
                "output": output_buffer,
            }
        },
    )
    
    # Get results
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, out_size)
    
    # Compute expected result using the created linear layer
    input_torch = torch.tensor(input_data)
    
    # Compute expected output
    linear_output = linear_layer(input_torch)
    expected = torch.relu(linear_output).detach().numpy()
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [32, 64, 128])
@pytest.mark.parametrize("out_size", [32, 64, 128])
def test_feed_forward_derivative(device, make_kernel, random_seed, in_size, out_size):
    np.random.seed(random_seed)
    
    # Create linear layer and parameters
    batch_size = 16
    linear_layer, parameters_data = create_linear_layer_data(in_size, out_size, random_seed)
    
    specialization_module = create_specialization_module(device, in_size, out_size)
    kernel = make_kernel("feed_forward_derivative", link_modules=[specialization_module])
    
    # Create input data (10 samples, 64 features each)
    input_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    
    # Create buffers
    input_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=in_size * 4,
        usage=spy.BufferUsage.shader_resource,
        data=input_data,
    )
    
    # Buffer for input gradients (same size as input)
    dinput_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=in_size * 4,
        usage=spy.BufferUsage.shader_resource,
        data=np.zeros_like(input_data),
    )
    
    parameters_buffer = device.create_buffer(
        size=parameters_data.nbytes,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource,
        data=parameters_data,
    )
    
    # Buffer for parameter gradients (same size as parameters)
    dparameters_buffer = device.create_buffer(
        size=parameters_data.nbytes,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource,
        data=np.zeros_like(parameters_data),
    )
    
    # Dispatch kernel
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "dinput": dinput_buffer,
                "parameters": parameters_buffer,
                "dparameters": dparameters_buffer,
            }
        },
    )
    
    # Get derivative results
    input_derivatives = dinput_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    parameter_derivatives = dparameters_buffer.to_numpy().view(np.float32).reshape(parameters_data.shape)
    
    # Compute expected derivatives using PyTorch autograd with the created linear layer
    input_torch = torch.tensor(input_data, requires_grad=True)
    
    # Enable gradient tracking for layer parameters
    linear_layer.weight.requires_grad_(True)
    linear_layer.bias.requires_grad_(True)
    
    # Compute forward pass
    output = torch.relu(linear_layer(input_torch))
    
    # Create gradient tensor of all ones (like Vec8(1.0) in Slang)
    grad_output = torch.ones_like(output)
    
    # Compute gradients
    output.backward(grad_output)
    
    # Ensure gradients were computed
    assert input_torch.grad is not None, "Input gradients were not computed"
    assert linear_layer.weight.grad is not None, "Weight gradients were not computed"
    assert linear_layer.bias.grad is not None, "Bias gradients were not computed"
    
    expected_input_derivatives = input_torch.grad.detach().numpy()
    # Convert back to our parameter layout (weights transposed back, bias reshaped)
    expected_weights_derivatives = linear_layer.weight.grad.detach().numpy().T
    expected_bias_derivatives = linear_layer.bias.grad.detach().numpy().reshape(1, -1)
    
    # Combine expected parameter derivatives to match the combined parameters_data layout
    expected_parameter_derivatives = np.concatenate((expected_weights_derivatives, expected_bias_derivatives), axis=0)
    
    # Compare results
    assert_close(input_derivatives, expected_input_derivatives, rtol=1e-5, atol=1e-6)
    assert_close(parameter_derivatives, expected_parameter_derivatives, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [32, 64])
@pytest.mark.parametrize("out_size", [32, 64])
@pytest.mark.parametrize("offset", [0, 16, 32, 64])
def test_feed_forward_address(device, make_kernel, random_seed, in_size, out_size, offset):
    np.random.seed(random_seed)
    
    # Create linear layer and parameters
    batch_size = 16
    linear_layer, parameters_data = create_linear_layer_data(in_size, out_size, random_seed)
    
    specialization_module = create_specialization_module(device, in_size, out_size)
    kernel = make_kernel("feed_forward_address", link_modules=[specialization_module])
    
    # Create input data (batch_size samples, in_size features each)
    input_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    
    # Create normal input buffer (no padding needed for input)
    input_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=in_size * 4,  # Structure size for InlineVector<float, In>
        usage=spy.BufferUsage.shader_resource,
        data=input_data,
    )
    
    output_buffer = device.create_buffer(
        size=batch_size * out_size * 4,
        struct_size=out_size * 4,
        usage=spy.BufferUsage.shader_resource,
    )
    
    # Create padded parameters buffer with offset
    # Parameters should also be offset in the buffer to test address functionality
    padded_parameters_data = np.zeros(parameters_data.size + offset, dtype=np.float32)
    if offset > 0:
        # Fill padding with random data
        padded_parameters_data[:offset] = np.random.rand(offset).astype(np.float32)
    # Place actual parameters after the offset
    padded_parameters_data[offset:] = parameters_data.flatten()
    
    parameters_buffer = device.create_buffer(
        size=padded_parameters_data.nbytes,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource,
        data=padded_parameters_data,
    )
    
    # Dispatch kernel with address offset
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),  # Process normal batch size
        vars={
            "globals": {
                "parameters": parameters_buffer,
                "input": input_buffer,
                "output": output_buffer,
                "address": offset,
            }
        },
    )
    
    # Get results
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, out_size)
    
    # Compute expected result using the created linear layer on the original (non-padded) input
    input_torch = torch.tensor(input_data)
    
    # Compute expected output
    linear_output = linear_layer(input_torch)
    expected = torch.relu(linear_output).detach().numpy()
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [32, 64])
@pytest.mark.parametrize("out_size", [32, 64])
@pytest.mark.parametrize("offset", [0, 16, 32, 64])
def test_feed_forward_address_derivative(device, make_kernel, random_seed, in_size, out_size, offset):
    np.random.seed(random_seed)
    
    # Create linear layer and parameters
    batch_size = 16
    linear_layer, parameters_data = create_linear_layer_data(in_size, out_size, random_seed)
    
    specialization_module = create_specialization_module(device, in_size, out_size)
    kernel = make_kernel("feed_forward_address_derivative", link_modules=[specialization_module])
    
    # Create input data (batch_size samples, in_size features each)
    input_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    
    # Create normal input buffer (no padding needed for input)
    input_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=in_size * 4,  # Structure size for InlineVector<float, In>
        usage=spy.BufferUsage.shader_resource,
        data=input_data,
    )
    
    # Buffer for input gradients (same size as input)
    dinput_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=in_size * 4,
        usage=spy.BufferUsage.shader_resource,
        data=np.zeros_like(input_data),
    )
    
    # Create padded parameters buffer with offset
    # Parameters should be offset in the buffer to test address functionality
    padded_parameters_data = np.zeros(parameters_data.size + offset, dtype=np.float32)
    if offset > 0:
        # Fill padding with random data
        padded_parameters_data[:offset] = np.random.rand(offset).astype(np.float32)
    # Place actual parameters after the offset
    padded_parameters_data[offset:] = parameters_data.flatten()
    
    parameters_buffer = device.create_buffer(
        size=padded_parameters_data.nbytes,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource,
        data=padded_parameters_data,
    )
    
    # Buffer for parameter gradients (same size as padded parameters)
    dparameters_buffer = device.create_buffer(
        size=padded_parameters_data.nbytes,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource,
        data=np.zeros_like(padded_parameters_data),
    )
    
    # Dispatch kernel with address offset
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),  # Process normal batch size
        vars={
            "globals": {
                "input": input_buffer,
                "dinput": dinput_buffer,
                "parameters": parameters_buffer,
                "dparameters": dparameters_buffer,
                "address": offset,
            }
        },
    )
    
    # Get derivative results
    input_derivatives = dinput_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    parameter_derivatives_padded = dparameters_buffer.to_numpy().view(np.float32).reshape(padded_parameters_data.shape)
    # Extract only the actual parameter derivatives (skip padding)
    parameter_derivatives = parameter_derivatives_padded[offset:].reshape(parameters_data.shape)
    
    # Compute expected derivatives using PyTorch autograd with the created linear layer
    input_torch = torch.tensor(input_data, requires_grad=True)
    
    # Enable gradient tracking for layer parameters
    linear_layer.weight.requires_grad_(True)
    linear_layer.bias.requires_grad_(True)
    
    # Compute forward pass
    output = torch.relu(linear_layer(input_torch))
    
    # Create gradient tensor of all ones (like Vec8(1.0) in Slang)
    grad_output = torch.ones_like(output)
    
    # Compute gradients
    output.backward(grad_output)
    
    # Ensure gradients were computed
    assert input_torch.grad is not None, "Input gradients were not computed"
    assert linear_layer.weight.grad is not None, "Weight gradients were not computed"
    assert linear_layer.bias.grad is not None, "Bias gradients were not computed"
    
    expected_input_derivatives = input_torch.grad.detach().numpy()
    # Convert back to our parameter layout (weights transposed back, bias reshaped)
    expected_weights_derivatives = linear_layer.weight.grad.detach().numpy().T
    expected_bias_derivatives = linear_layer.bias.grad.detach().numpy().reshape(1, -1)
    
    # Combine expected parameter derivatives to match the combined parameters_data layout
    expected_parameter_derivatives = np.concatenate((expected_weights_derivatives, expected_bias_derivatives), axis=0)
    
    # Compare results
    assert_close(input_derivatives, expected_input_derivatives, rtol=1e-5, atol=1e-6)
    assert_close(parameter_derivatives, expected_parameter_derivatives, rtol=1e-5, atol=1e-6)