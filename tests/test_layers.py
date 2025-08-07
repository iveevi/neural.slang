
import numpy as np
import pytest
import torch
import slangpy as spy
from .conftest import assert_close, RANDOM_SEEDS
from .test_utils import create_feed_forward_parameters


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_feed_forward_basic(device, make_kernel, random_seed):
    """Test feed forward layer using nn.Linear."""
    kernel = make_kernel("feed_forward")
    np.random.seed(random_seed)
    
    # Create parameters using shared helper
    weights_data, bias_data, parameters_data = create_feed_forward_parameters(4, 8, seed=random_seed)
    
    # Create input data (10 samples, 4 features each)
    input_data = 2 * np.random.rand(10, 4).astype(np.float32) - 1
    
    # Create buffers
    input_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=4 * 4,
        usage=spy.BufferUsage.shader_resource,
        data=input_data,
    )
    
    output_buffer = device.create_buffer(
        size=10 * 8 * 4,
        struct_size=8 * 4,
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
        thread_count=(10, 1, 1),
        vars={
            "globals": {
                "parameters": parameters_buffer,
                "input": input_buffer,
                "output": output_buffer,
            }
        },
    )
    
    # Get results
    output = output_buffer.to_numpy().view(np.float32).reshape(10, 8)
    
    # Compute expected result using nn.Linear
    input_torch = torch.tensor(input_data)
    linear_layer = torch.nn.Linear(4, 8)
    
    # Set the linear layer weights and bias to match our test data
    with torch.no_grad():
        linear_layer.weight.copy_(torch.tensor(weights_data.T))  # nn.Linear expects transposed weights
        linear_layer.bias.copy_(torch.tensor(bias_data.squeeze()))
    
    # Compute expected output
    linear_output = linear_layer(input_torch)
    expected = torch.relu(linear_output).detach().numpy()
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_feed_forward_derivative(device, make_kernel, random_seed):
    """Test FeedForward derivatives against PyTorch autograd using nn.Linear."""
    kernel = make_kernel("feed_forward_derivative")
    np.random.seed(random_seed)
    
    # Create parameters using shared helper
    weights_data, bias_data, parameters_data = create_feed_forward_parameters(4, 8, seed=random_seed)
    
    # Create input data (10 samples, 4 features each)
    input_data = 2 * np.random.rand(10, 4).astype(np.float32) - 1
    
    # Create buffers
    input_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=4 * 4,
        usage=spy.BufferUsage.shader_resource,
        data=input_data,
    )
    
    # Buffer for input gradients (same size as input)
    dinput_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=4 * 4,
        usage=spy.BufferUsage.shader_resource,
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
    )
    
    # Dispatch kernel
    kernel.dispatch(
        thread_count=(10, 1, 1),
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
    input_derivatives = dinput_buffer.to_numpy().view(np.float32).reshape(10, 4)
    parameter_derivatives = dparameters_buffer.to_numpy().view(np.float32).reshape(parameters_data.shape)
    
    # Compute expected derivatives using PyTorch autograd with nn.Linear
    input_torch = torch.tensor(input_data, requires_grad=True)
    
    # Create linear layer and set weights/bias
    linear_layer = torch.nn.Linear(4, 8)
    with torch.no_grad():
        linear_layer.weight.copy_(torch.tensor(weights_data.T))  # nn.Linear expects transposed weights
        linear_layer.bias.copy_(torch.tensor(bias_data.squeeze()))
    
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