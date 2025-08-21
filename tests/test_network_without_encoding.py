import numpy as np
import pytest
import torch
from .conftest import assert_close
from util import *


def create_specialization_module(device, in_size, hidden_size, out_size):
    source = f"""
    export static const int In = {in_size};
    export static const int Hidden = {hidden_size};
    export static const int Out = {out_size};
    """
    return device.load_module_from_source("specialization", source)


def create_network_layers(random_seed, in_size, hidden_size, out_size):
    """Create PyTorch network layers and extract their parameters."""
    torch.manual_seed(random_seed)
    
    network = torch.nn.Sequential(
        torch.nn.Linear(in_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size), 
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, out_size),
        torch.nn.ReLU()
    )
    
    # Extract parameters using utility functions
    layer1_params = linear_to_numpy(network[0])
    layer2_params = linear_to_numpy(network[2])
    layer3_params = linear_to_numpy(network[4])
    layer4_params = linear_to_numpy(network[6])
    
    return network, layer1_params, layer2_params, layer3_params, layer4_params


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64])
@pytest.mark.parametrize("hidden_size", [16, 32, 64])
@pytest.mark.parametrize("out_size", [16, 32, 64])
def test_network_without_encoding(device, make_kernel, random_seed, in_size, hidden_size, out_size):
    np.random.seed(random_seed)

    result = create_network_layers(random_seed, in_size, hidden_size, out_size)
    network, layer1_params, layer2_params, layer3_params, layer4_params = result
    
    batch_size = 64
    test_inputs = (np.random.rand(batch_size, in_size).astype(np.float32) - 0.5) * 2.0
    
    specialization_module = create_specialization_module(device, in_size, hidden_size, out_size)
    kernel = make_kernel("network_without_encoding", link_modules=[specialization_module])
    
    input_buffer = create_buffer_32b(device, test_inputs, in_size)
    output_buffer = create_batched_buffer_32b(device, batch_size, out_size)
    
    layer1_buffer = create_buffer_32b(device, layer1_params)
    layer2_buffer = create_buffer_32b(device, layer2_params)
    layer3_buffer = create_buffer_32b(device, layer3_params)
    layer4_buffer = create_buffer_32b(device, layer4_params)
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
                "layer1": layer1_buffer,
                "layer2": layer2_buffer,
                "layer3": layer3_buffer,
                "layer4": layer4_buffer,
            
                    "count": batch_size,}
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, out_size)
    input_torch = torch.tensor(test_inputs)
    expected = network(input_torch).detach().numpy()
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64])
@pytest.mark.parametrize("hidden_size", [16, 32, 64])
@pytest.mark.parametrize("out_size", [16, 32, 64])
def test_network_without_encoding_derivative(device, make_kernel, random_seed, in_size, hidden_size, out_size):
    np.random.seed(random_seed)
    
    # Create network and parameters
    result = create_network_layers(random_seed, in_size, hidden_size, out_size)
    network, layer1_params, layer2_params, layer3_params, layer4_params = result
    
    # Generate completely random test inputs for gradient testing
    batch_size = 64
    test_inputs = (np.random.rand(batch_size, in_size).astype(np.float32) - 0.5) * 1.0  # Random values in [-0.5, 0.5]

    specialization_module = create_specialization_module(device, in_size, hidden_size, out_size)
    kernel = make_kernel("network_without_encoding_derivative", link_modules=[specialization_module])
    
    # Create input buffers
    input_buffer = create_buffer_32b(device, test_inputs, in_size)
    dinput_buffer = create_batched_buffer_32b(device, batch_size, in_size)
    
    # Create parameter buffers
    layer1_buffer = create_buffer_32b(device, layer1_params)
    layer2_buffer = create_buffer_32b(device, layer2_params)
    layer3_buffer = create_buffer_32b(device, layer3_params)
    layer4_buffer = create_buffer_32b(device, layer4_params)
    
    # Create gradient buffers for parameters
    dlayer1_buffer = create_batched_buffer_32b(device, layer1_params.shape[0], layer1_params.shape[1])
    dlayer2_buffer = create_batched_buffer_32b(device, layer2_params.shape[0], layer2_params.shape[1])
    dlayer3_buffer = create_batched_buffer_32b(device, layer3_params.shape[0], layer3_params.shape[1])
    dlayer4_buffer = create_batched_buffer_32b(device, layer4_params.shape[0], layer4_params.shape[1])
    
    # Run kernel
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "dinput": dinput_buffer,
                "layer1": layer1_buffer,
                "dlayer1": dlayer1_buffer,
                "layer2": layer2_buffer,
                "dlayer2": dlayer2_buffer,
                "layer3": layer3_buffer,
                "dlayer3": dlayer3_buffer,
                "layer4": layer4_buffer,
                "dlayer4": dlayer4_buffer,
            
                    "count": batch_size,}
        },
    )
    
    # Get derivative results
    input_derivatives = dinput_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    layer1_derivatives = dlayer1_buffer.to_numpy().view(np.float32).reshape(layer1_params.shape)
    layer2_derivatives = dlayer2_buffer.to_numpy().view(np.float32).reshape(layer2_params.shape)
    layer3_derivatives = dlayer3_buffer.to_numpy().view(np.float32).reshape(layer3_params.shape)
    layer4_derivatives = dlayer4_buffer.to_numpy().view(np.float32).reshape(layer4_params.shape)
    
    # Compute expected derivatives using PyTorch autograd with nn.Linear layers
    input_torch = torch.tensor(test_inputs, requires_grad=True)
    
    # Enable gradient tracking for all parameters
    for param in network.parameters():
        param.requires_grad_(True)
    
    # Forward pass and compute gradients
    output = network(input_torch)
    
    # Sum over all outputs and batch to get scalar loss for backprop
    output.backward(torch.ones_like(output))
    
    # Get expected gradients
    expected_input_grad = input_torch.grad.detach().numpy()
    
    # Extract parameter gradients using utility functions
    expected_layer1_grad = linear_gradients_to_numpy(network[0])
    expected_layer2_grad = linear_gradients_to_numpy(network[2])
    expected_layer3_grad = linear_gradients_to_numpy(network[4])
    expected_layer4_grad = linear_gradients_to_numpy(network[6])
    
    # Combine expected parameter gradients to match layout

    
    # Test input gradients
    assert_close(input_derivatives, expected_input_grad)
    
    # Test parameter gradients
    assert_close(layer1_derivatives, expected_layer1_grad)
    assert_close(layer2_derivatives, expected_layer2_grad)
    assert_close(layer3_derivatives, expected_layer3_grad)
    assert_close(layer4_derivatives, expected_layer4_grad)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [32])
@pytest.mark.parametrize("hidden_size", [32, 64])
@pytest.mark.parametrize("out_size", [32])
@pytest.mark.parametrize("offset", [0, 16, 32, 64])
def test_network_without_encoding_address(device, make_kernel, random_seed, in_size, hidden_size, out_size, offset):
    """Test network with all layer weights stored in the same buffer at different offsets."""
    np.random.seed(random_seed)

    result = create_network_layers(random_seed, in_size, hidden_size, out_size)
    network, layer1_params, layer2_params, layer3_params, layer4_params = result
    
    batch_size = 16
    test_inputs = (np.random.rand(batch_size, in_size).astype(np.float32) - 0.5) * 2.0
    
    specialization_module = create_specialization_module(device, in_size, hidden_size, out_size)
    kernel = make_kernel("network_without_encoding_address", link_modules=[specialization_module])
    
    input_buffer = create_buffer_32b(device, test_inputs, in_size)
    output_buffer = create_batched_buffer_32b(device, batch_size, out_size)
    
    # Calculate total parameter size and addresses
    layer1_size = layer1_params.size
    layer2_size = layer2_params.size  
    layer3_size = layer3_params.size
    layer4_size = layer4_params.size
    
    # Calculate addresses within the combined buffer (after initial offset)
    layer1_address = offset
    layer2_address = layer1_address + layer1_size
    layer3_address = layer2_address + layer2_size
    layer4_address = layer3_address + layer3_size
    
    # Create combined parameter buffer with initial offset
    total_param_size = layer1_size + layer2_size + layer3_size + layer4_size
    combined_params = np.zeros(total_param_size + offset, dtype=np.float32)
    
    if offset > 0:
        # Fill padding with random data
        combined_params[:offset] = np.random.rand(offset).astype(np.float32)
    
    # Place layer parameters at their respective addresses
    combined_params[layer1_address:layer1_address + layer1_size] = layer1_params.flatten()
    combined_params[layer2_address:layer2_address + layer2_size] = layer2_params.flatten()
    combined_params[layer3_address:layer3_address + layer3_size] = layer3_params.flatten()
    combined_params[layer4_address:layer4_address + layer4_size] = layer4_params.flatten()
    
    parameters_buffer = create_buffer_32b(device, combined_params)
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
                "parameters": parameters_buffer,
                "layer1Address": layer1_address,
                "layer2Address": layer2_address,
                "layer3Address": layer3_address,
                "layer4Address": layer4_address,
            
                    "count": batch_size,}
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, out_size)
    input_torch = torch.tensor(test_inputs)
    expected = network(input_torch).detach().numpy()
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [32])
@pytest.mark.parametrize("hidden_size", [32, 64])
@pytest.mark.parametrize("out_size", [32])
@pytest.mark.parametrize("offset", [0, 16, 32, 64])
def test_network_without_encoding_address_derivative(device, make_kernel, random_seed, in_size, hidden_size, out_size, offset):
    """Test network derivative with all layer weights stored in the same buffer at different offsets."""
    np.random.seed(random_seed)
    
    # Create network and parameters
    result = create_network_layers(random_seed, in_size, hidden_size, out_size)
    network, layer1_params, layer2_params, layer3_params, layer4_params = result
    
    # Generate test inputs for gradient testing
    batch_size = 16
    test_inputs = (np.random.rand(batch_size, in_size).astype(np.float32) - 0.5) * 1.0

    specialization_module = create_specialization_module(device, in_size, hidden_size, out_size)
    kernel = make_kernel("network_without_encoding_address_derivative", link_modules=[specialization_module])
    
    # Create input buffers
    input_buffer = create_buffer_32b(device, test_inputs, in_size)
    dinput_buffer = create_batched_buffer_32b(device, batch_size, in_size)
    
    # Calculate total parameter size and addresses
    layer1_size = layer1_params.size
    layer2_size = layer2_params.size  
    layer3_size = layer3_params.size
    layer4_size = layer4_params.size
    
    # Calculate addresses within the combined buffer (after initial offset)
    layer1_address = offset
    layer2_address = layer1_address + layer1_size
    layer3_address = layer2_address + layer2_size
    layer4_address = layer3_address + layer3_size
    
    # Create combined parameter buffer with initial offset
    total_param_size = layer1_size + layer2_size + layer3_size + layer4_size
    combined_params = np.zeros(total_param_size + offset, dtype=np.float32)
    
    if offset > 0:
        # Fill padding with random data
        combined_params[:offset] = np.random.rand(offset).astype(np.float32)
    
    # Place layer parameters at their respective addresses
    combined_params[layer1_address:layer1_address + layer1_size] = layer1_params.flatten()
    combined_params[layer2_address:layer2_address + layer2_size] = layer2_params.flatten()
    combined_params[layer3_address:layer3_address + layer3_size] = layer3_params.flatten()
    combined_params[layer4_address:layer4_address + layer4_size] = layer4_params.flatten()
    
    parameters_buffer = create_buffer_32b(device, combined_params)
    
    # Create gradient buffer for combined parameters (same size as combined parameters)
    dparameters_buffer = create_buffer_32b(device, np.zeros_like(combined_params))
    
    # Run kernel
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "dinput": dinput_buffer,
                "parameters": parameters_buffer,
                "dparameters": dparameters_buffer,
                "layer1Address": layer1_address,
                "layer2Address": layer2_address,
                "layer3Address": layer3_address,
                "layer4Address": layer4_address,
            
                    "count": batch_size,}
        },
    )
    
    # Get derivative results
    input_derivatives = dinput_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    combined_param_derivatives = dparameters_buffer.to_numpy().view(np.float32).reshape(combined_params.shape)
    
    # Extract individual layer derivatives from the combined buffer
    layer1_derivatives = combined_param_derivatives[layer1_address:layer1_address + layer1_size].reshape(layer1_params.shape)
    layer2_derivatives = combined_param_derivatives[layer2_address:layer2_address + layer2_size].reshape(layer2_params.shape)
    layer3_derivatives = combined_param_derivatives[layer3_address:layer3_address + layer3_size].reshape(layer3_params.shape)
    layer4_derivatives = combined_param_derivatives[layer4_address:layer4_address + layer4_size].reshape(layer4_params.shape)
    
    # Compute expected derivatives using PyTorch autograd
    input_torch = torch.tensor(test_inputs, requires_grad=True)
    
    # Enable gradient tracking for all parameters
    for param in network.parameters():
        param.requires_grad_(True)
    
    # Forward pass and compute gradients
    output = network(input_torch)
    
    # Sum over all outputs and batch to get scalar loss for backprop
    output.backward(torch.ones_like(output))
    
    # Get expected gradients
    expected_input_grad = input_torch.grad.detach().numpy()
    
    # Extract parameter gradients using utility functions
    expected_layer1_grad = linear_gradients_to_numpy(network[0])
    expected_layer2_grad = linear_gradients_to_numpy(network[2])
    expected_layer3_grad = linear_gradients_to_numpy(network[4])
    expected_layer4_grad = linear_gradients_to_numpy(network[6])
    
    # Combine expected parameter gradients to match layout

    
    # Test input gradients
    assert_close(input_derivatives, expected_input_grad)
    
    # Test parameter gradients
    assert_close(layer1_derivatives, expected_layer1_grad)
    assert_close(layer2_derivatives, expected_layer2_grad)
    assert_close(layer3_derivatives, expected_layer3_grad)
    assert_close(layer4_derivatives, expected_layer4_grad)