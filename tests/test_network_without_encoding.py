import numpy as np
import pytest
import slangpy as spy
import torch
from .conftest import assert_close, RANDOM_SEEDS
from .test_utils import (
    create_buffer_for_data,
    create_output_buffer
)


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
    
    # Extract parameters in the format expected by the kernel
    layer1_weights = network[0].weight.detach().numpy().T  # Transpose back for kernel
    layer1_bias = network[0].bias.detach().numpy().reshape(1, -1)
    layer1_params = np.ascontiguousarray(np.concatenate((layer1_weights, layer1_bias), axis=0).astype(np.float32))
    
    layer2_weights = network[2].weight.detach().numpy().T
    layer2_bias = network[2].bias.detach().numpy().reshape(1, -1)
    layer2_params = np.ascontiguousarray(np.concatenate((layer2_weights, layer2_bias), axis=0).astype(np.float32))
    
    layer3_weights = network[4].weight.detach().numpy().T
    layer3_bias = network[4].bias.detach().numpy().reshape(1, -1)
    layer3_params = np.ascontiguousarray(np.concatenate((layer3_weights, layer3_bias), axis=0).astype(np.float32))
    
    layer4_weights = network[6].weight.detach().numpy().T
    layer4_bias = network[6].bias.detach().numpy().reshape(1, -1)
    layer4_params = np.ascontiguousarray(np.concatenate((layer4_weights, layer4_bias), axis=0).astype(np.float32))
    
    return network, layer1_params, layer2_params, layer3_params, layer4_params


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [16, 32, 64])
@pytest.mark.parametrize("hidden_size", [16, 32, 64])
@pytest.mark.parametrize("out_size", [16, 32, 64])
def test_network_without_encoding(device, make_kernel, random_seed, in_size, hidden_size, out_size):
    np.random.seed(random_seed)

    # Create network and parameters
    result = create_network_layers(random_seed, in_size, hidden_size, out_size)
    network, layer1_params, layer2_params, layer3_params, layer4_params = result
    
    # Generate completely random test inputs
    batch_size = 64
    test_inputs = (np.random.rand(batch_size, in_size).astype(np.float32) - 0.5) * 2.0
    
    specialization_module = create_specialization_module(device, in_size, hidden_size, out_size)
    kernel = make_kernel("network_without_encoding", link_modules=[specialization_module])
    
    # Create buffers
    input_buffer = create_buffer_for_data(device, test_inputs, 4 * in_size)
    output_buffer = create_output_buffer(device, batch_size, out_size)
    
    layer1_buffer = create_buffer_for_data(device, layer1_params, 4)
    layer2_buffer = create_buffer_for_data(device, layer2_params, 4)
    layer3_buffer = create_buffer_for_data(device, layer3_params, 4)
    layer4_buffer = create_buffer_for_data(device, layer4_params, 4)
    
    # Run kernel
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
            }
        },
    )
    
    # Get results
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, out_size)
    
    # Compute expected result using nn.Sequential with nn.Linear layers
    input_torch = torch.tensor(test_inputs)
    
    # Compute expected output
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
    input_buffer = create_buffer_for_data(device, test_inputs, 4 * in_size)
    dinput_buffer = create_output_buffer(device, batch_size, in_size)
    
    # Create parameter buffers
    layer1_buffer = create_buffer_for_data(device, layer1_params, 4)
    layer2_buffer = create_buffer_for_data(device, layer2_params, 4)
    layer3_buffer = create_buffer_for_data(device, layer3_params, 4)
    layer4_buffer = create_buffer_for_data(device, layer4_params, 4)
    
    # Create gradient buffers for parameters
    dlayer1_buffer = create_output_buffer(device, layer1_params.shape[0], layer1_params.shape[1])
    dlayer2_buffer = create_output_buffer(device, layer2_params.shape[0], layer2_params.shape[1])
    dlayer3_buffer = create_output_buffer(device, layer3_params.shape[0], layer3_params.shape[1])
    dlayer4_buffer = create_output_buffer(device, layer4_params.shape[0], layer4_params.shape[1])
    
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
            }
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
    total_loss = torch.sum(output)
    total_loss.backward()
    
    # Get expected gradients
    expected_input_grad = input_torch.grad.detach().numpy()
    
    # Extract parameter gradients and convert back to our parameter layout
    expected_w1_grad = network[0].weight.grad.detach().numpy().T
    expected_b1_grad = network[0].bias.grad.detach().numpy().reshape(1, -1)
    expected_w2_grad = network[2].weight.grad.detach().numpy().T
    expected_b2_grad = network[2].bias.grad.detach().numpy().reshape(1, -1)
    expected_w3_grad = network[4].weight.grad.detach().numpy().T
    expected_b3_grad = network[4].bias.grad.detach().numpy().reshape(1, -1)
    expected_w4_grad = network[6].weight.grad.detach().numpy().T
    expected_b4_grad = network[6].bias.grad.detach().numpy().reshape(1, -1)
    
    # Combine expected parameter gradients to match layout
    expected_layer1_grad = np.concatenate((expected_w1_grad, expected_b1_grad), axis=0)
    expected_layer2_grad = np.concatenate((expected_w2_grad, expected_b2_grad), axis=0)
    expected_layer3_grad = np.concatenate((expected_w3_grad, expected_b3_grad), axis=0)
    expected_layer4_grad = np.concatenate((expected_w4_grad, expected_b4_grad), axis=0)
    
    # Test input gradients
    assert_close(input_derivatives, expected_input_grad)
    
    # Test parameter gradients
    assert_close(layer1_derivatives, expected_layer1_grad)
    assert_close(layer2_derivatives, expected_layer2_grad)
    assert_close(layer3_derivatives, expected_layer3_grad)
    assert_close(layer4_derivatives, expected_layer4_grad)