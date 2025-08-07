import numpy as np
import pytest
import slangpy as spy
import torch
from .conftest import assert_close, RANDOM_SEEDS
from .test_utils import (
    create_buffer_for_data,
    create_output_buffer
)


def create_network_layers(random_seed):
    """Create PyTorch network layers and extract their parameters."""
    torch.manual_seed(random_seed)
    
    # Create the neural network (4 -> 8 -> 8 -> 4)
    network = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 8), 
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
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
    
    return network, layer1_params, layer2_params, layer3_params


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_network_without_encoding(device, make_kernel, random_seed):
    """Test the 4→8→8→4 network without encoding using nn.Linear layers."""
    kernel = make_kernel("network_without_encoding")
    
    # Create network and parameters
    network, layer1_params, layer2_params, layer3_params = create_network_layers(random_seed)
    
    # Test multiple scenarios in one batch - keep inputs in reasonable range
    np.random.seed(random_seed)
    test_inputs = np.vstack([
        0.5 * np.random.rand(3, 4).astype(np.float32) - 0.25,  # Small random data
        [[0.1, -0.1, 0.2, -0.05]],                             # Small specific values
        np.zeros((1, 4)),                                      # Zero input (bias test)
        0.1 * np.ones((1, 4)),                                 # Small positive values
    ]).astype(np.float32)
    
    batch_size = test_inputs.shape[0]
    
    # Create buffers
    input_buffer = create_buffer_for_data(device, test_inputs, 4 * 4)
    output_buffer = create_output_buffer(device, batch_size, 4)
    layer1_buffer = create_buffer_for_data(device, layer1_params, 4)
    layer2_buffer = create_buffer_for_data(device, layer2_params, 4)
    layer3_buffer = create_buffer_for_data(device, layer3_params, 4)
    
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
            }
        },
    )
    
    # Get results
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, 4)
    
    # Compute expected result using nn.Sequential with nn.Linear layers
    input_torch = torch.tensor(test_inputs)
    
    # Compute expected output
    expected = network(input_torch).detach().numpy()
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_network_without_encoding_derivative(device, make_kernel, random_seed):
    """Test network derivatives against PyTorch autograd using nn.Linear layers."""
    kernel = make_kernel("network_without_encoding_derivative")
    
    # Create network and parameters
    network, layer1_params, layer2_params, layer3_params = create_network_layers(random_seed)
    
    # Create smaller test inputs for gradient testing
    np.random.seed(random_seed)
    test_inputs = np.array([
        [0.1, -0.1, 0.2, -0.05],   # Small specific values
        [0.0, 0.0, 0.0, 0.0],      # Zero input (bias test)
        [0.05, 0.05, 0.05, 0.05],  # Small positive values
    ]).astype(np.float32)
    
    batch_size = test_inputs.shape[0]
    
    # Create input buffers
    input_buffer = create_buffer_for_data(device, test_inputs, 4 * 4)
    dinput_buffer = create_output_buffer(device, batch_size, 4)
    
    # Create parameter buffers
    layer1_buffer = create_buffer_for_data(device, layer1_params, 4)
    layer2_buffer = create_buffer_for_data(device, layer2_params, 4)
    layer3_buffer = create_buffer_for_data(device, layer3_params, 4)
    
    # Create gradient buffers for parameters
    dlayer1_buffer = create_output_buffer(device, layer1_params.shape[0], layer1_params.shape[1])
    dlayer2_buffer = create_output_buffer(device, layer2_params.shape[0], layer2_params.shape[1])
    dlayer3_buffer = create_output_buffer(device, layer3_params.shape[0], layer3_params.shape[1])
    
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
            }
        },
    )
    
    # Get derivative results
    input_derivatives = dinput_buffer.to_numpy().view(np.float32).reshape(batch_size, 4)
    layer1_derivatives = dlayer1_buffer.to_numpy().view(np.float32).reshape(layer1_params.shape)
    layer2_derivatives = dlayer2_buffer.to_numpy().view(np.float32).reshape(layer2_params.shape)
    layer3_derivatives = dlayer3_buffer.to_numpy().view(np.float32).reshape(layer3_params.shape)
    
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
    
    # Combine expected parameter gradients to match layout
    expected_layer1_grad = np.concatenate((expected_w1_grad, expected_b1_grad), axis=0)
    expected_layer2_grad = np.concatenate((expected_w2_grad, expected_b2_grad), axis=0)
    expected_layer3_grad = np.concatenate((expected_w3_grad, expected_b3_grad), axis=0)
    
    # Test input gradients
    assert_close(input_derivatives, expected_input_grad)
    
    # Test parameter gradients
    assert_close(layer1_derivatives, expected_layer1_grad)
    assert_close(layer2_derivatives, expected_layer2_grad)
    assert_close(layer3_derivatives, expected_layer3_grad)