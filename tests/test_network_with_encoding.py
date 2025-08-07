import numpy as np
import pytest
import slangpy as spy
import torch
from .conftest import assert_close, RANDOM_SEEDS
from .test_utils import (
    create_buffer_for_data,
    create_output_buffer
)


def pytorch_frequency_encoder(input_tensor, levels=4):
    """PyTorch implementation of FrequencyEncoder<float, 3, 4>."""
    batch_size = input_tensor.shape[0]
    encoded_parts = []
    
    for level in range(levels):  # For each frequency level
        k = 2.0 ** level  # k = 1, 2, 4, 8
        frequency = k * torch.pi
        
        # Compute sin and cos for all dimensions at this level
        sin_vals = torch.sin(frequency * input_tensor)  # Shape: (batch_size, 3)
        cos_vals = torch.cos(frequency * input_tensor)  # Shape: (batch_size, 3)
        
        # Concatenate sin and cos for this level
        level_encoding = torch.cat([sin_vals, cos_vals], dim=1)  # Shape: (batch_size, 6)
        encoded_parts.append(level_encoding)
    
    # Concatenate all levels
    output = torch.cat(encoded_parts, dim=1)  # Shape: (batch_size, 24)
    return output


def create_network_layers(random_seed):
    """Create PyTorch network layers and extract their parameters."""
    torch.manual_seed(random_seed)
    
    # Create the neural network (24 -> 8 -> 8 -> 4)
    network = torch.nn.Sequential(
        torch.nn.Linear(24, 8),
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
def test_network_with_encoding(device, make_kernel, random_seed):
    """Test the 3→[FrequencyEncoder]→24→8→8→4 network with frequency encoding."""
    kernel = make_kernel("network_with_encoding")
    
    # Create network and parameters
    network, layer1_params, layer2_params, layer3_params = create_network_layers(random_seed)
    
    # Create 3D input data (frequency encoder expects 3D input)
    np.random.seed(random_seed)
    test_inputs = np.vstack([
        0.5 * np.random.rand(3, 3).astype(np.float32) - 0.25,  # Small random 3D data
        [[0.1, -0.1, 0.2]],                                     # Small specific values
        np.zeros((1, 3)),                                       # Zero input (bias test)
        0.1 * np.ones((1, 3)),                                  # Small positive values
    ]).astype(np.float32)
    
    batch_size = test_inputs.shape[0]
    
    # Create buffers
    input_buffer = create_buffer_for_data(device, test_inputs, 3 * 4)  # 3D input
    output_buffer = create_output_buffer(device, batch_size, 4)        # 4D output
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
    
    # Compute expected result using PyTorch
    input_torch = torch.tensor(test_inputs)
    
    # Step 1: Apply frequency encoding
    encoded = pytorch_frequency_encoder(input_torch)
    
    # Step 2: Pass encoded input through network
    expected = network(encoded).detach().numpy()
    
    assert_close(output, expected)
    
    # Verify that encoding is working by checking zero input produces expected sin/cos pattern
    zero_idx = 2  # Index of zero input in test_inputs
    zero_output = output[zero_idx]
    
    # For zero input, frequency encoding should produce [sin(0), cos(0)] pattern
    # which is [0, 1, 0, 1, ...] for all frequency levels
    # After going through ReLU network, we expect some specific behavior
    assert not np.allclose(zero_output, 0), "Zero input should not produce zero output due to bias terms"


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_network_with_encoding_derivative(device, make_kernel, random_seed):
    """Test derivatives of the network with frequency encoding."""
    kernel = make_kernel("network_with_encoding_derivative")
    
    # Create network and parameters
    network, layer1_params, layer2_params, layer3_params = create_network_layers(random_seed)
    
    # Create 3D input data 
    np.random.seed(random_seed)
    test_inputs = np.vstack([
        0.3 * np.random.rand(2, 3).astype(np.float32) - 0.15,  # Small random 3D data
        [[0.1, -0.1, 0.2]],                                     # Small specific values
    ]).astype(np.float32)
    
    batch_size = test_inputs.shape[0]
    
    # Create buffers for forward pass
    input_buffer = create_buffer_for_data(device, test_inputs, 3 * 4)
    layer1_buffer = create_buffer_for_data(device, layer1_params, 4)
    layer2_buffer = create_buffer_for_data(device, layer2_params, 4)
    layer3_buffer = create_buffer_for_data(device, layer3_params, 4)
    
    # Create buffers for derivatives
    dinput_buffer = create_output_buffer(device, batch_size, 3)  # 3D input derivatives
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
    input_derivatives = dinput_buffer.to_numpy().view(np.float32).reshape(batch_size, 3)
    layer1_derivatives = dlayer1_buffer.to_numpy().view(np.float32).reshape(layer1_params.shape)
    layer2_derivatives = dlayer2_buffer.to_numpy().view(np.float32).reshape(layer2_params.shape)
    layer3_derivatives = dlayer3_buffer.to_numpy().view(np.float32).reshape(layer3_params.shape)
    
    # Compute expected derivatives using PyTorch autograd
    input_torch = torch.tensor(test_inputs, requires_grad=True)
    
    # Enable gradient tracking for all parameters
    for param in network.parameters():
        param.requires_grad_(True)
    
    # Create the network with frequency encoding
    def network_with_encoding(input_tensor):
        # Apply frequency encoding
        encoded = pytorch_frequency_encoder(input_tensor)
        # Pass through network
        return network(encoded)
    
    # Forward pass
    output = network_with_encoding(input_torch)
    
    # Sum over all outputs and batch to get scalar loss for backprop
    total_loss = torch.sum(output)
    total_loss.backward()
    
    # Get expected input derivatives
    expected_input_derivatives = input_torch.grad.detach().numpy()
    
    # Compare input derivatives
    assert_close(input_derivatives, expected_input_derivatives, rtol=1e-4, atol=1e-5)
    
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
    
    # Test parameter gradients
    assert_close(layer1_derivatives, expected_layer1_grad, rtol=1e-4, atol=1e-5)
    assert_close(layer2_derivatives, expected_layer2_grad, rtol=1e-4, atol=1e-5)
    assert_close(layer3_derivatives, expected_layer3_grad, rtol=1e-4, atol=1e-5)
    
    # Verify gradients have reasonable magnitudes (not zero, not exploding)
    assert np.any(np.abs(input_derivatives) > 1e-6), "Input derivatives should not be zero"
    assert np.any(np.abs(layer1_derivatives) > 1e-6), "Layer1 derivatives should not be zero"
    assert np.any(np.abs(layer2_derivatives) > 1e-6), "Layer2 derivatives should not be zero"
    assert np.any(np.abs(layer3_derivatives) > 1e-6), "Layer3 derivatives should not be zero"
    
    # Check for gradient explosion
    assert np.all(np.abs(input_derivatives) < 1e3), "Input derivatives should not explode"
    assert np.all(np.abs(layer1_derivatives) < 1e3), "Layer1 derivatives should not explode"
    assert np.all(np.abs(layer2_derivatives) < 1e3), "Layer2 derivatives should not explode"
    assert np.all(np.abs(layer3_derivatives) < 1e3), "Layer3 derivatives should not explode"