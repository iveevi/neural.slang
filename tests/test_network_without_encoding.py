import numpy as np
import pytest
import slangpy as spy
import torch
from .conftest import assert_close, RANDOM_SEEDS
from .test_utils import (
    create_feed_forward_parameters, 
    compute_feed_forward_expected,
    create_buffer_for_data,
    create_output_buffer
)


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_network_without_encoding(device, make_kernel, random_seed):
    """Test the 4→8→8→4 network without encoding with multiple scenarios."""
    kernel = make_kernel("network_without_encoding")
    
    # Create parameters for 4 -> 8 -> 8 -> 4 network
    weights1, bias1, params1 = create_feed_forward_parameters(4, 8, seed=random_seed)
    weights2, bias2, params2 = create_feed_forward_parameters(8, 8, seed=random_seed + 1) 
    weights3, bias3, params3 = create_feed_forward_parameters(8, 4, seed=random_seed + 2)
    
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
    layer1_buffer = create_buffer_for_data(device, params1, 4)
    layer2_buffer = create_buffer_for_data(device, params2, 4)
    layer3_buffer = create_buffer_for_data(device, params3, 4)
    
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
    
    # Get results and compute expected
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, 4)
    x1 = compute_feed_forward_expected(test_inputs, weights1, bias1)
    x2 = compute_feed_forward_expected(x1, weights2, bias2)
    expected = compute_feed_forward_expected(x2, weights3, bias3)
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_network_without_encoding_derivative(device, make_kernel, random_seed):
    """Test network derivatives against PyTorch autograd."""
    kernel = make_kernel("network_without_encoding_derivative")
    np.random.seed(random_seed)
    
    # Create parameters for 4 -> 8 -> 8 -> 4 network
    weights1, bias1, params1 = create_feed_forward_parameters(4, 8, seed=random_seed)
    weights2, bias2, params2 = create_feed_forward_parameters(8, 8, seed=random_seed + 1)
    weights3, bias3, params3 = create_feed_forward_parameters(8, 4, seed=random_seed + 2)
    
    # Create smaller test inputs for gradient testing
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
    layer1_buffer = create_buffer_for_data(device, params1, 4)
    layer2_buffer = create_buffer_for_data(device, params2, 4)
    layer3_buffer = create_buffer_for_data(device, params3, 4)
    
    # Create gradient buffers for parameters
    dlayer1_buffer = create_output_buffer(device, params1.shape[0], params1.shape[1])
    dlayer2_buffer = create_output_buffer(device, params2.shape[0], params2.shape[1])
    dlayer3_buffer = create_output_buffer(device, params3.shape[0], params3.shape[1])
    
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
    layer1_derivatives = dlayer1_buffer.to_numpy().view(np.float32).reshape(params1.shape)
    layer2_derivatives = dlayer2_buffer.to_numpy().view(np.float32).reshape(params2.shape)
    layer3_derivatives = dlayer3_buffer.to_numpy().view(np.float32).reshape(params3.shape)
    
    # Compute expected derivatives using PyTorch autograd
    def pytorch_network(inputs, w1, b1, w2, b2, w3, b3):
        """PyTorch implementation of the 4→8→8→4 network."""
        x1 = torch.relu(torch.matmul(inputs, w1) + b1)
        x2 = torch.relu(torch.matmul(x1, w2) + b2)
        output = torch.relu(torch.matmul(x2, w3) + b3)
        return output
    
    # Create PyTorch tensors with gradient tracking
    input_torch = torch.tensor(test_inputs, requires_grad=True)
    w1_torch = torch.tensor(weights1, requires_grad=True)
    b1_torch = torch.tensor(bias1.reshape(-1), requires_grad=True)
    w2_torch = torch.tensor(weights2, requires_grad=True)
    b2_torch = torch.tensor(bias2.reshape(-1), requires_grad=True)
    w3_torch = torch.tensor(weights3, requires_grad=True)
    b3_torch = torch.tensor(bias3.reshape(-1), requires_grad=True)
    
    # Forward pass and compute gradients
    output = pytorch_network(input_torch, w1_torch, b1_torch, w2_torch, b2_torch, w3_torch, b3_torch)
    
    # Sum over all outputs and batch to get scalar loss for backprop
    total_loss = torch.sum(output)
    total_loss.backward()
    
    # Get expected gradients
    expected_input_grad = input_torch.grad.numpy()
    expected_w1_grad = w1_torch.grad.numpy()
    expected_b1_grad = b1_torch.grad.numpy().reshape(1, -1)
    expected_w2_grad = w2_torch.grad.numpy()
    expected_b2_grad = b2_torch.grad.numpy().reshape(1, -1)
    expected_w3_grad = w3_torch.grad.numpy()
    expected_b3_grad = b3_torch.grad.numpy().reshape(1, -1)
    
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