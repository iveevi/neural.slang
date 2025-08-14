import numpy as np
import pytest
import slangpy as spy
import torch
from .conftest import assert_close
from common import *


def create_specialization_module(device, in_size, out_size):
    source = f"""
    export static const int In = {in_size};
    export static const int Out = {out_size};
    """
    return device.load_module_from_source("specialization", source)


def create_linear_layer_data(input_size, output_size, random_seed):
    torch.manual_seed(random_seed)
    linear_layer = torch.nn.Linear(input_size, output_size)
    parameters_data = linear_to_numpy(linear_layer)
    return linear_layer, parameters_data


def create_bindless_linear_layer_data(input_size, output_size, random_seed):
    torch.manual_seed(random_seed)
    linear_layer = torch.nn.Linear(input_size, output_size)
    weights, bias = linear_to_bindless_numpy(linear_layer)
    return linear_layer, weights, bias


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [32, 64, 128])
@pytest.mark.parametrize("out_size", [32, 64, 128])
def test_feed_forward(device, make_kernel, random_seed, in_size, out_size):
    np.random.seed(random_seed)
    
    batch_size = 16
    linear_layer, parameters_data = create_linear_layer_data(in_size, out_size, random_seed)
    
    specialization_module = create_specialization_module(device, in_size, out_size)
    kernel = make_kernel("feed_forward", link_modules=[specialization_module])
    
    input_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    
    input_buffer = create_buffer_32b(device, input_data, in_size)
    
    output_buffer = create_batched_buffer_32b(device, batch_size, out_size)
    
    parameters_buffer = create_buffer_32b(device, parameters_data)
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "parameters": parameters_buffer,
                "input": input_buffer,
                "output": output_buffer,
            
                    "count": batch_size,}
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, out_size)
    input_torch = torch.tensor(input_data)
    linear_output = linear_layer(input_torch)
    expected = torch.relu(linear_output).detach().numpy()
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [32, 64, 128])
@pytest.mark.parametrize("out_size", [32, 64, 128])
def test_feed_forward_derivative(device, make_kernel, random_seed, in_size, out_size):
    np.random.seed(random_seed)
    
    batch_size = 16
    linear_layer, parameters_data = create_linear_layer_data(in_size, out_size, random_seed)
    
    specialization_module = create_specialization_module(device, in_size, out_size)
    kernel = make_kernel("feed_forward_derivative", link_modules=[specialization_module])
    
    input_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    
    input_buffer = create_buffer_32b(device, input_data, in_size)
    
    dinput_buffer = create_buffer_32b(device, np.zeros_like(input_data), in_size)
    
    parameters_buffer = create_buffer_32b(device, parameters_data)
    
    dparameters_buffer = create_buffer_32b(device, np.zeros_like(parameters_data))
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "dinput": dinput_buffer,
                "parameters": parameters_buffer,
                "dparameters": dparameters_buffer,
            
                    "count": batch_size,}
        },
    )
    
    input_derivatives = dinput_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    parameter_derivatives = dparameters_buffer.to_numpy().view(np.float32).reshape(parameters_data.shape)
    
    input_torch = torch.tensor(input_data, requires_grad=True)
    
    linear_layer.weight.requires_grad_(True)
    linear_layer.bias.requires_grad_(True)
    
    output = torch.relu(linear_layer(input_torch))
    
    grad_output = torch.ones_like(output)
    
    output.backward(grad_output)
    
    assert input_torch.grad is not None, "Input gradients were not computed"
    assert linear_layer.weight.grad is not None, "Weight gradients were not computed"
    assert linear_layer.bias.grad is not None, "Bias gradients were not computed"
    
    expected_input_derivatives = input_torch.grad.detach().numpy()
    expected_parameter_derivatives = linear_gradients_to_numpy(linear_layer)
    
    assert_close(input_derivatives, expected_input_derivatives, rtol=1e-5, atol=1e-6)
    assert_close(parameter_derivatives, expected_parameter_derivatives, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [32, 64])
@pytest.mark.parametrize("out_size", [32, 64])
@pytest.mark.parametrize("offset", [0, 16, 32, 64])
def test_feed_forward_address(device, make_kernel, random_seed, in_size, out_size, offset):
    np.random.seed(random_seed)
    
    batch_size = 16
    linear_layer, parameters_data = create_linear_layer_data(in_size, out_size, random_seed)
    
    specialization_module = create_specialization_module(device, in_size, out_size)
    kernel = make_kernel("feed_forward_address", link_modules=[specialization_module])
    
    input_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    
    input_buffer = create_buffer_32b(device, input_data, in_size)
    
    output_buffer = create_batched_buffer_32b(device, batch_size, out_size)
    
    padded_parameters_data = np.zeros(parameters_data.size + offset, dtype=np.float32)
    if offset > 0:
        padded_parameters_data[:offset] = np.random.rand(offset).astype(np.float32)
    padded_parameters_data[offset:] = parameters_data.flatten()
    
    parameters_buffer = create_buffer_32b(device, padded_parameters_data)
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "parameters": parameters_buffer,
                "input": input_buffer,
                "output": output_buffer,
                "address": offset,
            
                    "count": batch_size,}
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, out_size)
    
    input_torch = torch.tensor(input_data)
    
    linear_output = linear_layer(input_torch)
    expected = torch.relu(linear_output).detach().numpy()
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [32, 64])
@pytest.mark.parametrize("out_size", [32, 64])
@pytest.mark.parametrize("offset", [0, 16, 32, 64])
def test_feed_forward_address_derivative(device, make_kernel, random_seed, in_size, out_size, offset):
    np.random.seed(random_seed)
    
    batch_size = 16
    linear_layer, parameters_data = create_linear_layer_data(in_size, out_size, random_seed)
    
    specialization_module = create_specialization_module(device, in_size, out_size)
    kernel = make_kernel("feed_forward_address_derivative", link_modules=[specialization_module])
    
    input_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    
    input_buffer = create_buffer_32b(device, input_data, in_size)
    
    dinput_buffer = create_buffer_32b(device, np.zeros_like(input_data), in_size)
    
    padded_parameters_data = np.zeros(parameters_data.size + offset, dtype=np.float32)
    if offset > 0:
        padded_parameters_data[:offset] = np.random.rand(offset).astype(np.float32)
    padded_parameters_data[offset:] = parameters_data.flatten()
    
    parameters_buffer = create_buffer_32b(device, padded_parameters_data)
    
    dparameters_buffer = create_buffer_32b(device, np.zeros_like(padded_parameters_data))
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "dinput": dinput_buffer,
                "parameters": parameters_buffer,
                "dparameters": dparameters_buffer,
                "address": offset,
            
                    "count": batch_size,}
        },
    )
    
    input_derivatives = dinput_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    parameter_derivatives_padded = dparameters_buffer.to_numpy().view(np.float32).reshape(padded_parameters_data.shape)
    parameter_derivatives = parameter_derivatives_padded[offset:].reshape(parameters_data.shape)
    
    input_torch = torch.tensor(input_data, requires_grad=True)
    
    linear_layer.weight.requires_grad_(True)
    linear_layer.bias.requires_grad_(True)
    
    output = torch.relu(linear_layer(input_torch))
    
    grad_output = torch.ones_like(output)
    
    output.backward(grad_output)
    
    assert input_torch.grad is not None, "Input gradients were not computed"
    assert linear_layer.weight.grad is not None, "Weight gradients were not computed"
    assert linear_layer.bias.grad is not None, "Bias gradients were not computed"
    
    expected_input_derivatives = input_torch.grad.detach().numpy()
    expected_parameter_derivatives = linear_gradients_to_numpy(linear_layer)
    
    assert_close(input_derivatives, expected_input_derivatives, rtol=1e-5, atol=1e-6)
    assert_close(parameter_derivatives, expected_parameter_derivatives, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [32, 64, 128])
@pytest.mark.parametrize("out_size", [32, 64, 128])
def test_feed_forward_bindless(device, make_kernel, in_size, out_size, random_seed):
    pytest.skip("Skipping bindless test")

    np.random.seed(random_seed)
    
    batch_size = 16
    linear_layer, weights, bias = create_bindless_linear_layer_data(in_size, out_size, random_seed)
    
    specialization_module = create_specialization_module(device, in_size, out_size)
    kernel = make_kernel("feed_forward_bindless", link_modules=[specialization_module])
    
    input_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    
    input_buffer = create_buffer_32b(device, input_data, in_size)
    
    output_buffer = create_batched_buffer_32b(device, batch_size, out_size)
    
    weights_buffer = create_buffer_32b(device, weights)
    
    bias_buffer = create_buffer_32b(device, bias)

    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "weights": weights_buffer.descriptor_handle_rw,
                "biases": bias_buffer.descriptor_handle_rw,
                "input": input_buffer,
                "output": output_buffer,
            
                    "count": batch_size,}
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, out_size)
    input_torch = torch.tensor(input_data)
    linear_output = linear_layer(input_torch)
    expected = torch.relu(linear_output).detach().numpy()
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [32, 64, 128])
@pytest.mark.parametrize("out_size", [32, 64, 128])
def test_feed_forward_bindless_derivative(device, make_kernel, in_size, out_size, random_seed):
    pytest.skip("Skipping bindless test")

    np.random.seed(random_seed)
    
    batch_size = 16
    linear_layer, weights, bias = create_bindless_linear_layer_data(in_size, out_size, random_seed)
    
    specialization_module = create_specialization_module(device, in_size, out_size)
    kernel = make_kernel("feed_forward_bindless_derivative", link_modules=[specialization_module])
    
    input_data = 2 * np.random.rand(batch_size, in_size).astype(np.float32) - 1
    
    input_buffer = create_buffer_32b(device, input_data, in_size)
    
    dinput_buffer = create_buffer_32b(device, np.zeros_like(input_data), in_size)
    
    weights_buffer = create_buffer_32b(device, weights)
    
    bias_buffer = create_buffer_32b(device, bias)
    
    dweights_buffer = create_buffer_32b(device, np.zeros_like(weights))
    
    dbias_buffer = create_buffer_32b(device, np.zeros_like(bias))

    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "weights": weights_buffer.descriptor_handle_rw,
                "biases": bias_buffer.descriptor_handle_rw,
                "dweights": dweights_buffer.descriptor_handle_rw,
                "dbiases": dbias_buffer.descriptor_handle_rw,
                "input": input_buffer,
                "dinput": dinput_buffer,
            
                    "count": batch_size,}
        },
    )
    
    input_derivatives = dinput_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    weights_derivatives = dweights_buffer.to_numpy().view(np.float32).reshape(weights.shape)
    bias_derivatives = dbias_buffer.to_numpy().view(np.float32).reshape(bias.shape)
    
    input_torch = torch.tensor(input_data, requires_grad=True)
    
    linear_layer.weight.requires_grad_(True)
    linear_layer.bias.requires_grad_(True)
    
    output = torch.relu(linear_layer(input_torch))
    
    grad_output = torch.ones_like(output)
    
    output.backward(grad_output)
    
    assert input_torch.grad is not None, "Input gradients were not computed"
    assert linear_layer.weight.grad is not None, "Weight gradients were not computed"
    assert linear_layer.bias.grad is not None, "Bias gradients were not computed"
    
    expected_input_derivatives = input_torch.grad.detach().numpy()
    expected_weights_derivatives, expected_bias_derivatives = linear_gradients_to_bindless_numpy(linear_layer)
    
    assert_close(input_derivatives, expected_input_derivatives, rtol=1e-5, atol=1e-6)
    assert_close(weights_derivatives, expected_weights_derivatives, rtol=1e-5, atol=1e-6)
    assert_close(bias_derivatives, expected_bias_derivatives, rtol=1e-5, atol=1e-6)