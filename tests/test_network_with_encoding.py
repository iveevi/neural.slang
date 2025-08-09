import numpy as np
import pytest
import slangpy as spy
import torch
from .conftest import assert_close, RANDOM_SEEDS
from .test_utils import create_buffer_for_data, create_output_buffer


def create_specialization_module(device, in_size, levels, hidden_size, out_size):
    source = f"""
    export static const int In = {in_size};
    export static const int Levels = {levels};
    export static const int Hidden = {hidden_size};
    export static const int Out = {out_size};
    """
    return device.load_module_from_source("specialization", source)


def frequency_encoder(input_tensor, levels=4):
    encoded_parts = []
    
    for level in range(levels):
        k = 2.0 ** level
        frequency = k * torch.pi
        
        sin_vals = torch.sin(frequency * input_tensor)
        cos_vals = torch.cos(frequency * input_tensor)
        
        level_encoding = torch.cat([sin_vals, cos_vals], dim=1)
        encoded_parts.append(level_encoding)
    
    return torch.cat(encoded_parts, dim=1)


def create_network_layers(random_seed, in_size, levels, hidden_size, out_size):
    torch.manual_seed(random_seed)

    encoded_size = 2 * levels * in_size
    
    network = torch.nn.Sequential(
        torch.nn.Linear(encoded_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size), 
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, out_size),
        torch.nn.ReLU()
    )
    
    layer1_weights = network[0].weight.detach().numpy().T
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
@pytest.mark.parametrize("in_size", [2, 3])
@pytest.mark.parametrize("levels", [6, 7, 8])
@pytest.mark.parametrize("hidden_size", [16, 32, 64])
@pytest.mark.parametrize("out_size", [16])
def test_network_with_encoding(device, make_kernel, random_seed, in_size, levels, hidden_size, out_size):
    np.random.seed(random_seed)
    
    result = create_network_layers(random_seed, in_size, levels, hidden_size, out_size)
    network, layer1_params, layer2_params, layer3_params, layer4_params = result
    
    batch_size = 64
    test_inputs = (np.random.rand(batch_size, in_size).astype(np.float32) - 0.5) * 2.0
    
    specialization_module = create_specialization_module(device, in_size, levels, hidden_size, out_size)
    kernel = make_kernel("network_with_encoding", link_modules=[specialization_module])
    
    input_buffer = create_buffer_for_data(device, test_inputs, in_size * 4)
    output_buffer = create_output_buffer(device, batch_size, out_size)
    layer1_buffer = create_buffer_for_data(device, layer1_params, 4)
    layer2_buffer = create_buffer_for_data(device, layer2_params, 4)
    layer3_buffer = create_buffer_for_data(device, layer3_params, 4)
    layer4_buffer = create_buffer_for_data(device, layer4_params, 4)
    
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
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, out_size)
    input_torch = torch.tensor(test_inputs)
    encoded = frequency_encoder(input_torch, levels)
    expected = network(encoded).detach().numpy()
    
    assert_close(output, expected)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("in_size", [2, 3])
@pytest.mark.parametrize("levels", [6, 7, 8])
@pytest.mark.parametrize("hidden_size", [16, 32, 64])
@pytest.mark.parametrize("out_size", [16])
def test_network_with_encoding_derivative(device, make_kernel, random_seed, in_size, levels, hidden_size, out_size):
    np.random.seed(random_seed)
    
    result = create_network_layers(random_seed, in_size, levels, hidden_size, out_size)
    network, layer1_params, layer2_params, layer3_params, layer4_params = result
    
    batch_size = 64
    test_inputs = (np.random.rand(batch_size, in_size).astype(np.float32) - 0.5) * 2.0
    
    specialization_module = create_specialization_module(device, in_size, levels, hidden_size, out_size)
    kernel = make_kernel("network_with_encoding_derivative", link_modules=[specialization_module])
    
    input_buffer = create_buffer_for_data(device, test_inputs, in_size * 4)
    layer1_buffer = create_buffer_for_data(device, layer1_params, 4)
    layer2_buffer = create_buffer_for_data(device, layer2_params, 4)
    layer3_buffer = create_buffer_for_data(device, layer3_params, 4)
    layer4_buffer = create_buffer_for_data(device, layer4_params, 4)
    
    dinput_buffer = create_output_buffer(device, batch_size, in_size)
    dlayer1_buffer = create_output_buffer(device, layer1_params.shape[0], layer1_params.shape[1])
    dlayer2_buffer = create_output_buffer(device, layer2_params.shape[0], layer2_params.shape[1])
    dlayer3_buffer = create_output_buffer(device, layer3_params.shape[0], layer3_params.shape[1])
    dlayer4_buffer = create_output_buffer(device, layer4_params.shape[0], layer4_params.shape[1])
    
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
    
    input_derivatives = dinput_buffer.to_numpy().view(np.float32).reshape(batch_size, in_size)
    layer1_derivatives = dlayer1_buffer.to_numpy().view(np.float32).reshape(layer1_params.shape)
    layer2_derivatives = dlayer2_buffer.to_numpy().view(np.float32).reshape(layer2_params.shape)
    layer3_derivatives = dlayer3_buffer.to_numpy().view(np.float32).reshape(layer3_params.shape)
    layer4_derivatives = dlayer4_buffer.to_numpy().view(np.float32).reshape(layer4_params.shape)
    
    input_torch = torch.tensor(test_inputs, requires_grad=True)
    
    for param in network.parameters():
        param.requires_grad_(True)
    
    def network_with_encoding(input_tensor):
        encoded = frequency_encoder(input_tensor, levels)
        return network(encoded)
    
    output = network_with_encoding(input_torch)
    
    total_loss = torch.sum(output)
    total_loss.backward()
    
    expected_input_derivatives = input_torch.grad.detach().numpy()
    
    expected_w1_grad = network[0].weight.grad.detach().numpy().T
    expected_b1_grad = network[0].bias.grad.detach().numpy().reshape(1, -1)
    expected_w2_grad = network[2].weight.grad.detach().numpy().T
    expected_b2_grad = network[2].bias.grad.detach().numpy().reshape(1, -1)
    expected_w3_grad = network[4].weight.grad.detach().numpy().T
    expected_b3_grad = network[4].bias.grad.detach().numpy().reshape(1, -1)
    expected_w4_grad = network[6].weight.grad.detach().numpy().T
    expected_b4_grad = network[6].bias.grad.detach().numpy().reshape(1, -1)
    
    expected_layer1_grad = np.concatenate((expected_w1_grad, expected_b1_grad), axis=0)
    expected_layer2_grad = np.concatenate((expected_w2_grad, expected_b2_grad), axis=0)
    expected_layer3_grad = np.concatenate((expected_w3_grad, expected_b3_grad), axis=0)
    expected_layer4_grad = np.concatenate((expected_w4_grad, expected_b4_grad), axis=0)

    assert_close(input_derivatives, expected_input_derivatives, rtol=1e-3, atol=1e-3) 
    assert_close(layer1_derivatives, expected_layer1_grad, rtol=1e-3, atol=1e-3)
    assert_close(layer2_derivatives, expected_layer2_grad, rtol=1e-3, atol=1e-3)
    assert_close(layer3_derivatives, expected_layer3_grad, rtol=1e-3, atol=1e-3)
    assert_close(layer4_derivatives, expected_layer4_grad, rtol=1e-3, atol=1e-3)