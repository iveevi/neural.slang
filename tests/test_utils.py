import numpy as np
import slangpy as spy


def create_feed_forward_parameters(input_size, output_size, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Create weights and bias
    weights_data = 2 * np.random.rand(input_size, output_size).astype(np.float32) - 1
    bias_data = 2 * np.random.rand(1, output_size).astype(np.float32) - 1
    
    # Combine parameters: weights followed by bias
    parameters_data = np.concatenate((weights_data, bias_data), axis=0)
    
    return weights_data, bias_data, parameters_data


def compute_feed_forward_expected(input_data, weights, bias):
    linear_output = np.matmul(input_data, weights) + bias
    return np.where(linear_output > 0, linear_output, 0)  # ReLU activation


def create_buffer_for_data(device, data, struct_size, usage_type="shader_resource"):
    usage_map = {
        "shader_resource": spy.BufferUsage.shader_resource,
        "constant_buffer": spy.BufferUsage.constant_buffer,
    }
    
    return device.create_buffer(
        size=data.nbytes,
        struct_size=struct_size,
        usage=usage_map.get(usage_type, spy.BufferUsage.shader_resource),
        data=data,
    )


def create_output_buffer(device, batch_size, output_size, element_size=4):
    return device.create_buffer(
        size=batch_size * output_size * element_size,
        struct_size=output_size * element_size,
        usage=spy.BufferUsage.shader_resource,
    )