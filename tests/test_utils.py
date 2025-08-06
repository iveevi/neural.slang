"""Shared utilities for neural network tests."""

import numpy as np


def create_feed_forward_parameters(input_size, output_size, seed=None):
    """
    Create parameters for a feed forward layer.
    Returns weights and bias as separate arrays, and combined parameters.
    
    Args:
        input_size: Number of input features
        output_size: Number of output neurons  
        seed: Random seed for reproducible results
    
    Returns:
        tuple: (weights_data, bias_data, parameters_data)
            - weights_data: Shape (input_size, output_size)
            - bias_data: Shape (1, output_size) 
            - parameters_data: Combined weights and bias, shape (input_size + 1, output_size)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create weights and bias
    weights_data = 2 * np.random.rand(input_size, output_size).astype(np.float32) - 1
    bias_data = 2 * np.random.rand(1, output_size).astype(np.float32) - 1
    
    # Combine parameters: weights followed by bias
    parameters_data = np.concatenate((weights_data, bias_data), axis=0)
    
    return weights_data, bias_data, parameters_data


def compute_feed_forward_expected(input_data, weights, bias):
    """
    Compute expected output for a feed forward layer with ReLU activation.
    
    Args:
        input_data: Input array, shape (batch_size, input_size)
        weights: Weight matrix, shape (input_size, output_size)
        bias: Bias vector, shape (1, output_size)
    
    Returns:
        Expected output after linear transformation and ReLU activation
    """
    linear_output = np.matmul(input_data, weights) + bias
    return np.where(linear_output > 0, linear_output, 0)  # ReLU activation


def create_buffer_for_data(device, data, struct_size, usage_type="shader_resource"):
    """
    Create a buffer from numpy data with appropriate size and structure.
    
    Args:
        device: Slang device
        data: Numpy array data
        struct_size: Size of each struct element in bytes
        usage_type: Buffer usage type (default: "shader_resource")
    
    Returns:
        Created buffer
    """
    import slangpy as spy
    
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
    """
    Create an output buffer for network results.
    
    Args:
        device: Slang device
        batch_size: Number of samples
        output_size: Number of output features
        element_size: Size of each element in bytes (default: 4 for float32)
    
    Returns:
        Created buffer
    """
    import slangpy as spy
    
    return device.create_buffer(
        size=batch_size * output_size * element_size,
        struct_size=output_size * element_size,
        usage=spy.BufferUsage.shader_resource,
    )