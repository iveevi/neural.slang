
import numpy as np
import pytest
import slangpy as spy
from .conftest import assert_close


def test_feed_forward_basic(device, make_kernel):
    kernel = make_kernel("feed_forward_main")
    np.random.seed(42)
    
    # Create weights (4x8) and bias (1x8) 
    weights_data = 2 * np.random.rand(4, 8).astype(np.float32) - 1
    bias_data = 2 * np.random.rand(1, 8).astype(np.float32) - 1
    
    # Combine parameters: weights followed by bias
    parameters_data = np.concatenate((weights_data, bias_data), axis=0)
    
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
            "feed_forward_globals": {
                "parameters": parameters_buffer,
                "input": input_buffer,
                "output": output_buffer,
            }
        },
    )
    
    # Get results
    output = output_buffer.to_numpy().view(np.float32).reshape(10, 8)
    
    # Compute expected result: linear layer followed by ReLU
    linear_output = np.matmul(input_data, weights_data) + bias_data
    expected = np.where(linear_output > 0, linear_output, 0)  # ReLU activation
    
    assert_close(output, expected)