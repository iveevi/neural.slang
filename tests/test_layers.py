
import numpy as np
import pytest
import slangpy as spy
from .conftest import assert_close, RANDOM_SEEDS
from .test_utils import create_feed_forward_parameters, compute_feed_forward_expected


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_feed_forward_basic(device, make_kernel, random_seed):
    kernel = make_kernel("feed_forward")
    np.random.seed(random_seed)
    
    # Create parameters using shared helper
    weights_data, bias_data, parameters_data = create_feed_forward_parameters(4, 8, seed=random_seed)
    
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
            "globals": {
                "parameters": parameters_buffer,
                "input": input_buffer,
                "output": output_buffer,
            }
        },
    )
    
    # Get results
    output = output_buffer.to_numpy().view(np.float32).reshape(10, 8)
    
    # Compute expected result using shared helper
    expected = compute_feed_forward_expected(input_data, weights_data, bias_data)
    
    assert_close(output, expected)