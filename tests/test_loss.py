
import numpy as np
import pytest
import slangpy as spy
from .conftest import assert_close


def test_mse_basic(device, make_kernel):
    kernel = make_kernel("mse_main")
    np.random.seed(42)
    input_data = 2 * np.random.rand(10, 16).astype(np.float32) - 1
    target_data = 2 * np.random.rand(10, 16).astype(np.float32) - 1
    
    input_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=16 * 4,
        usage=spy.BufferUsage.shader_resource,
        data=input_data,
    )
    
    target_buffer = device.create_buffer(
        size=target_data.nbytes,
        struct_size=16 * 4,
        usage=spy.BufferUsage.shader_resource,
        data=target_data,
    )
    
    output_buffer = device.create_buffer(
        size=10 * 4,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource,
    )
    
    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "mse_globals": {
                "input": input_buffer,
                "target": target_buffer,
                "output": output_buffer,
            }
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32)
    expected = np.mean(np.square(input_data - target_data), axis=1)
    
    assert_close(output, expected)