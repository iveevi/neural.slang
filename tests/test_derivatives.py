
import numpy as np
import pytest
import slangpy as spy
from .conftest import assert_close, RANDOM_SEEDS


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_vector_relu_derivative(device, make_kernel, random_seed):
    kernel = make_kernel("relu_derivative_vector")
    np.random.seed(random_seed)
    data = 2 * np.random.rand(10, 2).astype(np.float32) - 1
    
    input_buffer = device.create_buffer(
        size=data.nbytes,
        struct_size=8,
        usage=spy.BufferUsage.shader_resource,
        data=data,
    )
    
    output_buffer = device.create_buffer(
        size=data.nbytes,
        struct_size=8,
        usage=spy.BufferUsage.shader_resource,
    )
    
    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
            }
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(10, 2)
    # ReLU derivative: 1 if input > 0, 0 otherwise
    expected = np.where(data > 0, 1, 0).astype(np.float32)
    
    assert_close(output, expected)