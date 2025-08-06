import numpy as np
import pytest
import slangpy as spy
from .conftest import assert_close, RANDOM_SEEDS


@pytest.mark.parametrize("random_seed", RANDOM_SEEDS)
def test_frequency_encoder(device, make_kernel, random_seed):
    """Test FrequencyEncoder with 3D input and 4 levels."""
    kernel = make_kernel("frequency_encoder")
    np.random.seed(random_seed)
    
    # Create input data: 10 samples of 3D vectors including edge cases
    input_data = np.array([
        [0.0, 0.0, 0.0],      # All zeros
        [1.0, 1.0, 1.0],      # All ones
        [0.5, -0.5, 0.25],    # Mixed values
        *np.random.rand(7, 3).tolist()  # Random samples
    ], dtype=np.float32)
    
    # Create buffers
    input_buffer = device.create_buffer(
        size=input_data.nbytes,
        struct_size=3 * 4,  # 3 floats * 4 bytes each
        usage=spy.BufferUsage.shader_resource,
        data=input_data,
    )
    
    # Output size: 2 * Levels * Dim = 2 * 4 * 3 = 24 floats per sample
    output_size = 10 * 24 * 4  # 10 samples * 24 floats * 4 bytes
    output_buffer = device.create_buffer(
        size=output_size,
        struct_size=24 * 4,  # 24 floats * 4 bytes each
        usage=spy.BufferUsage.shader_resource,
    )
    
    # Dispatch kernel
    kernel.dispatch(
        thread_count=(10, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
            }
        },
    )
    
    # Get results
    output = output_buffer.to_numpy().view(np.float32).reshape(10, 24)
    
    # Compute expected result manually
    # FrequencyEncoder<float, 3, 4> means:
    # - 3D input
    # - 4 frequency levels
    # - Output: 2 * 4 * 3 = 24 dimensions (sin and cos for each level and dimension)
    
    expected = np.zeros((10, 24), dtype=np.float32)
    
    for sample_idx in range(10):
        input_sample = input_data[sample_idx]
        
        for dim in range(3):  # For each input dimension
            k = 1.0
            
            for level in range(4):  # For each frequency level
                frequency = k * np.pi
                sin_val = np.sin(frequency * input_sample[dim])
                cos_val = np.cos(frequency * input_sample[dim])
                
                # Output layout: [sin_0_0, sin_0_1, sin_0_2, cos_0_0, cos_0_1, cos_0_2, sin_1_0, ...]
                # where first index is level, second is dimension
                sin_idx = 2 * level * 3 + dim
                cos_idx = 2 * level * 3 + dim + 3
                
                expected[sample_idx, sin_idx] = sin_val
                expected[sample_idx, cos_idx] = cos_val
                
                k *= 2.0
    
    assert_close(output, expected)
    
    # Verify specific known values for zero input
    zero_output = output[0]
    for level in range(4):
        for dim in range(3):
            sin_idx = 2 * level * 3 + dim
            cos_idx = 2 * level * 3 + dim + 3
            assert abs(zero_output[sin_idx]) < 1e-6, f"Sin should be ~0 for zero input at level {level}, dim {dim}"
            assert abs(zero_output[cos_idx] - 1.0) < 1e-6, f"Cos should be ~1 for zero input at level {level}, dim {dim}"