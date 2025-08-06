import numpy as np
import pytest
import slangpy as spy
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