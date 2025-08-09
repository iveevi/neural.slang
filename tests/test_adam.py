import numpy as np
import pytest
import torch
import slangpy as spy
from .conftest import assert_close
from .test_utils import create_buffer_for_data


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("num_params", [64, 128, 256])
def test_adam(device, make_kernel, seed, num_params):
    """Test Adam optimizer against PyTorch's Adam over multiple iterations."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    num_iterations = 10
    
    # Adam hyperparameters (matching neural.slang defaults)
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    # Initialize parameters and state
    initial_params = np.random.randn(num_params).astype(np.float32) * 0.1
    
    # Initialize Adam state as flat array: [m0, v0, t0, m1, v1, t1, ...]
    # The AdamState struct has: float m, float v, int t
    # We'll pack this as interleaved floats (treating t as float for now)
    adam_state_data = np.zeros(num_params * 3, dtype=np.float32)  # [m, v, t] for each param
    
    # Create GPU kernel
    kernel = make_kernel("adam")
    
    # Create buffers with proper initialization
    parameters_buffer = create_buffer_for_data(device, initial_params.copy(), 4)
    state_buffer = create_buffer_for_data(device, adam_state_data, 3 * 4)  # 3 floats per state
    
    # PyTorch reference
    pytorch_params = torch.tensor(initial_params.copy(), requires_grad=True)
    pytorch_optimizer = torch.optim.Adam([pytorch_params], lr=lr, betas=(beta1, beta2), eps=eps)
    
    # Run optimization for multiple iterations
    for iteration in range(num_iterations):
        # Generate random gradients for this iteration
        gradients = np.random.randn(num_params).astype(np.float32) * 0.01
        gradients_buffer = create_buffer_for_data(device, gradients, 4)
        
        # Run GPU Adam step
        kernel.dispatch(
            thread_count=(num_params, 1, 1),
            vars={
                "globals": {
                    "state": state_buffer,
                    "parameters": parameters_buffer,
                    "gradients": gradients_buffer,
                }
            },
        )
        
        # Run PyTorch Adam step
        pytorch_optimizer.zero_grad()
        pytorch_params.grad = torch.tensor(gradients)
        pytorch_optimizer.step()
        
        # Check closeness at each iteration
        gpu_params = parameters_buffer.to_numpy().view(np.float32)
        pytorch_params_current = pytorch_params.detach().numpy()

        print(f"Iteration {iteration}:")
        print(f"GPU params: {gpu_params}")
        print(f"PyTorch params: {pytorch_params_current}")
        
        # Use more relaxed tolerances for intermediate iterations
        tolerance_scale = 1.0 + iteration * 0.1  # Gradually increase tolerance
        assert_close(
            gpu_params, 
            pytorch_params_current, 
            rtol=1e-5 * tolerance_scale, 
            atol=1e-5 * tolerance_scale
        )