import numpy as np
import pytest
import torch
import slangpy as spy
from .conftest import assert_close
from .test_utils import create_buffer_for_data


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("num_params", [64, 128, 256])
def test_adam(device, make_kernel, seed, num_params):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    num_iterations = 10
    
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    initial_params = np.random.randn(num_params).astype(np.float32) * 0.1
    
    adam_state_data = np.zeros(num_params * 3, dtype=np.float32)  # [m, v, t] for each param
    
    kernel = make_kernel("adam")
    
    parameters_buffer = create_buffer_for_data(device, initial_params.copy(), 4)
    state_buffer = create_buffer_for_data(device, adam_state_data, 3 * 4)  # 3 floats per state
    
    pytorch_params = torch.tensor(initial_params.copy(), requires_grad=True)
    pytorch_optimizer = torch.optim.Adam([pytorch_params], lr=lr, betas=(beta1, beta2), eps=eps)
    
    for iteration in range(num_iterations):
        gradients = np.random.randn(num_params).astype(np.float32) * 0.01
        gradients_buffer = create_buffer_for_data(device, gradients, 4)
        
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
        
        pytorch_optimizer.zero_grad()
        pytorch_params.grad = torch.tensor(gradients)
        pytorch_optimizer.step()
        
        gpu_params = parameters_buffer.to_numpy().view(np.float32)
        pytorch_params_current = pytorch_params.detach().numpy()
        
        tolerance_scale = 1.0 + iteration * 0.1
        assert_close(
            gpu_params, 
            pytorch_params_current, 
            rtol=1e-5 * tolerance_scale, 
            atol=1e-5 * tolerance_scale
        )