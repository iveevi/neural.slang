import numpy as np
import pytest
import torch
from .conftest import assert_close
from common import *


def create_specialization_module(device, features, bank_size):
    source = f"""
    export static const int Features = {features};
    """
    return device.load_module_from_source("specialization", source)


def feature_bank_pytorch(parameters: torch.Tensor, indices: torch.Tensor, features: int) -> torch.Tensor:
    return parameters.view(-1, features)[indices]


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("features", [8, 16, 32])
@pytest.mark.parametrize("bank_size", [16, 32, 64])
def test_feature_bank(device, make_kernel, random_seed, features, bank_size):
    batch_size = 16
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    specialization_module = create_specialization_module(device, features, bank_size)
    kernel = make_kernel("feature_bank", link_modules=[specialization_module])
    
    parameters_data = np.random.randn(bank_size * features).astype(np.float32)
    indices_data = np.random.randint(0, bank_size, size=batch_size, dtype=np.uint32)
    
    parameters_buffer = create_buffer_32b(device, parameters_data)
    indices_buffer = create_buffer_32b(device, indices_data.astype(np.uint32))
    output_buffer = create_batched_buffer_32b(device, batch_size, features)
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "parameters": parameters_buffer,
                "indices": indices_buffer,
                "output": output_buffer,
                "count": batch_size,
            }
        },
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, features)
    
    parameters_torch = torch.tensor(parameters_data)
    indices_torch = torch.tensor(indices_data.astype(np.int64))
    expected = feature_bank_pytorch(parameters_torch, indices_torch, features).numpy()
    
    assert_close(output, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("features", [8, 16, 32])
@pytest.mark.parametrize("bank_size", [16, 32, 64])
def test_feature_bank_derivative(device, make_kernel, random_seed, features, bank_size):
    batch_size = 16
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    specialization_module = create_specialization_module(device, features, bank_size)
    kernel = make_kernel("feature_bank_derivative", link_modules=[specialization_module])
    
    parameters_data = np.random.randn(bank_size * features).astype(np.float32)
    indices_data = np.random.randint(0, bank_size, size=batch_size, dtype=np.uint32)
    
    parameters_buffer = create_buffer_32b(device, parameters_data)
    dparameters_buffer = create_buffer_32b(device, np.zeros_like(parameters_data))
    indices_buffer = create_buffer_32b(device, indices_data.astype(np.uint32))
    
    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "parameters": parameters_buffer,
                "dparameters": dparameters_buffer,
                "indices": indices_buffer,
                "count": batch_size,
            }
        },
    )
    
    parameter_derivatives = dparameters_buffer.to_numpy().view(np.float32).reshape(parameters_data.shape)
    
    parameters_torch = torch.tensor(parameters_data, requires_grad=True)
    indices_torch = torch.tensor(indices_data.astype(np.int64))
    
    output = feature_bank_pytorch(parameters_torch, indices_torch, features)
    grad_output = torch.ones_like(output)
    output.backward(grad_output)
    
    expected_parameter_derivatives = parameters_torch.grad.detach().numpy()
    
    assert_close(parameter_derivatives, expected_parameter_derivatives, rtol=1e-5, atol=1e-6)
