import slangpy as spy
import numpy as np
import torch.nn as nn


def create_buffer(
    device: spy.Device,
    data: np.ndarray,
    struct_size: int = 4,
    usage: spy.BufferUsage = spy.BufferUsage.shader_resource,
) -> spy.Buffer:
    return device.create_buffer(
        size=data.nbytes,
        struct_size=struct_size,
        usage=usage,
        data=data,
    )


def linear_to_numpy(linear: nn.Linear) -> np.ndarray:
    """Convert PyTorch Linear layer weights and bias to numpy array."""
    weights = linear.weight.cpu().detach().numpy().T
    bias = linear.bias.cpu().detach().numpy().reshape(1, -1)
    return np.ascontiguousarray(np.concatenate((weights, bias), axis=0).astype(np.float32))


def linear_gradients_to_numpy(linear: nn.Linear) -> np.ndarray:
    """Convert PyTorch Linear layer gradients to numpy array."""
    assert linear.weight.grad is not None
    assert linear.bias.grad is not None
    weights = linear.weight.grad.cpu().detach().numpy().T
    bias = linear.bias.grad.cpu().detach().numpy().reshape(1, -1)
    return np.ascontiguousarray(np.concatenate((weights, bias), axis=0).astype(np.float32))