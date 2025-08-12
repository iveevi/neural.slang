import numpy as np
import torch.nn as nn


def linear_to_numpy(linear: nn.Linear) -> np.ndarray:
    weights = linear.weight.cpu().detach().numpy().T
    bias = linear.bias.cpu().detach().numpy().reshape(1, -1)
    return np.ascontiguousarray(np.concatenate((weights, bias), axis=0).astype(np.float32))


def linear_to_bindless_numpy(linear: nn.Linear, dtype: np.dtype = np.float32) -> tuple[np.ndarray, np.ndarray]:
    weights = linear.weight.cpu().detach().numpy().T.ravel().astype(dtype)
    bias = linear.bias.cpu().detach().numpy().ravel().astype(dtype)
    return np.ascontiguousarray(weights), np.ascontiguousarray(bias)


def linear_gradients_to_numpy(linear: nn.Linear) -> np.ndarray:
    assert linear.weight.grad is not None
    assert linear.bias.grad is not None
    weights = linear.weight.grad.cpu().detach().numpy().T
    bias = linear.bias.grad.cpu().detach().numpy().reshape(1, -1)
    return np.ascontiguousarray(np.concatenate((weights, bias), axis=0).astype(np.float32))
