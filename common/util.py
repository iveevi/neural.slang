import slangpy as spy
import numpy as np
import torch.nn as nn
import pathlib
from typing import List, Optional


ROOT = pathlib.Path(__file__).parent.parent.absolute()


def create_device(
    additional_include_paths: Optional[List[pathlib.Path]] = None,
) -> spy.Device:
    device_types = [spy.DeviceType.vulkan, spy.DeviceType.metal]
    
    # Always include neural path
    neural_path = ROOT / "neural"
    include_paths = [neural_path]
    
    if additional_include_paths:
        include_paths.extend(additional_include_paths)
    
    last_error = None
    for device_type in device_types:
        try:
            return spy.create_device(
                device_type,
                enable_debug_layers=True,
                include_paths=include_paths,
            )
        except Exception as e:
            last_error = e
            continue
    
    raise RuntimeError(f"Failed to create device with any of {device_types}. Last error: {last_error}")


def create_float_buffer(
    device: spy.Device,
    data: np.ndarray,
) -> spy.Buffer:
    return device.create_buffer(
        size=data.nbytes,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=data,
    )


def create_float_tensor_buffer(
    device: spy.Device,
    data: np.ndarray,
    elements_per_struct: int,
) -> spy.Buffer:
    return device.create_buffer(
        size=data.nbytes,
        struct_size=elements_per_struct * 4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=data,
    )


def linear_to_numpy(linear: nn.Linear) -> np.ndarray:
    weights = linear.weight.cpu().detach().numpy().T
    bias = linear.bias.cpu().detach().numpy().reshape(1, -1)
    return np.ascontiguousarray(np.concatenate((weights, bias), axis=0).astype(np.float32))


def linear_gradients_to_numpy(linear: nn.Linear) -> np.ndarray:
    assert linear.weight.grad is not None
    assert linear.bias.grad is not None
    weights = linear.weight.grad.cpu().detach().numpy().T
    bias = linear.bias.grad.cpu().detach().numpy().reshape(1, -1)
    return np.ascontiguousarray(np.concatenate((weights, bias), axis=0).astype(np.float32))


def create_buffer_for_data(device: spy.Device, data: np.ndarray, struct_size: int, usage_type: str = "shader_resource") -> spy.Buffer:
    usage_map = {
        "shader_resource": spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        "constant_buffer": spy.BufferUsage.constant_buffer,
    }
    
    return device.create_buffer(
        size=data.nbytes,
        struct_size=struct_size,
        usage=usage_map.get(usage_type, spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access),
        data=data,
    )


def create_output_buffer(device: spy.Device, batch_size: int, output_size: int, element_size: int = 4) -> spy.Buffer:
    return device.create_buffer(
        size=batch_size * output_size * element_size,
        struct_size=output_size * element_size,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )