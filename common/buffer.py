import slangpy as spy
import numpy as np
import pathlib
from typing import List, Optional


ROOT = pathlib.Path(__file__).parent.parent.absolute()


def create_device(
    additional_include_paths: Optional[List[pathlib.Path]] = None,
) -> spy.Device:
    device_types = [spy.DeviceType.metal, spy.DeviceType.vulkan]
    
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


def create_buffer_32b(
    device: spy.Device,
    data: np.ndarray,
    elements_per_struct: int = 1,
) -> spy.Buffer:
    return device.create_buffer(
        size=data.nbytes,
        struct_size=elements_per_struct * 4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=data,
    )


def create_batched_buffer_32b(device: spy.Device, batch_size: int, elements_per_struct: int = 1) -> spy.Buffer:
    return device.create_buffer(
        size=batch_size * elements_per_struct * 4,
        struct_size=elements_per_struct * 4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
