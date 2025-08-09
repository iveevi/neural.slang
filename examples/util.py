import slangpy as spy
import numpy as np


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