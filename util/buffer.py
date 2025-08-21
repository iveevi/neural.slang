import slangpy as spy
import numpy as np


def create_buffer_32b(
    device: spy.Device,
    data: np.ndarray,
    elements_per_struct: int = 1,
    usage: spy.BufferUsage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
) -> spy.Buffer:
    return device.create_buffer(
        size=data.nbytes,
        struct_size=elements_per_struct * 4,
        usage=usage,
        data=data,
    )


def create_batched_buffer_32b(device: spy.Device, batch_size: int, elements_per_struct: int = 1) -> spy.Buffer:
    return device.create_buffer(
        size=batch_size * elements_per_struct * 4,
        struct_size=elements_per_struct * 4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
