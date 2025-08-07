import numpy as np
import slangpy as spy


def create_buffer_for_data(device, data, struct_size, usage_type="shader_resource"):
    usage_map = {
        "shader_resource": spy.BufferUsage.shader_resource,
        "constant_buffer": spy.BufferUsage.constant_buffer,
    }
    
    return device.create_buffer(
        size=data.nbytes,
        struct_size=struct_size,
        usage=usage_map.get(usage_type, spy.BufferUsage.shader_resource),
        data=data,
    )


def create_output_buffer(device, batch_size, output_size, element_size=4):
    return device.create_buffer(
        size=batch_size * output_size * element_size,
        struct_size=output_size * element_size,
        usage=spy.BufferUsage.shader_resource,
    )