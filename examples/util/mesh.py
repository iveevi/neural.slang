import numpy as np
import trimesh
import slangpy as spy
from dataclasses import dataclass
from common.buffer import create_buffer_32b


@dataclass
class TriangleMesh:
    vertices: spy.Buffer
    normals: spy.Buffer
    uvs: spy.Buffer
    triangles: spy.Buffer
    triangle_count: int
    vertex_count: int

    def blas(self, device: spy.Device):
        blas_geometry_desc = spy.AccelerationStructureBuildInputTriangles()
        blas_geometry_desc.flags = spy.AccelerationStructureGeometryFlags.opaque
        blas_geometry_desc.vertex_buffers = [ spy.BufferOffsetPair(self.vertices) ]
        blas_geometry_desc.vertex_format = spy.Format.rgb32_float
        blas_geometry_desc.vertex_count = self.vertex_count
        blas_geometry_desc.vertex_stride = 3 * 4
        blas_geometry_desc.index_buffer = self.triangles
        blas_geometry_desc.index_format = spy.IndexFormat.uint32
        blas_geometry_desc.index_count = 3 * self.triangle_count

        blas_build_desc = spy.AccelerationStructureBuildDesc()
        blas_build_desc.inputs = [ blas_geometry_desc ]

        blas_sizes = device.get_acceleration_structure_sizes(blas_build_desc)

        blas_scratch_buffer = device.create_buffer(
            size=blas_sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access,
        )

        blas_buffer = device.create_buffer(
            size=blas_sizes.acceleration_structure_size,
            usage=spy.BufferUsage.acceleration_structure,
        )

        blas = device.create_acceleration_structure(
            size=blas_buffer.size,
        )

        command_encoder = device.create_command_encoder()
        command_encoder.build_acceleration_structure(
            desc=blas_build_desc,
            dst=blas,
            src=None,
            scratch_buffer=blas_scratch_buffer,
        )
        device.submit_command_buffer(command_encoder.finish())
        
        return blas

    @staticmethod
    def new(device: spy.Device, mesh: trimesh.Trimesh):
        vertices = create_buffer_32b(
            device,
            mesh.vertices.astype(np.float32),
            3,
            spy.BufferUsage.shader_resource | spy.BufferUsage.vertex_buffer,
        )
        
        normals = create_buffer_32b(
            device,
            mesh.vertex_normals.astype(np.float32),
            3,
            spy.BufferUsage.shader_resource | spy.BufferUsage.vertex_buffer,
        )
        
        if mesh.visual is not None and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uv_data = mesh.visual.uv.astype(np.float32)
            uv_data[:, 1] = 1.0 - uv_data[:, 1]  # Flip V coordinate
            uvs = create_buffer_32b(
                device,
                uv_data,
                2,
                spy.BufferUsage.shader_resource | spy.BufferUsage.vertex_buffer,
            )
        else:
            default_uvs = np.zeros((len(mesh.vertices), 2), dtype=np.float32)
            uvs = create_buffer_32b(
                device,
                default_uvs,
                2,
                spy.BufferUsage.shader_resource | spy.BufferUsage.vertex_buffer,
            )
        
        triangles = create_buffer_32b(
            device,
            mesh.faces.astype(np.uint32),
            3,
            spy.BufferUsage.shader_resource | spy.BufferUsage.index_buffer,
        )
        
        return TriangleMesh(vertices, normals, uvs, triangles, len(mesh.faces), len(mesh.vertices))

    def dict(self):
        return {
            "vertices": self.vertices,
            "normals": self.normals,
            "uvs": self.uvs,
            "triangles": self.triangles,
            "triangleCount": self.triangle_count,
            "vertexCount": self.vertex_count,
        }