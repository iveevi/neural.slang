import numpy as np
import trimesh
import slangpy as spy
from dataclasses import dataclass
from common.buffer import create_buffer_32b


@dataclass
class TriangleMesh:
    vertices: spy.Buffer
    triangles: spy.Buffer
    triangle_count: int

    @staticmethod
    def new(device: spy.Device, mesh: trimesh.Trimesh):
        vertices = create_buffer_32b(device, mesh.vertices.astype(np.float32), 3)
        triangles = create_buffer_32b(device, mesh.faces.astype(np.uint32), 3)
        return TriangleMesh(vertices, triangles, len(mesh.faces))

    def dict(self):
        return {
            "vertices": self.vertices,
            "triangles": self.triangles,
            "triangleCount": self.triangle_count,
        }