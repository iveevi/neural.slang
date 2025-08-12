import trimesh
import slangpy as spy
from common import *


class Pipeline:
    def __init__(self, device: spy.Device):
        SOURCE = ROOT / "examples" / "slang" / "sdf_rendering.slang"
        
        self.device = device
        self.module = device.load_module(str(SOURCE))

        # Reference rendering kernel
        reference_entry_point = self.module.entry_point("reference")
        reference_program = device.link_program(
            modules=[self.module],
            entry_points=[reference_entry_point],
        )
        self.reference_pipeline = device.create_compute_pipeline(reference_program)

    @property
    def rayframe(self):
        return self.module.RayFrame


def main():
    mesh = trimesh.load_mesh(ROOT / "resources" /"bunny.obj")

    print('vertices', mesh.vertices.shape)
    print('faces', mesh.faces.shape)

    device = create_device()

    vertex_buffer = create_buffer_32b(device, mesh.vertices, 3)
    face_buffer = create_buffer_32b(device, mesh.faces, 3)

    print('vertex_buffer', vertex_buffer.size)
    print('face_buffer', face_buffer.size)

    pipeline = Pipeline(device)

    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba8_unorm,
        width=1024,
        height=1024,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        data=None,
    )

    print('rayframe', dir(pipeline.module))


if __name__ == "__main__":
    main()