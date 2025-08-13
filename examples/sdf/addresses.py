import trimesh
import slangpy as spy
import numpy as np
from dataclasses import dataclass
from common import *


@dataclass
class RayFrame:
    origin: np.ndarray
    lower_left: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray
    
    def dict(self):
        return {
            "origin": spy.float3(self.origin.tolist()),
            "lower_left": spy.float3(self.lower_left.tolist()),
            "horizontal": spy.float3(self.horizontal.tolist()),
            "vertical": spy.float3(self.vertical.tolist()),
        }


@dataclass
class Transform:
    position: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray

    @staticmethod
    def new(position: np.ndarray = np.array((0.0, 0.0, 0.0)),
            rotation: np.ndarray = np.array((0.0, 0.0, 0.0)),
            scale: np.ndarray = np.array((1.0, 1.0, 1.0))):
        return Transform(position, rotation, scale)
    
    def _rotation_matrix(self) -> np.ndarray:
        rx, ry, rz = np.radians(self.rotation)
        
        cos_x, sin_x = np.cos(rx), np.sin(rx)
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        cos_z, sin_z = np.cos(rz), np.sin(rz)
        
        Rx = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        
        Ry = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        
        Rz = np.array([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx
    
    @property
    def forward(self) -> np.ndarray:
        rotation_matrix = self._rotation_matrix()
        return -rotation_matrix[:, 2]
    
    @property
    def right(self) -> np.ndarray:
        rotation_matrix = self._rotation_matrix()
        return rotation_matrix[:, 0]
    
    @property
    def up(self) -> np.ndarray:
        rotation_matrix = self._rotation_matrix()
        return rotation_matrix[:, 1]
    
    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rotation_matrix = self._rotation_matrix()
        right = rotation_matrix[:, 0]
        up = rotation_matrix[:, 1]
        forward = -rotation_matrix[:, 2]
        return right, up, forward


@dataclass
class Camera:
    transform: Transform
    fov: float
    aspect_ratio: float
    near: float
    far: float
    
    @staticmethod
    def new(transform: Transform = Transform.new(),
            fov: float = 45,
            aspect_ratio: float = 1,
            near: float = 0.1,
            far: float = 100):
        return Camera(transform, fov, aspect_ratio, near, far)
    
    def rayframe(self) -> RayFrame:
        normalize = lambda x: x / np.linalg.norm(x)
        
        _, up, forward = self.transform.axes()
        
        vfov = np.radians(self.fov)
        h = np.tan(vfov / 2)
        vheight = 2 * h
        vwidth = vheight * self.aspect_ratio
        
        w = normalize(-forward)
        u = normalize(np.cross(up, w))
        v = np.cross(w, u)

        horizontal = u * vwidth
        vertical = v * vheight

        return RayFrame(
            origin=self.transform.position,
            lower_left=self.transform.position - horizontal/2 - vertical/2 - w,
            horizontal=horizontal,
            vertical=vertical,
        )


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

    def render_reference(self, mesh: dict, rayframe: RayFrame, target_texture: spy.Texture):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.reference_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mesh = mesh
            cursor.rayFrame = rayframe.dict()
            cursor.targetTexture = target_texture
            cursor.targetResolution = (target_texture.width, target_texture.height)
            cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))
        
        self.device.submit_command_buffer(command_encoder.finish())


def main():
    mesh = trimesh.load_mesh(ROOT / "resources" /"suzanne.obj")

    print("vertices", mesh.vertices.shape)
    print("faces", mesh.faces.shape)

    device = create_device()
    
    # Single triangle
    # vertex_buffer = create_buffer_32b(device, np.array((
    #     0.0, 0.0, 0.0,
    #     1.0, 0.0, 0.0,
    #     0.0, 1.0, 0.0,
    # ), dtype=np.float32), 3)
    # face_buffer = create_buffer_32b(device, np.array((
    #     0, 1, 2,
    # ), dtype=np.uint32), 3)

    vertex_buffer = create_buffer_32b(device, mesh.vertices.astype(np.float32), 3)
    face_buffer = create_buffer_32b(device, mesh.faces.astype(np.uint32), 3)

    mesh = {
        "vertices": vertex_buffer,
        "triangles": face_buffer,
        "triangleCount": len(mesh.faces),
        # "triangleCount": 1,
    }

    print("vertex_buffer", vertex_buffer.size)
    print("face_buffer", face_buffer.size)

    pipeline = Pipeline(device)

    # Define consistent dimensions
    WINDOW_WIDTH = 512
    WINDOW_HEIGHT = 512
    
    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba8_unorm,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        data=None,
    )
    
    # Calculate aspect ratio from actual dimensions
    aspect_ratio = WINDOW_WIDTH / WINDOW_HEIGHT
    camera = Camera.new(aspect_ratio=1)
    
    window = spy.Window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    surface = device.create_surface(window)
    surface.configure(WINDOW_WIDTH, WINDOW_HEIGHT)
    
    def keyboard_handler(event: spy.KeyboardEvent):
        # TODO: map each key to a function...
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                window.close()
        
    window.on_keyboard_event = keyboard_handler
    
    # TODO: app class with custom render loop and event handling
    frame = 0
    while not window.should_close():
        window.process_events()
        
        camera.transform.position = np.array((0.0, 0.0, 2.0 + 10.0 * np.fmod(frame, 500) / 500.0))
        # print("camera", camera.transform.position)
        
        image = surface.acquire_next_image()

        pipeline.render_reference(mesh, camera.rayframe(), texture)
        
        command_encoder = device.create_command_encoder()
        command_encoder.blit(image, texture)
        command_encoder.set_texture_state(image, spy.ResourceState.present)
        device.submit_command_buffer(command_encoder.finish())
        
        surface.present()
        
        frame += 1


if __name__ == "__main__":
    main()