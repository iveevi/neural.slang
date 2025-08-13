from __future__ import annotations
import trimesh
import slangpy as spy
import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import polyscope as ps
from scipy.ndimage import gaussian_filter
from common import *
from ..networks.addresses import Network
from ..render_util import *

class Pipeline:
    @staticmethod
    def load_specialization_module(device: spy.Device, network: Network):
        source = f"""
        export static const int Hidden = {network.hidden};
        export static const int HiddenLayers = {network.hidden_layers};
        export static const int Levels = {network.levels};
        """
        return device.load_module_from_source("specialization", source)

    # TODO: move to common/device.py
    @staticmethod
    def load_pipeline(device: spy.Device, module: spy.SlangModule, extra: List[spy.SlangModule], entry_point: str):
        program = device.link_program(
            modules=[module] + extra,
            entry_points=[module.entry_point(entry_point)],
        )
        return device.create_compute_pipeline(program)

    def __init__(self, device: spy.Device, network: Network):
        SOURCE = ROOT / "examples" / "slang" / "neural_sdf.slang"
        
        self.device = device
        self.module = device.load_module(str(SOURCE))
        self.specialization_module = self.load_specialization_module(device, network)

        self.reference_pipeline = self.load_pipeline(device, self.module, [self.specialization_module], "reference")
        self.neural_pipeline = self.load_pipeline(device, self.module, [self.specialization_module], "neural")
        self.sample_neural_pipeline = self.load_pipeline(device, self.module, [self.specialization_module], "sample_neural")
        self.sample_reference_pipeline = self.load_pipeline(device, self.module, [self.specialization_module], "sample_reference")
        self.gradient_pipeline = self.load_pipeline(device, self.module, [self.specialization_module], "gradient")
        self.optimize_pipeline = self.load_pipeline(device, self.module, [self.specialization_module], "optimize")

    # def render_reference(self, mesh: TriangleMesh, rayframe: RayFrame, target_texture: spy.Texture):
    #     command_encoder = self.device.create_command_encoder()

    #     with command_encoder.begin_compute_pass() as cmd:
    #         shader_object = cmd.bind_pipeline(self.reference_pipeline)
    #         cursor = spy.ShaderCursor(shader_object)
    #         cursor.mesh = mesh.dict()
    #         cursor.rayFrame = rayframe.dict()
    #         cursor.targetTexture = target_texture
    #         cursor.targetResolution = (target_texture.width, target_texture.height)
    #         cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))
        
    #     self.device.submit_command_buffer(command_encoder.finish())

    def render_neural(self, network: Network, rayframe: RayFrame, target_texture: spy.Texture):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.neural_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.network = network.dict()
            cursor.rayFrame = rayframe.dict()
            cursor.targetTexture = target_texture
            cursor.targetResolution = (target_texture.width, target_texture.height)
            cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    def sample_neural(self, network: Network, sample_buffer: spy.Buffer, sdf_buffer: spy.Buffer, sample_count: int):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.sample_neural_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.network = network.dict()
            cursor.sampleBuffer = sample_buffer
            cursor.sdfBuffer = sdf_buffer
            cmd.dispatch(thread_count=(sample_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())
        
    # def sample_reference(self, mesh: TriangleMesh, sample_buffer: spy.Buffer, sdf_buffer: spy.Buffer, sample_count: int):
    #     command_encoder = self.device.create_command_encoder()
        
    #     with command_encoder.begin_compute_pass() as cmd:
    #         shader_object = cmd.bind_pipeline(self.sample_reference_pipeline)
    #         cursor = spy.ShaderCursor(shader_object)
    #         cursor.mesh = mesh.dict()
    #         cursor.sampleBuffer = sample_buffer
    #         cursor.sdfBuffer = sdf_buffer
    #         cmd.dispatch(thread_count=(sample_count, 1, 1))

    #     self.device.submit_command_buffer(command_encoder.finish())

    def evaluate_gradient(self, network: Network, sample_buffer: spy.Buffer, sdf_buffer: spy.Buffer, sample_count: int):
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.gradient_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.network = network.dict()
            cursor.sampleBuffer = sample_buffer
            cursor.sdfBuffer = sdf_buffer
            cursor.lossBoost = 1.0 / sample_count
            cmd.dispatch(thread_count=(sample_count, 1, 1))
        
        self.device.submit_command_buffer(command_encoder.finish())

    def optimize(self, network: Network):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.optimize_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.network = network.dict()
            cmd.dispatch(thread_count=(network.parameter_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    # def layout_of(self, name: str) -> spy.TypeLayoutReflection:
    #     type_layout = self.module.layout.find_type_by_name(name)
    #     return self.module.layout.get_type_layout(type_layout)


# TODO: design a main method agnostic to the pipeline... also for the other examples...
def main():
    mesh = trimesh.load_mesh(ROOT / "resources" /"suzanne.obj")

    print("max", mesh.vertices.max(axis=0))
    print("min", mesh.vertices.min(axis=0))

    device = create_device()
    
    network = Network(device, hidden=32, hidden_layers=2, levels=8, input=3, output=1)
    pipeline = Pipeline(device, network)
    mesh = TriangleMesh.new(device, mesh)

    # Allocate loss buffer
    SAMPLE_COUNT = 1 << 14
    sample_buffer = create_buffer_32b(device, np.zeros((SAMPLE_COUNT, 3), dtype=np.float32))
    sdf_buffer = create_buffer_32b(device, np.zeros(SAMPLE_COUNT, dtype=np.float32))

    ps.init()
    
    length = lambda x: np.linalg.norm(x, axis=1)

    history = []
    for frame in tqdm.trange(1000, desc="training"):
        pipeline.sample_neural(network, sample_buffer, sdf_buffer, SAMPLE_COUNT)
        sdf_neural = sdf_buffer.to_numpy().view(np.float32)
        
        samples = 3 * (np.random.rand(SAMPLE_COUNT, 3).astype(np.float32) * 2 - 1)
        sample_buffer.copy_from_numpy(samples)
        
        sdf = length(samples) - 1.0
        sdf_buffer.copy_from_numpy(sdf)
        
        loss = np.square(sdf_neural - sdf).mean()
        history.append(loss)
        
        # cloud = ps.register_point_cloud("samples", samples)
        # cloud.add_scalar_quantity("sdf", sdf)
        # ps.show()

        pipeline.evaluate_gradient(network, sample_buffer, sdf_buffer, SAMPLE_COUNT)
        pipeline.optimize(network)

    history = np.array(history)
    sns.lineplot(history, alpha=0.5)

    smoothed = gaussian_filter(history, 2.5)
    sns.lineplot(smoothed, linewidth=2.5)

    plt.yscale("log")
    plt.show()

    # Sample and visualize the learned SDF
    x = np.linspace(-1, 1, 64) * 2
    y = np.linspace(-1, 1, 64) * 2
    z = np.linspace(-1, 1, 64) * 2
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")
    
    samples = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    sdf = np.zeros(samples.shape[0])
    print("np samples", samples.shape)
    print("np sdf", sdf.shape)

    samples = create_buffer_32b(device, samples.astype(np.float32), 3)
    sdf = create_buffer_32b(device, sdf.astype(np.float32), 1)
    
    pipeline.sample_neural(network, samples, sdf, SAMPLE_COUNT)

    samples_np = samples.to_numpy().view(np.float32).reshape(-1, 3)
    sdf_np = sdf.to_numpy().view(np.float32)
    print("samples", samples_np.shape)
    print("sdf", sdf_np.shape)

    ps.init()
    cloud = ps.register_point_cloud("samples", samples_np)
    cloud.add_scalar_quantity("sdf", sdf_np)
    ps.show()
    
    # # Visualize the reference mesh
    # pipeline.sample_reference(mesh, samples, sdf, SAMPLE_COUNT)
    
    # samples_np = samples.to_numpy().view(np.float32).reshape(-1, 3)
    # sdf_np = sdf.to_numpy().view(np.float32)
    
    # print("samples", samples_np.shape)
    # print("sdf", sdf_np.shape)
    
    # cloud = ps.register_point_cloud("samples", samples_np)
    # cloud.add_scalar_quantity("sdf", sdf_np)
    # ps.show()

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
    camera = Camera.new(aspect_ratio=aspect_ratio)
    camera.transform.position = np.array((0.0, 0.0, 5.0))
    
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

        # pipeline.render_reference(mesh, camera.rayframe(), texture)
        pipeline.render_neural(network, camera.rayframe(), texture)

        # pipeline.evaluate_gradient(network, mesh, loss_buffer, SAMPLE_COUNT, frame)
        # losses = loss_buffer.to_numpy().view(np.float32)
        # history.append(losses.mean())
        # print("losses", losses.mean())

        # pipeline.optimize(network)

        command_encoder = device.create_command_encoder()
        command_encoder.blit(image, texture)
        command_encoder.set_texture_state(image, spy.ResourceState.present)
        device.submit_command_buffer(command_encoder.finish())
        
        surface.present()
        
        frame += 1

if __name__ == "__main__":
    sns.set_theme()
    sns.set_palette("pastel")

    main()