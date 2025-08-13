from __future__ import annotations
import slangpy as spy
import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import perf_counter
import pytorch_volumetric as pv
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

        self.render_neural_pipeline = self.load_pipeline(device, self.module, [self.specialization_module], "render_neural")
        self.sample_neural_pipeline = self.load_pipeline(device, self.module, [self.specialization_module], "sample_neural")
        self.evaluate_gradients_pipeline = self.load_pipeline(device, self.module, [self.specialization_module], "evaluate_gradients")
        self.optimize_pipeline = self.load_pipeline(device, self.module, [self.specialization_module], "optimize")

    def render_neural(self, network: Network, rayframe: RayFrame, target_texture: spy.Texture):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.render_neural_pipeline)
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

    def evaluate_gradients(self, network: Network, sample_buffer: spy.Buffer, sdf_buffer: spy.Buffer, sample_count: int):
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.evaluate_gradients_pipeline)
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


# TODO: design a main method agnostic to the pipeline... also for the other examples...
def main():
    device = create_device()
    
    network = Network(device, hidden=32, hidden_layers=2, levels=8, input=3, output=1)
    pipeline = Pipeline(device, network)

    # Allocate loss buffer
    SAMPLE_COUNT = 1 << 14
    sample_buffer = create_buffer_32b(device, np.zeros((SAMPLE_COUNT, 3), dtype=np.float32))
    sdf_buffer = create_buffer_32b(device, np.zeros(SAMPLE_COUNT, dtype=np.float32))

    mesh_obj = pv.MeshObjectFactory(ROOT / "resources" / "bunny.obj")
    mesh_sdf = pv.MeshSDF(mesh_obj)

    def target_sdf(samples: np.ndarray) -> np.ndarray:
        return mesh_sdf(samples)[0].numpy()

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

    context = spy.ui.Context(device)
    
    def keyboard_handler(event: spy.KeyboardEvent):
        # TODO: map each key to a function...
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                window.close()

    def mouse_handler(event: spy.MouseEvent):
        context.handle_mouse_event(event)
        
    window.on_keyboard_event = keyboard_handler
    window.on_mouse_event = mouse_handler
    
    # TODO: app class with custom render loop and event handling
    frame = 0
    history = []

    ui_window = spy.ui.Window(context.screen, 'Info')
    loss_text = spy.ui.Text(ui_window)
    time_text = spy.ui.Text(ui_window)

    last_time = perf_counter()

    while not window.should_close():
        window.process_events()
        
        # TODO: mouse controlled orbital camera

        time = frame * 0.01
        radius = 5.0
        center = np.array((0.0, 0.0, 0.0))
        
        camera.transform.position = np.array((
            radius * np.cos(time),
            0.0,
            radius * np.sin(time)
        ))
        
        camera.transform.look_at(center)
        
        image = surface.acquire_next_image()

        pipeline.render_neural(network, camera.rayframe(), texture)

        for _ in range(10):
            samples = 3 * (np.random.rand(SAMPLE_COUNT, 3).astype(np.float32) * 2 - 1)
            sample_buffer.copy_from_numpy(samples)

            pipeline.sample_neural(network, sample_buffer, sdf_buffer, SAMPLE_COUNT)
            sdf_neural = sdf_buffer.to_numpy().view(np.float32)

            sdf = target_sdf(samples)
            sdf_buffer.copy_from_numpy(sdf)

            loss = np.square(sdf_neural - sdf).mean()
            history.append(loss)

            pipeline.evaluate_gradients(network, sample_buffer, sdf_buffer, SAMPLE_COUNT)
            pipeline.optimize(network)

        command_encoder = device.create_command_encoder()
        command_encoder.blit(image, texture)
        command_encoder.set_texture_state(image, spy.ResourceState.present)

        context.new_frame(WINDOW_WIDTH, WINDOW_HEIGHT)

        loss_text.text = f"loss: {history[-1]:.4f}"
        time_text.text = f"time: {perf_counter() - last_time:.2f}s"
        last_time = perf_counter()

        context.render(image, command_encoder)
        
        device.submit_command_buffer(command_encoder.finish())
        
        surface.present()
        
        frame += 1

    history = np.array(history)
    sns.lineplot(history, alpha=0.5)

    smoothed = gaussian_filter(history, 2.5)
    sns.lineplot(smoothed, linewidth=2.5)

    plt.yscale("log")
    plt.show()

if __name__ == "__main__":
    sns.set_theme()
    sns.set_palette("pastel")

    main()