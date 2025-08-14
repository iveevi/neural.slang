import argparse
import slangpy as spy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import perf_counter
import pytorch_volumetric as pv
from scipy.ndimage import gaussian_filter
from common import *
from ..networks.addresses import Network, TrainingPipeline
from ..render_util import *
from .addresses import RenderingPipeline

def main(Network, RenderingPipeline, TrainingPipeline):
    device = create_device()

    hidden = 32
    hidden_layers = 2
    
    network = Network(device, hidden=hidden, hidden_layers=hidden_layers, levels=0, input=3, output=1)
    rendering_pipeline = RenderingPipeline(device, network)
    training_pipeline = TrainingPipeline(device, network)

    # Allocate loss buffer
    SAMPLE_COUNT = 1 << 10
    sample_buffer = create_buffer_32b(device, np.zeros((SAMPLE_COUNT, 3), dtype=np.float32), 3)
    sdf_buffer = create_buffer_32b(device, np.zeros(SAMPLE_COUNT, dtype=np.float32))

    mesh_obj = pv.MeshObjectFactory(ROOT / "resources" / "spot.obj")
    mesh_sdf = pv.MeshSDF(mesh_obj)

    def target_sdf(samples: np.ndarray) -> np.ndarray:
        return mesh_sdf(samples)[0].numpy()
        # return np.linalg.norm(samples, axis=1) - 1.0

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
        camera.transform.position = 5.0 * np.array((np.cos(time), 0.0, np.sin(time)))
        camera.transform.look_at(np.array((0.0, 0.0, 0.0)))
        
        image = surface.acquire_next_image()

        # Rendering
        # TODO: render at a lower resolution than the window (render to texture, then interpolate texture)
        rendering_pipeline.render_neural(network, camera.rayframe(), texture)
        # TODO: heatmap of the number of steps in sphere tracing

        # Training
        samples = (2 * np.random.rand(SAMPLE_COUNT, 3).astype(np.float32) - 1)
        sample_buffer.copy_from_numpy(samples)

        training_pipeline.forward(network, sample_buffer, sdf_buffer)
        sdf_neural = sdf_buffer.to_numpy().view(np.float32)

        sdf = target_sdf(samples)
        sdf_buffer.copy_from_numpy(sdf)

        # loss = np.square(sdf_neural - sdf).mean()
        loss = np.abs(sdf_neural - sdf).mean()
        history.append(loss)

        training_pipeline.backward(network, sample_buffer, sdf_buffer)
        training_pipeline.optimize(network)

        # UI
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

    sns.lineplot(history, alpha=0.5, color="green")

    sns.lineplot(gaussian_filter(history, 5), linewidth=2.5, label="Slang", color="green")

    plt.yscale("log")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, choices=[
        "addresses",
    ], default="addresses")
    args = parser.parse_args()

    sns.set_theme()
    sns.set_palette("pastel")

    match args.network:
        case "addresses":
            from .addresses import Network, RenderingPipeline, TrainingPipeline
            main(Network, RenderingPipeline, TrainingPipeline)
        case _:
            raise ValueError(f"Invalid network: {args.network}")