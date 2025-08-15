import slangpy as spy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_volumetric as pv
from scipy.ndimage import gaussian_filter
from common import *
from ..networks.addresses import Network, TrainingPipeline
from ..util import *
from .addresses import RenderingPipeline

def main():
    device = create_device()
    
    if not device.has_feature(spy.Feature.bindless):
        raise ValueError("Bindless is not supported on this device")

    hidden = 32
    hidden_layers = 2
    
    network = Network(device, hidden=hidden, hidden_layers=hidden_layers, levels=0, input=3, output=1)
    rendering_pipeline = RenderingPipeline(device, network)
    training_pipeline = TrainingPipeline(device, network)
    
    # MLP buffer
    mlp_buffer = device.create_buffer(
        size=rendering_pipeline.mlp_layout.stride,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    
    mlp_buffer_cursor = spy.BufferCursor(rendering_pipeline.mlp_layout, mlp_buffer)
    # mlp_buffer_cursor[0].parameterBuffer = network.parameters.descriptor_handle_rw
    # mlp_buffer_cursor[0].gradientBuffer = network.gradients.descriptor_handle_rw
    mlp_buffer_cursor[0].layerAddressBuffer = network.layer_addresses.descriptor_handle_ro
    mlp_buffer_cursor[0].parameterCount = network.parameter_count

    # Allocate loss buffer
    SAMPLE_COUNT = 1 << 10
    sample_buffer = create_buffer_32b(device, np.zeros((SAMPLE_COUNT, 3), dtype=np.float32), 3)
    sdf_buffer = create_buffer_32b(device, np.zeros(SAMPLE_COUNT, dtype=np.float32))

    mesh_obj = pv.MeshObjectFactory(str(ROOT / "resources" / "spot.obj"))
    mesh_sdf = pv.MeshSDF(mesh_obj)

    def target_sdf(samples: np.ndarray) -> np.ndarray:
        return mesh_sdf(samples)[0].numpy()
        # return np.linalg.norm(samples, axis=1) - 1.0
        
    render_heatmap = True
    
    def keyboard_hook(event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.tab:
                nonlocal render_heatmap
                render_heatmap = not render_heatmap

    app = App(device, keyboard_hook=keyboard_hook)

    # Define consistent dimensions
    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba8_unorm,
        width=256,
        height=256,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        data=None,
    )
    
    # Calculate aspect ratio from actual dimensions
    aspect_ratio = app.width / app.height
    camera = Camera.new(aspect_ratio=aspect_ratio)
    camera.transform.position = np.array((0.0, 0.0, 5.0))
    
    history = []

    def loop(frame: Frame):
        # TODO: mouse controlled orbital camera
        time = frame.count[0] * 0.01
        camera.transform.position = 5.0 * np.array((np.cos(time), 0.0, np.sin(time)))
        camera.transform.look_at(np.array((0.0, 0.0, 0.0)))
        
        # Rendering
        # TODO: render at a lower resolution than the window (render to texture, then interpolate texture)
        if render_heatmap:
            rendering_pipeline.render_heatmap(network, camera.rayframe(), texture)
        else:
            rendering_pipeline.render_normal(network, camera.rayframe(), texture)

        # Training
        samples = (2 * np.random.rand(SAMPLE_COUNT, 3).astype(np.float32) - 1)
        sample_buffer.copy_from_numpy(samples)

        training_pipeline.forward(network, sample_buffer, sdf_buffer)
        sdf_neural = sdf_buffer.to_numpy().view(np.float32)

        sdf = target_sdf(samples)
        sdf_buffer.copy_from_numpy(sdf)

        loss = np.abs(sdf_neural - sdf).mean()
        history.append(loss)

        training_pipeline.backward(network, sample_buffer, sdf_buffer)
        training_pipeline.optimize(network)
        
        frame.blit(texture)

    app.run(loop)

    # Plot loss
    history = np.array(history)
    sns.lineplot(history, alpha=0.5, color="green")
    sns.lineplot(gaussian_filter(history, 5), linewidth=2.5, label="Slang", color="green")
    plt.yscale("log")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    sns.set_theme()
    sns.set_palette("pastel")

    main()