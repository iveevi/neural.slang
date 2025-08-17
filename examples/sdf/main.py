import slangpy as spy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_volumetric as pv
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
from common import *
from ..networks.addresses import Network
from ..util import *
from ..ngp.objects import MLP, Adam, FeatureGrid
from .addresses import *


@dataclass
class AddressBasedMLP(MLP):
    parameter_buffer: spy.Buffer
    gradient_buffer: spy.Buffer

    def dict(self):
        return {
            "parameterBuffer": self.parameter_buffer,
            "gradientBuffer": self.gradient_buffer,
        }


def main():
    device = create_device()

    hidden = 32
    hidden_layers = 2

    network = Network(device, hidden=hidden, hidden_layers=hidden_layers, levels=0, input=2, output=1)
    rendering_pipeline = RenderingPipeline(device, network)

    # TODO: proper constructor
    mlp = AddressBasedMLP(
        parameter_buffer=network.parameters,
        gradient_buffer=network.gradients,
    )

    optimizer = Adam(alpha=1e-2)
    grid = FeatureGrid.new(device, dimension=3, features=2, resolution=32)
    grid_optimizer_states = grid.alloc_optimizer_states(device, optimizer)

    # Allocate loss buffer
    SAMPLE_COUNT = 1 << 14
    sample_buffer = create_buffer_32b(device, np.zeros((SAMPLE_COUNT, 3), dtype=np.float32), 3)
    sdf_buffer = create_buffer_32b(device, np.zeros(SAMPLE_COUNT, dtype=np.float32))
    loss_buffer = create_buffer_32b(device, np.zeros(SAMPLE_COUNT, dtype=np.float32))

    mesh_obj = pv.MeshObjectFactory(str(ROOT / "resources" / "bunny.obj"))
    mesh_sdf = pv.MeshSDF(mesh_obj)

    def target_sdf(samples: np.ndarray) -> np.ndarray:
        return mesh_sdf(samples)[0].numpy()

    render_heatmap = True
    pause_rotation = False

    def keyboard_hook(event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.tab:
                nonlocal render_heatmap
                render_heatmap = not render_heatmap
            elif event.key == spy.KeyCode.space:
                nonlocal pause_rotation
                pause_rotation = not pause_rotation

    app = App(device, keyboard_hook=keyboard_hook)

    # Define consistent dimensions
    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba8_unorm,
        width=400,
        height=400,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        data=None,
    )

    # Calculate aspect ratio from actual dimensions
    aspect_ratio = app.width / app.height
    camera = Camera.new(aspect_ratio=aspect_ratio)
    camera.transform.position = np.array((0.0, 0.0, 5.0))

    history = []
    alphas = []
    counter = 0
    
    # TODO: decay the lr

    def loop(frame: Frame):
        nonlocal counter

        # TODO: mouse controlled orbital camera
        time = counter * 0.01
        camera.transform.position = 3.0 * np.array((np.cos(time), 0.0, np.sin(time)))
        camera.transform.look_at(np.array((0.0, 0.0, 0.0)))
        if not pause_rotation:
            counter += 1

        # Rendering
        if render_heatmap:
            rendering_pipeline.render_heatmap(mlp, grid, camera.rayframe(), texture)
        else:
            rendering_pipeline.render_normal(mlp, grid, camera.rayframe(), texture)

        # Training
        samples = (2 * np.random.rand(SAMPLE_COUNT, 3).astype(np.float32) - 1)
        sample_buffer.copy_from_numpy(samples)

        sdf = target_sdf(samples)
        sdf_buffer.copy_from_numpy(sdf)

        rendering_pipeline.backward(mlp, grid, sample_buffer, sdf_buffer, loss_buffer, SAMPLE_COUNT)

        loss = loss_buffer.to_numpy().view(np.float32).mean()
        history.append(loss)
        alphas.append(optimizer.alpha)

        rendering_pipeline.update_mlp(mlp, optimizer, network.optimizer_states, network.parameter_count)
        rendering_pipeline.update_grid(grid, optimizer, grid_optimizer_states, grid.parameter_count)

        frame.blit(texture)
        
        optimizer.alpha *= (1 - 1e-3)

    app.run(loop)

    # Plot loss
    history = np.array(history)
    alphas = np.array(alphas)
    sns.lineplot(history, alpha=0.5, color="green")
    sns.lineplot(gaussian_filter(history, 5), linewidth=2.5, label="Slang", color="green")
    sns.lineplot(alphas, alpha=0.5, color="blue")
    plt.yscale("log")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    sns.set_theme()
    sns.set_palette("pastel")

    main()
