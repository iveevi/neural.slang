import slangpy as spy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_volumetric as pv
from scipy.ndimage import gaussian_filter
from common import *

from ..util import *
from ngp import *


HERE = ROOT / "examples" / "sdf"


# TODO: move to main as a mlp agnostic pipeline
# TODO: decorators for extracting each part of the pipeline -- reflection to get the global variables
class Pipeline:
    @staticmethod
    def load_specialization_module(device: spy.Device, mlp: MLP, grid: Grid):
        source = f"""
        export static const int Hidden = {mlp.hidden};
        export static const int HiddenLayers = {mlp.hidden_layers};
        export static const int Features = {grid.features};
        """
        return device.load_module_from_source("specialization", source)

    def __init__(self, device: spy.Device, mlp: MLP, grid: Grid):
        SOURCE = HERE / "slang" / "main.slang"
        
        self.device = device
        self.module = device.load_module(str(SOURCE))
        self.specialization_module = self.load_specialization_module(device, mlp, grid)

        self.render_heatmap_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "render_heatmap")
        self.render_normal_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "render_normal")

        self.backward_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "backward")
        
    def render_heatmap(self, mlp: MLP, grid: Grid, mldg: MultiLevelDenseGrid, rayframe: RayFrame, target_texture: spy.Texture):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.render_heatmap_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.scene.mlp = mlp.dict()
            cursor.scene.grid = grid.dict()
            cursor.scene.mldg = mldg.dict()
            cursor.rayFrame = rayframe.dict()
            cursor.targetTexture = target_texture
            cursor.targetResolution = (target_texture.width, target_texture.height)
            cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))

        self.device.submit_command_buffer(command_encoder.finish())
        
    def render_normal(self, mlp: MLP, grid: Grid, mldg: MultiLevelDenseGrid, rayframe: RayFrame, target_texture: spy.Texture):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.render_normal_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.scene.mlp = mlp.dict()
            cursor.scene.grid = grid.dict()
            cursor.scene.mldg = mldg.dict()
            cursor.rayFrame = rayframe.dict()
            cursor.targetTexture = target_texture
            cursor.targetResolution = (target_texture.width, target_texture.height)
            cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    def backward(self, mlp: MLP, grid: Grid, mldg: MultiLevelDenseGrid, input_buffer: spy.Buffer, expected_buffer: spy.Buffer, loss_buffer: spy.Buffer, sample_count: int):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.backward_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.scene.mlp = mlp.dict()
            cursor.scene.grid = grid.dict()
            cursor.scene.mldg = mldg.dict()
            cursor.inputBuffer = input_buffer
            cursor.expectedBuffer = expected_buffer
            cursor.lossBuffer = loss_buffer
            cursor.boost = 1.0 / sample_count
            cmd.dispatch(thread_count=(sample_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())


def main():
    device = create_device()

    features = 2
    mlp = AddressBasedMLP.new(device, hidden=32, hidden_layers=2, input=3 * features, output=1)
    grid = DenseGrid.new(device, dimension=3, features=features, resolution=16)
    mldg = MultiLevelDenseGrid.new(device, dimension=3, features=features, resolutions=[16, 32, 64])
    rendering_pipeline = Pipeline(device, mlp, grid)

    optimizer = Adam(alpha=1e-2)
    mlp_optimizer_states = mlp.alloc_optimizer_states(device, optimizer)
    grid_optimizer_states = grid.alloc_optimizer_states(device, optimizer)
    mldg_optimizer_states = mldg.alloc_optimizer_states(device, optimizer)

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
    
    def loop(frame: Frame):
        nonlocal counter
        nonlocal history
        nonlocal alphas

        # TODO: mouse controlled orbital camera
        time = counter * 0.01
        camera.transform.position = 3.0 * np.array((np.cos(time), 0.0, np.sin(time)))
        camera.transform.look_at(np.array((0.0, 0.0, 0.0)))
        if not pause_rotation:
            counter += 1

        # Rendering
        if render_heatmap:
            rendering_pipeline.render_heatmap(mlp, grid, mldg, camera.rayframe(), texture)
        else:
            rendering_pipeline.render_normal(mlp, grid, mldg, camera.rayframe(), texture)

        # Training
        samples = (2 * np.random.rand(SAMPLE_COUNT, 3).astype(np.float32) - 1)
        sample_buffer.copy_from_numpy(samples)

        sdf = target_sdf(samples)
        sdf_buffer.copy_from_numpy(sdf)

        rendering_pipeline.backward(mlp, grid, mldg, sample_buffer, sdf_buffer, loss_buffer, SAMPLE_COUNT)

        loss = loss_buffer.to_numpy().view(np.float32).mean()
        history.append(loss)
        alphas.append(optimizer.alpha)

        mlp.update(optimizer, mlp_optimizer_states)
        grid.update(optimizer, grid_optimizer_states)
        mldg.update(optimizer, mldg_optimizer_states)

        frame.blit(texture)
        
        optimizer.alpha *= (1 - 1e-3)

    app.run(loop)

    # Plot loss
    sns.set_theme()
    sns.set_palette("pastel")
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
