import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import slangpy as spy
from PIL import Image
from common import *
from ..util import *
from ..networks.addresses import Network, TrainingPipeline
from ngp import AddressBasedMLP, Adam


HERE = ROOT / "examples" / "texture"


class RenderingPipeline:
    @staticmethod
    def load_specialization_module(device: spy.Device, mlp: AddressBasedMLP, levels: int):
        source = f"""
        export static const int Hidden = {mlp.hidden};
        export static const int HiddenLayers = {mlp.hidden_layers};
        export static const int Levels = {levels};
        """
        return device.load_module_from_source("specialization", source)

    def __init__(self, device: spy.Device, mlp: AddressBasedMLP, levels: int):
        SOURCE = HERE / "slang" / "main.slang"
        
        self.device = device
        self.module = device.load_module(str(SOURCE))
        self.specialization_module = self.load_specialization_module(device, mlp, levels)

        self.render_neural_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "render_neural")
        self.backward_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "backward")

    def render_neural(self, mlp: AddressBasedMLP, target_texture: spy.Texture):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.render_neural_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mlp = mlp.dict()
            cursor.targetTexture = target_texture
            cursor.targetResolution = (target_texture.width, target_texture.height)
            cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    def backward(self, mlp: AddressBasedMLP, samples: spy.Buffer, expected: spy.Buffer, sample_count: int):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.backward_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mlp = mlp.dict()
            cursor.samples = samples
            cursor.expected = expected
            cursor.boost = 1.0 / sample_count
            cmd.dispatch(thread_count=[sample_count, 1, 1])

        self.device.submit_command_buffer(command_encoder.finish())

def main():
    image = Image.open(ROOT / "resources" / "yellowstone.png")
    image = np.array(image)
    image = image[..., :3].astype(np.float32) / 255.0

    # Bilinear sampling
    # TODO: just pass texture...
    def sample(uv: np.ndarray) -> np.ndarray:
        x = uv[..., 1] * (image.shape[1] - 1)
        y = uv[..., 0] * (image.shape[0] - 1)

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.ceil(x).astype(np.int32)
        y1 = np.ceil(y).astype(np.int32)

        dx = (x - x0)[..., None]
        dy = (y - y0)[..., None]

        c00 = image[x0, y0] * (1 - dx) * (1 - dy)
        c01 = image[x0, y1] * (1 - dx) * dy
        c10 = image[x1, y0] * dx * (1 - dy)
        c11 = image[x1, y1] * dx * dy

        return c00 + c01 + c10 + c11

    device = create_device()

    levels = 8
    optimizer = Adam()
    mlp = AddressBasedMLP.new(device, hidden=64, hidden_layers=2, input=4 * levels, output=3)
    mlp_optimizer_states = mlp.alloc_optimizer_states(device, optimizer)

    # training_pipeline = TrainingPipeline(device, network)
    pipeline = RenderingPipeline(device, mlp, 8)

    app = App(device)

    target_texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba8_unorm,
        width=app.width,
        height=app.height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        data=None,
    )

    SAMPLE_COUNT = 1 << 10
    sample_buffer = create_buffer_32b(device, np.zeros((SAMPLE_COUNT, 2), dtype=np.float32), 2)
    color_buffer = create_buffer_32b(device, np.zeros((SAMPLE_COUNT, 3), dtype=np.float32), 3)

    def loop(frame: Frame):
        pipeline.render_neural(mlp, target_texture)

        samples = np.random.rand(SAMPLE_COUNT, 2).astype(np.float32)
        colors = sample(samples).astype(np.float32)
        sample_buffer.copy_from_numpy(samples)
        color_buffer.copy_from_numpy(colors)

        pipeline.backward(mlp, sample_buffer, color_buffer, SAMPLE_COUNT)

        mlp.update(optimizer, mlp_optimizer_states)

        frame.blit(target_texture)

    app.run(loop)


if __name__ == "__main__":
    sns.set_theme()
    sns.set_palette("pastel")

    main()
