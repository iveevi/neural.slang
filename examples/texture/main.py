import argparse
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import slangpy as spy
from PIL import Image
from common import *
from ..util import *


HERE = ROOT / "examples" / "texture"


def main(Network, TrainingPipeline, RenderingPipeline):
    image = Image.open(ROOT / "resources" / "yellowstone.png")
    image = np.array(image)
    image = image[..., :3].astype(np.float32) / 255.0

    # Bilinear sampling
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

    network = Network(device,
        hidden=64,
        hidden_layers=2,
        levels=8,
        input=2,
        output=3,
    )

    training_pipeline = TrainingPipeline(device, network)
    rendering_pipeline = RenderingPipeline(device, network)

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

    history = []

    def loop(frame: Frame):
        rendering_pipeline.render_neural(network, target_texture)

        frame.cmd.blit(frame.image, target_texture)
        frame.cmd.set_texture_state(frame.image, spy.ResourceState.present)

        samples = np.random.rand(SAMPLE_COUNT, 2).astype(np.float32)
        colors = sample(samples).astype(np.float32)
        sample_buffer.copy_from_numpy(samples)

        training_pipeline.forward(network, sample_buffer, color_buffer)
        predicted = color_buffer.to_numpy().view(np.float32).reshape(SAMPLE_COUNT, 3)
        
        loss = np.mean(np.square(predicted - colors))
        history.append(loss)

        color_buffer.copy_from_numpy(colors)

        training_pipeline.backward(network, sample_buffer, color_buffer)
        training_pipeline.optimize(network)

    app.run(loop)

    # Plot loss
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
            from .addresses import Network, TrainingPipeline, RenderingPipeline
            main(Network, TrainingPipeline, RenderingPipeline)
        case _:
            raise ValueError(f"Invalid network: {args.network}")
