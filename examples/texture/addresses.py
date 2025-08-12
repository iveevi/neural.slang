import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import slangpy as spy
import pathlib

from PIL import Image
from tqdm import tqdm

from common.util import *
from ..network_with_addresses import Network, TrainingPipeline





def main():
    image = Image.open(ROOT / "examples" / "media" / "texture-128.png")
    image = np.array(image)
    image = image[..., :3].astype(np.float32) / 255.0
    print(image.shape)

    # TODO: take stochastic samples from uv
    uv = np.linspace(0, 1, image.shape[0])
    uv = np.stack(np.meshgrid(uv, uv, indexing='xy'), axis=-1)
    uv = uv.reshape(-1, 2)
    uv = uv.astype(np.float32)
    print(uv.shape)

    device = create_device()

    network = Network(device,
        hidden=64,
        hidden_layers=3,
        levels=8,
        input=2,
        output=3,
    )

    pipeline = TrainingPipeline(device, network)

    input_buffer = network.input_vec(uv)
    texture_buffer = network.output_vec(image.reshape(-1, 3))
    output_buffer = network.output_vec(np.zeros_like(image.reshape(-1, 3)))

    pipeline.forward(network, input_buffer, output_buffer)
    output = output_buffer.to_numpy().view(np.float32).reshape(image.shape)

    # Training loop
    history = []
    for _ in tqdm(range(1000)):
        pipeline.forward(network, input_buffer, output_buffer)
        output = output_buffer.to_numpy().view(np.float32).reshape(image.shape)
        loss = np.mean(np.square(output - image))
        history.append(loss)

        pipeline.backward(network, input_buffer, texture_buffer)
        pipeline.optimize(network)

    pipeline.forward(network, input_buffer, output_buffer)
    output = output_buffer.to_numpy().view(np.float32).reshape(image.shape)

    _, ax = plt.subplots(2, 1)
    ax[0].imshow(np.clip(output, 0, 1))
    ax[0].axis('off')
    ax[1].plot(history)
    ax[1].set_yscale('log')
    plt.show()

if __name__ == "__main__":
    sns.set_theme()
    sns.set_palette("pastel")

    main()
