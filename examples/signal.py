import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import slangpy as spy
import pathlib
import argparse

from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d


ROOT = pathlib.Path(__file__).parent.parent.absolute()


def generate_random_signal(length: int) -> np.ndarray:
    signal = 2 * np.random.rand(length) - 1
    signal = gaussian_filter1d(signal, sigma=2)
    return signal


def main(address_mode: bool = True):
    length = 1024
    time = np.linspace(0, 1, length)
    signal = generate_random_signal(length)

    device = spy.create_device(
        spy.DeviceType.vulkan,
        enable_debug_layers=True,
        include_paths=[
            ROOT / "neural",
        ],
    )

    if address_mode:
        from .network_with_addresses import Network, Pipeline
    else:
        from .network_with_separate_buffers import Network, Pipeline

    network = Network(
        device,
        hidden=64,
        hidden_layers=2,
        levels=8,
        input=1,
        output=1,
    )

    pipeline = Pipeline(device, network)

    input = np.array(time, dtype=np.float32).reshape(-1, 1)
    signal = np.array(signal, dtype=np.float32).reshape(-1, 1)

    input_buffer = network.input_vec(input)
    signal_buffer = network.output_vec(signal)
    output_buffer = network.output_vec(np.zeros_like(signal))

    # Training loop
    history = []
    for _ in tqdm(range(1000)):
        pipeline.forward(network, input_buffer, output_buffer)
        output = output_buffer.to_numpy().view(np.float32).reshape(-1, 1)
        loss = np.mean(np.square(output - signal))
        history.append(loss)

        pipeline.backward(network, input_buffer, signal_buffer)
        pipeline.optimize(network)

    pipeline.forward(network, input_buffer, output_buffer)
    output = output_buffer.to_numpy().view(np.float32)

    _, ax = plt.subplots(2, 1)
    ax[0].plot(time, output, label='output')
    ax[0].plot(time, signal, label='signal')
    ax[0].legend()
    ax[1].plot(history, label='loss')
    ax[1].set_yscale('log')
    ax[1].legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address-mode', action='store_true', default=False)
    args = parser.parse_args()

    sns.set_theme()
    sns.set_palette("pastel")

    main(args.address_mode)
