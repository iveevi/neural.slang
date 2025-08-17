import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import slangpy as spy
import pathlib
from common import *
from examples.ngp import Optimizer, FeatureGrid


class BaseFeatureGridPipeline:
    def __init__(self, device: spy.Device, slang_file: str):
        HERE = pathlib.Path(__file__).parent
        SOURCE = HERE / "slang" / slang_file

        self.device = device
        self.module = device.load_module(str(SOURCE))

        self.forward_pipeline = create_compute_pipeline(device, self.module, [], "forward_pass")
        self.backward_pipeline = create_compute_pipeline(device, self.module, [], "backward_pass")
        self.update_pipeline = create_compute_pipeline(device, self.module, [], "update_parameters")

    def forward(self, feature_grid: FeatureGrid, input_buffer: spy.Buffer, output_buffer: spy.Buffer, sample_count: int):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.forward_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.featureGrid = feature_grid.dict()
            cursor.inputBuffer = input_buffer
            cursor.outputBuffer = output_buffer
            cursor.sampleCount = sample_count
            cmd.dispatch(thread_count=(sample_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    def backward(self, feature_grid: FeatureGrid, input_buffer: spy.Buffer, target_buffer: spy.Buffer, loss_buffer: spy.Buffer, sample_count: int):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.backward_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.featureGrid = feature_grid.dict()
            cursor.inputBuffer = input_buffer
            cursor.targetBuffer = target_buffer
            cursor.lossBuffer = loss_buffer
            cursor.sampleCount = sample_count
            cmd.dispatch(thread_count=(sample_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    def update(self, feature_grid: FeatureGrid, optimizer: Optimizer, optimizer_states: spy.Buffer):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.update_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.featureGrid = feature_grid.dict()
            cursor.optimizer = optimizer.dict()
            cursor.optimizerStates = optimizer_states
            cmd.dispatch(thread_count=(feature_grid.parameter_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())


class FeatureGridPipeline(BaseFeatureGridPipeline):
    @staticmethod
    def load_specialization_module(device: spy.Device, dimension: int, features: int):
        source = f"""
        export static const int Dimension = {dimension};
        export static const int Features = {features};
        """
        return device.load_module_from_source("specialization", source)

    def __init__(self, device: spy.Device, dimension: int, features: int):
        HERE = pathlib.Path(__file__).parent
        SOURCE = HERE / "slang" / "main.slang"

        self.device = device
        self.dimension = dimension
        self.features = features

        self.module = device.load_module(str(SOURCE))
        self.specialization_module = self.load_specialization_module(device, dimension, features)

        self.forward_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "forward_pass")
        self.backward_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "backward_pass")
        self.update_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "update_parameters")

    def forward(self, feature_grid: FeatureGrid, input_buffer: spy.Buffer, output_buffer: spy.Buffer, sample_count: int):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.forward_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.featureGrid = feature_grid.dict()
            cursor.inputBuffer = input_buffer
            cursor.outputBuffer = output_buffer
            cursor.sampleCount = sample_count
            cmd.dispatch(thread_count=(sample_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    def backward(self, feature_grid: FeatureGrid, input_buffer: spy.Buffer, target_buffer: spy.Buffer, loss_buffer: spy.Buffer, sample_count: int):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.backward_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.featureGrid = feature_grid.dict()
            cursor.inputBuffer = input_buffer
            cursor.targetBuffer = target_buffer
            cursor.lossBuffer = loss_buffer
            cursor.sampleCount = sample_count
            cmd.dispatch(thread_count=(sample_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    def optimize(self, feature_grid: FeatureGrid, optimizer: Optimizer):
        optimizer_states = create_buffer_32b(self.device, np.zeros(feature_grid.parameter_count, dtype=np.float32), 1)

        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.update_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.featureGrid = feature_grid.dict()
            cursor.optimizer = optimizer.dict()
            cursor.optimizerStates = optimizer_states
            cmd.dispatch(thread_count=(feature_grid.parameter_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    def get_output(self, output_buffer: spy.Buffer) -> np.ndarray:
        sample_count = output_buffer.size // (self.features * 4)  # features floats per sample, 4 bytes per float
        return output_buffer.to_numpy().view(np.float32).reshape(sample_count, self.features)


def plot_feature_grid_results(
    original_data,
    pytorch_initial_output,
    slang_initial_output,
    pytorch_output,
    slang_output,
    pytorch_losses,
    slang_losses,
    pytorch_params_history,
    slang_params_history,
    initial_params,
    resolution,
    data_shape,
    is_2d=False
):
    sns.set_theme()
    sns.set_palette("pastel")

    if len(data_shape) == 2:  # 2D case
        height, width = data_shape
        sample_count = height * width

        # Plot results
        fig, axs = plt.subplots(2, 5, figsize=(25, 10))
        axs = axs.flatten()

        # Original texture
        axs[0].set_title("Original Texture")
        axs[0].imshow(original_data)
        axs[0].axis('off')

        # PyTorch initial reconstruction
        axs[1].set_title("PyTorch Initial")
        pytorch_initial_recon = np.clip(pytorch_initial_output.reshape(height, width, 3), 0, 1)
        axs[1].imshow(pytorch_initial_recon)
        axs[1].axis('off')

        # SlangPy initial reconstruction
        axs[2].set_title("SlangPy Initial")
        slang_initial_recon = np.clip(slang_initial_output.reshape(height, width, 3), 0, 1)
        axs[2].imshow(slang_initial_recon)
        axs[2].axis('off')

        # PyTorch final reconstruction
        axs[3].set_title("PyTorch Final")
        pytorch_recon = np.clip(pytorch_output.reshape(height, width, 3), 0, 1)
        axs[3].imshow(pytorch_recon)
        axs[3].axis('off')

        # SlangPy final reconstruction
        axs[4].set_title("SlangPy Final")
        slang_recon = np.clip(slang_output.reshape(height, width, 3), 0, 1)
        axs[4].imshow(slang_recon)
        axs[4].axis('off')

        # Loss comparison
        axs[5].set_title("Training Loss")
        axs[5].plot(pytorch_losses, label='PyTorch', linewidth=2)
        axs[5].plot(slang_losses, label='SlangPy', linewidth=2)
        axs[5].set_xlabel('Epoch')
        axs[5].set_ylabel('Loss')
        axs[5].legend()
        axs[5].grid(True, alpha=0.3)
        axs[5].set_yscale('log')

        # Output deltas over time
        output_delta = np.mean(np.abs(pytorch_output - slang_output))
        output_deltas = [output_delta] * len(pytorch_params_history)
        axs[6].set_title("Output Deltas")
        axs[6].plot(output_deltas, label='Output', linewidth=2)
        axs[6].set_xlabel('Epoch')
        axs[6].set_ylabel('Mean Absolute Difference')
        axs[6].legend()
        axs[6].grid(True, alpha=0.3)
        axs[6].set_yscale('log')

        # Parameter deltas over time
        param_deltas = []
        for i in range(min(len(pytorch_params_history), len(slang_params_history))):
            pytorch_params_flat = pytorch_params_history[i].flatten()
            param_delta = np.mean(np.abs(pytorch_params_flat - slang_params_history[i]))
            param_deltas.append(param_delta)

        axs[7].set_title("Parameter Deltas")
        axs[7].plot(param_deltas, label='Parameters', linewidth=2)
        axs[7].set_xlabel('Epoch')
        axs[7].set_ylabel('Mean Absolute Difference')
        axs[7].legend()
        axs[7].grid(True, alpha=0.3)
        axs[7].set_yscale('log')

        # Feature grid visualization (first channel)
        axs[8].set_title("Feature Grid (R channel)")
        pytorch_grid_viz = pytorch_params_history[-1][:, :, 0]
        slang_grid_viz = slang_params_history[-1].reshape(resolution, resolution, 3)[:, :, 0]
        axs[8].imshow(pytorch_grid_viz, cmap='viridis')
        axs[8].axis('off')

        # Difference visualization
        axs[9].set_title("PyTorch - SlangPy Difference")
        diff = pytorch_recon - slang_recon
        diff_img = axs[9].imshow(np.sum(np.abs(diff), axis=-1), cmap='RdBu')
        fig.colorbar(diff_img, ax=axs[9], fraction=0.046, pad=0.04)
        axs[9].axis('off')

    elif len(data_shape) == 3:  # 3D case
        depth, height, width = data_shape
        sample_count = depth * height * width

        # Plot results
        fig, axs = plt.subplots(2, 5, figsize=(25, 10))
        axs = axs.flatten()

        # Show skewed slice of original volume for more interesting visualization
        def extract_skewed_slice(volume_data, slice_idx=None):
            if slice_idx is None:
                slice_idx = depth // 2

            # Create a skewed slice by taking different z-indices for different x,y positions
            slice_data = np.zeros((height, width, 3))
            for i in range(height):
                for j in range(width):
                    # Skew the slice: z varies with x and y
                    z_idx = int((slice_idx + i * 0.3 + j * 0.2) % depth)
                    slice_data[i, j] = volume_data[z_idx, i, j]
            return slice_data

        skewed_slice = extract_skewed_slice(original_data)
        axs[0].set_title("Original Volume (Skewed Slice)")
        axs[0].imshow(skewed_slice)
        axs[0].axis('off')

        # PyTorch initial reconstruction (skewed slice)
        axs[1].set_title("PyTorch Initial")
        pytorch_initial_recon = np.clip(pytorch_initial_output.reshape(depth, height, width, 3), 0, 1)
        pytorch_skewed = extract_skewed_slice(pytorch_initial_recon)
        axs[1].imshow(pytorch_skewed)
        axs[1].axis('off')

        # SlangPy initial reconstruction (skewed slice)
        axs[2].set_title("SlangPy Initial")
        slang_initial_recon = np.clip(slang_initial_output.reshape(depth, height, width, 3), 0, 1)
        slang_skewed = extract_skewed_slice(slang_initial_recon)
        axs[2].imshow(slang_skewed)
        axs[2].axis('off')

        # PyTorch final reconstruction (skewed slice)
        axs[3].set_title("PyTorch Final")
        pytorch_recon = np.clip(pytorch_output.reshape(depth, height, width, 3), 0, 1)
        pytorch_final_skewed = extract_skewed_slice(pytorch_recon)
        axs[3].imshow(pytorch_final_skewed)
        axs[3].axis('off')

        # SlangPy final reconstruction (skewed slice)
        axs[4].set_title("SlangPy Final")
        slang_recon = np.clip(slang_output.reshape(depth, height, width, 3), 0, 1)
        slang_final_skewed = extract_skewed_slice(slang_recon)
        axs[4].imshow(slang_final_skewed)
        axs[4].axis('off')

        # Loss comparison
        axs[5].set_title("Training Loss")
        axs[5].plot(pytorch_losses, label='PyTorch', linewidth=2)
        axs[5].plot(slang_losses, label='SlangPy', linewidth=2)
        axs[5].set_xlabel('Epoch')
        axs[5].set_ylabel('Loss')
        axs[5].legend()
        axs[5].grid(True, alpha=0.3)
        axs[5].set_yscale('log')

        # Output deltas over time
        output_delta = np.mean(np.abs(pytorch_output - slang_output))
        output_deltas = [output_delta] * len(pytorch_params_history)
        axs[6].set_title("Output Deltas")
        axs[6].plot(output_deltas, label='Output', linewidth=2)
        axs[6].set_xlabel('Epoch')
        axs[6].set_ylabel('Mean Absolute Difference')
        axs[6].legend()
        axs[6].grid(True, alpha=0.3)
        axs[6].set_yscale('log')

        # Parameter deltas over time
        param_deltas = []
        for i in range(min(len(pytorch_params_history), len(slang_params_history))):
            pytorch_params_flat = pytorch_params_history[i].flatten()
            param_delta = np.mean(np.abs(pytorch_params_flat - slang_params_history[i]))
            param_deltas.append(param_delta)

        axs[7].set_title("Parameter Deltas")
        axs[7].plot(param_deltas, label='Parameters', linewidth=2)
        axs[7].set_xlabel('Epoch')
        axs[7].set_ylabel('Mean Absolute Difference')
        axs[7].legend()
        axs[7].grid(True, alpha=0.3)
        axs[7].set_yscale('log')

        # Feature grid visualization (middle slice, first channel)
        axs[8].set_title("Feature Grid (R channel, middle slice)")
        pytorch_grid_viz = pytorch_params_history[-1][resolution//2, :, :, 0]
        slang_grid_viz = slang_params_history[-1].reshape(resolution, resolution, resolution, 3)[resolution//2, :, :, 0]
        axs[8].imshow(pytorch_grid_viz, cmap='viridis')
        axs[8].axis('off')

        # Difference visualization (skewed slice)
        axs[9].set_title("PyTorch - SlangPy Difference")
        diff = pytorch_final_skewed - slang_final_skewed
        diff_img = axs[9].imshow(np.sum(np.abs(diff), axis=-1), cmap='RdBu')
        fig.colorbar(diff_img, ax=axs[9], fraction=0.046, pad=0.04)
        axs[9].axis('off')

    else:  # 1D case
        # 1D plotting
        time = np.linspace(0, 1, len(original_data))

        # Plot results
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        axs = axs.flatten()

        # Initial reconstruction comparison
        axs[0].set_title("Initial Reconstruction")
        axs[0].plot(time, original_data, label='Target Signal', linewidth=2, alpha=0.8)
        axs[0].plot(time, pytorch_initial_output, label='PyTorch Initial', linewidth=2, alpha=0.8)
        axs[0].plot(time, slang_initial_output, label='SlangPy Initial', linewidth=2, alpha=0.8)
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Amplitude')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # Final reconstruction comparison
        axs[1].set_title("Final Reconstruction")
        axs[1].plot(time, original_data, label='Target Signal', linewidth=2, alpha=0.8)
        axs[1].plot(time, pytorch_output, label='PyTorch Final', linewidth=2, alpha=0.8)
        axs[1].plot(time, slang_output, label='SlangPy Final', linewidth=2, alpha=0.8)
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Amplitude')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        # Loss comparison
        axs[2].set_title("Training Loss")
        axs[2].plot(pytorch_losses, label='PyTorch', linewidth=2)
        axs[2].plot(slang_losses, label='SlangPy', linewidth=2)
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Loss')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)
        axs[2].set_yscale('log')

        # Output deltas over time
        output_delta = np.mean(np.abs(pytorch_output - slang_output))
        output_deltas = [output_delta] * len(pytorch_params_history)
        axs[3].set_title("Output Deltas")
        axs[3].plot(output_deltas, label='Output', linewidth=2)
        axs[3].set_xlabel('Epoch')
        axs[3].set_ylabel('Mean Absolute Difference')
        axs[3].legend()
        axs[3].grid(True, alpha=0.3)
        axs[3].set_yscale('log')

        # Parameter deltas over time
        param_deltas = []
        for i in range(min(len(pytorch_params_history), len(slang_params_history))):
            param_delta = np.mean(np.abs(pytorch_params_history[i] - slang_params_history[i]))
            param_deltas.append(param_delta)

        axs[4].set_title("Parameter Deltas")
        axs[4].plot(param_deltas, label='Parameters', linewidth=2)
        axs[4].set_xlabel('Epoch')
        axs[4].set_ylabel('Mean Absolute Difference')
        axs[4].legend()
        axs[4].grid(True, alpha=0.3)
        axs[4].set_yscale('log')

        # Feature grid elements (Initial)
        axs[5].set_title("Feature Grid Elements (Initial)")
        grid_indices = np.arange(resolution)
        axs[5].plot(grid_indices, initial_params, label='Initial', linewidth=2, alpha=0.8)
        axs[5].set_xlabel('Grid Index')
        axs[5].set_ylabel('Parameter Value')
        axs[5].legend()
        axs[5].grid(True, alpha=0.3)

        # Feature grid elements (Final)
        axs[6].set_title("Feature Grid Elements (Final)")
        grid_indices = np.arange(resolution)
        axs[6].plot(grid_indices, pytorch_params_history[-1], label='PyTorch', linewidth=2, alpha=0.8)
        axs[6].plot(grid_indices, slang_params_history[-1], label='SlangPy', linewidth=2, alpha=0.8)
        axs[6].set_xlabel('Grid Index')
        axs[6].set_ylabel('Parameter Value')
        axs[6].legend()
        axs[6].grid(True, alpha=0.3)

        # Initial vs Final comparison
        axs[7].set_title("Initial vs Final")
        axs[7].plot(time, original_data, label='Target', linewidth=2, alpha=0.8)
        axs[7].plot(time, pytorch_initial_output, label='PyTorch Initial', linewidth=2, alpha=0.8)
        axs[7].plot(time, pytorch_output, label='PyTorch Final', linewidth=2, alpha=0.8)
        axs[7].set_xlabel('Time')
        axs[7].set_ylabel('Amplitude')
        axs[7].legend()
        axs[7].grid(True, alpha=0.3)

        # Difference visualization
        axs[8].set_title("PyTorch - SlangPy Difference")
        diff = np.mean(np.abs(pytorch_output - slang_output), axis=-1)
        axs[8].plot(time, diff, label='Difference', linewidth=2, alpha=0.8)
        axs[8].set_xlabel('Time')
        axs[8].set_ylabel('Difference')
        axs[8].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return output_delta
