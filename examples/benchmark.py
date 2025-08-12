import numpy as np
import seaborn as sns
import slangpy as spy
import pathlib
import torch
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import time
from functools import wraps
from collections import defaultdict

from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

from .network_with_separate_buffers import Network as SeparateBuffersNetwork, TrainingPipeline as SeparateBuffersTrainingPipeline
from .network_with_addresses import Network as AddressesNetwork, TrainingPipeline as AddressesTrainingPipeline
from .pytorch_networks import PyTorchNetwork
from common.util import *





class ProfilerState:
    def __init__(self):
        self.timing_data = defaultdict(list)
        self.iteration_count = 0
        self.skip_first_iteration = True
    
    def reset(self):
        self.timing_data.clear()
        self.iteration_count = 0


# Global profiler instance
profiler = ProfilerState()


def profile(func_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global profiler
            
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Skip recording the first iteration to avoid initialization overhead
            if not (profiler.skip_first_iteration and profiler.iteration_count == 0):
                profiler.timing_data[func_name].append(execution_time)
            
            return result
        return wrapper
    return decorator


def reset_profiler():
    global profiler
    profiler.reset()


def set_iteration_count(count):
    global profiler
    profiler.iteration_count = count


def plot_profiling_results():
    # Prepare data
    phases = ['forward', 'backward', 'optimize', 'inference']
    pytorch_medians = []
    slangpy_medians = []
    pytorch_q1 = []
    pytorch_q3 = []
    slangpy_q1 = []
    slangpy_q3 = []
    
    # Calculate statistics for each phase
    for phase in phases:
        pytorch_key = f"pytorch_{phase}"
        slangpy_key = f"slangpy_{phase}"
        
        # Check if data exists for this phase
        if pytorch_key in profiler.timing_data and slangpy_key in profiler.timing_data:
            pytorch_times = np.array(profiler.timing_data[pytorch_key])
            slangpy_times = np.array(profiler.timing_data[slangpy_key])
            
            # Calculate quartiles
            pytorch_medians.append(np.median(pytorch_times))
            slangpy_medians.append(np.median(slangpy_times))
            
            pytorch_q1.append(np.percentile(pytorch_times, 25))
            pytorch_q3.append(np.percentile(pytorch_times, 75))
            slangpy_q1.append(np.percentile(slangpy_times, 25))
            slangpy_q3.append(np.percentile(slangpy_times, 75))
        else:
            # If no data, append zeros
            pytorch_medians.append(0)
            slangpy_medians.append(0)
            pytorch_q1.append(0)
            pytorch_q3.append(0)
            slangpy_q1.append(0)
            slangpy_q3.append(0)
    
    # Create subplots - 4x2 layout to accommodate all time series including inference
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 24), layout='constrained')
    title = f'PyTorch vs SlangPy Performance Comparison'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Time series for forward pass
    iterations = range(len(profiler.timing_data['pytorch_forward']))
    ax1.plot(iterations, profiler.timing_data['pytorch_forward'], label='PyTorch', alpha=0.8, linewidth=2)
    ax1.plot(iterations, profiler.timing_data['slangpy_forward'], label='SlangPy', alpha=0.8, linewidth=2)
    ax1.set_title('Forward Pass Time Over Iterations')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Time (ms)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time series for backward pass
    ax2.plot(iterations, profiler.timing_data['pytorch_backward'], label='PyTorch', alpha=0.8, linewidth=2)
    ax2.plot(iterations, profiler.timing_data['slangpy_backward'], label='SlangPy', alpha=0.8, linewidth=2)
    ax2.set_title('Backward Pass Time Over Iterations')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Time (ms)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time series for optimize pass
    ax3.plot(iterations, profiler.timing_data['pytorch_optimize'], label='PyTorch', alpha=0.8, linewidth=2)
    ax3.plot(iterations, profiler.timing_data['slangpy_optimize'], label='SlangPy', alpha=0.8, linewidth=2)
    ax3.set_title('Optimize Pass Time Over Iterations')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Time (ms)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time series for inference pass (switched to ax4)
    if 'pytorch_inference' in profiler.timing_data and 'slangpy_inference' in profiler.timing_data:
        inference_iterations = range(len(profiler.timing_data['pytorch_inference']))
        ax4.plot(inference_iterations, profiler.timing_data['pytorch_inference'], label='PyTorch', alpha=0.8, linewidth=2)
        ax4.plot(inference_iterations, profiler.timing_data['slangpy_inference'], label='SlangPy', alpha=0.8, linewidth=2)
        ax4.set_title('Inference Pass Time Over Iterations')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Time (ms)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No inference data available', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
        ax4.set_title('Inference Pass Time Over Iterations')
    
    # Plot 4: Bar chart comparison with quartile ranges
    x = np.arange(len(phases))
    width = 0.35
    
    # Calculate error bar ranges (Q3 - median, median - Q1)
    pytorch_yerr = [np.array(pytorch_medians) - np.array(pytorch_q1), 
                    np.array(pytorch_q3) - np.array(pytorch_medians)]
    slangpy_yerr = [np.array(slangpy_medians) - np.array(slangpy_q1), 
                    np.array(slangpy_q3) - np.array(slangpy_medians)]
    
    bars1 = ax6.bar(x - width/2, pytorch_medians, width, yerr=pytorch_yerr, 
                    label='PyTorch', alpha=0.8, capsize=5)
    bars2 = ax6.bar(x + width/2, slangpy_medians, width, yerr=slangpy_yerr, 
                    label='SlangPy', alpha=0.8, capsize=5)
    
    ax6.set_title('Median Execution Time by Phase (with Quartiles)')
    ax6.set_xlabel('Phase')
    ax6.set_ylabel('Time (ms)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(phases)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars above quartile ranges
    for i, (bars, q3_vals) in enumerate([(bars1, pytorch_q3), (bars2, slangpy_q3)]):
        for j, bar in enumerate(bars):
            median = bar.get_height()
            q3_top = q3_vals[j]  # Position above the Q3 quartile
            ax6.annotate(f'{median:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, q3_top),
                        xytext=(0, 5),  # 5 points vertical offset above Q3
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot 5: Performance deviation analysis
    speedup_ratios = []
    phase_labels = []
    
    for i, phase in enumerate(phases):
        if slangpy_medians[i] > 0 and pytorch_medians[i] > 0:  # Avoid division by zero and skip missing data
            ratio = pytorch_medians[i] / slangpy_medians[i]
            speedup_ratios.append(ratio)
            phase_labels.append(phase)
    
    colors = ['green' if r > 1 else 'red' for r in speedup_ratios]
    bars = ax5.bar(phase_labels, speedup_ratios, color=colors, alpha=0.7)
    ax5.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Performance')
    ax5.set_title('Performance Ratio (PyTorch/SlangPy)')
    ax5.set_xlabel('Phase')
    ax5.set_ylabel('Speedup Ratio')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, speedup_ratios):
        height = bar.get_height()
        ax5.annotate(f'{ratio:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    for i, phase in enumerate(phases):
        pytorch_median = pytorch_medians[i]
        slangpy_median = slangpy_medians[i]
        
        # Skip phases with no data
        if pytorch_median == 0 and slangpy_median == 0:
            print(f"\n{phase.upper()} PASS:")
            print("  No data available")
            continue
            
        speedup = pytorch_median / slangpy_median if slangpy_median > 0 else 0
        
        # Calculate IQR (Interquartile Range)
        pytorch_iqr = pytorch_q3[i] - pytorch_q1[i]
        slangpy_iqr = slangpy_q3[i] - slangpy_q1[i]
        
        print(f"\n{phase.upper()} PASS:")
        print(f"  PyTorch:  {pytorch_median:.3f} ms (Q1: {pytorch_q1[i]:.3f}, Q3: {pytorch_q3[i]:.3f}, IQR: {pytorch_iqr:.3f})")
        print(f"  SlangPy:  {slangpy_median:.3f} ms (Q1: {slangpy_q1[i]:.3f}, Q3: {slangpy_q3[i]:.3f}, IQR: {slangpy_iqr:.3f})")
        print(f"  Speedup:  {speedup:.2f}x {'(SlangPy faster)' if speedup > 1 else '(PyTorch faster)'}")


def generate_random_signal(length: int) -> np.ndarray:
    signal = 2 * np.random.rand(length) - 1
    signal = gaussian_filter1d(signal, sigma=2)
    return signal


def run_benchmark(iterations=200, hidden_size=64, hidden_layers=0, levels=8):
    print(f"\n{'='*60}")
    print(f"Running benchmark with hidden size: {hidden_size}")
    print(f"{'='*60}")
    
    # Prepare data
    length = 1024
    time_data = np.linspace(0, 1, length)
    signal = generate_random_signal(length)
    time_data = np.array(time_data, dtype=np.float32).reshape(-1, 1)
    signal = np.array(signal, dtype=np.float32).reshape(-1, 1)

    # Configuration
    levels = 8

    # Prepare SlangPy
    slangpy_device = create_device()

    slangpy_network = AddressesNetwork(
        slangpy_device,
        hidden=hidden_size,
        hidden_layers=hidden_layers,
        levels=levels,
        input=1,
        output=1,
    )
    slangpy_pipeline = AddressesTrainingPipeline(slangpy_device, slangpy_network)
    slangpy_input = slangpy_network.input_vec(time_data)
    slangpy_signal = slangpy_network.output_vec(signal)
    slangpy_output = slangpy_network.output_vec(np.zeros_like(signal))

    # Prepare PyTorch
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_network = PyTorchNetwork(
        hidden=hidden_size,
        levels=levels,
        input=1,
        output=1,
        hidden_layers=hidden_layers,
    ).to(torch_device)
    torch_optimizer = torch.optim.Adam(torch_network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    torch_input = torch.from_numpy(time_data).to(torch_device)
    torch_signal = torch.from_numpy(signal).to(torch_device)

    # Copy weights from PyTorch to SlangPy
    for i, layer in enumerate(torch_network.layers):
        slangpy_network.copy_weights(i, layer)

    # Profiled phases using the global profiler
    @profile('pytorch_forward')
    def pytorch_forward():
        torch_network_output = torch_network(torch_input)
        loss = F.mse_loss(torch_network_output, torch_signal)
        torch.cuda.synchronize()
        return loss
    
    @profile('slangpy_forward')
    def slangpy_forward():
        slangpy_pipeline.forward(slangpy_network, slangpy_input, slangpy_output)
        slangpy_device.wait_for_idle()
        return slangpy_output
    
    @profile('pytorch_backward')
    def pytorch_backward(loss):
        torch_network.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
    
    @profile('slangpy_backward')
    def slangpy_backward():
        slangpy_pipeline.backward(slangpy_network, slangpy_input, slangpy_signal)
        slangpy_device.wait_for_idle()

    @profile('pytorch_optimize')
    def pytorch_optimize():
        torch_optimizer.step()
        torch.cuda.synchronize()

    @profile('slangpy_optimize')
    def slangpy_optimize():
        slangpy_pipeline.optimize(slangpy_network)
        slangpy_device.wait_for_idle()

    @profile('pytorch_inference')
    def pytorch_inference():
        with torch.no_grad():
            torch_network_output = torch_network(torch_input)
        torch.cuda.synchronize()
        return torch_network_output

    @profile('slangpy_inference')
    def slangpy_inference():
        slangpy_pipeline.forward(slangpy_network, slangpy_input, slangpy_output)
        slangpy_device.wait_for_idle()
        return slangpy_output

    # Training loops
    print("Running training benchmark...")
    for i in tqdm(range(iterations), desc="Training"):
        set_iteration_count(i)
        
        # SlangPy training
        slangpy_forward()
        slangpy_backward()
        slangpy_optimize()
        
    for i in tqdm(range(iterations), desc="Training"):
        set_iteration_count(i)

        # PyTorch training
        pytorch_loss = pytorch_forward()
        pytorch_backward(pytorch_loss)
        pytorch_optimize()

    # Set PyTorch to eval mode for inference
    torch_network.eval()
    
    print("Running inference benchmark...")
    for i in tqdm(range(iterations), desc="Inference"):
        set_iteration_count(i)
        
        # SlangPy inference
        slangpy_inference()
    
    for i in tqdm(range(iterations), desc="Inference"):
        set_iteration_count(i)

        # PyTorch inference
        pytorch_inference()


def main():
    # Reset profiler and run benchmark
    reset_profiler()
    run_benchmark(iterations=10000, hidden_size=64, hidden_layers=2, levels=8)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS...")
    print("="*80)
    plot_profiling_results()

if __name__ == "__main__":
    sns.set_theme()
    sns.set_palette("pastel")

    main()
