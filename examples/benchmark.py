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

from .network_with_separate_buffers import Network, Pipeline
from .pytorch_networks import PyTorchNetwork


ROOT = pathlib.Path(__file__).parent.parent.absolute()


class ProfilerState:
    """Global state for the profiler"""
    def __init__(self):
        self.timing_data = defaultdict(list)
        self.iteration_count = 0
        self.skip_first_iteration = True
    
    def reset(self):
        """Reset profiling state"""
        self.timing_data.clear()
        self.iteration_count = 0


# Global profiler instance
profiler = ProfilerState()


def profile(func_name):
    """Decorator to profile function execution time"""
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
    """Reset the global profiler state"""
    global profiler
    profiler.reset()


def set_iteration_count(count):
    """Set the current iteration count"""
    global profiler
    profiler.iteration_count = count


def plot_profiling_results(title_suffix=""):
    """Plot timing results comparing PyTorch and SlangPy performance"""
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
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(15, 24), layout='constrained')
    title = f'PyTorch vs SlangPy Performance Comparison{title_suffix}'
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
    
    # Plot 7: Time series for inference pass
    if 'pytorch_inference' in profiler.timing_data and 'slangpy_inference' in profiler.timing_data:
        inference_iterations = range(len(profiler.timing_data['pytorch_inference']))
        ax7.plot(inference_iterations, profiler.timing_data['pytorch_inference'], label='PyTorch', alpha=0.8, linewidth=2)
        ax7.plot(inference_iterations, profiler.timing_data['slangpy_inference'], label='SlangPy', alpha=0.8, linewidth=2)
        ax7.set_title('Inference Pass Time Over Iterations')
        ax7.set_xlabel('Iteration')
        ax7.set_ylabel('Time (ms)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'No inference data available', horizontalalignment='center', verticalalignment='center', transform=ax7.transAxes)
        ax7.set_title('Inference Pass Time Over Iterations')
    
    # Plot 4: Bar chart comparison with quartile ranges
    x = np.arange(len(phases))
    width = 0.35
    
    # Calculate error bar ranges (Q3 - median, median - Q1)
    pytorch_yerr = [np.array(pytorch_medians) - np.array(pytorch_q1), 
                    np.array(pytorch_q3) - np.array(pytorch_medians)]
    slangpy_yerr = [np.array(slangpy_medians) - np.array(slangpy_q1), 
                    np.array(slangpy_q3) - np.array(slangpy_medians)]
    
    bars1 = ax4.bar(x - width/2, pytorch_medians, width, yerr=pytorch_yerr, 
                    label='PyTorch', alpha=0.8, capsize=5)
    bars2 = ax4.bar(x + width/2, slangpy_medians, width, yerr=slangpy_yerr, 
                    label='SlangPy', alpha=0.8, capsize=5)
    
    ax4.set_title('Median Execution Time by Phase (with Quartiles)')
    ax4.set_xlabel('Phase')
    ax4.set_ylabel('Time (ms)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(phases)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars above quartile ranges
    for i, (bars, q3_vals) in enumerate([(bars1, pytorch_q3), (bars2, slangpy_q3)]):
        for j, bar in enumerate(bars):
            median = bar.get_height()
            q3_top = q3_vals[j]  # Position above the Q3 quartile
            ax4.annotate(f'{median:.2f}',
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
    
    # Plot 6: Combined training phases time series comparison
    ax6.plot(iterations, profiler.timing_data['pytorch_forward'], label='PyTorch Forward', alpha=0.8, linewidth=2)
    ax6.plot(iterations, profiler.timing_data['pytorch_backward'], label='PyTorch Backward', alpha=0.8, linewidth=2)
    ax6.plot(iterations, profiler.timing_data['pytorch_optimize'], label='PyTorch Optimize', alpha=0.8, linewidth=2)
    ax6.plot(iterations, profiler.timing_data['slangpy_forward'], label='SlangPy Forward', alpha=0.8, linewidth=2, linestyle='--')
    ax6.plot(iterations, profiler.timing_data['slangpy_backward'], label='SlangPy Backward', alpha=0.8, linewidth=2, linestyle='--')
    ax6.plot(iterations, profiler.timing_data['slangpy_optimize'], label='SlangPy Optimize', alpha=0.8, linewidth=2, linestyle='--')
    ax6.set_title('Training Phases Combined Time Series')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Time (ms)')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    # Plot 8: Combined inference comparison (if available)
    if 'pytorch_inference' in profiler.timing_data and 'slangpy_inference' in profiler.timing_data:
        inference_iterations = range(len(profiler.timing_data['pytorch_inference']))
        ax8.plot(inference_iterations, profiler.timing_data['pytorch_inference'], label='PyTorch Inference', alpha=0.8, linewidth=2)
        ax8.plot(inference_iterations, profiler.timing_data['slangpy_inference'], label='SlangPy Inference', alpha=0.8, linewidth=2, linestyle='--')
        ax8.set_title('Inference Phase Comparison')
        ax8.set_xlabel('Iteration')
        ax8.set_ylabel('Time (ms)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'No inference data available', horizontalalignment='center', verticalalignment='center', transform=ax8.transAxes)
        ax8.set_title('Inference Phase Comparison')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY (First iteration excluded)")
    print("="*80)
    print(f"Total iterations analyzed: {len(profiler.timing_data['pytorch_forward'])}/999 per framework")
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


def plot_hidden_size_scaling(hidden_sizes, timing_results):
    """Plot violin plots showing performance scaling with hidden layer sizes"""
    
    # Prepare data for violin plots
    phases = ['forward', 'backward', 'optimize', 'inference']
    frameworks = ['pytorch', 'slangpy']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), layout='constrained')
    axes = axes.flatten()
    
    fig.suptitle('Performance Scaling Across Hidden Layer Sizes', fontsize=16, fontweight='bold')
    
    for phase_idx, phase in enumerate(phases):
        ax = axes[phase_idx]
        
        # Collect data for each framework and hidden size
        plot_data = []
        labels = []
        colors = []
        
        for framework in frameworks:
            for i, hidden_size in enumerate(hidden_sizes):
                key = f"{framework}_{phase}_{hidden_size}"
                if key in timing_results and len(timing_results[key]) > 0:
                    plot_data.append(timing_results[key])
                    labels.append(f"{framework.title()}\n{hidden_size}")
                    colors.append('lightblue' if framework == 'pytorch' else 'lightcoral')
        
        if plot_data:
            # Create violin plot
            parts = ax.violinplot(plot_data, positions=range(len(plot_data)), 
                                showmeans=True, showmedians=True, showextrema=True)
            
            # Color the violin plots
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
            
            # Set labels and title
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Time (ms)')
            ax.set_title(f'{phase.title()} Pass Performance')
            ax.grid(True, alpha=0.3)
            
            # Add median values as text
            for i, data in enumerate(plot_data):
                median_val = np.median(data)
                ax.text(i, median_val, f'{median_val:.2f}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'No {phase} data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{phase.title()} Pass Performance')
    
    plt.tight_layout()
    plt.show()
    
    # Print scaling summary
    print("\n" + "="*80)
    print("HIDDEN SIZE SCALING SUMMARY")
    print("="*80)
    
    for phase in phases:
        print(f"\n{phase.upper()} PASS SCALING:")
        for framework in frameworks:
            medians = []
            for hidden_size in hidden_sizes:
                key = f"{framework}_{phase}_{hidden_size}"
                if key in timing_results and len(timing_results[key]) > 0:
                    median = np.median(timing_results[key])
                    medians.append(median)
                    print(f"  {framework.title()} (hidden={hidden_size}): {median:.3f} ms")
                else:
                    print(f"  {framework.title()} (hidden={hidden_size}): No data")
            
            # Calculate scaling factor if we have enough data
            if len(medians) >= 2:
                scaling_factor = medians[-1] / medians[0] if medians[0] > 0 else 0
                size_factor = hidden_sizes[-1] / hidden_sizes[0]
                print(f"  {framework.title()} scaling: {scaling_factor:.2f}x time for {size_factor:.1f}x hidden size")


def generate_random_signal(length: int) -> np.ndarray:
    signal = 2 * np.random.rand(length) - 1
    signal = gaussian_filter1d(signal, sigma=2)
    return signal


def run_single_benchmark(hidden_size, address_mode, iterations=100):
    """Run benchmark for a single hidden layer size"""
    print(f"\n{'='*60}")
    print(f"Benchmarking hidden size: {hidden_size}")
    print(f"{'='*60}")
    
    # Prepare data
    length = 1024
    time_data = np.linspace(0, 1, length)
    signal = generate_random_signal(length)
    time_data = np.array(time_data, dtype=np.float32).reshape(-1, 1)
    signal = np.array(signal, dtype=np.float32).reshape(-1, 1)

    # Configuration
    levels = 0

    # Prepare SlangPy
    slangpy_device = spy.create_device(
        spy.DeviceType.vulkan,
        enable_debug_layers=False,
        include_paths=[
            ROOT / "neural",
        ],
    )

    if address_mode:
        from .network_with_addresses import Network, Pipeline
    else:
        from .network_with_separate_buffers import Network, Pipeline

    slangpy_network = Network(slangpy_device, hidden=hidden_size, hidden_layers=2, levels=levels, input=1, output=1)
    slangpy_pipeline = Pipeline(slangpy_device, slangpy_network)
    slangpy_input = slangpy_network.input_vec(time_data)
    slangpy_signal = slangpy_network.output_vec(signal)
    slangpy_output = slangpy_network.output_vec(np.zeros_like(signal))

    # Prepare PyTorch
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_network = PyTorchNetwork(hidden=hidden_size, levels=levels, input=1, output=1).to(torch_device)
    torch_optimizer = torch.optim.Adam(torch_network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    torch_input = torch.from_numpy(time_data).to(torch_device)
    torch_signal = torch.from_numpy(signal).to(torch_device)

    # Copy weights from PyTorch to SlangPy
    slangpy_network.layers[0].copy_weights(torch_network.layer1)
    slangpy_network.layers[1].copy_weights(torch_network.layer2)
    slangpy_network.layers[2].copy_weights(torch_network.layer3)
    slangpy_network.layers[3].copy_weights(torch_network.layer4)

    # Dictionary to store timing results for this hidden size
    timing_results = defaultdict(list)

    # Phases to profile
    def pytorch_forward():
        start_time = time.perf_counter()
        torch_network_output = torch_network(torch_input)
        loss = F.mse_loss(torch_network_output, torch_signal)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        timing_results[f"pytorch_forward_{hidden_size}"].append((end_time - start_time) * 1000)
        return loss
    
    def slangpy_forward():
        start_time = time.perf_counter()
        slangpy_pipeline.forward(slangpy_network, slangpy_input, slangpy_output)
        slangpy_device.wait_for_idle()
        end_time = time.perf_counter()
        timing_results[f"slangpy_forward_{hidden_size}"].append((end_time - start_time) * 1000)
        return slangpy_output
    
    def pytorch_backward(loss):
        start_time = time.perf_counter()
        torch_network.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        timing_results[f"pytorch_backward_{hidden_size}"].append((end_time - start_time) * 1000)
    
    def slangpy_backward():
        start_time = time.perf_counter()
        slangpy_pipeline.backward(slangpy_network, slangpy_input, slangpy_signal)
        slangpy_device.wait_for_idle()
        end_time = time.perf_counter()
        timing_results[f"slangpy_backward_{hidden_size}"].append((end_time - start_time) * 1000)

    def pytorch_optimize():
        start_time = time.perf_counter()
        torch_optimizer.step()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        timing_results[f"pytorch_optimize_{hidden_size}"].append((end_time - start_time) * 1000)

    def slangpy_optimize():
        start_time = time.perf_counter()
        slangpy_pipeline.optimize(slangpy_network)
        slangpy_device.wait_for_idle()
        end_time = time.perf_counter()
        timing_results[f"slangpy_optimize_{hidden_size}"].append((end_time - start_time) * 1000)

    def pytorch_inference():
        start_time = time.perf_counter()
        with torch.no_grad():
            torch_network_output = torch_network(torch_input)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        timing_results[f"pytorch_inference_{hidden_size}"].append((end_time - start_time) * 1000)
        return torch_network_output

    def slangpy_inference():
        start_time = time.perf_counter()
        slangpy_pipeline.forward(slangpy_network, slangpy_input, slangpy_output)
        slangpy_device.wait_for_idle()
        end_time = time.perf_counter()
        timing_results[f"slangpy_inference_{hidden_size}"].append((end_time - start_time) * 1000)
        return slangpy_output

    # Training loops - separate SlangPy and PyTorch
    print("Running SlangPy training...")
    for i in tqdm(range(iterations), desc="SlangPy"):
        if i == 0:  # Skip first iteration to avoid initialization overhead
            slangpy_forward()
            slangpy_backward()
            slangpy_optimize()
            # Remove the first timing measurement
            for key in timing_results:
                if timing_results[key]:
                    timing_results[key].pop()
        else:
            slangpy_forward()
            slangpy_backward()
            slangpy_optimize()
    
    print("Running PyTorch training...")
    for i in tqdm(range(iterations), desc="PyTorch"):
        if i == 0:  # Skip first iteration to avoid initialization overhead
            pytorch_loss = pytorch_forward()
            pytorch_backward(pytorch_loss)
            pytorch_optimize()
            # Remove the first timing measurement
            for key in timing_results:
                if 'pytorch' in key and timing_results[key]:
                    timing_results[key].pop()
        else:
            pytorch_loss = pytorch_forward()
            pytorch_backward(pytorch_loss)
            pytorch_optimize()

    # Set PyTorch to eval mode for inference
    torch_network.eval()
    
    print("Running SlangPy inference...")
    for i in tqdm(range(iterations), desc="SlangPy Inference"):
        if i == 0:  # Skip first iteration
            slangpy_inference()
            if f"slangpy_inference_{hidden_size}" in timing_results and timing_results[f"slangpy_inference_{hidden_size}"]:
                timing_results[f"slangpy_inference_{hidden_size}"].pop()
        else:
            slangpy_inference()

    print("Running PyTorch inference...")
    for i in tqdm(range(iterations), desc="PyTorch Inference"):
        if i == 0:  # Skip first iteration
            pytorch_inference()
            if f"pytorch_inference_{hidden_size}" in timing_results and timing_results[f"pytorch_inference_{hidden_size}"]:
                timing_results[f"pytorch_inference_{hidden_size}"].pop()
        else:
            pytorch_inference()

    return timing_results


def main(address_mode: bool = True):
    """Main benchmark function that runs across multiple hidden layer sizes"""
    # Hidden layer sizes to benchmark
    hidden_sizes = [8, 16, 32, 64, 128]
    iterations_per_size = 100  # Reduced for faster execution across multiple sizes
    
    print("Starting multi-size benchmarking...")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Iterations per size: {iterations_per_size}")
    
    # Collect all timing results across hidden sizes
    all_timing_results = defaultdict(list)
    
    # Run benchmarks for each hidden size
    for hidden_size in hidden_sizes:
        size_results = run_single_benchmark(hidden_size, address_mode, iterations_per_size)
        
        # Merge results into the global timing collection
        for key, values in size_results.items():
            all_timing_results[key] = values
    
    print("\n" + "="*80)
    print("GENERATING PLOTS...")
    print("="*80)
    
    # Generate violin plots for hidden size scaling
    plot_hidden_size_scaling(hidden_sizes, all_timing_results)
    
    # Also run one final detailed benchmark with the middle hidden size for traditional plots
    print(f"\nRunning detailed benchmark for hidden size {hidden_sizes[2]} for traditional plots...")
    reset_profiler()
    
    # Run a single benchmark with the profiler enabled for traditional plots
    detailed_results = run_single_benchmark(hidden_sizes[2], address_mode, 200)
    
    # Convert detailed results to profiler format for traditional plotting
    for key, values in detailed_results.items():
        if f"_{hidden_sizes[2]}" in key:
            base_key = key.replace(f"_{hidden_sizes[2]}", "")
            profiler.timing_data[base_key] = values
    
    # Generate traditional plots
    plot_profiling_results(title_suffix=f" - Hidden Size {hidden_sizes[2]}")

if __name__ == "__main__":
    # TODO: move to util
    parser = argparse.ArgumentParser()
    parser.add_argument("--address-mode", action="store_true")
    args = parser.parse_args()

    sns.set_theme()
    sns.set_palette("pastel")

    main(address_mode=args.address_mode)
