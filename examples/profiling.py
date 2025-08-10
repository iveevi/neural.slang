"""
Profiling utilities for benchmarking PyTorch vs SlangPy performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from functools import wraps
from collections import defaultdict


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


def plot_profiling_results(title_suffix=""):
    """Plot timing results comparing PyTorch and SlangPy performance"""
    # Prepare data
    phases = ['forward', 'backward', 'optimize']
    pytorch_means = []
    slangpy_means = []
    pytorch_stds = []
    slangpy_stds = []
    
    # Calculate statistics for each phase
    for phase in phases:
        pytorch_key = f"pytorch_{phase}"
        slangpy_key = f"slangpy_{phase}"
        
        pytorch_times = np.array(profiler.timing_data[pytorch_key])
        slangpy_times = np.array(profiler.timing_data[slangpy_key])
        
        pytorch_means.append(np.mean(pytorch_times))
        slangpy_means.append(np.mean(slangpy_times))
        pytorch_stds.append(np.std(pytorch_times))
        slangpy_stds.append(np.std(slangpy_times))
    
    # Create subplots - 3x2 layout to accommodate all time series
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
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
    
    # Plot 4: Bar chart comparison with error bars
    x = np.arange(len(phases))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, pytorch_means, width, yerr=pytorch_stds, 
                    label='PyTorch', alpha=0.8, capsize=5)
    bars2 = ax4.bar(x + width/2, slangpy_means, width, yerr=slangpy_stds, 
                    label='SlangPy', alpha=0.8, capsize=5)
    
    ax4.set_title('Average Execution Time by Phase')
    ax4.set_xlabel('Phase')
    ax4.set_ylabel('Time (ms)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(phases)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars above error bars
    for i, (bars, stds) in enumerate([(bars1, pytorch_stds), (bars2, slangpy_stds)]):
        for j, bar in enumerate(bars):
            height = bar.get_height()
            error_top = height + stds[j]  # Position above the error bar
            ax4.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, error_top),
                        xytext=(0, 5),  # 5 points vertical offset above error bar
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot 5: Performance deviation analysis
    speedup_ratios = []
    phase_labels = []
    
    for i, phase in enumerate(phases):
        if slangpy_means[i] > 0:  # Avoid division by zero
            ratio = pytorch_means[i] / slangpy_means[i]
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
    
    # Plot 6: Combined time series comparison
    ax6.plot(iterations, profiler.timing_data['pytorch_forward'], label='PyTorch Forward', alpha=0.8, linewidth=2)
    ax6.plot(iterations, profiler.timing_data['pytorch_backward'], label='PyTorch Backward', alpha=0.8, linewidth=2)
    ax6.plot(iterations, profiler.timing_data['pytorch_optimize'], label='PyTorch Optimize', alpha=0.8, linewidth=2)
    ax6.plot(iterations, profiler.timing_data['slangpy_forward'], label='SlangPy Forward', alpha=0.8, linewidth=2, linestyle='--')
    ax6.plot(iterations, profiler.timing_data['slangpy_backward'], label='SlangPy Backward', alpha=0.8, linewidth=2, linestyle='--')
    ax6.plot(iterations, profiler.timing_data['slangpy_optimize'], label='SlangPy Optimize', alpha=0.8, linewidth=2, linestyle='--')
    ax6.set_title('All Phases Combined Time Series')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Time (ms)')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY (First iteration excluded)")
    print("="*70)
    print(f"Total iterations analyzed: {len(profiler.timing_data['pytorch_forward'])}/999 per framework")
    for i, phase in enumerate(phases):
        pytorch_mean = pytorch_means[i]
        slangpy_mean = slangpy_means[i]
        speedup = pytorch_mean / slangpy_mean if slangpy_mean > 0 else 0
        
        print(f"\n{phase.upper()} PASS:")
        print(f"  PyTorch:  {pytorch_mean:.3f} ± {pytorch_stds[i]:.3f} ms")
        print(f"  SlangPy:  {slangpy_mean:.3f} ± {slangpy_stds[i]:.3f} ms")
        print(f"  Speedup:  {speedup:.2f}x {'(SlangPy faster)' if speedup > 1 else '(PyTorch faster)'}")


def reset_profiler():
    """Reset the global profiler state"""
    global profiler
    profiler.reset()


def set_iteration_count(count):
    """Set the current iteration count"""
    global profiler
    profiler.iteration_count = count
