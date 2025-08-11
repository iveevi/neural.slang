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
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(15, 24))
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


def reset_profiler():
    """Reset the global profiler state"""
    global profiler
    profiler.reset()


def set_iteration_count(count):
    """Set the current iteration count"""
    global profiler
    profiler.iteration_count = count
