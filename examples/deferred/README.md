# Deferred Shading Example

Neural deferred shading that learns lighting and shading effects in real-time.

## Overview

This example demonstrates how neural networks can be integrated into a deferred rendering pipeline to learn complex lighting and shading effects. The system trains a neural network to approximate global illumination by comparing against reference path-traced images.

## Controls

- `Tab` or `1`/`2`/`3`: Switch between reference/neural shading/neural illumination modes
- `Space`: Pause/resume camera orbit
- `Up`/`Down` arrows: Adjust camera vertical position

## Usage

```bash
python -m examples.deferred.main
```

## Details

The example demonstrates neural deferred shading by training a network to produce coherent illumination from noisy ray-traced samples:

1. **Reference Pipeline**:
   - Full rasterization pipeline that performs ray-traced direct illumination
   - Uses blue noise sampling patterns to reduce correlation in noise
   - Produces noisy but physically accurate lighting at each pixel
   - Serves as ground truth for training the neural network

2. **G-buffer Generation**:
   - Standard deferred rendering first pass: rasterizes geometry to multiple render targets
   - Stores world position, surface normal, and albedo in separate textures
   - Provides geometric context for the neural network to reason about lighting

3. **Neural Shading**:
   - Takes G-buffer data as input and produces final shaded color
   - Network learns to denoise and approximate global illumination effects
   - Outputs temporally stable, coherent illumination without per-pixel noise
   - Two modes: full shading or just indirect illumination component

4. **Training Process**:
   - Reference image rendered at low resolution $(256 \times 256)$ for efficiency
   - Neural network evaluates same G-buffer and compares against reference
   - Gradients computed per-pixel drive network to match ray-traced quality
   - Stable parameters use exponential moving average for flicker-free display

5. **Real-time Rendering**:
   - Display resolution $(1024 \times 1024)$ uses stable network parameters
   - Achieves ray-traced quality lighting at real-time framerates
   - Network generalizes to novel viewpoints during camera movement
