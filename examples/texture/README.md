# Texture Synthesis Example

Neural texture synthesis and compression using coordinate-based networks.

## Overview

This example demonstrates how neural networks can learn to represent and synthesize textures. A small MLP learns to map 2D coordinates to RGB colors, effectively compressing the texture into network weights while enabling resolution-independent sampling.

## Usage

```bash
python -m examples.texture.main
```

## Details

The example demonstrates neural texture synthesis as a continuous function:

1. **Coordinate-based Representation**:
   - Network learns to map UV coordinates $[0,1]^2$ to RGB colors
   - Texture stored as neural network weights instead of pixel grid
   - Can be sampled at any resolution without pixelation
   - Compression achieved by using fewer parameters than pixels

2. **Fourier Feature Encoding**:
   - Raw UV coordinates cannot represent high-frequency details
   - Fourier encoding maps 2D positions to higher dimensional space
   - Uses sin/cos at exponentially increasing frequencies $2^0$ to $2^7$
   - Enables network to learn sharp edges and fine texture patterns

3. **Progressive Learning**:
   - Network starts outputting average color of the texture
   - Gradually learns low-frequency patterns first
   - High-frequency details emerge as training progresses
   - Real-time visualization shows texture forming from neural weights

4. **Training Strategy**:
   - Random UV coordinates sampled uniformly across image
   - Target colors obtained via bilinear interpolation
   - Network learns to match these color samples
   - Converges to continuous representation of discrete texture

5. **Rendering Process**:
   - Each pixel evaluates the network at its UV coordinate
   - Fourier encoding computed on-the-fly in shader
   - MLP forward pass produces final RGB color
   - Entire image generated from ~10K network parameters
