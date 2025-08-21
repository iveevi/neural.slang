# SDF Learning Example

Neural network learning to represent 3D shapes as signed distance fields.

## Overview

This example demonstrates how neural networks can learn implicit 3D shape representations by approximating signed distance fields (SDFs). The network is trained on point samples from a target mesh and visualized using sphere tracing.

## Controls

- `Tab`: Toggle between heatmap and normal rendering modes
- `Space`: Pause/resume camera rotation

## Usage

```bash
python -m examples.sdf.main
```

## Details

The example visualizes neural SDF learning through real-time sphere tracing:

1. **SDF Representation**:
   - Neural network learns to output signed distance at any 3D point
   - Negative values inside the shape, positive outside, zero at surface
   - Multi-resolution feature grids capture fine geometric details
   - Network trained on random point samples from ground truth mesh SDF

2. **Sphere Tracing Visualization**:
   - Each pixel casts a ray and marches using the learned SDF
   - Steps forward by the distance value until hitting the surface
   - Heatmap mode colors points by SDF value (red=inside, blue=outside)
   - Normal mode computes surface normals via automatic differentiation

3. **Multi-resolution Encoding**:
   - Multi-level grids $(16^3, 32^3, 64^3)$ provide progressive detail
   - Each resolution captures different frequency components of the shape
   - Grids store learnable features queried by 3D position
   - Features from all levels concatenated and processed by MLP to predict SDF

4. **Real-time Training**:
   - Random 3D points sampled uniformly in $[-1, 1]^3$ space
   - Ground truth SDF computed from input mesh using exact distance
   - Network learns to match these values across entire volume
   - Loss visualization shows convergence over time

5. **Rendering Quality**:
   - Smooth surfaces emerge as network learns the shape
   - Automatic differentiation provides accurate normals for shading
   - Real-time feedback allows watching the learning process
