# neural.slang

**neural.slang** is a [Slang](https://shader-slang.com/) initiative that demonstrates the capabilities of differentiable programming and GPU-accelerated machine learning, with a focus on neural graphics primitives and real-time rendering applications.

## Goal

**neural.slang** provides a high-performance, GPU-native implementation of neural networks that seamlessly integrates with graphics pipelines. By leveraging Slang's automatic differentiation capabilities and GPU compute shaders, this framework enables:

- **Real-time neural rendering**: Train and evaluate neural networks directly on the GPU within rendering pipelines
- **Differentiable programming**: Automatic gradient computation for neural network training
- **Graphics-ML integration**: Native support for neural graphics primitives like Neural Radiance Fields (NeRF) and neural SDFs
- **High performance**: Fully GPU-accelerated training and inference with minimal CPU-GPU transfers

## Features

- Modular neural network components (layers, activations, optimizers, losses)
- Support for both inline and cooperative vector operations for different GPU architectures
- Neural graphics primitives including dense grids and multi-level feature grids
- Integration with traditional graphics pipelines (rasterization, ray tracing)
- Mirrored Python interfaces via slangpy for easy experimentation

## Requirements

- Python 3.10+
- Vulkan-compatible GPU (tested on Vulkan-compatible systems)
- Dependencies listed in `requirements.txt`

## Getting Started

Installing dependencies:
```bash
pip install -r requirements.txt
```

Running examples:
```bash
python -m examples.<example_name>.main
```

## Organization

| Directory | Description |
|-----------|-------------|
| [**neural**](neural) | Core neural network framework modules implementing layers, activations, optimizers, and losses in Slang |
| [**ngp**](ngp) | Neural Graphics Primitives implementation including dense grids and multi-level feature grids |
| [**examples**](examples) | Example applications demonstrating various use cases (SDF learning, deferred shading, texture learning) |
| [**tests**](tests) | Unit tests for validating the framework components |
| [**util**](util) | Utility modules for graphics operations, camera handling, and mesh processing |
| [**resources**](resources) | 3D models and textures used by the example applications |
