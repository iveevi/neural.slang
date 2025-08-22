# Neural Graphics Primitives (NGP)

The NGP module provides Python interfaces for neural graphics primitives - GPU-accelerated components for neural rendering applications like NeRF and SDF learning.

## Overview

NGP implements the building blocks for instant neural graphics primitives, providing Python wrappers around high-performance Slang shader implementations. The module focuses on spatial data structures and neural networks optimized for real-time rendering.

## Components

### Core Classes (`objects.py`)
- `Object`: Base class for shader-bindable objects
- `Optimizer`: Base class for ML optimizers  
- `Optimizable`: Base class for trainable components with GPU parameter updates
- `MLP`: Base class for multi-layer perceptrons
- `Grid`: Base class for spatial feature storage

### Grids (`grids.py`)

#### DenseGrid
Dense voxel grid for spatial feature storage.

```python
import slangpy as spy
from ngp import DenseGrid

# Create a 3D dense grid with 32³ resolution and 8 features per voxel
device = spy.Device()
grid = DenseGrid.new(
    device=device,
    dimension=3,
    features=8,
    resolution=32
)
```

**Parameters:**
- `device`: GPU device
- `dimension`: Spatial dimension (2 or 3)
- `features`: Number of features per voxel
- `resolution`: Grid resolution per dimension

**Key Methods:**
- `parameter_count`: Total number of parameters
- `alloc_optimizer_states(device, optimizer)`: Allocate optimizer state buffers
- `update(optimizer, optimizer_states)`: Update parameters using optimizer

**Slang Integration:**
```slang
DenseGrid<3, 8> grid;
InlineVector<float, 8> features = grid.sample(position);
```

#### MultiLevelDenseGrid
Multi-resolution grid with 3 hierarchical levels.

```python
from ngp import MultiLevelDenseGrid

grid = MultiLevelDenseGrid.new(
    device=device,
    dimension=3,
    features=32,
    resolutions=[16, 32, 64]  # Must be exactly 3 resolutions
)
```

**Parameters:**
- `resolutions`: List of exactly 3 resolutions (one per level)

**Note:** Returns concatenated features from all 3 levels (3 × features).

**Slang Integration:**
```slang
MultiLevelDenseGrid<32> grid;
InlineVector<float, 96> features = grid.sample(position);  // 3 × 32 features
```

### Networks (`mlps.py`)

#### AddressBasedMLP
GPU-optimized multi-layer perceptron with ReLU activations.

```python
from ngp import AddressBasedMLP

# Create an MLP with 3 inputs, 4 outputs, 64 hidden units, and 2 hidden layers
mlp = AddressBasedMLP.new(
    device=device,
    input=3,
    output=4,
    hidden=64,
    hidden_layers=2
)
```

**Architecture:**
- ReLU activation for hidden layers
- Identity activation for output layer
- Parameters stored in flattened format for efficient GPU access

**Slang Integration:**
```slang
AddressBasedMLP<3, 4, 64, 2, ReLU<float>, Identity<float>> mlp;
float4 output = mlp.forward(input);
```

### Optimizers (`optimizers.py`)

#### Adam
Adam optimizer with standard hyperparameters.

```python
from ngp import Adam

# Create optimizer
optimizer = Adam(alpha=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)

# Allocate optimizer states for a component
states = mlp.alloc_optimizer_states(device, optimizer)

# Update parameters (runs on GPU)
mlp.update(optimizer, states)
```

**Slang Integration:**
```slang
Adam<float> optimizer;
optimizer.update(params, gradients, states, parameterIndex);
```

### Encoders (`encoders.py`)

#### RandomFourierEncoder
Random Fourier features for coordinate encoding.

```python
from ngp import RandomFourierEncoder

encoder = RandomFourierEncoder(input=3, features=32)
```