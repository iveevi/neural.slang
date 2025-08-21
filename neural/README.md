# neural.slang Documentation

The **neural.slang** module provides a comprehensive set of interfaces and implementations for building differentiable neural networks in Slang. This module is designed to work with automatic differentiation and supports both inline and cooperative vector operations.

## Core Interfaces

### `IStorage<T>`
Defines the interface for parameter storage systems.
- **Associated Types:**
  - `Address`: Type used for addressing elements in storage
- **Methods:**
  - `read(Address)`: Read a value from storage
  - `add(Address, T)`: Atomically add to a value in storage
  - `write(Address, T)`: Write a value to storage
  - `getOffset(Address, int)`: Calculate offset address

### `IBindlessStorage<T>`
Interface for bindless buffer storage systems.
- **Associated Types:**
  - `BufferHandle`: Handle type for the underlying buffer
- **Methods:**
  - `read()`: Read value at current offset
  - `write(T)`: Write value at current offset
  - `add(T)`: Atomically add to value at current offset
  - `getOffset(int)`: Get storage at offset
  - `getBufferHandle()`: Get underlying buffer handle

### `IVector<T, N>`
Core interface for differentiable vector operations.
- **Type Parameters:**
  - `T`: Floating-point element type
  - `N`: Vector dimension
- **Methods:**
  - Arithmetic: `add`, `sub`, `mul`, `div`, `neg`
  - ML operations: `max`, `step`, `sin`, `cos`, `abs`, `exp`, `sum`
  - Matrix operations: `apply` (for matrix-vector multiplication)
  - Conversions: `toVector`, `fromVector`

### `IActivation<T>`
Interface for activation functions.
- **Methods:**
  - `eval<N, Vector>(Vector input)`: Apply activation function

### `ILayer<T, In, Out, Storage>`
Interface for neural network layers.
- **Type Parameters:**
  - `T`: Element type
  - `In`: Input dimension
  - `Out`: Output dimension
  - `Storage`: Storage type for parameters
- **Methods:**
  - `eval(Storage, InputVector)`: Forward pass through the layer

### `ILoss<T>`
Interface for loss functions.
- **Methods:**
  - `eval(Vector predicted, Vector expected)`: Compute loss value

### `IOptimizer<T>`
Interface for optimization algorithms.
- **Associated Types:**
  - `State`: Optimizer state type
- **Methods:**
  - `step(State, parameter, gradient)`: Update parameter using gradient

### `IEncoder<T, In, Out>`
Interface for input encoders (non-learnable).
- **Methods:**
  - `eval(InputVector)`: Encode input vector

### `ILearnableEncoder<T, In, Out, Storage>`
Interface for learnable encoders with parameters.
- **Methods:**
  - `eval(Storage, InputVector)`: Encode using stored parameters

## Implementations

### Storage Systems

| Implementation | File | Status | Testing | Description | Notes |
|----------------|------|--------|---------|-------------|-------|
| `StructuredBufferStorage<T>` | [storages.slang](storages.slang) | ✅ Fully implemented | ✅ Tested | Implements `IStorage<T>` using `RWStructuredBuffer`, supports atomic operations | None |
| `BindlessBufferStorage<T>` | [storages.slang](storages.slang) | ✅ Fully implemented | ⚠️ Testing paused | Implements `IBindlessStorage<T>` using buffer handles, enables dynamic buffer access | None |

### Vector Types

| Implementation | File | Status | Testing | Description | Notes |
|----------------|------|--------|---------|-------------|-------|
| `InlineVector<T, N>` | [inline_vector.slang](inline_vector.slang) | ✅ Fully implemented | ✅ Tested | Stack-allocated vector with full differentiability | Best for small vectors (N < 32) |
| `CoopVector<T, N>` | [cooperative_vector.slang](cooperative_vector.slang) | ⚠️ Partially implemented | ❌ Not tested | Uses cooperative matrix operations for performance | `apply` method not implemented for ordinary storage; `applyBindless` fully implemented |

### Activation Functions

| Implementation | File | Status | Testing | Description | Notes |
|----------------|------|--------|---------|-------------|-------|
| `Identity<T>` | [activations.slang](activations.slang) | ✅ Fully implemented | ❌ Not tested | Pass-through activation | None |
| `ReLU<T>` | [activations.slang](activations.slang) | ✅ Fully implemented | ✅ Tested | Rectified Linear Unit with custom backward | None |
| `LeakyReLU<T>` | [activations.slang](activations.slang) | ✅ Fully implemented | ✅ Tested | Leaky ReLU with configurable alpha | None |
| `Sine<T>` | [activations.slang](activations.slang) | ✅ Fully implemented | ✅ Tested | Sine activation (useful for neural fields) | None |
| `Exp<T>` | [activations.slang](activations.slang) | ✅ Fully implemented | ✅ Tested | Exponential activation | None |
| `Sigmoid<T>` | - | ❌ Not implemented | ➖ N/A | Sigmoid activation function | None |
| `Tanh<T>` | - | ❌ Not implemented | ➖ N/A | Hyperbolic tangent activation | None |
| `Softmax<T>` | - | ❌ Not implemented | ➖ N/A | Softmax activation for multi-class outputs | None |

### Layers

| Implementation | File | Status | Testing | Description | Notes |
|----------------|------|--------|---------|-------------|-------|
| `FFLayer<T, In, Out, Storage, Activation>` | [layers.slang](layers.slang) | ✅ Fully implemented | ✅ Tested | Standard fully-connected layer with activation, supports bias terms | None |
| `ResidualFFLayer<T, Size, Storage, Activation>` | [layers.slang](layers.slang) | ✅ Fully implemented | ❌ Not tested | Feed-forward layer with residual connection, input and output dimensions must match | None |

### Loss Functions

| Implementation | File | Status | Testing | Description | Notes |
|----------------|------|--------|---------|-------------|-------|
| `MeanSquaredError<T>` / `MSE<T>` | [losses.slang](losses.slang) | ✅ Fully implemented | ✅ Tested | L2 loss | None |
| `MeanAbsoluteError<T>` / `MAE<T>` | [losses.slang](losses.slang) | ✅ Fully implemented | ✅ Tested | L1 loss | None |
| `MeanAbsolutePercentageError<T>` / `MAPE<T>` | [losses.slang](losses.slang) | ✅ Fully implemented | ❌ Not tested | Percentage-based loss | None |

### Optimizers

| Implementation | File | Status | Testing | Description | Notes |
|----------------|------|--------|---------|-------------|-------|
| `SGD<T>` | [optimizers.slang](optimizers.slang) | ✅ Fully implemented | ✅ Tested | Stochastic Gradient Descent with optional momentum | None |
| `Adam<T>` | [optimizers.slang](optimizers.slang) | ✅ Fully implemented | ✅ Tested | Adaptive learning rates with configurable beta1, beta2, epsilon | None |
| `AdamW<T>` | - | ❌ Not implemented | ➖ N/A | Adam with weight decay | None |
| `NAdam<T>` | - | ❌ Not implemented | ➖ N/A | Nesterov-accelerated Adam | None |

### Encoders (Non-learnable)

| Implementation | File | Status | Testing | Description | Notes |
|----------------|------|--------|---------|-------------|-------|
| `IdentityEncoder<T, N>` | [encoders.slang](encoders.slang) | ✅ Fully implemented | ❌ Not tested | Pass-through encoder | None |
| `FrequencyEncoder<T, Dim, Levels>` | [encoders.slang](encoders.slang) | ✅ Fully implemented | ✅ Tested | Fourier feature encoding with custom backward pass | None |
| `OneBlobEncoder<T, N, K>` | [encoders.slang](encoders.slang) | ✅ Fully implemented | ❌ Not tested | Gaussian kernel encoding for continuous representations | None |
| `SphericalHarmonicsEncoder<T, Levels>` | [encoders.slang](encoders.slang) | ❌ Not implemented | ➖ N/A | Placeholder for spherical harmonics encoding | None |

### Learnable Encoders

| Implementation | File | Status | Testing | Description | Notes |
|----------------|------|--------|---------|-------------|-------|
| `RandomFourierEncoder<T, Dim, Features, Storage>` | [encoders.slang](encoders.slang) | ✅ Fully implemented | ❌ Not tested | Learnable random Fourier features | None |
| `DenseGrid<Dimension, Features>` | [../ngp/slang/dense_grid.slang](../ngp/slang/dense_grid.slang) | ⚠️ Partially implemented | ❌ Not tested | Dense grid encoder | None |
| `MultiLevelDenseGrid<Features>` | [../ngp/slang/multi_level_dense_grid.slang](../ngp/slang/multi_level_dense_grid.slang) | ⚠️ Partially implemented | ❌ Not tested | Multi-resolution dense grid | Lacks generality due to Slang limitations |
| `HashGrid<Dimension, Features>` | - | ❌ Not implemented | ➖ N/A | Hash-based grid encoder | None |
| `MultiLevelHashGrid<Dimension, Features>` | - | ❌ Not implemented | ➖ N/A | Multi-resolution hash grid | None |

### Utility Functions

| Function | File | Status | Testing | Description | Notes |
|----------|------|--------|---------|-------------|-------|
| `concat` | [vector_ops.slang](vector_ops.slang) | ✅ Fully implemented | ❌ Not tested | Concatenate two vectors with full differentiability | None |
