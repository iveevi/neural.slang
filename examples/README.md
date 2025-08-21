# **neural.slang** Examples

This directory contains example applications demonstrating various use cases of the **neural.slang** framework.

## Examples

| Example | Description |
|---------|-------------|
| [**benchmark**](./benchmark/) | Performance comparison between PyTorch and SlangPy implementations of neural networks |
| [**deferred**](./deferred/) | Neural deferred shading that learns lighting and shading effects in real-time |
| [**sdf**](./sdf/) | Neural network learning to represent 3D shapes as signed distance fields |
| [**signal**](./signal/) | 1D signal reconstruction with Fourier encoding, comparing PyTorch and SlangPy implementations |
| [**texture**](./texture/) | Neural texture synthesis and compression |

## Running Examples

```bash
# Run the SDF learning example
python -m examples.sdf.main

# Run the deferred shading example
python -m examples.deferred.main

# Run the texture synthesis example
python -m examples.texture.main
```
