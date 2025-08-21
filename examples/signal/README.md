# Signal Reconstruction Example

Validation test comparing PyTorch and SlangPy implementations.

## Overview

Trains identical networks on both frameworks to fit a 1D signal, tracking numerical differences in weights, gradients, and outputs to verify correctness.

## Usage

```bash
python -m examples.signal.main
```

## Expected Results

Weight/gradient deltas should remain small, demonstrating numerical equivalence between implementations.
