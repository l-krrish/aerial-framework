# Tensor Power Normalization Module

## Overview
This module adds Lp power normalization for 1D floating-point input vectors.

Given an input vector $x$ and parameters $p$ and $\epsilon$, the output is:

$$
y_i = \frac{x_i}{\left(\sum_j |x_j|^p + \epsilon\right)^{1/p}}
$$

The module supports:
- Returning a normalized copy
- In-place normalization

## What Was Implemented
- A public configuration type with:
  - `p`: norm order
  - `epsilon`: numerical stability term
- A normalization API that returns a new vector
- An in-place normalization API
- Shared denominator computation logic used by both APIs
- Input/config validation with exceptions for invalid cases

## Validation Rules
The implementation throws invalid argument errors when:
- Input is empty
- `p` is less than or equal to zero
- `epsilon` is negative

## Numerical Behavior
- Absolute value is used before applying power: $|x|^p$
- Epsilon is added before root extraction: $(\text{sum} + \epsilon)^{1/p}$
- This improves stability for near-zero or all-zero inputs

## Test Coverage
The test suite validates:
- L2 normalization correctness using a known vector
- L1 normalization with signed values
- In-place behavior
- Zero-vector stability with epsilon
- Exception behavior for invalid input/config

## Design Notes
- Validation and denominator logic are centralized to avoid divergence between APIs
- The implementation uses transform-style element-wise normalization for clarity and consistency
- API design matches existing tensor module conventions

## Future Improvements
- Support for additional floating-point types
- Batched normalization helpers
- Optional SIMD/GPU-accelerated backend paths
- Axis-based normalization for higher-dimensional tensors
