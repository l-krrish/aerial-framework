# GPU Signal Processing Module — NVIDIA Aerial Framework

A custom real-time signal processing module built in C++20 and CUDA inside [NVIDIA's Aerial Framework](https://github.com/NVIDIA/aerial-framework), the open-source software stack powering AI-native 6G base stations.

## Overview

This project implements a GPU-accelerated **ScaleModule** — a signal processing pipeline block that applies element-wise scaling to float tensors at microsecond latency. It follows the full production module architecture of the Aerial Framework, including memory management, CUDA graph integration, and stream-mode execution.

Built as part of the [NVIDIA 6G Developer Program](https://developer.nvidia.com/6g-program), verified on an RTX 4090 (compute capability 8.9).

## What It Does

```
input[i]  ──►  output[i] = input[i] * scale
```

Each element of the input tensor is multiplied by a configurable scale factor. The operation runs in parallel across thousands of GPU threads, with one thread per element.

## Files

| File | Description |
|------|-------------|
| `framework/pipeline/samples/src/my_scale_module_kernel.cuh` | CUDA kernel header — defines `StaticKernelParams` (output ptr, size, scale) and `DynamicKernelParams` (input ptr), declares the kernel |
| `framework/pipeline/samples/src/my_scale_module_kernel.cu` | CUDA kernel implementation — each thread computes `output[idx] = input[idx] * scale` |
| `framework/pipeline/samples/src/my_scale_module.hpp` | C++ module class declaration — inherits from `IModule`, `IAllocationInfoProvider`, `IGraphNodeProvider`, `IStreamExecutor` |
| `framework/pipeline/samples/src/my_scale_module.cpp` | Full module implementation — memory management, kernel launch configuration, stream and graph mode execution |

## Architecture

The Aerial Framework organizes signal processing as a pipeline of modules. Each module:

1. **Declares its ports** — named input/output connections with tensor shape information
2. **Allocates memory once** in `setup_memory()` — no dynamic allocation in the hot path
3. **Updates per-iteration state** in `configure_io()` — input pointer and dynamic parameters
4. **Executes** via `execute()` in stream mode or `add_node_to_graph()` in CUDA graph mode

ScaleModule separates parameters into two structs:
- **Static params** — output pointer, tensor size, scale factor (set once at setup)
- **Dynamic params** — input pointer (updated each iteration)

This indirection pattern allows CUDA graphs to capture the kernel launch and replay it efficiently with minimal CPU overhead.

## Key CUDA Concepts Used

- **Thread indexing** — `idx = blockIdx.x * blockDim.x + threadIdx.x` maps each thread to one element
- **Grid/block dimensions** — grid size computed as `(tensor_size + 255) / 256`, block size fixed at 256
- **CUDA streams** — async kernel execution without blocking the CPU
- **CUDA graphs** — captured kernel execution replayed with deterministic timing
- **Kernel descriptor pattern** — static/dynamic parameter indirection for graph-mode compatibility

## Build

Requires the Aerial Framework container and a GPU with compute capability ≥ 8.0.

```bash
git clone https://github.com/l-krrish/aerial-framework.git
cd aerial-framework

# Using the Aerial Framework Docker container
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/aerial/aerial-framework-base:v0.1.0 \
  bash -c "cmake --preset gcc-release -DCMAKE_CUDA_ARCHITECTURES=89 && \
           cmake --build out/build/gcc-release --target sample-pipeline -j8"
```

Syntax verification (no full build required):

```bash
# C++ module
g++ -std=c++20 [includes...] -fsyntax-only my_scale_module.cpp

# CUDA kernel
nvcc -std=c++20 -arch=sm_89 -dc my_scale_module_kernel.cu -o kernel.o
```

Both verified clean on RTX 4090 (sm_89).

## Tech Stack

- **C++20** — module class, pipeline interfaces, memory management
- **CUDA 12.9** — GPU kernel, stream and graph mode execution
- **CMake** — build system integration
- **NVIDIA Aerial Framework v0.1.0** — production RAN pipeline architecture
- **RTX 4090** — verified on WATcloud compute cluster

## Context

This module was built as part of learning GPU systems programming using NVIDIA's real production codebase for 6G wireless infrastructure. The Aerial Framework is used by telecom operators building AI-native RAN (Radio Access Network) systems.

Future modules planned:
- Power Normalization Module (parallel reduction pattern)
- Noise Injection Module (cuRAND GPU random number generation)
- Complex Number Scaling Module (struct-based tensor types)
- Full pipeline chaining all modules
