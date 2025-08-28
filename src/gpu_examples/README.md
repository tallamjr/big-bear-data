# GPU Examples for Polars

This directory contains comprehensive examples for using Polars' GPU acceleration features. These examples demonstrate the latest API usage and best practices for GPU-accelerated data processing.

## Quick Start

```bash
# Basic GPU usage
python gpu_basic.py

# Advanced GPU configuration
python gpu_advanced.py

# GPU monitoring and debugging
python gpu_monitoring.py

# Performance benchmarking
python gpu_benchmarks.py
```

## Prerequisites

- NVIDIA GPU with compute capability 7.0+ (Volta architecture or newer)
- CUDA 12 (CUDA 11 support deprecated after RAPIDS v25.06)
- Linux or WSL2
- 24GB+ VRAM recommended for large datasets

### Installation

```bash
# Install GPU dependencies
../install-gpu-deps.sh

# Or manually:
pip install polars[gpu] --extra-index-url=https://pypi.nvidia.com
pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.2.*"
```

## Examples Overview

### 1. `gpu_basic.py` - Fundamental GPU Usage

**Purpose**: Learn the basics of GPU acceleration with automatic fallback.

**Features**:
- GPU detection and fallback to streaming engine
- Basic aggregation operations
- Engine comparison demonstrations

**Key Concepts**:
```python
# Dynamic engine selection
if is_nvidia_gpu_available():
    engine = "gpu"
else:
    engine = "streaming"

result = query.collect(engine=engine)
```

### 2. `gpu_advanced.py` - Advanced Configuration

**Purpose**: Master advanced GPU engine configuration and multi-GPU setups.

**Features**:
- `GPUEngine` configuration with device selection
- GPU streaming executor (RAPIDS 25.06+)
- Multi-GPU distribution with Dask
- Fallback behavior control

**Key Concepts**:
```python
# Advanced GPU configuration
gpu_engine = GPUEngine(
    device=0,                    # GPU device selection
    raise_on_fail=True,         # Disable CPU fallback
)

# GPU streaming for larger-than-VRAM datasets
streaming_gpu = GPUEngine(
    executor="streaming",
    executor_options={"scheduler": "synchronous"}
)
```

### 3. `gpu_monitoring.py` - Debugging and Monitoring

**Purpose**: Monitor GPU usage, debug performance issues, and troubleshoot problems.

**Features**:
- Real-time GPU memory monitoring
- Verbose debugging with `pl.Config().set_verbose(True)`
- CUDA environment validation
- Performance profiling

**Key Concepts**:
```python
# Enable verbose debugging
with pl.Config() as cfg:
    cfg.set_verbose(True)
    result = query.collect(engine="gpu")

# GPU memory monitoring
with gpu_monitoring() as monitor:
    result = query.collect(engine="gpu")
    monitor.print_summary()
```

### 4. `gpu_benchmarks.py` - Performance Analysis

**Purpose**: Comprehensive benchmarking suite to measure GPU acceleration benefits.

**Features**:
- Multi-engine performance comparison
- Statistical analysis of execution times
- Memory usage profiling
- Query type optimization recommendations

**Key Concepts**:
```python
# Benchmark different engines
engines = ["cpu", "streaming", "gpu"]
for engine in engines:
    time = benchmark_query(query, engine)
    print(f"{engine}: {time:.3f}s")
```

## Engine Selection Guide

| Engine | Use Case | Best For |
|--------|----------|----------|
| `"gpu"` | GPU acceleration with CPU fallback | Grouped aggregations, joins |
| `"streaming"` | Larger-than-memory datasets | Very large datasets, limited RAM |
| `"cpu"` | Standard in-memory processing | Small to medium datasets |
| `GPUEngine(raise_on_fail=True)` | Strict GPU validation | Development, testing |

## Performance Expectations

Based on our benchmarking with NYC taxi data (1.5B rows):

| Operation Type | GPU Speedup | Recommendation |
|----------------|-------------|----------------|
| **Grouped Aggregations** | 5-15x | PASS Use GPU |
| **Joins** | 3-10x | PASS Use GPU |
| **Filters & Selections** | 2-5x | PASS Use GPU |
| **String Processing** | 2-5x | WARNING Mixed results |
| **I/O Operations** | 1-2x | FAIL Minimal benefit |

## Environment Variables

Configure GPU behavior through environment variables:

```bash
# Enable verbose debugging
export POLARS_VERBOSE=1

# Use GPU engine affinity
export POLARS_ENGINE_AFFINITY=gpu

# Custom GPU configuration (main.py specific)
export POLARS_GPU_CONFIG=strict    # No CPU fallback
export POLARS_GPU_CONFIG=verbose   # Enable debugging
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Test Polars GPU installation
python -c "import polars as pl; print(pl.LazyFrame({'a': [1,2,3]}).collect(engine='gpu'))"
```

### Performance Issues

1. **Memory Errors**: Use streaming engine for larger datasets
2. **Slow Performance**: Check if operations are GPU-supported
3. **Fallback Warnings**: Enable verbose mode to identify unsupported operations

### Common Error Patterns

```python
# Handle GPU limitations gracefully
try:
    result = query.collect(engine="gpu")
except pl.exceptions.ComputeError:
    result = query.collect(engine="streaming")
```

## API Migration Notes

Recent API changes for streaming:

```python
# Old API (deprecated)
result = lf.collect(streaming=True)

# New API (current)
result = lf.collect(engine="streaming")
```

## Next Steps

1. Start with `gpu_basic.py` to understand fundamentals
2. Explore `gpu_advanced.py` for production configurations
3. Use `gpu_monitoring.py` to debug performance issues
4. Run `gpu_benchmarks.py` to measure your workload performance

For more information, see the [main README GPU section](../../README.md#gpu-acceleration) and [official Polars GPU documentation](https://docs.pola.rs/user-guide/gpu-support/).
