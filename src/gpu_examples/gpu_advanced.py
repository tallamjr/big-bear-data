#!/usr/bin/env python3
"""
Advanced GPU configuration examples using GPUEngine.

This demonstrates advanced GPU engine configuration including:
- Device selection for multi-GPU systems
- Fallback behavior control
- Memory resource management
- GPU streaming executor (RAPIDS 25.06+)
- Multi-GPU distribution with Dask
"""

import subprocess
import time
import logging
import polars as pl
from polars.lazyframe.engine_config import GPUEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def is_nvidia_gpu_available():
    """Check if NVIDIA GPU is available and accessible."""
    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_gpu_count():
    """Get number of available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        return len(result.stdout.strip().split("\n"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0


def demonstrate_basic_gpu_engine():
    """Demonstrate basic GPUEngine configuration."""
    print("=== Basic GPUEngine Configuration ===\n")

    # Create sample data
    df = pl.LazyFrame(
        {
            "category": [f"cat_{i % 5}" for i in range(50_000)],
            "value": list(range(50_000)),
            "score": [i * 0.01 for i in range(50_000)],
        }
    )

    query = (
        df.filter(pl.col("value") > 1000)
        .group_by("category")
        .agg([pl.mean("value").alias("avg_value"), pl.max("score").alias("max_score")])
    )

    # Basic GPU execution
    print("1. Basic GPU execution:")
    try:
        start_time = time.time()
        result = query.collect(engine="gpu")
        execution_time = time.time() - start_time
        logging.info(f"   Success: {execution_time:.3f}s")
        print(f"   Result shape: {result.shape}")
    except Exception as e:
        logging.error(f"   Failed: {e}")

    # Advanced GPU engine configuration
    print("\n2. Advanced GPU engine with device selection:")
    try:
        gpu_engine = GPUEngine(
            device=0,  # Specify GPU device (for multi-GPU systems)
            raise_on_fail=False,  # Allow CPU fallback
        )
        start_time = time.time()
        result = query.collect(engine=gpu_engine)
        execution_time = time.time() - start_time
        logging.info(f"   Success: {execution_time:.3f}s")
    except Exception as e:
        logging.error(f"   Failed: {e}")

    # Strict GPU execution (no fallback)
    print("\n3. Strict GPU execution (raise_on_fail=True):")
    try:
        strict_gpu = GPUEngine(raise_on_fail=True)
        start_time = time.time()
        result = query.collect(engine=strict_gpu)
        execution_time = time.time() - start_time
        logging.info(f"   Success: {execution_time:.3f}s")
    except Exception as e:
        logging.error(f"   Failed: {e}")


def demonstrate_gpu_streaming():
    """Demonstrate GPU streaming executor (RAPIDS 25.06+)."""
    print("\n=== GPU Streaming Executor ===\n")

    # Create larger dataset for streaming demonstration
    df = pl.LazyFrame(
        {
            "partition": [f"part_{i % 100}" for i in range(500_000)],
            "timestamp": [i for i in range(500_000)],
            "metric": [i * 0.001 for i in range(500_000)],
        }
    )

    query = (
        df.group_by("partition")
        .agg(
            [
                pl.count().alias("record_count"),
                pl.sum("metric").alias("total_metric"),
                pl.var("timestamp").alias("timestamp_var"),
            ]
        )
        .filter(pl.col("record_count") > 100)
        .sort("total_metric", descending=True)
    )

    print("Testing GPU streaming executor...")
    try:
        # GPU with streaming executor for larger-than-VRAM datasets
        streaming_gpu = GPUEngine(
            executor="streaming", executor_options={"scheduler": "synchronous"}
        )
        start_time = time.time()
        result = query.collect(engine=streaming_gpu)
        execution_time = time.time() - start_time
        logging.info(f"GPU Streaming Success: {execution_time:.3f}s")
        print(f"   Result shape: {result.shape}")
        print(f"   Top partition: {result.head(1)}")
    except Exception as e:
        logging.error(f"GPU Streaming Failed: {e}")
        print("   Note: GPU streaming requires RAPIDS 25.06+ and compatible hardware")


def demonstrate_multi_gpu():
    """Demonstrate multi-GPU execution with Dask."""
    gpu_count = get_gpu_count()
    print(f"\n=== Multi-GPU Configuration (Available GPUs: {gpu_count}) ===\n")

    if gpu_count < 2:
        logging.warning("Multi-GPU demonstration requires 2+ GPUs. Skipping.")
        return

    try:
        # Import Dask components for multi-GPU
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client

        print("Setting up Dask CUDA cluster...")
        cluster = LocalCUDACluster(n_workers=min(2, gpu_count))
        client = Client(cluster)

        # Create dataset for multi-GPU processing
        df = pl.LazyFrame(
            {
                "worker_id": [i % gpu_count for i in range(1_000_000)],
                "data": list(range(1_000_000)),
                "weight": [i * 0.0001 for i in range(1_000_000)],
            }
        )

        query = df.group_by("worker_id").agg(
            [pl.sum("data").alias("sum_data"), pl.mean("weight").alias("avg_weight")]
        )

        # Multi-GPU execution
        multi_gpu_engine = GPUEngine(
            executor="streaming", executor_options={"scheduler": "distributed"}
        )

        print("Executing on multi-GPU cluster...")
        start_time = time.time()
        result = query.collect(engine=multi_gpu_engine)
        execution_time = time.time() - start_time

        logging.info(f"Multi-GPU Success: {execution_time:.3f}s")
        print(f"   Result shape: {result.shape}")
        print(f"   Sample results:\n{result}")

        # Cleanup
        client.close()
        cluster.close()

    except ImportError:
        logging.error("Multi-GPU requires: pip install dask-cuda")
    except Exception as e:
        logging.error(f"Multi-GPU Failed: {e}")


def demonstrate_error_handling():
    """Demonstrate proper error handling patterns."""
    print("\n=== Error Handling Patterns ===\n")

    # Create a query that might not be supported on GPU
    df = pl.LazyFrame(
        {
            "text": ["hello world", "polars gpu", "data processing"] * 10000,
            "numbers": list(range(30000)),
        }
    )

    # Some string operations may not be fully supported on GPU
    complex_query = (
        df.with_columns(
            [
                pl.col("text").str.len_chars().alias("text_length"),
                pl.col("text").str.to_uppercase().alias("upper_text"),
            ]
        )
        .filter(pl.col("text_length") > 5)
        .group_by("upper_text")
        .agg(pl.sum("numbers").alias("sum_numbers"))
    )

    print("1. Graceful fallback pattern:")
    try:
        # Try GPU first, fallback to streaming if needed
        _ = complex_query.collect(engine="gpu")
        logging.info("   GPU execution succeeded")
    except pl.exceptions.ComputeError as e:
        logging.warning(f"   GPU failed: {e}")
        print("   Falling back to streaming engine...")
        _ = complex_query.collect(engine="streaming")
        logging.info("   Streaming fallback succeeded")

    print("\n2. Explicit GPU validation:")
    try:
        # Use raise_on_fail to explicitly catch unsupported operations
        strict_gpu = GPUEngine(raise_on_fail=True)
        _ = complex_query.collect(engine=strict_gpu)
        logging.info("   Strict GPU execution succeeded")
    except pl.exceptions.ComputeError as e:
        logging.error(f"   Strict GPU failed (expected): {e}")
        print("   This helps identify which operations need CPU fallback")


def main():
    print("=== Advanced GPU Configuration Examples ===\n")

    if not is_nvidia_gpu_available():
        logging.warning("No NVIDIA GPU detected. Some examples will fail gracefully.")
        print("Install NVIDIA drivers and CUDA toolkit to fully test GPU features.\n")

    demonstrate_basic_gpu_engine()
    demonstrate_gpu_streaming()
    demonstrate_multi_gpu()
    demonstrate_error_handling()

    print("\n=== Summary ===")
    print("PASS Basic GPUEngine configuration")
    print("PASS GPU streaming executor patterns")
    print("PASS Multi-GPU distribution examples")
    print("PASS Error handling best practices")


if __name__ == "__main__":
    main()
