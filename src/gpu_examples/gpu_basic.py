#!/usr/bin/env python3
"""
Basic GPU usage example with automatic fallback to streaming engine.

This demonstrates the fundamental pattern for GPU acceleration in Polars,
including proper GPU detection and graceful fallback to streaming mode
when GPU is unavailable or unsuitable.
"""

import subprocess
import time
import logging
import polars as pl

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


def main():
    print("=== Basic GPU Usage Example ===\n")

    # Create sample data for demonstration
    print("Creating sample dataset...")
    df = pl.LazyFrame(
        {
            "group": [f"group_{i % 10}" for i in range(100_000)],
            "value": list(range(100_000)),
            "weight": [i * 0.1 for i in range(100_000)],
        }
    )

    # Dynamic engine selection based on GPU availability
    collect_args = {}
    if is_nvidia_gpu_available():
        collect_args["engine"] = "gpu"
        logging.info("NVIDIA GPU detected, using GPU engine for collection.")
    else:
        collect_args["engine"] = "streaming"
        logging.warning(
            "No NVIDIA GPU detected, using streaming engine for collection."
        )

    print(f"Selected engine: {collect_args['engine']}\n")

    # Basic aggregation query
    query = (
        df.group_by("group")
        .agg(
            [
                pl.count().alias("count"),
                pl.sum("value").alias("sum_value"),
                pl.mean("weight").alias("avg_weight"),
            ]
        )
        .sort("sum_value", descending=True)
    )

    print("Executing query...")
    start_time = time.time()
    result = query.collect(**collect_args)
    execution_time = time.time() - start_time

    print(f"Query completed in {execution_time:.3f} seconds")
    print(f"Result shape: {result.shape}")
    print("\nTop 5 groups by sum_value:")
    print(result.head())

    # Demonstrate different engine options
    print("\n=== Engine Comparison ===")
    engines = ["cpu", "streaming"]
    if is_nvidia_gpu_available():
        engines.append("gpu")

    for engine in engines:
        print(f"\nTesting engine: {engine}")
        start_time = time.time()
        try:
            result = query.collect(engine=engine)
            execution_time = time.time() - start_time
            logging.info(f"  Success: {execution_time:.3f}s")
        except Exception as e:
            logging.error(f"  Failed: {e}")


if __name__ == "__main__":
    main()
