#!/usr/bin/env python3
"""
Advanced GPU patterns for production workloads.

This module contains comprehensive GPU management patterns that were
previously embedded in documentation but are better suited as
executable examples.
"""

import subprocess
import polars as pl
import psutil
import time
import logging
from typing import Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class GPUManager:
    """Intelligent GPU management with comprehensive fallback strategies."""

    def __init__(self, prefer_gpu: bool = True, memory_fraction: float = 0.8):
        self.prefer_gpu = prefer_gpu
        self.memory_fraction = memory_fraction
        self.gpu_available = self._detect_gpu()

    def _detect_gpu(self) -> bool:
        """Detect NVIDIA GPU with proper error handling."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )

            gpu_info = result.stdout.strip().split("\n")
            logger.info(f"Detected GPUs: {len(gpu_info)}")
            for i, info in enumerate(gpu_info):
                name, memory = info.split(", ")
                logger.info(f"  GPU {i}: {name} ({memory})")

            return True

        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ) as e:
            logger.warning(f"GPU detection failed: {e}")
            return False

    def get_optimal_engine(
        self, query_complexity: str = "medium"
    ) -> Union[str, pl.GPUEngine]:
        """Select optimal engine based on query complexity and GPU availability."""

        if not self.gpu_available or not self.prefer_gpu:
            return "streaming"

        if query_complexity == "simple":
            return pl.GPUEngine(
                device=0, raise_on_fail=False, memory_resource="managed"
            )
        elif query_complexity == "complex":
            return pl.GPUEngine(device=0, raise_on_fail=True, memory_resource="pool")
        else:  # medium complexity
            return "gpu"


def execute_with_gpu_monitoring(
    query: pl.LazyFrame, gpu_manager: GPUManager, query_complexity: str = "medium"
) -> pl.DataFrame:
    """Execute query with comprehensive GPU monitoring and fallback."""

    engine = gpu_manager.get_optimal_engine(query_complexity)
    start_time = time.time()

    process = psutil.Process()
    start_memory = process.memory_info().rss / 1e9

    try:
        logger.info(f"Executing with engine: {engine}")

        with pl.Config() as cfg:
            cfg.set_verbose(True)
            result = query.collect(engine=engine)

        end_time = time.time()
        end_memory = process.memory_info().rss / 1e9

        logger.info("GPU execution successful:")
        logger.info(f"   Time: {end_time - start_time:.2f}s")
        logger.info(f"   Memory delta: {end_memory - start_memory:+.2f} GB")
        logger.info(f"   Result shape: {result.shape}")

        return result

    except pl.exceptions.ComputeError as e:
        logger.error(f"GPU execution failed: {e}")

        if "out of memory" in str(e).lower():
            logger.info("Falling back to streaming due to memory constraints")
            return query.collect(engine="streaming")
        elif "not implemented" in str(e).lower():
            logger.info("Falling back to CPU due to unsupported operation")
            return query.collect()
        else:
            logger.info("Generic fallback to streaming")
            return query.collect(engine="streaming")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.info("Emergency fallback to standard CPU execution")
        return query.collect()


def benchmark_gpu_vs_cpu(query: pl.LazyFrame, iterations: int = 3):
    """Benchmark GPU vs CPU performance."""

    results = {"gpu": [], "cpu": [], "streaming": []}

    logger.info("Running GPU vs CPU benchmark...")

    for engine_name, engine in [
        ("gpu", "gpu"),
        ("cpu", None),
        ("streaming", "streaming"),
    ]:
        logger.info(f"Testing {engine_name}...")

        for i in range(iterations):
            try:
                start_time = time.time()
                _ = query.collect(engine=engine) if engine else query.collect()
                end_time = time.time()

                execution_time = end_time - start_time
                results[engine_name].append(execution_time)
                logger.debug(f"  Run {i+1}: {execution_time:.2f}s")

            except Exception as e:
                logger.debug(f"  Run {i+1}: FAILED ({e})")
                results[engine_name].append(None)

    logger.info("Benchmark Results:")
    for engine_name, times in results.items():
        valid_times = [t for t in times if t is not None]
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            logger.info(
                f"{engine_name:10}: {avg_time:.2f}s avg ({len(valid_times)}/{iterations} successful)"
            )
        else:
            logger.warning(f"{engine_name:10}: All runs failed")


class ProductionGPUPipeline:
    """Production-ready GPU acceleration pipeline."""

    def __init__(self):
        self.gpu_manager = GPUManager(prefer_gpu=True, memory_fraction=0.8)

    def process_large_dataset(
        self, file_pattern: str, filters: dict = None
    ) -> pl.DataFrame:
        """Process large datasets with intelligent GPU acceleration."""

        query = pl.scan_parquet(file_pattern)

        if filters:
            for col, condition in filters.items():
                query = query.filter(pl.col(col) > condition)

        query = (
            query.group_by(["category", "region"])
            .agg(
                [
                    pl.col("sales").sum().alias("total_sales"),
                    pl.col("sales").mean().alias("avg_sales"),
                    pl.col("quantity").sum().alias("total_quantity"),
                    pl.len().alias("transaction_count"),
                ]
            )
            .sort("total_sales", descending=True)
        )

        complexity = "medium"  # Could be determined by query analysis

        return execute_with_gpu_monitoring(query, self.gpu_manager, complexity)


if __name__ == "__main__":
    # Example usage
    pipeline = ProductionGPUPipeline()

    # Create test data
    test_df = pl.LazyFrame(
        {
            "category": ["A", "B", "C"] * 1000,
            "region": ["North", "South"] * 1500,
            "sales": range(3000),
            "quantity": range(1, 3001),
        }
    )

    # Test GPU execution
    try:
        result = execute_with_gpu_monitoring(
            test_df.group_by("category").agg(pl.col("sales").sum()),
            GPUManager(),
            "simple",
        )
        logger.info(f"Test completed: {result.shape}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
