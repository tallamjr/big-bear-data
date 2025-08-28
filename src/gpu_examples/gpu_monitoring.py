#!/usr/bin/env python3
"""
GPU monitoring and debugging utilities for Polars.

This demonstrates:
- GPU memory monitoring during query execution
- Verbose debugging to understand GPU fallbacks
- Performance profiling for GPU vs CPU engines
- CUDA runtime diagnostics
"""

import subprocess
import time
import threading
import logging
from contextlib import contextmanager
from typing import List, Dict, Any
import polars as pl
from polars.lazyframe.engine_config import GPUEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GPUMonitor:
    """Monitor GPU memory usage during query execution."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.monitoring = False
        self.memory_history: List[Dict[str, Any]] = []
        self.monitor_thread = None

    def _get_gpu_memory(self) -> Dict[str, Any]:
        """Get current GPU memory usage."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            lines = result.stdout.strip().split("\n")
            gpu_stats = []

            for i, line in enumerate(lines):
                used, total, util = line.split(", ")
                gpu_stats.append(
                    {
                        "device": i,
                        "memory_used_mb": int(used),
                        "memory_total_mb": int(total),
                        "memory_percent": (int(used) / int(total)) * 100,
                        "gpu_utilization": int(util),
                        "timestamp": time.time(),
                    }
                )

            return {"gpus": gpu_stats, "timestamp": time.time()}
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return {
                "gpus": [],
                "timestamp": time.time(),
                "error": "nvidia-smi unavailable",
            }

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            memory_info = self._get_gpu_memory()
            self.memory_history.append(memory_info)
            time.sleep(self.interval)

    def start_monitoring(self):
        """Start background GPU monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.memory_history.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop background GPU monitoring."""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()

    def get_peak_memory_usage(self) -> Dict[str, Any]:
        """Get peak memory usage during monitoring period."""
        if not self.memory_history:
            return {}

        peak_usage = {}
        for entry in self.memory_history:
            if "gpus" in entry:
                for gpu in entry["gpus"]:
                    device = gpu["device"]
                    if (
                        device not in peak_usage
                        or gpu["memory_used_mb"] > peak_usage[device]["memory_used_mb"]
                    ):
                        peak_usage[device] = gpu

        return peak_usage

    def print_summary(self):
        """Print monitoring summary."""
        if not self.memory_history:
            print("No monitoring data available")
            return

        peak_usage = self.get_peak_memory_usage()
        print("\n=== GPU Memory Summary ===")

        for device, stats in peak_usage.items():
            print(f"GPU {device}:")
            print(
                f"  Peak Memory: {stats['memory_used_mb']:,} MB / {stats['memory_total_mb']:,} MB ({stats['memory_percent']:.1f}%)"
            )
            print(f"  Peak GPU Utilization: {stats['gpu_utilization']}%")


@contextmanager
def gpu_monitoring(interval: float = 0.5):
    """Context manager for GPU monitoring."""
    monitor = GPUMonitor(interval)
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()


def check_cuda_environment():
    """Check CUDA environment and GPU capabilities."""
    print("=== CUDA Environment Check ===\n")

    # Check NVIDIA driver
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        )
        logging.info("NVIDIA Driver: Available")

        # Extract driver version
        lines = result.stdout.split("\n")
        for line in lines:
            if "Driver Version:" in line:
                driver_version = line.split("Driver Version:")[1].split()[0]
                print(f"   Driver Version: {driver_version}")
                break
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("NVIDIA Driver: Not available")
        return False

    # Check CUDA toolkit
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        )
        logging.info("CUDA Toolkit: Available")

        # Extract CUDA version
        for line in result.stdout.split("\n"):
            if "release" in line.lower():
                cuda_version = line.split("release")[1].split(",")[0].strip()
                print(f"   CUDA Version: {cuda_version}")
                break
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning(
            "CUDA Toolkit: Not in PATH (may still work with conda/pip CUDA)"
        )

    # Check GPU details
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        print("\nGPU Details:")
        for i, line in enumerate(result.stdout.strip().split("\n")):
            name, memory, compute_cap = line.split(", ")
            print(f"   GPU {i}: {name}")
            print(f"     Memory: {memory}")
            print(f"     Compute Capability: {compute_cap}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return True


def demonstrate_verbose_debugging():
    """Demonstrate verbose debugging for GPU execution."""
    print("\n=== Verbose Debugging ===\n")

    # Create a query that might have GPU limitations
    df = pl.LazyFrame(
        {
            "group": [f"group_{i % 10}" for i in range(100_000)],
            "values": list(range(100_000)),
            "text": [f"string_{i}" for i in range(100_000)],
        }
    )

    query = (
        df.with_columns(
            [
                pl.col("text").str.len_chars().alias("text_length"),
                pl.col("values").rolling_mean(window_size=100).alias("rolling_avg"),
            ]
        )
        .group_by("group")
        .agg(
            [
                pl.sum("values").alias("sum_values"),
                pl.mean("text_length").alias("avg_text_length"),
            ]
        )
    )

    print("1. Standard execution (may show warnings):")
    with pl.Config() as cfg:
        cfg.set_verbose(True)
        try:
            _ = query.collect(engine="gpu")
            logging.info("   GPU execution completed")
        except Exception as e:
            logging.error(f"   GPU execution failed: {e}")

    print("\n2. Explicit fallback detection:")
    try:
        gpu_engine = GPUEngine(raise_on_fail=True)
        _ = query.collect(engine=gpu_engine)
        logging.info("   All operations supported on GPU")
    except pl.exceptions.ComputeError as e:
        logging.warning(f"   GPU limitation detected: {e}")
        logging.info("   This query requires CPU fallback for some operations")


def benchmark_engines():
    """Benchmark different engines with monitoring."""
    print("\n=== Engine Performance Benchmark ===\n")

    # Create substantial dataset for meaningful benchmarks
    df = pl.LazyFrame(
        {
            "category": [f"cat_{i % 50}" for i in range(1_000_000)],
            "timestamp": list(range(1_000_000)),
            "value": [i * 0.001 for i in range(1_000_000)],
            "weight": [(i % 1000) * 0.01 for i in range(1_000_000)],
        }
    )

    # Complex query for benchmarking
    query = (
        df.filter(pl.col("timestamp") > 50_000)
        .with_columns(
            [
                (pl.col("value") * pl.col("weight")).alias("weighted_value"),
                pl.col("timestamp").cast(pl.Float64).alias("ts_float"),
            ]
        )
        .group_by("category")
        .agg(
            [
                pl.count().alias("count"),
                pl.sum("weighted_value").alias("total_weighted"),
                pl.mean("ts_float").alias("avg_timestamp"),
                pl.var("value").alias("value_variance"),
            ]
        )
        .filter(pl.col("count") > 1000)
        .sort("total_weighted", descending=True)
    )

    engines = ["cpu", "streaming"]
    if check_cuda_environment():
        engines.append("gpu")

    results = {}

    for engine in engines:
        print(f"\nTesting {engine} engine:")

        with gpu_monitoring() as monitor:
            start_time = time.time()
            try:
                result = query.collect(engine=engine)
                execution_time = time.time() - start_time

                logging.info(f"   Success: {execution_time:.3f}s")
                print(f"   Result shape: {result.shape}")

                results[engine] = {
                    "time": execution_time,
                    "success": True,
                    "shape": result.shape,
                }

                if engine == "gpu":
                    monitor.print_summary()

            except Exception as e:
                execution_time = time.time() - start_time
                logging.error(f"   Failed: {e}")
                results[engine] = {
                    "time": execution_time,
                    "success": False,
                    "error": str(e),
                }

    # Print benchmark summary
    print("\n=== Benchmark Summary ===")
    successful_results = {k: v for k, v in results.items() if v["success"]}

    if successful_results:
        fastest_engine = min(
            successful_results.keys(), key=lambda x: successful_results[x]["time"]
        )
        print(f"Fastest engine: {fastest_engine}")

        for engine, result in successful_results.items():
            speedup = ""
            if engine != fastest_engine:
                ratio = result["time"] / successful_results[fastest_engine]["time"]
                speedup = f" ({ratio:.2f}x slower)"
            print(f"   {engine}: {result['time']:.3f}s{speedup}")


def main():
    """Main demonstration function."""
    print("=== GPU Monitoring and Debugging Utilities ===\n")

    check_cuda_environment()
    demonstrate_verbose_debugging()
    benchmark_engines()

    print("\n=== Monitoring Tips ===")
    print("Use POLARS_VERBOSE=1 environment variable for debugging")
    print("Monitor GPU memory with: nvidia-smi -l 1")
    print("Use GPUEngine(raise_on_fail=True) to catch unsupported operations")
    print("Consider streaming engine for larger-than-VRAM datasets")


if __name__ == "__main__":
    main()
