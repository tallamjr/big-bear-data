#!/usr/bin/env python3
"""
GPU performance benchmarking suite for Polars.

This provides comprehensive benchmarking tools to:
- Compare engine performance across different query types
- Measure GPU acceleration benefits
- Identify optimal engine choices for different workloads
- Profile memory usage patterns
"""

import subprocess
import time
import statistics
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import polars as pl
from polars.lazyframe.engine_config import GPUEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    engine: str
    query_name: str
    execution_time: float
    memory_peak_mb: Optional[int]
    success: bool
    error_message: Optional[str]
    result_shape: Optional[tuple]


class PolarsGPUBenchmark:
    """Comprehensive benchmarking suite for Polars GPU performance."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for benchmarking."""
        try:
            subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _get_peak_memory(self) -> Optional[int]:
        """Get current GPU memory usage."""
        if not self.gpu_available:
            return None

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return int(result.stdout.strip().split("\n")[0])
        except Exception:
            return None

    def run_benchmark(
        self, query: pl.LazyFrame, query_name: str, engines: List[str], runs: int = 3
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmark across multiple engines and iterations."""
        print(f"\n=== Benchmarking: {query_name} ===")

        results_by_engine = {engine: [] for engine in engines}

        for engine in engines:
            if engine == "gpu" and not self.gpu_available:
                print(f"  Skipping {engine}: GPU not available")
                continue

            print(f"  Testing {engine} engine...")

            for run in range(runs):
                result = self._single_benchmark_run(query, query_name, engine, run + 1)
                results_by_engine[engine].append(result)
                self.results.append(result)

                if result.success:
                    logging.info(f"    Run {run + 1}: {result.execution_time:.3f}s")
                else:
                    logging.error(f"    Run {run + 1}: FAILED - {result.error_message}")

        return results_by_engine

    def _single_benchmark_run(
        self, query: pl.LazyFrame, query_name: str, engine: str, run_number: int
    ) -> BenchmarkResult:
        """Execute a single benchmark run."""
        memory_before = self._get_peak_memory()

        try:
            start_time = time.time()

            if engine == "gpu_strict":
                # Use strict GPU mode (no fallback)
                gpu_engine = GPUEngine(raise_on_fail=True)
                result = query.collect(engine=gpu_engine)
                engine_used = "gpu"
            else:
                result = query.collect(engine=engine)
                engine_used = engine

            execution_time = time.time() - start_time
            memory_after = self._get_peak_memory()

            memory_peak = None
            if memory_before and memory_after:
                memory_peak = max(memory_before, memory_after)

            return BenchmarkResult(
                engine=engine_used,
                query_name=query_name,
                execution_time=execution_time,
                memory_peak_mb=memory_peak,
                success=True,
                error_message=None,
                result_shape=result.shape,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return BenchmarkResult(
                engine=engine,
                query_name=query_name,
                execution_time=execution_time,
                memory_peak_mb=None,
                success=False,
                error_message=str(e),
                result_shape=None,
            )

    def print_summary(self):
        """Print comprehensive benchmark summary."""
        if not self.results:
            print("No benchmark results available.")
            return

        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        # Group results by query
        results_by_query = {}
        for result in self.results:
            if result.query_name not in results_by_query:
                results_by_query[result.query_name] = {}
            if result.engine not in results_by_query[result.query_name]:
                results_by_query[result.query_name][result.engine] = []
            results_by_query[result.query_name][result.engine].append(result)

        for query_name, engine_results in results_by_query.items():
            print(f"\nBenchmark: {query_name}")
            print("-" * 40)

            # Calculate statistics for each engine
            engine_stats = {}
            for engine, results in engine_results.items():
                successful_results = [r for r in results if r.success]
                if successful_results:
                    times = [r.execution_time for r in successful_results]
                    engine_stats[engine] = {
                        "mean_time": statistics.mean(times),
                        "min_time": min(times),
                        "max_time": max(times),
                        "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                        "success_rate": len(successful_results) / len(results),
                        "runs": len(results),
                    }

            # Find fastest engine
            if engine_stats:
                fastest_engine = min(
                    engine_stats.keys(), key=lambda x: engine_stats[x]["mean_time"]
                )
                fastest_time = engine_stats[fastest_engine]["mean_time"]

                for engine, stats in engine_stats.items():
                    speedup = fastest_time / stats["mean_time"]
                    if engine == fastest_engine:
                        speedup_text = "FASTEST"
                    else:
                        speedup_text = f"{1/speedup:.2f}x slower"

                    print(
                        f"  {engine:12s}: {stats['mean_time']:6.3f}s ± {stats['std_time']:5.3f}s  {speedup_text}"
                    )
                    if stats["success_rate"] < 1.0:
                        print(
                            f"               Success rate: {stats['success_rate']*100:.1f}% ({stats['runs']} runs)"
                        )

            # Show failures
            failed_engines = set()
            for engine, results in engine_results.items():
                failed_results = [r for r in results if not r.success]
                if failed_results:
                    failed_engines.add(engine)
                    print(f"  {engine:12s}: FAILED - {failed_results[0].error_message}")

        # Overall performance summary
        print("\nOverall Performance Analysis")
        print("-" * 40)

        successful_results = [r for r in self.results if r.success]
        if successful_results:
            gpu_results = [r for r in successful_results if r.engine == "gpu"]
            cpu_results = [r for r in successful_results if r.engine == "cpu"]
            streaming_results = [
                r for r in successful_results if r.engine == "streaming"
            ]

            if gpu_results and cpu_results:
                gpu_avg = statistics.mean([r.execution_time for r in gpu_results])
                cpu_avg = statistics.mean([r.execution_time for r in cpu_results])
                speedup = cpu_avg / gpu_avg
                print(f"  Average GPU speedup vs CPU: {speedup:.2f}x")

            if gpu_results and streaming_results:
                gpu_avg = statistics.mean([r.execution_time for r in gpu_results])
                streaming_avg = statistics.mean(
                    [r.execution_time for r in streaming_results]
                )
                speedup = streaming_avg / gpu_avg
                print(f"  Average GPU speedup vs Streaming: {speedup:.2f}x")


def create_benchmark_queries() -> Dict[str, pl.LazyFrame]:
    """Create various benchmark queries for different operation types."""

    # Dataset 1: Simple aggregations (GPU-friendly)
    df_simple = pl.LazyFrame(
        {
            "group": [f"group_{i % 100}" for i in range(1_000_000)],
            "value": list(range(1_000_000)),
            "weight": [i * 0.001 for i in range(1_000_000)],
        }
    )

    # Dataset 2: Complex transformations
    df_complex = pl.LazyFrame(
        {
            "category": [f"cat_{i % 50}" for i in range(500_000)],
            "timestamp": list(range(500_000)),
            "metric_a": [i * 0.01 for i in range(500_000)],
            "metric_b": [(i % 1000) * 0.1 for i in range(500_000)],
        }
    )

    # Dataset 3: String operations (potentially challenging for GPU)
    df_strings = pl.LazyFrame(
        {
            "text": [f"sample_text_{i}_end" for i in range(100_000)],
            "category": [f"cat_{i % 20}" for i in range(100_000)],
            "number": list(range(100_000)),
        }
    )

    queries = {
        "Simple Aggregation": (
            df_simple.group_by("group")
            .agg(
                [
                    pl.sum("value").alias("sum_value"),
                    pl.mean("weight").alias("avg_weight"),
                    pl.count().alias("count"),
                ]
            )
            .filter(pl.col("count") > 5)
            .sort("sum_value", descending=True)
        ),
        "Complex Analytics": (
            df_complex.with_columns(
                [
                    (pl.col("metric_a") * pl.col("metric_b")).alias("product"),
                    pl.col("timestamp").cast(pl.Float64).alias("ts_float"),
                ]
            )
            .filter(pl.col("product") > 0.5)
            .group_by("category")
            .agg(
                [
                    pl.sum("product").alias("total_product"),
                    pl.var("ts_float").alias("timestamp_var"),
                    pl.quantile("metric_a", 0.95).alias("metric_a_p95"),
                ]
            )
            .filter(pl.col("total_product") > 100)
        ),
        "String Processing": (
            df_strings.with_columns(
                [
                    pl.col("text").str.len_chars().alias("text_length"),
                    pl.col("text").str.slice(0, 10).alias("text_prefix"),
                ]
            )
            .filter(pl.col("text_length") > 15)
            .group_by("category")
            .agg(
                [
                    pl.count().alias("count"),
                    pl.mean("text_length").alias("avg_length"),
                    pl.sum("number").alias("sum_number"),
                ]
            )
        ),
        "Join Operations": (
            df_simple.select(["group", "value"])
            .join(
                df_complex.select(["category", "metric_a"]).rename(
                    {"category": "group"}
                ),
                on="group",
                how="inner",
            )
            .group_by("group")
            .agg(
                [
                    pl.sum("value").alias("total_value"),
                    pl.mean("metric_a").alias("avg_metric"),
                ]
            )
        ),
        "Window Functions": (
            df_complex.with_columns(
                [
                    pl.col("metric_a")
                    .rolling_mean(window_size=100)
                    .alias("rolling_avg"),
                    pl.col("metric_b").rank().over("category").alias("category_rank"),
                ]
            )
            .filter(pl.col("rolling_avg").is_not_null())
            .group_by("category")
            .agg(
                [
                    pl.mean("rolling_avg").alias("avg_rolling"),
                    pl.max("category_rank").alias("max_rank"),
                ]
            )
        ),
    }

    return queries


def main():
    """Main benchmarking function."""
    print("=== Polars GPU Performance Benchmark Suite ===\n")

    benchmark = PolarsGPUBenchmark()

    if not benchmark.gpu_available:
        logging.warning("GPU not available. Running CPU-only benchmarks.")
        engines = ["cpu", "streaming"]
    else:
        logging.info("GPU detected. Running full benchmark suite.")
        engines = ["cpu", "streaming", "gpu", "gpu_strict"]

    # Create benchmark queries
    queries = create_benchmark_queries()

    print("\nBenchmark Configuration:")
    print(f"   Engines: {', '.join(engines)}")
    print(f"   Queries: {len(queries)}")
    print("   Runs per query: 3")

    # Run benchmarks
    for query_name, query in queries.items():
        benchmark.run_benchmark(query, query_name, engines, runs=3)

    # Print results
    benchmark.print_summary()

    print("\nPerformance Tips:")
    print("   - GPU excels at grouped aggregations and joins")
    print("   - Use streaming engine for larger-than-memory datasets")
    print("   - String operations may require CPU fallback")
    print("   - Consider GPUEngine(raise_on_fail=True) to identify unsupported ops")


if __name__ == "__main__":
    main()
