"""Benchmark framework with timing and memory measurement"""

import importlib.util
import platform
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import polars as pl
from memory_profiler import memory_usage

# Removed pandera dependency for simpler validation


class BenchmarkFramework:
    def __init__(self, results_file: str = "results.parquet"):
        self.results_file = Path(results_file)
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> str:
        """Get system information for benchmark context"""
        cpu_count = psutil.cpu_count(logical=False)
        memory_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        system = platform.system()
        machine = platform.machine()

        gpu_info = "No GPU"
        if importlib.util.find_spec("cudf") is not None:
            gpu_info = "cuDF available"

        return f"{system}-{machine}, {cpu_count}C/{memory_gb}GB, {gpu_info}"

    def measure_execution(
        self,
        func: Callable,
        library: str,
        query_name: str,
        dataset_size: str = "tpch-10",
        runs: int = 3,
    ) -> Dict[str, Any]:
        """Measure execution time and memory usage of a function"""

        execution_times = []
        memory_peaks = []
        result_row_count = 0

        for run in range(runs):
            # Measure memory usage during execution
            def execute_func():
                nonlocal result_row_count
                result = func()
                if hasattr(result, "__len__"):
                    result_row_count = len(result)
                elif hasattr(result, "shape"):
                    result_row_count = result.shape[0]
                return result

            # Memory profiler measures peak memory usage
            start_time = time.perf_counter()
            mem_usage = memory_usage(execute_func, interval=0.01, timeout=300)
            end_time = time.perf_counter()

            execution_times.append((end_time - start_time) * 1000)  # Convert to ms
            memory_peaks.append(max(mem_usage) if mem_usage else 0)

        # Use median values for more stable results
        median_time = sorted(execution_times)[runs // 2]
        median_memory = sorted(memory_peaks)[runs // 2]

        return {
            "timestamp": datetime.now(),
            "library": library,
            "query_name": query_name,
            "dataset_size": dataset_size,
            "execution_time_ms": median_time,
            "peak_memory_mb": median_memory,
            "row_count": result_row_count,
            "system_info": self.system_info,
        }

    def save_result(self, result: Dict[str, Any]):
        """Save benchmark result to parquet file"""
        result_df = pl.DataFrame([result])

        # Simple validation - just ensure we have the expected columns
        expected_cols = {
            "timestamp",
            "library",
            "query_name",
            "dataset_size",
            "execution_time_ms",
            "peak_memory_mb",
            "row_count",
            "system_info",
        }
        df_cols = set(result_df.columns)
        missing_cols = expected_cols - df_cols
        if missing_cols:
            print(
                f"Warning: Missing result columns {missing_cols} (results still saved)"
            )

        # Always save results regardless of validation
        if self.results_file.exists():
            # Append to existing results
            existing_df = pl.read_parquet(self.results_file)
            combined_df = pl.concat([existing_df, result_df])
            combined_df.write_parquet(self.results_file)
        else:
            # Create new results file
            result_df.write_parquet(self.results_file)

    def load_results(self) -> Optional[pl.DataFrame]:
        """Load benchmark results from parquet file"""
        if self.results_file.exists():
            return pl.read_parquet(self.results_file)
        return None

    def benchmark_query(
        self,
        query_func: Callable,
        library: str,
        query_name: str,
        dataset_size: str = "tpch-10",
        runs: int = 3,
    ):
        """Run complete benchmark for a query"""
        print(f"Benchmarking {library} - {query_name}...")

        try:
            result = self.measure_execution(
                query_func, library, query_name, dataset_size, runs
            )
            self.save_result(result)

            print(f"  Time: {result['execution_time_ms']:.2f}ms")
            print(f"  Memory: {result['peak_memory_mb']:.1f}MB")
            print(f"  Rows: {result['row_count']:,}")

        except Exception as e:
            print(f"  FAILED: {e}")
            # Still save the failure for analysis
            failure_result = {
                "timestamp": datetime.now(),
                "library": library,
                "query_name": query_name,
                "dataset_size": dataset_size,
                "execution_time_ms": -1.0,  # Indicates failure
                "peak_memory_mb": 0.0,
                "row_count": 0,
                "system_info": f"{self.system_info} - ERROR: {str(e)}",
            }
            self.save_result(failure_result)
