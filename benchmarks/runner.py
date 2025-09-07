"""Main benchmark runner"""

import argparse
import logging
import subprocess

from .framework import BenchmarkFramework
from .polars_bench import PolarsBenchmark
from .duckdb_bench import DuckDBBenchmark
from .cudf_bench import CuDFBenchmark
from .pandas_bench import PandasBenchmark
from .visualizer import BenchmarkVisualizer

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    def __init__(
        self,
        data_path: str = "benchmarks/data/tpch-10",
        results_file: str = "results.parquet",
    ):
        self.data_path = data_path
        self.framework = BenchmarkFramework(results_file)
        self.gpu_available = self._check_gpu_availability()

        # Initialize benchmark implementations
        self.polars = PolarsBenchmark(data_path)
        self.duckdb = DuckDBBenchmark(data_path)
        if self.gpu_available:
            self.cudf = CuDFBenchmark(data_path)
        else:
            self.cudf = None
            logger.info("GPU not available - skipping cuDF initialization")
        self.pandas = PandasBenchmark(data_path)

        # Query methods mapping
        self.queries = [
            "simple_aggregation",
            "customer_segments",
            "order_customer_join",
            "supplier_revenue",
            "detailed_orders",
            "revenue_ranking",
        ]

    def _check_gpu_availability(self):
        """Check if GPU is available via nvidia-smi"""
        try:
            subprocess.run(["nvidia-smi"], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def run_polars_benchmarks(self):
        """Run Polars CPU streaming benchmarks"""
        print("=== Running Polars CPU Streaming Benchmarks ===")

        for query in self.queries:
            streaming_method = getattr(self.polars, f"{query}_streaming")
            self.framework.benchmark_query(streaming_method, "polars_streaming", query)

    def run_polars_gpu_benchmarks(self):
        """Run Polars GPU benchmarks"""
        if not self.gpu_available:
            logger.info("Skipping Polars GPU benchmarks - GPU not available")
            return

        print("=== Running Polars GPU Benchmarks ===")

        for query in self.queries:
            try:
                gpu_method = getattr(self.polars, f"{query}_gpu")
                self.framework.benchmark_query(gpu_method, "polars_gpu", query)
            except Exception as e:
                logger.error(f"GPU benchmark failed for {query}: {e}")

    def run_duckdb_benchmarks(self):
        """Run DuckDB benchmarks"""
        print("=== Running DuckDB Benchmarks ===")

        for query in self.queries:
            method = getattr(self.duckdb, query)
            self.framework.benchmark_query(method, "duckdb", query)

    def run_cudf_benchmarks(self):
        """Run cuDF benchmarks"""
        if not self.gpu_available or self.cudf is None:
            logger.info("Skipping cuDF benchmarks - GPU not available")
            return

        print("=== Running cuDF Benchmarks ===")

        for query in self.queries:
            try:
                method = getattr(self.cudf, query)
                self.framework.benchmark_query(method, "cudf", query)
            except Exception as e:
                logger.error(f"cuDF benchmark failed for {query}: {e}")

    def run_pandas_benchmarks(self):
        """Run Pandas benchmarks"""
        print("=== Running Pandas Benchmarks ===")

        for query in self.queries:
            method = getattr(self.pandas, query)
            self.framework.benchmark_query(method, "pandas", query)

    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("Starting comprehensive benchmark suite...")
        print(f"Data path: {self.data_path}")
        print(f"Results file: {self.framework.results_file}")
        print(f"System info: {self.framework.system_info}")
        print()

        # Run all benchmark suites
        self.run_pandas_benchmarks()
        self.run_polars_benchmarks()
        self.run_duckdb_benchmarks()
        self.run_cudf_benchmarks()
        self.run_polars_gpu_benchmarks()

        print("\nBenchmark suite completed!")

        # Generate visualizations
        print("Generating visualizations...")
        visualizer = BenchmarkVisualizer(self.framework.results_file)
        visualizer.create_dashboard()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run TPC-H benchmarks")
    parser.add_argument(
        "--data-path",
        type=str,
        default="benchmarks/data/tpch-10",
        help="Path to TPC-H data directory (default: benchmarks/data/tpch-10)",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results.parquet",
        help="Output file for benchmark results (default: results.parquet)",
    )

    args = parser.parse_args()

    runner = BenchmarkRunner(data_path=args.data_path, results_file=args.results_file)
    runner.run_all_benchmarks()


if __name__ == "__main__":
    main()
