#!/usr/bin/env python3
"""
Advanced streaming patterns for out-of-core computation.

This module demonstrates sophisticated streaming processing techniques
for datasets that don't fit in memory.
"""

import polars as pl
import psutil
import time
import threading
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class StreamingProcessor:
    """Advanced streaming processor with memory monitoring and optimization."""

    def __init__(self, memory_limit_gb: float = 4.0):
        self.memory_limit_gb = memory_limit_gb

    def process_with_memory_control(
        self, query: pl.LazyFrame, description: str = "Streaming query"
    ) -> pl.DataFrame:
        """Process query with comprehensive memory monitoring."""

        import gc

        process = psutil.Process()

        gc.collect()
        start_memory = process.memory_info().rss / 1e9
        start_time = time.time()

        logger.info(f"Starting {description}")
        logger.info(f"Initial memory: {start_memory:.2f} GB")
        logger.info(f"Memory limit: {self.memory_limit_gb} GB")

        try:
            result = query.collect(engine="streaming")

            end_memory = process.memory_info().rss / 1e9
            elapsed_time = time.time() - start_time

            logger.info("Streaming completed successfully:")
            logger.info(f"Peak memory: {end_memory:.2f} GB")
            logger.info(f"Memory delta: {end_memory - start_memory:+.2f} GB")
            logger.info(f"Execution time: {elapsed_time:.2f}s")
            logger.info(f"Result shape: {result.shape}")

            return result

        except Exception as e:
            error_memory = process.memory_info().rss / 1e9
            logger.error(f"Streaming failed: {e}")
            logger.error(f"Error memory: {error_memory:.2f} GB")
            raise


def analyze_streaming_support(query: pl.LazyFrame) -> dict:
    """Analyze query plan for streaming compatibility."""

    analysis = {
        "streaming_supported": True,
        "optimized_operations": [],
        "limitations": [],
        "recommendations": [],
    }

    try:
        streaming_plan = query.explain(streaming=True)
        regular_plan = query.explain(optimized=True)

        print("=== Regular Query Plan ===")
        print(regular_plan)
        print("\\n=== Streaming Query Plan ===")
        print(streaming_plan)

        if "STREAMING" in streaming_plan:
            analysis["optimized_operations"].append(
                "Streaming-compatible operations detected"
            )

        breaking_patterns = ["SORT", "PIVOT", "WINDOW", "COLLECT"]
        for pattern in breaking_patterns:
            if pattern in streaming_plan and "STREAMING" not in streaming_plan:
                analysis["limitations"].append(
                    f"{pattern} operation may break streaming"
                )
                analysis["streaming_supported"] = False

        if analysis["streaming_supported"]:
            analysis["recommendations"].append("Query is fully streaming compatible")
        else:
            analysis["recommendations"].append(
                "Consider restructuring query to maintain streaming"
            )

    except Exception as e:
        analysis["limitations"].append(f"Analysis failed: {e}")

    return analysis


def stream_with_progress_monitoring(
    file_pattern: str, operation_name: str = "Large Dataset Processing"
):
    """Stream with detailed progress monitoring."""

    monitoring_active = True

    def monitor_memory():
        process = psutil.Process()

        while monitoring_active:
            current_memory = process.memory_info().rss / 1e9
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Memory: {current_memory:.2f} GB")
            time.sleep(30)

    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()

    try:
        processor = StreamingProcessor(memory_limit_gb=8.0)

        query = (
            pl.scan_parquet(file_pattern)
            .filter(pl.col("timestamp").dt.year() >= 2020)
            .select(["customer_id", "product_id", "amount", "timestamp", "country"])
            .with_columns(
                [
                    pl.col("timestamp").dt.date().alias("date"),
                    (pl.col("amount") * 1.2).alias("amount_with_tax"),
                ]
            )
            .group_by(["date", "country"])
            .agg(
                [
                    pl.col("amount_with_tax").sum().alias("daily_revenue"),
                    pl.col("customer_id").n_unique().alias("unique_customers"),
                    pl.len().alias("transaction_count"),
                ]
            )
            .sort("date")
        )

        result = processor.process_with_memory_control(query, operation_name)
        return result

    finally:
        monitoring_active = False
        monitor_thread.join(timeout=1)


def recommend_execution_strategy(query: pl.LazyFrame, estimated_size_gb: float) -> str:
    """Recommend optimal execution strategy based on query and data size."""

    available_memory_gb = psutil.virtual_memory().available / 1e9

    recommendations = []

    if estimated_size_gb > available_memory_gb * 0.5:
        recommendations.append("Use streaming: Dataset > 50% of available memory")
        strategy = "streaming"
    elif estimated_size_gb > available_memory_gb * 0.8:
        recommendations.append("Use streaming: Dataset > 80% of available memory")
        strategy = "streaming"
    else:
        recommendations.append("Regular collection feasible")
        strategy = "regular"

    try:
        plan = query.explain()
        if any(op in plan for op in ["SORT BY", "PIVOT", "WINDOW"]):
            recommendations.append("Warning: Operations may break streaming efficiency")
            if strategy == "streaming":
                recommendations.append("Consider chunking or restructuring query")
    except Exception:
        pass

    logger.info(f"Execution Strategy Recommendation: {strategy.upper()}")
    logger.info(f"Estimated data size: {estimated_size_gb:.1f} GB")
    logger.info(f"Available memory: {available_memory_gb:.1f} GB")
    for rec in recommendations:
        logger.info(f"   • {rec}")

    return strategy


def process_very_large_dataset_with_chunking(
    file_pattern: str, chunk_size: int = 1000000
):
    """Process extremely large datasets with manual chunking fallback."""

    import glob

    files = glob.glob(file_pattern)
    results = []

    print(f"Processing {len(files)} files in chunks of {chunk_size} rows")

    for i, file in enumerate(files):
        try:
            chunk_result = (
                pl.scan_parquet(file)
                .head(chunk_size)
                .filter(pl.col("active"))
                .group_by("category")
                .agg(
                    [
                        pl.col("amount").sum(),
                        pl.col("count").sum(),
                        pl.len().alias("records"),
                    ]
                )
                .collect(engine="streaming")
            )

            results.append(chunk_result)
            print(f"   Chunk {i+1}/{len(files)}: {chunk_result.shape} processed")

        except Exception as e:
            print(f"   Chunk {i+1} failed: {e}")
            continue

    if results:
        combined = (
            pl.concat(results)
            .group_by("category")
            .agg(
                [pl.col("amount").sum(), pl.col("count").sum(), pl.col("records").sum()]
            )
        )
        return combined
    else:
        raise RuntimeError("All chunks failed to process")


if __name__ == "__main__":
    # Example: Create test data
    test_query = (
        pl.LazyFrame(
            {
                "category": ["A", "B", "C"] * 10000,
                "amount": range(30000),
                "active": [True, False] * 15000,
                "count": [1] * 30000,
            }
        )
        .group_by("category")
        .agg(pl.col("amount").sum())
    )

    # Analyze streaming support
    analysis = analyze_streaming_support(test_query)
    print(f"Streaming analysis: {analysis}")

    # Test execution strategy recommendation
    strategy = recommend_execution_strategy(test_query, estimated_size_gb=2.0)
    print(f"Recommended strategy: {strategy}")

    # Test streaming processor
    processor = StreamingProcessor()
    result = processor.process_with_memory_control(test_query, "Test streaming query")
    print(f"Test result: {result}")
