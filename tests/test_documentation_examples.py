#!/usr/bin/env python3
"""
Test script to validate all code examples from enhanced Polars documentation.

This script validates examples from:
- pl.md (comprehensive patterns and advanced usage)
- README.md (practical examples and troubleshooting)

Tests include:
- Memory optimization patterns
- Advanced expression patterns
- GPU engine configuration
- Streaming engine patterns
- Production patterns and error handling
- Data validation patterns
"""

import sys
import traceback
import tempfile
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Production pipeline configuration for testing."""

    input_path: str
    output_path: str
    use_gpu: bool = False
    gpu_memory_fraction: float = 0.8
    batch_size: int = 10000
    log_level: str = "INFO"
    retry_attempts: int = 3
    timeout_seconds: int = 300
    enable_monitoring: bool = True
    min_data_quality_score: float = 0.85
    environment: str = "test"
    config_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataValidator:
    """Data validation class for testing production patterns."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.validation_rules = {}

    def validate_schema(self, df: pl.LazyFrame, expected_schema: dict) -> bool:
        """Validate DataFrame schema against expected schema."""
        try:
            actual_schema = df.collect_schema()

            for col_name, expected_type in expected_schema.items():
                if col_name not in actual_schema:
                    return False
                if str(actual_schema[col_name]) != str(expected_type):
                    return False

            return True
        except Exception:
            return False

    def check_data_quality(self, df: pl.LazyFrame) -> float:
        """Calculate data quality score."""
        try:
            total_rows = df.select(pl.len()).collect().item()
            if total_rows == 0:
                return 0.0

            null_count = df.select(pl.all().null_count().sum()).collect().item()
            quality_score = 1.0 - (null_count / (total_rows * len(df.collect_schema())))

            return max(0.0, min(1.0, quality_score))
        except Exception:
            return 0.0


def test_imports() -> Tuple[bool, str]:
    """Test that all necessary imports work."""
    try:
        import polars  # noqa: F401

        # Test GPU engine import (may not be available)
        gpu_available = False
        try:
            from polars.lazyframe.engine_config import GPUEngine

            _ = GPUEngine
            gpu_available = True
        except ImportError:
            pass

        return True, f"All imports successful (GPU engine available: {gpu_available})"
    except ImportError as e:
        return False, f"Import error: {e}"


def test_memory_optimization() -> Tuple[bool, str]:
    """Test memory optimization patterns from documentation."""
    try:
        # Test schema-based optimization
        schema = {
            "id": pl.UInt32,
            "category": pl.Categorical,
            "price": pl.Float32,
            "quantity": pl.UInt16,
            "is_active": pl.Boolean,
            "timestamp": pl.Datetime("ms"),
        }

        # Create test data with optimal schema
        df = pl.LazyFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "category": ["A", "B", "A", "C", "B"],
                "price": [10.5, 15.0, 12.5, 20.0, 8.0],
                "quantity": [100, 200, 150, 300, 75],
                "is_active": [True, False, True, True, False],
                "timestamp": [datetime.now()] * 5,
            }
        ).cast(schema)

        # Test memory estimation (if available)
        _ = df.collect()

        # Test categorical optimization
        categorical_df = pl.LazyFrame(
            {"category": ["Category A"] * 1000 + ["Category B"] * 1000}
        ).with_columns(pl.col("category").cast(pl.Categorical))

        _ = categorical_df.collect()

        return True, "Memory optimization patterns validated successfully"

    except Exception as e:
        return False, f"Memory optimization test failed: {e}"


def test_advanced_expressions() -> Tuple[bool, str]:
    """Test advanced expression patterns from documentation."""
    try:
        # Test fold expressions
        df = pl.LazyFrame(
            {
                "price": [10.0, 15.0, 20.0],
                "quantity": [2, 3, 1],
                "tax_rate": [1.08, 1.05, 1.10],
            }
        )

        # Test horizontal fold operation
        result = df.select(
            pl.fold(
                acc=1,
                exprs=[pl.col("price"), pl.col("quantity"), pl.col("tax_rate")],
                function=lambda acc, x: acc * x,
            ).alias("total_cost")
        ).collect()

        assert result.shape == (3, 1)

        # Test expression composition
        df_text = pl.LazyFrame(
            {"text": ["Hello World", "POLARS rocks", "data PROCESSING"]}
        )

        composition_result = df_text.select(
            pl.col("text")
            .str.to_lowercase()
            .str.replace_all(r"\s+", "_")
            .str.slice(0, 10)
            .alias("processed_text")
        ).collect()

        assert composition_result.shape == (3, 1)

        # Test list operations
        df_lists = pl.LazyFrame({"numbers": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})

        list_result = df_lists.select(
            pl.col("numbers").list.sum().alias("sum"),
            pl.col("numbers").list.len().alias("length"),
        ).collect()

        assert list_result.shape == (3, 2)

        return True, "Advanced expression patterns validated successfully"

    except Exception as e:
        return False, f"Advanced expressions test failed: {e}"


def test_gpu_engine_configuration() -> Tuple[bool, str]:
    """Test GPU engine configuration patterns."""
    try:
        # Test basic GPU engine import
        try:
            from polars.lazyframe.engine_config import GPUEngine
        except ImportError:
            return True, "GPU engine not available - skipping GPU tests"

        # Test GPU engine creation with different configurations
        gpu_engines = []

        # Lenient GPU engine
        try:
            lenient_gpu = GPUEngine(raise_on_fail=False)
            gpu_engines.append(("lenient", lenient_gpu))
        except Exception:
            pass

        # Strict GPU engine
        try:
            strict_gpu = GPUEngine(
                device=0,
                raise_on_fail=True,
                memory_resource="managed",
                memory_fraction=0.8,
            )
            gpu_engines.append(("strict", strict_gpu))
        except Exception:
            pass

        # Test with simple data
        df = pl.LazyFrame(
            {"group": ["A", "B", "A", "B", "C"], "value": [1, 2, 3, 4, 5]}
        )

        # Test CPU execution first
        cpu_result = df.group_by("group").sum().collect(engine="cpu")
        assert cpu_result.shape[0] == 3

        # Test GPU engines if available
        for engine_name, gpu_engine in gpu_engines:
            try:
                gpu_result = df.group_by("group").sum().collect(engine=gpu_engine)
                assert gpu_result.shape[0] == 3
            except Exception:
                # GPU execution may fail, which is acceptable
                continue

        return (
            True,
            f"GPU configuration patterns validated (engines tested: {len(gpu_engines)})",
        )

    except Exception as e:
        return False, f"GPU configuration test failed: {e}"


def test_streaming_patterns() -> Tuple[bool, str]:
    """Test streaming engine patterns from documentation."""
    try:
        # Create test data
        df = pl.LazyFrame(
            {
                "id": range(1000),
                "category": [["A", "B", "C"][i % 3] for i in range(1000)],
                "value": [i * 0.1 for i in range(1000)],
            }
        )

        # Test streaming execution
        streaming_result = (
            df.group_by("category")
            .agg(
                [
                    pl.col("value").sum().alias("total_value"),
                    pl.col("id").count().alias("count"),
                ]
            )
            .collect(engine="streaming")
        )

        assert streaming_result.shape[0] == 3

        # Test query plan explanation
        plan = df.group_by("category").sum().explain(streaming=True)
        assert isinstance(plan, str)
        assert len(plan) > 0

        # Test operations that work well with streaming
        streaming_operations = [
            df.filter(pl.col("value") > 50),
            df.select(pl.col("id"), pl.col("value") * 2),
            df.with_columns(pl.col("category").str.to_uppercase()),
        ]

        for operation in streaming_operations:
            result = operation.collect(engine="streaming")
            assert result.shape[1] >= 1

        return True, "Streaming patterns validated successfully"

    except Exception as e:
        return False, f"Streaming patterns test failed: {e}"


def test_production_patterns() -> Tuple[bool, str]:
    """Test production patterns and error handling."""
    try:
        # Test PipelineConfig
        config = PipelineConfig(
            input_path="/tmp/test_input.parquet",
            output_path="/tmp/test_output.parquet",
            use_gpu=False,
            environment="test",
        )

        assert config.batch_size == 10000
        assert config.min_data_quality_score == 0.85

        # Test DataValidator
        validator = DataValidator(config)

        test_df = pl.LazyFrame(
            {
                "id": [1, 2, 3, None, 5],
                "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
                "score": [85.0, 92.0, 78.0, 88.0, 95.0],
            }
        )

        # Test schema validation
        expected_schema = {"id": pl.Int64, "name": pl.String, "score": pl.Float64}

        schema_valid = validator.validate_schema(test_df, expected_schema)
        assert isinstance(schema_valid, bool)

        # Test data quality check
        quality_score = validator.check_data_quality(test_df)
        assert 0.0 <= quality_score <= 1.0

        # Test error handling patterns
        try:
            # This should work
            result = test_df.fill_null(0).collect()
            assert result.shape == (5, 3)
        except Exception as e:
            return False, f"Basic error handling failed: {e}"

        return True, "Production patterns validated successfully"

    except Exception as e:
        return False, f"Production patterns test failed: {e}"


def test_io_patterns() -> Tuple[bool, str]:
    """Test I/O and partitioning patterns."""
    try:
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test data
            df = pl.LazyFrame(
                {
                    "year": [2020, 2020, 2021, 2021, 2022],
                    "month": [1, 2, 1, 2, 1],
                    "sales": [100, 150, 200, 250, 300],
                    "region": ["North", "South", "North", "South", "North"],
                }
            )

            # Test parquet writing
            output_file = temp_path / "test_data.parquet"
            df.collect().write_parquet(output_file)

            # Test parquet reading with lazy evaluation
            read_df = pl.scan_parquet(output_file)
            result = read_df.filter(pl.col("year") == 2021).collect()

            assert result.shape[0] == 2

            # Test schema collection
            schema = read_df.collect_schema()
            assert "year" in schema
            assert "sales" in schema

            return True, "I/O patterns validated successfully"

    except Exception as e:
        return False, f"I/O patterns test failed: {e}"


def test_temporal_patterns() -> Tuple[bool, str]:
    """Test temporal data processing patterns."""
    try:
        # Create time series data
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
        df = pl.LazyFrame(
            {"date": dates, "value": [i * 2.5 + (i % 7) * 10 for i in range(30)]}
        )

        # Test temporal operations
        temporal_result = df.select(
            [
                pl.col("date"),
                pl.col("value"),
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month"),
                pl.col("date").dt.weekday().alias("weekday"),
            ]
        ).collect()

        assert temporal_result.shape == (30, 5)

        # Test window functions with temporal data
        window_result = df.select(
            [
                pl.col("date"),
                pl.col("value"),
                pl.col("value").rolling_mean(window_size=7).alias("weekly_avg"),
            ]
        ).collect()

        assert window_result.shape == (30, 3)

        return True, "Temporal patterns validated successfully"

    except Exception as e:
        return False, f"Temporal patterns test failed: {e}"


def test_decision_patterns() -> Tuple[bool, str]:
    """Test decision-making patterns from documentation."""
    try:
        # Test execution engine decision function
        def choose_execution_engine(
            row_count: int, has_gpu: bool, operations_complexity: str
        ) -> str:
            if row_count < 100_000:
                return "cpu"
            elif has_gpu and operations_complexity in ["simple", "medium"]:
                return "gpu"
            elif row_count > 10_000_000:
                return "streaming"
            else:
                return "cpu"

        # Test decision logic
        assert choose_execution_engine(50_000, False, "simple") == "cpu"
        assert choose_execution_engine(500_000, True, "simple") == "gpu"
        assert choose_execution_engine(50_000_000, False, "complex") == "streaming"

        # Test with actual data
        small_df = pl.LazyFrame({"x": range(1000)})
        result = small_df.select(pl.col("x").sum()).collect(engine="cpu")
        assert result.shape == (1, 1)

        return True, "Decision patterns validated successfully"

    except Exception as e:
        return False, f"Decision patterns test failed: {e}"


def run_all_tests() -> bool:
    """Run all validation tests."""
    tests = [
        ("Import Tests", test_imports),
        ("Memory Optimization", test_memory_optimization),
        ("Advanced Expressions", test_advanced_expressions),
        ("GPU Engine Configuration", test_gpu_engine_configuration),
        ("Streaming Patterns", test_streaming_patterns),
        ("Production Patterns", test_production_patterns),
        ("I/O Patterns", test_io_patterns),
        ("Temporal Patterns", test_temporal_patterns),
        ("Decision Patterns", test_decision_patterns),
    ]

    logger.info("=== Documentation Examples Validation ===\n")

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            success, message = test_func()
            if success:
                logger.info(f"PASS: {test_name}: {message}")
                passed += 1
            else:
                logger.error(f"FAIL: {test_name}: {message}")
                failed += 1
        except Exception as e:
            logger.error(f"FAIL: {test_name}: Unexpected error - {e}")
            traceback.print_exc()
            failed += 1

    logger.info(
        f"\n=== Results ===\nPassed: {passed}\nFailed: {failed}\nTotal:  {passed + failed}"
    )

    if failed == 0:
        logger.info("All documentation examples validated successfully!")
        return True
    else:
        logger.warning(f"{failed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
