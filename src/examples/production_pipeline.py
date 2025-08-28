#!/usr/bin/env python3
"""
Production pipeline patterns with comprehensive error handling and monitoring.

This module demonstrates enterprise-ready data pipeline patterns that were
previously embedded in documentation.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import polars as pl


@dataclass
class PipelineConfig:
    """Configuration management for production pipelines"""

    # Data sources
    input_path: str
    output_path: str
    backup_path: Optional[str] = None

    # Processing settings
    chunk_size: int = 1_000_000
    max_memory_gb: int = 32
    use_gpu: bool = False
    gpu_memory_fraction: float = 0.8

    # Quality thresholds
    min_data_quality_score: float = 0.85
    max_null_percentage: float = 0.1

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Monitoring
    enable_profiling: bool = True
    log_level: str = "INFO"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, config_path: str) -> "PipelineConfig":
        """Load configuration from JSON file"""
        with open(config_path) as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def validate(self) -> List[str]:
        """Validate configuration settings"""
        errors = []

        if not Path(self.input_path).exists():
            errors.append(f"Input path does not exist: {self.input_path}")

        if self.chunk_size <= 0:
            errors.append("Chunk size must be positive")

        if not 0 < self.min_data_quality_score <= 1:
            errors.append("Data quality score must be between 0 and 1")

        return errors


class DataValidator:
    """Comprehensive data validation with detailed reporting"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.validation_rules = {}
        self.issues = []

    def validate_schema(self, df: pl.LazyFrame, expected_schema: dict) -> bool:
        """Validate DataFrame schema against expected schema."""
        try:
            actual_schema = df.collect_schema()

            for col_name, expected_type in expected_schema.items():
                if col_name not in actual_schema:
                    self.issues.append(f"Missing column: {col_name}")
                    return False
                if str(actual_schema[col_name]) != str(expected_type):
                    self.issues.append(
                        f"Type mismatch in {col_name}: got {actual_schema[col_name]}, expected {expected_type}"
                    )
                    return False

            return True
        except Exception as e:
            self.issues.append(f"Schema validation error: {e}")
            return False

    def check_data_quality(self, df: pl.LazyFrame) -> float:
        """Calculate comprehensive data quality score."""
        try:
            stats = df.select(
                [
                    pl.len().alias("total_rows"),
                    pl.all().null_count().sum().alias("total_nulls"),
                    pl.all().is_duplicated().sum().alias("total_duplicates"),
                ]
            ).collect()

            total_rows = stats["total_rows"][0]
            if total_rows == 0:
                return 0.0

            total_cells = total_rows * (
                len(df.collect_schema()) - 1
            )  # Exclude length column
            null_count = stats["total_nulls"][0]
            duplicate_count = stats["total_duplicates"][0]

            # Quality score: penalize nulls and duplicates
            quality_score = 1.0 - ((null_count + duplicate_count * 0.5) / total_cells)

            return max(0.0, min(1.0, quality_score))
        except Exception as e:
            self.issues.append(f"Data quality check failed: {e}")
            return 0.0

    def validate_business_rules(
        self, df: pl.LazyFrame, rules: Dict[str, pl.Expr]
    ) -> bool:
        """Validate custom business rules"""
        try:
            for rule_name, rule_expr in rules.items():
                violations = df.filter(~rule_expr).select(pl.len()).collect().item()
                if violations > 0:
                    self.issues.append(
                        f"Business rule violation '{rule_name}': {violations} rows"
                    )
                    return False
            return True
        except Exception as e:
            self.issues.append(f"Business rule validation failed: {e}")
            return False

    def get_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        return {
            "timestamp": time.time(),
            "issues_found": len(self.issues),
            "issues": self.issues,
            "config": self.config,
        }


class ProductionPipelineBase:
    """Base class for production data pipelines with comprehensive error handling"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.engine = self._setup_engine()
        self.metrics = {}

        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")

    def _setup_logging(self) -> logging.Logger:
        """Configure comprehensive logging"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(getattr(logging, self.config.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _setup_engine(self) -> Union[str, pl.GPUEngine]:
        """Setup optimal execution engine with fallback strategies"""
        if self.config.use_gpu:
            try:
                import subprocess

                subprocess.run(["nvidia-smi"], check=True, capture_output=True)

                engine = pl.GPUEngine(
                    device=0,
                    raise_on_fail=False,
                    memory_fraction=self.config.gpu_memory_fraction,
                    memory_resource="managed",
                )

                # Test GPU engine
                test_df = pl.DataFrame({"test": [1, 2, 3]}).lazy()
                test_df.collect(engine=engine)

                self.logger.info("GPU engine configured successfully")
                return engine

            except Exception as e:
                self.logger.warning(f"GPU setup failed: {e}, falling back to streaming")

        return "streaming"

    @contextmanager
    def performance_monitor(self, operation_name: str):
        """Context manager for performance monitoring"""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        self.logger.info(f"Starting {operation_name}")

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            self.logger.info(
                f"Completed {operation_name}: {duration:.2f}s, {memory_delta:+.2f}MB memory"
            )

            # Store metrics
            self.metrics[operation_name] = {
                "duration": duration,
                "memory_delta": memory_delta,
                "timestamp": end_time,
            }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil

            return psutil.Process().memory_info().rss / 1e6
        except ImportError:
            return 0.0

    def execute_with_retry(
        self, query: pl.LazyFrame, operation_name: str
    ) -> pl.DataFrame:
        """Execute query with retry logic and comprehensive error handling"""

        for attempt in range(self.config.max_retries + 1):
            try:
                with self.performance_monitor(
                    f"{operation_name}_attempt_{attempt + 1}"
                ):
                    result = query.collect(engine=self.engine)

                self.logger.info(
                    f"Successfully executed {operation_name} on attempt {attempt + 1}"
                )
                return result

            except pl.exceptions.ComputeError as e:
                self.logger.warning(
                    f"Compute error in {operation_name} (attempt {attempt + 1}): {e}"
                )

                if attempt < self.config.max_retries:
                    # Try fallback engines
                    fallback_engines = ["streaming", None]  # None = CPU
                    for fallback_engine in fallback_engines:
                        try:
                            self.logger.info(
                                f"Trying fallback engine: {fallback_engine or 'CPU'}"
                            )
                            result = query.collect(engine=fallback_engine)
                            return result
                        except Exception as fallback_error:
                            self.logger.warning(
                                f"Fallback engine failed: {fallback_error}"
                            )
                            continue

                    time.sleep(self.config.retry_delay)
                else:
                    self.logger.error(f"All retry attempts failed for {operation_name}")
                    raise

            except Exception as e:
                self.logger.error(f"Unexpected error in {operation_name}: {e}")
                if attempt >= self.config.max_retries:
                    raise
                time.sleep(self.config.retry_delay)

        raise RuntimeError(
            f"Failed to execute {operation_name} after {self.config.max_retries + 1} attempts"
        )


class ProductionETLPipeline(ProductionPipelineBase):
    """Complete ETL pipeline with validation and monitoring"""

    def process_dataset(
        self, expected_schema: dict, business_rules: dict = None
    ) -> pl.DataFrame:
        """Process dataset with comprehensive validation and error handling"""

        # Initialize validator
        validator = DataValidator(self.config)

        # Load and validate data
        with self.performance_monitor("data_loading"):
            query = pl.scan_parquet(self.config.input_path)

            # Schema validation
            if not validator.validate_schema(query, expected_schema):
                raise ValueError("Schema validation failed")

        # Data quality check
        with self.performance_monitor("quality_check"):
            quality_score = validator.check_data_quality(query)

            if quality_score < self.config.min_data_quality_score:
                raise ValueError(
                    f"Data quality too low: {quality_score:.3f} < {self.config.min_data_quality_score}"
                )

            self.logger.info(f"Data quality score: {quality_score:.3f}")

        # Business rules validation
        if business_rules:
            with self.performance_monitor("business_rules"):
                if not validator.validate_business_rules(query, business_rules):
                    raise ValueError("Business rule validation failed")

        # Apply transformations
        with self.performance_monitor("transformations"):
            processed_query = self._apply_transformations(query)

        # Execute with retry
        result = self.execute_with_retry(processed_query, "final_processing")

        # Save results
        with self.performance_monitor("save_results"):
            self._save_results(result)

        # Generate report
        self._generate_report(validator, result)

        return result

    def _apply_transformations(self, query: pl.LazyFrame) -> pl.LazyFrame:
        """Apply data transformations - override in subclasses"""
        return query  # Base implementation does nothing

    def _save_results(self, result: pl.DataFrame):
        """Save processing results"""
        result.write_parquet(self.config.output_path)

        if self.config.backup_path:
            result.write_parquet(self.config.backup_path)

    def _generate_report(self, validator: DataValidator, result: pl.DataFrame):
        """Generate processing report"""
        report = {
            "pipeline_config": self.config,
            "validation_report": validator.get_validation_report(),
            "result_stats": {
                "shape": result.shape,
                "memory_usage": result.estimated_size("mb"),
                "columns": list(result.columns),
            },
            "performance_metrics": self.metrics,
        }

        self.logger.info(f"Pipeline completed successfully: {report}")


if __name__ == "__main__":
    # Example usage
    config = PipelineConfig(
        input_path="test_data.parquet",
        output_path="processed_data.parquet",
        use_gpu=False,
        min_data_quality_score=0.8,
    )

    expected_schema = {"id": pl.Int64, "name": pl.String, "value": pl.Float64}

    business_rules = {
        "positive_values": pl.col("value") > 0,
        "non_empty_names": pl.col("name").str.len_chars() > 0,
    }

    # Create test data
    test_data = pl.DataFrame(
        {
            "id": range(1000),
            "name": [f"item_{i}" for i in range(1000)],
            "value": [i * 0.1 for i in range(1000)],
        }
    )
    test_data.write_parquet("test_data.parquet")

    try:
        pipeline = ProductionETLPipeline(config)
        result = pipeline.process_dataset(expected_schema, business_rules)
        print(f"Pipeline completed: {result.shape}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
    finally:
        # Cleanup
        Path("test_data.parquet").unlink(missing_ok=True)
        Path("processed_data.parquet").unlink(missing_ok=True)
