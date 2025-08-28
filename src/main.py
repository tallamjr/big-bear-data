import os
import resource
import subprocess
import logging
from datetime import datetime
from pprint import pprint

import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))


def is_nvidia_gpu_available():
    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# Advanced GPU configuration with fallback handling
def get_optimal_engine():
    """Get optimal execution engine based on environment and data characteristics."""
    if not is_nvidia_gpu_available():
        logger.warning("No NVIDIA GPU detected, using streaming engine for collection")
        return "streaming"

    # Check for advanced GPU configuration via environment
    gpu_config = os.getenv("POLARS_GPU_CONFIG", "auto")

    if gpu_config == "strict":
        logger.info("Using strict GPU mode (no CPU fallback)")
        return pl.GPUEngine(raise_on_fail=True)
    elif gpu_config == "verbose":
        logger.debug("Using verbose GPU mode for debugging")
        return "gpu"  # Will use with verbose config below
    else:
        logger.info("NVIDIA GPU detected, using GPU engine with automatic fallback")
        return "gpu"


# Set up collection arguments based on GPU availability and configuration
engine = get_optimal_engine()
collect_args = {"engine": engine}

# Enable verbose mode if requested
if os.getenv("POLARS_GPU_CONFIG") == "verbose" or os.getenv("POLARS_VERBOSE") == "1":
    logger.debug("Enabling verbose mode for GPU debugging")
    pl.Config.set_verbose(True)

# Lazy load the Parquet file (does NOT load into memory)
df = pl.scan_parquet("./data/yellow.hive/**/*.parquet", hive_partitioning=True)

# Count total rows without loading the full dataset
row_count = df.select(pl.len()).collect(**collect_args)

print(row_count)

pprint(df.collect_schema())
# Schema([('vendorID', String),
#         ('tpepPickupDateTime', Datetime(time_unit='ns', time_zone=None)),
#         ('tpepDropoffDateTime', Datetime(time_unit='ns', time_zone=None)),
#         ('passengerCount', Int32),
#         ('tripDistance', Float64),
#         ('puLocationId', String),
#         ('doLocationId', String),
#         ('startLon', Float64),
#         ('startLat', Float64),
#         ('endLon', Float64),
#         ('endLat', Float64),
#         ('rateCodeId', Int32),
#         ('storeAndFwdFlag', String),
#         ('paymentType', String),
#         ('fareAmount', Float64),
#         ('extra', Float64),
#         ('mtaTax', Float64),
#         ('improvementSurcharge', String),
#         ('tipAmount', Float64),
#         ('tollsAmount', Float64),
#         ('totalAmount', Float64)])

print(df.limit(10).collect(**collect_args))

# Group by pickup location and count occurrences
popular_pickups = (
    df.filter(pl.col("puLocationId").is_not_null())
    .group_by("puLocationId")
    .agg(pl.len().alias("num_trips"))
    .sort("num_trips", descending=True)
    .limit(10)
    .collect(**collect_args)
)

print(popular_pickups)

# Filter for 2016 only and compute daily total fares
daily_revenue = (
    df.filter(
        pl.col("tpepPickupDateTime").is_between(
            datetime(2016, 1, 1), datetime(2016, 12, 31)
        )
    )
    .with_columns(pl.col("tpepPickupDateTime").dt.date().alias("date"))
    .group_by("date")
    .agg(pl.sum("fareAmount").alias("total_fare"))
    .sort("date")
    .collect(**collect_args)
)

print(daily_revenue)

# Filter for trips longer than 50 miles, sorted by distance
longest_trips = (
    df.filter(pl.col("tripDistance") > 50)
    .select(["tripDistance", "fareAmount"])
    .sort("tripDistance", descending=True)
    .limit(10)
    .collect(**collect_args)
)

print(longest_trips)

# Define a bounding box for NYC (approximate)
min_lat, max_lat = 40.5, 40.9
min_lon, max_lon = -74.25, -73.70

result = (
    df
    # Filter out rows with null puLocationId and restrict to 2016
    # .filter(pl.col("puLocationId").is_not_null())
    .filter(
        pl.col("tpepPickupDateTime").is_between(
            datetime(2010, 1, 1), datetime(2018, 1, 1)
        )
    )
    # Filter trips by the NYC bounding box (based on startLat and startLon)
    .filter(
        (pl.col("startLat") >= min_lat)
        & (pl.col("startLat") <= max_lat)
        & (pl.col("startLon") >= min_lon)
        & (pl.col("startLon") <= max_lon)
    )
    # # Compute trip duration in minutes (convert nanoseconds to minutes)
    .with_columns(
        (
            (pl.col("tpepDropoffDateTime") - pl.col("tpepPickupDateTime")).cast(
                pl.Int64
            )
            / 1e9
            / 60
        ).alias("trip_duration")
    )
    # # Calculate average speed in mph: (tripDistance miles) / (duration in hours)
    .with_columns(
        (pl.col("tripDistance") * 60 / pl.col("trip_duration")).alias("avg_speed")
    )
    # Additional filtering on computed metrics:
    #   - Ensure positive trip duration,
    #   - Keep trips longer than 0.5 miles,
    #   - Fare amount below 150
    .filter((pl.col("trip_duration") > 0) & (pl.col("fareAmount") < 150))
    # Extract the date from the pickup datetime
    .with_columns(pl.col("tpepPickupDateTime").dt.date().alias("date"))
    # # Group by date and paymentType and compute aggregates
    .group_by(["date", "paymentType"])
    .agg(
        [
            pl.len().alias("num_trips"),
            pl.mean("tripDistance").alias("avg_trip_distance"),
            pl.sum("fareAmount").alias("total_fare"),
            pl.mean("trip_duration").alias("avg_duration"),
            pl.mean("avg_speed").alias("avg_speed"),
            pl.mean("tipAmount").alias("avg_tip"),
        ]
    )
    .sort(["date", "paymentType"])
)

qplan = result.explain(format="plain", optimized=False)
print(f"NAIVE Q-PLAN:\n {qplan}")
print(qplan)

qplan = result.explain(format="plain", optimized=True)
print(f"OPTIMIZED Q-PLAN:\n {qplan}")

result.show_graph(show=False, output_path="query-plan.png")

# Execute final complex query with advanced error handling
logger.info("Executing complex aggregation query")
try:
    if isinstance(collect_args["engine"], pl.GPUEngine):
        # Strict GPU mode - will raise on unsupported operations
        final_result = result.collect(**collect_args)
        logger.info("Complex query executed successfully on GPU (strict mode)")
    else:
        final_result = result.collect(**collect_args)
        logger.info(
            f"Complex query executed successfully on {collect_args['engine']} engine"
        )

    print(final_result)

except pl.exceptions.ComputeError as e:
    logger.warning(f"GPU execution failed: {e}")
    logger.info("Falling back to streaming engine for complex aggregation")

    # Fallback to streaming for memory-intensive operations
    final_result = result.collect(engine="streaming")
    logger.info("Fallback to streaming engine successful")
    print(final_result)

except Exception as e:
    logger.error(f"Unexpected error: {e}")
    logger.info("Using CPU engine as last resort")
    final_result = result.collect(engine="cpu")
    print(final_result)

# Print usage examples for advanced GPU features
print("\n" + "=" * 60)
print("GPU CONFIGURATION EXAMPLES")
print("=" * 60)
print("Environment variables for advanced GPU usage:")
print("• POLARS_GPU_CONFIG=strict    - Strict GPU mode (no CPU fallback)")
print("• POLARS_GPU_CONFIG=verbose   - Enable verbose debugging")
print("• POLARS_VERBOSE=1            - Enable verbose mode")
print("")
print("Example usage:")
print("  POLARS_GPU_CONFIG=verbose python src/main.py")
print("  POLARS_GPU_CONFIG=strict python src/main.py")
print("")
print("For more GPU examples, see: src/gpu_examples/")
print("• gpu_basic.py      - Basic GPU usage patterns")
print("• gpu_advanced.py   - Advanced GPUEngine configuration")
print("• gpu_monitoring.py - GPU memory monitoring and debugging")
print("• gpu_benchmarks.py - Performance benchmarking suite")
