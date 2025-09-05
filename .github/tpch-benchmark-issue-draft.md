# TPC-H Benchmark Suite Implementation

## Overview
Implement a comprehensive benchmarking framework to compare performance across 5 data processing scenarios using TPC-H dataset (scale factor 10):

1. **Polars (CPU Streaming)** - Using `collect(streaming=True)` for optimal memory usage
2. **Polars (GPU)** - Using `collect(engine="gpu")` for GPU acceleration
3. **DuckDB** - Using DataFrame API for analytical queries
4. **cuDF** - NVIDIA GPU-accelerated DataFrame operations
5. **Pandas** - Traditional pandas operations as baseline

## Motivation
With the rapid evolution of data processing libraries, we need empirical performance comparisons on realistic workloads. The TPC-H benchmark provides standardised queries that test different aspects of data processing: aggregations, joins, filtering, and complex analytics.

## Implementation Plan

### 1. Benchmark Framework
- [x] Timing and memory measurement infrastructure (`benchmarks/framework.py`)
- [x] Pandera validation schemas for result consistency (`benchmarks/schemas.py`)
- [x] Parquet-based results storage for analysis and trends
- [x] System information capture (CPU, memory, GPU availability)

### 2. Query Implementations
Six equivalent TPC-H-inspired queries across all libraries:

- **Simple Aggregation**: Revenue by order status
- **Customer Segments**: High-value customer analysis by market segment
- **Order-Customer Join**: Large order analysis with customer details
- **Supplier Revenue**: Top suppliers by total revenue with complex calculations
- **Detailed Order Analysis**: Multi-table joins with date filtering
- **Revenue Ranking**: Time-series analysis with window functions

### 3. Library-Specific Implementations
- [x] `benchmarks/polars_bench.py` - CPU streaming + GPU variants
- [x] `benchmarks/duckdb_bench.py` - SQL-based DataFrame operations
- [x] `benchmarks/cudf_bench.py` - GPU-accelerated operations with graceful fallback
- [x] `benchmarks/pandas_bench.py` - Optimised pandas operations
- [x] All queries validated with Pandera schemas for consistency

### 4. Visualization and Analysis
- [x] Interactive Plotly dashboards (`benchmarks/visualizer.py`)
- [x] Static matplotlib/seaborn plots for presentations
- [x] Performance matrix heatmaps
- [x] Speedup comparison charts (relative to pandas baseline)
- [x] Memory usage analysis
- [x] Automated summary reports

### 5. Data and Infrastructure
- [x] TPC-H scale factor 10 dataset ready (`data/tpch/tpch-10/`)
- [x] Complete benchmark runner (`benchmarks/runner.py`)
- [x] Dependency management via uv (pyproject.toml updated)

## Expected Results

### Performance Characteristics
- **Polars Streaming**: Excellent memory efficiency, good CPU utilisation
- **Polars GPU**: Fastest on supported operations, limited query coverage
- **DuckDB**: Superior analytical query optimisation, excellent SQL pushdown
- **cuDF**: GPU memory constraints but very fast when data fits
- **Pandas**: Baseline performance, highest memory usage

### Benchmark Outputs
- `benchmark_results.parquet` - Structured results with timing, memory, system info
- `benchmark_plots/` - Complete visualisation dashboard
- Interactive HTML plots for detailed analysis
- Performance trends ready for scaling to tpch-100, tpch-1000

## Usage

```bash
# Run complete benchmark suite
cd benchmarks && python runner.py

# Generate visualisations only
python -c "from benchmarks.visualizer import BenchmarkVisualizer; BenchmarkVisualizer().create_dashboard()"

# View results
python -c "from benchmarks.visualizer import BenchmarkVisualizer; BenchmarkVisualizer().generate_summary_report()"
```

## Technical Details

### Query Validation
- Pandera schemas ensure identical results across libraries
- Automatic validation of column types, constraints, and data integrity
- Failed validations logged for debugging

### Memory Profiling
- `memory_profiler` integration for peak usage measurement
- Multiple runs with median values for stability
- Background memory monitoring during execution

### Result Schema
```python
{
    'timestamp': datetime,
    'library': str,           # 'polars_streaming', 'polars_gpu', 'duckdb', 'cudf', 'pandas'
    'query_name': str,        # 'simple_aggregation', 'supplier_revenue', etc.
    'dataset_size': str,      # 'tpch-10', ready for 'tpch-100'
    'execution_time_ms': float,
    'peak_memory_mb': float,
    'row_count': int,
    'system_info': str        # CPU/memory/GPU configuration
}
```

## Future Extensions
- [ ] TPC-H scale factors 100, 1000 for scalability analysis
- [ ] Multi-GPU cuDF benchmarks with Dask
- [ ] Cloud storage integration (S3, Azure Blob)
- [ ] Automated CI/CD benchmarking on PRs
- [ ] Apache Arrow integration benchmarks
- [ ] Real-time streaming scenarios

## Dependencies
```toml
pandas = ">=2.3.2"
polars = ">=1.26.0"
duckdb = ">=1.3.2"
# cudf via conda install -c rapidsai-nightly cudf
pandera = ">=0.26.1"
matplotlib = ">=3.10.6"
plotly = ">=6.3.0"
memory-profiler = ">=0.61.0"
```

## Acceptance Criteria
- [ ] All 6 queries implemented across 5 scenarios
- [ ] Results validated with Pandera schemas
- [ ] Memory and timing measurements working
- [ ] Visualisation dashboard generates successfully
- [ ] Performance summary report available
- [ ] Framework ready for larger datasets
- [ ] Documentation and usage examples complete

This benchmark suite will provide valuable insights for choosing optimal data processing tools based on workload characteristics, dataset size, and available hardware.
