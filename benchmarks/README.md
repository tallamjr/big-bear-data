# Big Bear Data Benchmarks

A comprehensive benchmarking suite comparing data processing frameworks using TPC-H queries.

## Overview

This benchmark suite evaluates the performance of modern data processing libraries across a standardized set of analytical queries. We compare:

- **Pandas**: Traditional Python data analysis library
- **Polars**: Modern DataFrame library with lazy evaluation (CPU streaming + GPU)
- **DuckDB**: In-process analytical database
- **cuDF**: GPU-accelerated DataFrame library from RAPIDS

## TPC-H Benchmark Background

### What is TPC-H?

The **TPC-H** (Transaction Processing Performance Council - Ad Hoc) benchmark is an industry-standard decision support benchmark established in 1993. It simulates a business environment where users execute complex analytical queries against a database.

### History and Purpose

- **Created**: 1993 by the Transaction Processing Performance Council
- **Purpose**: Measure performance of decision support systems (data warehouses, analytics platforms)
- **Industry adoption**: Widely used by database vendors (Oracle, SQL Server, PostgreSQL, etc.) and cloud providers (AWS, GCP, Azure)
- **Data model**: Models a wholesale supplier managing orders and shipments

### TPC-H Schema

The benchmark uses 8 interconnected tables representing a business scenario:

```
├── NATION (25 rows) - Countries
├── REGION (5 rows) - Geographic regions
├── SUPPLIER (10K rows) - Parts suppliers
├── CUSTOMER (150K rows) - Customers
├── PART (200K rows) - Parts catalog
├── PARTSUPP (800K rows) - Supplier-part relationships
├── ORDERS (1.5M rows) - Customer orders
└── LINEITEM (6M rows) - Order line items (largest table)
```

*Row counts shown are for Scale Factor 1 (SF-1)*

### Scale Factors

TPC-H data is generated at different **Scale Factors (SF)**:
- **SF-0.1**: ~36MB total (development/testing)
- **SF-1**: ~360MB total (small production)
- **SF-10**: ~3.6GB total (medium production)
- **SF-100**: ~36GB total (large production)

## Our Benchmark Queries

We've selected 6 queries that represent common analytical patterns while testing different performance characteristics:

### 1. Simple Aggregation (`simple_aggregation`)
```sql
-- Revenue by order status
SELECT o_orderstatus, SUM(o_totalprice), COUNT(*)
FROM orders
GROUP BY o_orderstatus
ORDER BY SUM(o_totalprice) DESC;
```
**Tests**: Basic GROUP BY and aggregation performance

### 2. Customer Segments (`customer_segments`)
```sql
-- High-balance customers by market segment
SELECT c_mktsegment, AVG(c_acctbal), COUNT(*)
FROM customer
WHERE c_acctbal > 5000
GROUP BY c_mktsegment;
```
**Tests**: Filtering with aggregation, predicate pushdown optimization

### 3. Order Customer Join (`order_customer_join`)
```sql
-- High-value orders with customer details
SELECT o_orderkey, c_name, o_totalprice, c_mktsegment
FROM orders o
JOIN customer c ON o.o_custkey = c.c_custkey
WHERE o_totalprice > 100000
ORDER BY o_totalprice DESC
LIMIT 1000;
```
**Tests**: JOIN performance between medium-sized tables

### 4. Supplier Revenue (`supplier_revenue`)
```sql
-- Top suppliers by revenue
SELECT s_name, s_nationkey, SUM(l_extendedprice * (1 - l_discount))
FROM lineitem l
JOIN supplier s ON l.l_suppkey = s.s_suppkey
GROUP BY s_name, s_nationkey
ORDER BY revenue DESC
LIMIT 100;
```
**Tests**: Large table aggregation with JOIN (most compute-intensive)

### 5. Detailed Orders (`detailed_orders`)
```sql
-- 1995 order details with customer info
SELECT c_name, o_orderdate,
       l_extendedprice * (1 - l_discount) * (1 + l_tax) as line_total,
       c_mktsegment
FROM lineitem l
JOIN orders o ON l.l_orderkey = o.o_orderkey
JOIN customer c ON o.o_custkey = c.c_custkey
WHERE YEAR(o_orderdate) = 1995
ORDER BY line_total DESC
LIMIT 1000;
```
**Tests**: 3-way JOINs, date filtering, complex calculations

### 6. Revenue Ranking (`revenue_ranking`)
```sql
-- Monthly revenue trends with ranking
SELECT DATE_FORMAT(o_orderdate, '%Y-%m') as year_month,
       SUM(o_totalprice) as monthly_revenue,
       RANK() OVER (ORDER BY SUM(o_totalprice) DESC) as revenue_rank
FROM orders
GROUP BY year_month
ORDER BY revenue_rank;
```
**Tests**: Time-series aggregation, window functions

## Performance Characteristics Tested

| Query | Data Volume | JOIN Complexity | Operations | Optimization Opportunities |
|-------|-------------|-----------------|------------|---------------------------|
| Simple Aggregation | Small | None | GROUP BY, SUM, COUNT | Basic aggregation |
| Customer Segments | Small | None | Filter, GROUP BY, AVG | Predicate pushdown |
| Order Customer Join | Medium | 2-table | Inner JOIN, Filter | Hash vs other joins |
| Supplier Revenue | Large | 2-table | JOIN on largest table | Large data aggregation |
| Detailed Orders | Large | 3-table | Multiple JOINs, dates | Join reordering, lazy eval |
| Revenue Ranking | Medium | None | GROUP BY, window functions | Time-series optimization |

## Usage

### Basic Usage
```bash
# Run with default TPC-H SF-10 dataset
python benchmarks/runner.py

# Use smaller dataset for faster testing
python benchmarks/runner.py --data-path benchmarks/data/tpch-1

# Custom output file
python benchmarks/runner.py --data-path benchmarks/data/tpch-0.1 --results-file test_results.parquet
```

### Command Line Options
```bash
python benchmarks/runner.py --help
```

## Expected Performance Patterns

Based on the design characteristics of each framework:

### Pandas
- **Strengths**: Familiar API, mature ecosystem
- **Weaknesses**: Single-threaded, memory-intensive, no lazy evaluation
- **Expected**: Slowest overall, high memory usage

### Polars (CPU Streaming)
- **Strengths**: Lazy evaluation, query optimization, multi-threaded
- **Weaknesses**: Newer ecosystem
- **Expected**: Significantly faster than pandas, lower memory usage

### Polars (GPU)
- **Strengths**: GPU acceleration for parallel operations
- **Weaknesses**: Limited GPU query engine coverage
- **Expected**: Fastest for supported operations, may fall back to CPU

### DuckDB
- **Strengths**: Vectorized execution, columnar storage, SQL optimization
- **Weaknesses**: In-memory processing limits
- **Expected**: Excellent analytical query performance, memory efficient

### cuDF
- **Strengths**: Full GPU DataFrame API, RAPIDS ecosystem
- **Weaknesses**: GPU memory constraints, CUDA dependency
- **Expected**: Very fast for GPU-optimized operations

## Output

The benchmark produces:
1. **Performance metrics**: Execution time, peak memory usage, result row counts
2. **Parquet results file**: Structured data for further analysis
3. **Visualization dashboard**: Comparative performance charts

## Data Generation

TPC-H data can be generated using standard TPC-H tools:
- Use `dbgen` utility with appropriate scale factor
- Convert to Parquet format for optimal performance
- Ensure consistent schema across all test datasets

## System Requirements

- **CPU**: Multi-core recommended for parallel frameworks
- **Memory**: Minimum 16GB for SF-10, 32GB recommended
- **GPU**: NVIDIA GPU with CUDA support for GPU benchmarks
- **Storage**: Fast SSD recommended for large datasets
