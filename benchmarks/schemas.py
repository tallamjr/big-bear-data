"""Pandera validation schemas for benchmark result consistency"""

import polars as pl
import pandera.polars as pa
from pandera.polars import Column, DataFrameSchema
from pandera.engines.polars_engine import (
    DateTime,
    String,
    Int64,
    Float64,
)


# Results schema for parquet storage
BenchmarkResultsSchema = DataFrameSchema(
    {
        "timestamp": Column(DateTime),
        "library": Column(
            String,
            checks=pa.Check.isin(
                ["polars_streaming", "polars_gpu", "duckdb", "cudf", "pandas"]
            ),
        ),
        "query_name": Column(String),
        "dataset_size": Column(String),
        "execution_time_ms": Column(Float64, checks=pa.Check.ge(0)),
        "peak_memory_mb": Column(Float64, checks=pa.Check.ge(0)),
        "row_count": Column(Int64, checks=pa.Check.ge(0)),
        "system_info": Column(String),
    },
    strict=True,
)


# Query result validation schemas
SimpleAggregationSchema = DataFrameSchema(
    {
        "o_orderstatus": Column(str),
        "total_revenue": Column(pl.Float64, checks=pa.Check.ge(0), coerce=True),
        "order_count": Column(pl.Int64, checks=pa.Check.ge(0), coerce=True),
    }
)

CustomerSegmentSchema = DataFrameSchema(
    {
        "c_mktsegment": Column(str),
        "avg_acctbal": Column(pl.Float64, coerce=True),
        "customer_count": Column(pl.Int64, checks=pa.Check.ge(0), coerce=True),
    }
)

OrderCustomerJoinSchema = DataFrameSchema(
    {
        "o_orderkey": Column(pl.Int64, coerce=True),
        "c_name": Column(str),
        "o_totalprice": Column(pl.Float64, checks=pa.Check.ge(0), coerce=True),
        "c_mktsegment": Column(str),
    }
)

SupplierRevenueSchema = DataFrameSchema(
    {
        "s_name": Column(str),
        "s_nationkey": Column(pl.Int64, coerce=True),
        "total_revenue": Column(pl.Float64, checks=pa.Check.ge(0), coerce=True),
    }
)

DetailedOrderSchema = DataFrameSchema(
    {
        "c_name": Column(str),
        "o_orderdate": Column(pl.Date, coerce=True),
        "line_total": Column(pl.Float64, checks=pa.Check.ge(0), coerce=True),
        "c_mktsegment": Column(str),
    }
)

RevenueRankingSchema = DataFrameSchema(
    {
        "year_month": Column(str),
        "monthly_revenue": Column(pl.Float64, checks=pa.Check.ge(0), coerce=True),
        "revenue_rank": Column(pl.Int64, checks=pa.Check.ge(1), coerce=True),
    }
)


def validate_query_result(result_df, query_name: str, library: str):
    """Validate query results using appropriate schema, converting to Polars format"""
    schema_map = {
        "simple_aggregation": SimpleAggregationSchema,
        "customer_segments": CustomerSegmentSchema,
        "order_customer_join": OrderCustomerJoinSchema,
        "supplier_revenue": SupplierRevenueSchema,
        "detailed_orders": DetailedOrderSchema,
        "revenue_ranking": RevenueRankingSchema,
    }

    if query_name not in schema_map:
        raise ValueError(f"Unknown query name: {query_name}")

    schema = schema_map[query_name]

    # Convert result to Polars DataFrame for validation
    if isinstance(result_df, pl.DataFrame):
        # Already a Polars DataFrame
        polars_df = result_df
    elif hasattr(result_df, "to_polars"):
        # cuDF DataFrame
        polars_df = result_df.to_polars()
    else:
        # pandas DataFrame (DuckDB results) or other pandas-like
        polars_df = pl.from_pandas(result_df)

    try:
        schema.validate(polars_df)
        return True
    except pa.errors.SchemaError as e:
        print(f"Validation failed for {library} - {query_name}: {e}")
        return False
