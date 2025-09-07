"""Pandera validation schemas for benchmark result consistency"""

import pandera as pa
from pandera import Column, DataFrameSchema


# Results schema for parquet storage - Universal pandera schema
BenchmarkResultsSchema = DataFrameSchema(
    {
        "timestamp": Column(pa.DateTime),
        "library": Column(
            pa.String,
            checks=pa.Check.isin(
                ["polars_streaming", "polars_gpu", "duckdb", "cudf", "pandas"]
            ),
        ),
        "query_name": Column(pa.String),
        "dataset_size": Column(pa.String),
        "execution_time_ms": Column(pa.Float64, checks=pa.Check.ge(0)),
        "peak_memory_mb": Column(pa.Float64, checks=pa.Check.ge(0)),
        "row_count": Column(pa.Int64, checks=pa.Check.ge(0)),
        "system_info": Column(pa.String),
    },
    strict=True,
)


# Query result validation schemas - Universal schemas work across pandas/polars/cudf
SimpleAggregationSchema = DataFrameSchema(
    {
        "o_orderstatus": Column(pa.String),
        "total_revenue": Column(pa.Float64, checks=pa.Check.ge(0)),
        "order_count": Column(pa.Int64, checks=pa.Check.ge(0)),
    }
)

CustomerSegmentSchema = DataFrameSchema(
    {
        "c_mktsegment": Column(pa.String),
        "avg_acctbal": Column(pa.Float64),
        "customer_count": Column(pa.Int64, checks=pa.Check.ge(0)),
    }
)

OrderCustomerJoinSchema = DataFrameSchema(
    {
        "o_orderkey": Column(pa.Int64),
        "c_name": Column(pa.String),
        "o_totalprice": Column(pa.Float64, checks=pa.Check.ge(0)),
        "c_mktsegment": Column(pa.String),
    }
)

SupplierRevenueSchema = DataFrameSchema(
    {
        "s_name": Column(pa.String),
        "s_nationkey": Column(pa.Int64),
        "total_revenue": Column(pa.Float64, checks=pa.Check.ge(0)),
    }
)

DetailedOrderSchema = DataFrameSchema(
    {
        "c_name": Column(pa.String),
        "o_orderdate": Column(pa.DateTime),
        "line_total": Column(pa.Float64, checks=pa.Check.ge(0)),
        "c_mktsegment": Column(pa.String),
    }
)

RevenueRankingSchema = DataFrameSchema(
    {
        "year_month": Column(pa.String),
        "monthly_revenue": Column(pa.Float64, checks=pa.Check.ge(0)),
        "revenue_rank": Column(pa.Int64, checks=pa.Check.ge(1)),
    }
)


def validate_query_result(result_df, query_name: str, library: str):
    """
    Validate query results using pandera's universal backend support.
    Works with pandas, polars, cudf DataFrames automatically.
    """
    schema_map = {
        "simple_aggregation": SimpleAggregationSchema,
        "customer_segments": CustomerSegmentSchema,
        "order_customer_join": OrderCustomerJoinSchema,
        "supplier_revenue": SupplierRevenueSchema,
        "detailed_orders": DetailedOrderSchema,
        "revenue_ranking": RevenueRankingSchema,
    }

    if query_name not in schema_map:
        print(f"No schema defined for query: {query_name}")
        return True  # Skip validation for undefined queries

    schema = schema_map[query_name]

    try:
        # Pandera automatically detects the DataFrame backend and validates accordingly
        schema.validate(result_df)
        return True
    except pa.errors.SchemaError as e:
        print(f"Query result validation failed for {library} - {query_name}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected validation error for {library} - {query_name}: {e}")
        return False
