"""Pandera validation schemas for benchmark result consistency"""

import pandera as pa
from pandera import Column, DataFrameSchema


# Results schema for parquet storage
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
        "execution_time_ms": Column(pa.Float, checks=pa.Check.ge(0)),
        "peak_memory_mb": Column(pa.Float, checks=pa.Check.ge(0)),
        "row_count": Column(pa.Int, checks=pa.Check.ge(0)),
        "system_info": Column(pa.String),
    },
    strict=True,
)


# Query result validation schemas
SimpleAggregationSchema = DataFrameSchema(
    {
        "o_orderstatus": Column(pa.String),
        "total_revenue": Column(pa.Float, checks=pa.Check.ge(0)),
        "order_count": Column(pa.Int, checks=pa.Check.ge(0)),
    }
)

CustomerSegmentSchema = DataFrameSchema(
    {
        "c_mktsegment": Column(pa.String),
        "avg_acctbal": Column(pa.Float),
        "customer_count": Column(pa.Int, checks=pa.Check.ge(0)),
    }
)

OrderCustomerJoinSchema = DataFrameSchema(
    {
        "o_orderkey": Column(pa.Int),
        "c_name": Column(pa.String),
        "o_totalprice": Column(pa.Float, checks=pa.Check.ge(0)),
        "c_mktsegment": Column(pa.String),
    }
)

SupplierRevenueSchema = DataFrameSchema(
    {
        "s_name": Column(pa.String),
        "s_nationkey": Column(pa.Int),
        "total_revenue": Column(pa.Float, checks=pa.Check.ge(0)),
    }
)

DetailedOrderSchema = DataFrameSchema(
    {
        "c_name": Column(pa.String),
        "o_orderdate": Column(pa.Date),
        "line_total": Column(pa.Float, checks=pa.Check.ge(0)),
        "c_mktsegment": Column(pa.String),
    }
)

RevenueRankingSchema = DataFrameSchema(
    {
        "year_month": Column(pa.String),
        "monthly_revenue": Column(pa.Float, checks=pa.Check.ge(0)),
        "revenue_rank": Column(pa.Int, checks=pa.Check.ge(1)),
    }
)


def validate_query_result(result_df, query_name: str, library: str):
    """Validate query results using appropriate schema"""
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

    try:
        schema.validate(result_df)
        return True
    except pa.errors.SchemaError as e:
        print(f"Validation failed for {library} - {query_name}: {e}")
        return False
