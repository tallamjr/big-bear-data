"""Polars benchmark implementations (CPU streaming and GPU)"""

from pathlib import Path

import polars as pl

from .schemas import validate_query_result


class PolarsBenchmark:
    def __init__(self, data_path: str = "data/tpch/tpch-10"):
        self.data_path = Path(data_path)

    def _load_tables(self):
        """Load TPC-H tables lazily for optimal performance"""
        return {
            "lineitem": pl.scan_parquet(self.data_path / "lineitem.parquet"),
            "orders": pl.scan_parquet(self.data_path / "orders.parquet"),
            "customer": pl.scan_parquet(self.data_path / "customer.parquet"),
            "supplier": pl.scan_parquet(self.data_path / "supplier.parquet"),
            "part": pl.scan_parquet(self.data_path / "part.parquet"),
            "nation": pl.scan_parquet(self.data_path / "nation.parquet"),
            "region": pl.scan_parquet(self.data_path / "region.parquet"),
        }

    # Query 1: Simple Aggregation - Revenue by order status
    def simple_aggregation_streaming(self):
        tables = self._load_tables()

        result = (
            tables["orders"]
            .group_by("o_orderstatus")
            .agg(
                [
                    pl.sum("o_totalprice").alias("total_revenue"),
                    pl.count().alias("order_count"),
                ]
            )
            .sort("total_revenue", descending=True)
            .collect(engine="streaming")
        )

        validate_query_result(result, "simple_aggregation", "polars_streaming")
        return result

    def simple_aggregation_gpu(self):
        tables = self._load_tables()

        result = (
            tables["orders"]
            .group_by("o_orderstatus")
            .agg(
                [
                    pl.sum("o_totalprice").alias("total_revenue"),
                    pl.count().alias("order_count"),
                ]
            )
            .sort("total_revenue", descending=True)
            .collect(engine="gpu")
        )

        validate_query_result(result, "simple_aggregation", "polars_gpu")
        return result

    # Query 2: Customer segments with high account balance
    def customer_segments_streaming(self):
        tables = self._load_tables()

        result = (
            tables["customer"]
            .filter(pl.col("c_acctbal") > 5000)
            .group_by("c_mktsegment")
            .agg(
                [
                    pl.mean("c_acctbal").alias("avg_acctbal"),
                    pl.count().alias("customer_count"),
                ]
            )
            .sort("avg_acctbal", descending=True)
            .collect(engine="streaming")
        )

        validate_query_result(result, "customer_segments", "polars_streaming")
        return result

    def customer_segments_gpu(self):
        tables = self._load_tables()

        result = (
            tables["customer"]
            .filter(pl.col("c_acctbal") > 5000)
            .group_by("c_mktsegment")
            .agg(
                [
                    pl.mean("c_acctbal").alias("avg_acctbal"),
                    pl.count().alias("customer_count"),
                ]
            )
            .sort("avg_acctbal", descending=True)
            .collect(engine="gpu")
        )

        validate_query_result(result, "customer_segments", "polars_gpu")
        return result

    # Query 3: Orders with customer information
    def order_customer_join_streaming(self):
        tables = self._load_tables()

        result = (
            tables["orders"]
            .join(tables["customer"], left_on="o_custkey", right_on="c_custkey")
            .select(["o_orderkey", "c_name", "o_totalprice", "c_mktsegment"])
            .filter(pl.col("o_totalprice") > 100000)
            .sort("o_totalprice", descending=True)
            .head(1000)
            .collect(engine="streaming")
        )

        validate_query_result(result, "order_customer_join", "polars_streaming")
        return result

    def order_customer_join_gpu(self):
        tables = self._load_tables()

        result = (
            tables["orders"]
            .join(tables["customer"], left_on="o_custkey", right_on="c_custkey")
            .select(["o_orderkey", "c_name", "o_totalprice", "c_mktsegment"])
            .filter(pl.col("o_totalprice") > 100000)
            .sort("o_totalprice", descending=True)
            .head(1000)
            .collect(engine="gpu")
        )

        validate_query_result(result, "order_customer_join", "polars_gpu")
        return result

    # Query 4: Top suppliers by revenue
    def supplier_revenue_streaming(self):
        tables = self._load_tables()

        result = (
            tables["lineitem"]
            .with_columns(
                [
                    (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias(
                        "revenue"
                    )
                ]
            )
            .group_by("l_suppkey")
            .agg([pl.sum("revenue").alias("total_revenue")])
            .join(tables["supplier"], left_on="l_suppkey", right_on="s_suppkey")
            .select(["s_name", "s_nationkey", "total_revenue"])
            .sort("total_revenue", descending=True)
            .head(100)
            .collect(engine="streaming")
        )

        validate_query_result(result, "supplier_revenue", "polars_streaming")
        return result

    def supplier_revenue_gpu(self):
        tables = self._load_tables()

        result = (
            tables["lineitem"]
            .with_columns(
                [
                    (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias(
                        "revenue"
                    )
                ]
            )
            .group_by("l_suppkey")
            .agg([pl.sum("revenue").alias("total_revenue")])
            .join(tables["supplier"], left_on="l_suppkey", right_on="s_suppkey")
            .select(["s_name", "s_nationkey", "total_revenue"])
            .sort("total_revenue", descending=True)
            .head(100)
            .collect(engine="gpu")
        )

        validate_query_result(result, "supplier_revenue", "polars_gpu")
        return result

    # Query 5: Detailed order analysis
    def detailed_orders_streaming(self):
        tables = self._load_tables()

        result = (
            tables["lineitem"]
            .with_columns(
                [
                    (
                        pl.col("l_extendedprice")
                        * (1 - pl.col("l_discount"))
                        * (1 + pl.col("l_tax"))
                    ).alias("line_total")
                ]
            )
            .join(tables["orders"], left_on="l_orderkey", right_on="o_orderkey")
            .join(tables["customer"], left_on="o_custkey", right_on="c_custkey")
            .filter(pl.col("o_orderdate").dt.year() == 1995)
            .select(["c_name", "o_orderdate", "line_total", "c_mktsegment"])
            .sort("line_total", descending=True)
            .head(1000)
            .collect(engine="streaming")
        )

        validate_query_result(result, "detailed_orders", "polars_streaming")
        return result

    def detailed_orders_gpu(self):
        tables = self._load_tables()

        result = (
            tables["lineitem"]
            .with_columns(
                [
                    (
                        pl.col("l_extendedprice")
                        * (1 - pl.col("l_discount"))
                        * (1 + pl.col("l_tax"))
                    ).alias("line_total")
                ]
            )
            .join(tables["orders"], left_on="l_orderkey", right_on="o_orderkey")
            .join(tables["customer"], left_on="o_custkey", right_on="c_custkey")
            .filter(pl.col("o_orderdate").dt.year() == 1995)
            .select(["c_name", "o_orderdate", "line_total", "c_mktsegment"])
            .sort("line_total", descending=True)
            .head(1000)
            .collect(engine="gpu")
        )

        validate_query_result(result, "detailed_orders", "polars_gpu")
        return result

    # Query 6: Revenue trends with ranking
    def revenue_ranking_streaming(self):
        tables = self._load_tables()

        result = (
            tables["orders"]
            .with_columns(
                [pl.col("o_orderdate").dt.strftime("%Y-%m").alias("year_month")]
            )
            .group_by("year_month")
            .agg([pl.sum("o_totalprice").alias("monthly_revenue")])
            .with_columns(
                [
                    pl.col("monthly_revenue")
                    .rank(method="ordinal", descending=True)
                    .alias("revenue_rank")
                ]
            )
            .sort("revenue_rank")
            .collect(engine="streaming")
        )

        validate_query_result(result, "revenue_ranking", "polars_streaming")
        return result

    def revenue_ranking_gpu(self):
        tables = self._load_tables()

        result = (
            tables["orders"]
            .with_columns(
                [pl.col("o_orderdate").dt.strftime("%Y-%m").alias("year_month")]
            )
            .group_by("year_month")
            .agg([pl.sum("o_totalprice").alias("monthly_revenue")])
            .with_columns(
                [
                    pl.col("monthly_revenue")
                    .rank(method="ordinal", descending=True)
                    .alias("revenue_rank")
                ]
            )
            .sort("revenue_rank")
            .collect(engine="gpu")
        )

        validate_query_result(result, "revenue_ranking", "polars_gpu")
        return result
