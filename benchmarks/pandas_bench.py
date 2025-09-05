"""Pandas benchmark implementations"""

import pandas as pd
from pathlib import Path
from .schemas import validate_query_result


class PandasBenchmark:
    def __init__(self, data_path: str = "data/tpch/tpch-10"):
        self.data_path = Path(data_path)

    def _load_tables(self):
        """Load TPC-H tables using pandas"""
        return {
            "lineitem": pd.read_parquet(self.data_path / "lineitem.parquet"),
            "orders": pd.read_parquet(self.data_path / "orders.parquet"),
            "customer": pd.read_parquet(self.data_path / "customer.parquet"),
            "supplier": pd.read_parquet(self.data_path / "supplier.parquet"),
            "part": pd.read_parquet(self.data_path / "part.parquet"),
            "nation": pd.read_parquet(self.data_path / "nation.parquet"),
            "region": pd.read_parquet(self.data_path / "region.parquet"),
        }

    # Query 1: Simple Aggregation - Revenue by order status
    def simple_aggregation(self):
        tables = self._load_tables()
        orders = tables["orders"]

        result = (
            orders.groupby("o_orderstatus")
            .agg({"o_totalprice": ["sum", "count"]})
            .reset_index()
        )

        # Flatten column names
        result.columns = ["o_orderstatus", "total_revenue", "order_count"]
        result = result.sort_values("total_revenue", ascending=False)

        validate_query_result(result, "simple_aggregation", "pandas")
        return result

    # Query 2: Customer segments with high account balance
    def customer_segments(self):
        tables = self._load_tables()
        customer = tables["customer"]

        high_balance = customer[customer["c_acctbal"] > 5000]
        result = (
            high_balance.groupby("c_mktsegment")
            .agg({"c_acctbal": "mean", "c_custkey": "count"})
            .reset_index()
        )

        result.columns = ["c_mktsegment", "avg_acctbal", "customer_count"]
        result = result.sort_values("avg_acctbal", ascending=False)

        validate_query_result(result, "customer_segments", "pandas")
        return result

    # Query 3: Orders with customer information
    def order_customer_join(self):
        tables = self._load_tables()
        orders = tables["orders"]
        customer = tables["customer"]

        # Filter high-value orders first for better performance
        high_orders = orders[orders["o_totalprice"] > 100000]

        result = (
            high_orders.merge(
                customer, left_on="o_custkey", right_on="c_custkey", how="inner"
            )[["o_orderkey", "c_name", "o_totalprice", "c_mktsegment"]]
            .sort_values("o_totalprice", ascending=False)
            .head(1000)
            .reset_index(drop=True)
        )

        validate_query_result(result, "order_customer_join", "pandas")
        return result

    # Query 4: Top suppliers by revenue
    def supplier_revenue(self):
        tables = self._load_tables()
        lineitem = tables["lineitem"]
        supplier = tables["supplier"]

        # Calculate line revenue
        lineitem["revenue"] = lineitem["l_extendedprice"] * (1 - lineitem["l_discount"])

        supplier_revenue = (
            lineitem.groupby("l_suppkey").agg({"revenue": "sum"}).reset_index()
        )
        supplier_revenue.columns = ["l_suppkey", "total_revenue"]

        result = (
            supplier_revenue.merge(
                supplier, left_on="l_suppkey", right_on="s_suppkey", how="inner"
            )[["s_name", "s_nationkey", "total_revenue"]]
            .sort_values("total_revenue", ascending=False)
            .head(100)
            .reset_index(drop=True)
        )

        validate_query_result(result, "supplier_revenue", "pandas")
        return result

    # Query 5: Detailed order analysis
    def detailed_orders(self):
        tables = self._load_tables()
        lineitem = tables["lineitem"]
        orders = tables["orders"]
        customer = tables["customer"]

        # Calculate line total
        lineitem["line_total"] = (
            lineitem["l_extendedprice"]
            * (1 - lineitem["l_discount"])
            * (1 + lineitem["l_tax"])
        )

        # Convert date column and filter for 1995
        orders["o_orderdate"] = pd.to_datetime(orders["o_orderdate"])
        orders_1995 = orders[orders["o_orderdate"].dt.year == 1995]

        # Perform joins
        result = (
            lineitem.merge(
                orders_1995, left_on="l_orderkey", right_on="o_orderkey", how="inner"
            )
            .merge(customer, left_on="o_custkey", right_on="c_custkey", how="inner")[
                ["c_name", "o_orderdate", "line_total", "c_mktsegment"]
            ]
            .sort_values("line_total", ascending=False)
            .head(1000)
            .reset_index(drop=True)
        )

        validate_query_result(result, "detailed_orders", "pandas")
        return result

    # Query 6: Revenue trends with ranking
    def revenue_ranking(self):
        tables = self._load_tables()
        orders = tables["orders"]

        # Convert date and create year-month
        orders["o_orderdate"] = pd.to_datetime(orders["o_orderdate"])
        orders["year_month"] = orders["o_orderdate"].dt.strftime("%Y-%m")

        monthly_revenue = (
            orders.groupby("year_month").agg({"o_totalprice": "sum"}).reset_index()
        )
        monthly_revenue.columns = ["year_month", "monthly_revenue"]

        # Add ranking
        monthly_revenue["revenue_rank"] = (
            monthly_revenue["monthly_revenue"]
            .rank(method="first", ascending=False)
            .astype(int)
        )

        result = monthly_revenue.sort_values("revenue_rank").reset_index(drop=True)

        validate_query_result(result, "revenue_ranking", "pandas")
        return result
