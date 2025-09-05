"""cuDF benchmark implementations for GPU acceleration"""

from pathlib import Path
from .schemas import validate_query_result

try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False


class cuDFBenchmark:
    def __init__(self, data_path: str = "data/tpch/tpch-10"):
        self.data_path = Path(data_path)
        if not CUDF_AVAILABLE:
            raise ImportError(
                "cuDF not available. Install via conda from rapidsai channel."
            )

    def _load_tables(self):
        """Load TPC-H tables using cuDF"""
        return {
            "lineitem": cudf.read_parquet(self.data_path / "lineitem.parquet"),
            "orders": cudf.read_parquet(self.data_path / "orders.parquet"),
            "customer": cudf.read_parquet(self.data_path / "customer.parquet"),
            "supplier": cudf.read_parquet(self.data_path / "supplier.parquet"),
            "part": cudf.read_parquet(self.data_path / "part.parquet"),
            "nation": cudf.read_parquet(self.data_path / "nation.parquet"),
            "region": cudf.read_parquet(self.data_path / "region.parquet"),
        }

    # Query 1: Simple Aggregation - Revenue by order status
    def simple_aggregation(self):
        if not CUDF_AVAILABLE:
            raise RuntimeError("cuDF not available")

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

        validate_query_result(result.to_pandas(), "simple_aggregation", "cudf")
        return result

    # Query 2: Customer segments with high account balance
    def customer_segments(self):
        if not CUDF_AVAILABLE:
            raise RuntimeError("cuDF not available")

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

        validate_query_result(result.to_pandas(), "customer_segments", "cudf")
        return result

    # Query 3: Orders with customer information
    def order_customer_join(self):
        if not CUDF_AVAILABLE:
            raise RuntimeError("cuDF not available")

        tables = self._load_tables()
        orders = tables["orders"]
        customer = tables["customer"]

        # Filter high-value orders first
        high_orders = orders[orders["o_totalprice"] > 100000]

        result = (
            high_orders.merge(customer, left_on="o_custkey", right_on="c_custkey")[
                ["o_orderkey", "c_name", "o_totalprice", "c_mktsegment"]
            ]
            .sort_values("o_totalprice", ascending=False)
            .head(1000)
        )

        validate_query_result(result.to_pandas(), "order_customer_join", "cudf")
        return result

    # Query 4: Top suppliers by revenue
    def supplier_revenue(self):
        if not CUDF_AVAILABLE:
            raise RuntimeError("cuDF not available")

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
            supplier_revenue.merge(supplier, left_on="l_suppkey", right_on="s_suppkey")[
                ["s_name", "s_nationkey", "total_revenue"]
            ]
            .sort_values("total_revenue", ascending=False)
            .head(100)
        )

        validate_query_result(result.to_pandas(), "supplier_revenue", "cudf")
        return result

    # Query 5: Detailed order analysis
    def detailed_orders(self):
        if not CUDF_AVAILABLE:
            raise RuntimeError("cuDF not available")

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

        # Filter for 1995 orders
        orders_1995 = orders[orders["o_orderdate"].dt.year == 1995]

        result = (
            lineitem.merge(orders_1995, left_on="l_orderkey", right_on="o_orderkey")
            .merge(customer, left_on="o_custkey", right_on="c_custkey")[
                ["c_name", "o_orderdate", "line_total", "c_mktsegment"]
            ]
            .sort_values("line_total", ascending=False)
            .head(1000)
        )

        validate_query_result(result.to_pandas(), "detailed_orders", "cudf")
        return result

    # Query 6: Revenue trends with ranking
    def revenue_ranking(self):
        if not CUDF_AVAILABLE:
            raise RuntimeError("cuDF not available")

        tables = self._load_tables()
        orders = tables["orders"]

        # Create year-month column
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

        result = monthly_revenue.sort_values("revenue_rank")

        validate_query_result(result.to_pandas(), "revenue_ranking", "cudf")
        return result


# Fallback implementation when cuDF is not available
class MockCuDFBenchmark:
    def __init__(self, data_path: str = "data/tpch/tpch-10"):
        self.data_path = Path(data_path)

    def _mock_method(self):
        raise RuntimeError(
            "cuDF not available. Install via conda from rapidsai channel."
        )

    simple_aggregation = _mock_method
    customer_segments = _mock_method
    order_customer_join = _mock_method
    supplier_revenue = _mock_method
    detailed_orders = _mock_method
    revenue_ranking = _mock_method


# Export the appropriate class based on availability
CuDFBenchmark = cuDFBenchmark if CUDF_AVAILABLE else MockCuDFBenchmark
