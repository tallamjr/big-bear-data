"""DuckDB benchmark implementations using DataFrame API"""

import duckdb
from pathlib import Path
from .schemas import validate_query_result


class DuckDBBenchmark:
    def __init__(self, data_path: str = "data/tpch/tpch-10"):
        self.data_path = Path(data_path)
        self.conn = duckdb.connect()

    def _register_tables(self):
        """Register parquet files as DuckDB tables"""
        tables = [
            "lineitem",
            "orders",
            "customer",
            "supplier",
            "part",
            "nation",
            "region",
        ]
        for table in tables:
            file_path = self.data_path / f"{table}.parquet"
            self.conn.execute(
                f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM '{file_path}'"
            )

    # Query 1: Simple Aggregation - Revenue by order status
    def simple_aggregation(self):
        self._register_tables()

        result = self.conn.execute(
            """
            SELECT
                o_orderstatus,
                SUM(o_totalprice) as total_revenue,
                COUNT(*) as order_count
            FROM orders
            GROUP BY o_orderstatus
            ORDER BY total_revenue DESC
        """
        ).df()

        validate_query_result(result, "simple_aggregation", "duckdb")
        return result

    # Query 2: Customer segments with high account balance
    def customer_segments(self):
        self._register_tables()

        result = self.conn.execute(
            """
            SELECT
                c_mktsegment,
                AVG(c_acctbal) as avg_acctbal,
                COUNT(*) as customer_count
            FROM customer
            WHERE c_acctbal > 5000
            GROUP BY c_mktsegment
            ORDER BY avg_acctbal DESC
        """
        ).df()

        validate_query_result(result, "customer_segments", "duckdb")
        return result

    # Query 3: Orders with customer information
    def order_customer_join(self):
        self._register_tables()

        result = self.conn.execute(
            """
            SELECT
                o.o_orderkey,
                c.c_name,
                o.o_totalprice,
                c.c_mktsegment
            FROM orders o
            JOIN customer c ON o.o_custkey = c.c_custkey
            WHERE o.o_totalprice > 100000
            ORDER BY o.o_totalprice DESC
            LIMIT 1000
        """
        ).df()

        validate_query_result(result, "order_customer_join", "duckdb")
        return result

    # Query 4: Top suppliers by revenue
    def supplier_revenue(self):
        self._register_tables()

        result = self.conn.execute(
            """
            SELECT
                s.s_name,
                s.s_nationkey,
                SUM(l.l_extendedprice * (1 - l.l_discount)) as total_revenue
            FROM lineitem l
            JOIN supplier s ON l.l_suppkey = s.s_suppkey
            GROUP BY s.s_name, s.s_nationkey
            ORDER BY total_revenue DESC
            LIMIT 100
        """
        ).df()

        validate_query_result(result, "supplier_revenue", "duckdb")
        return result

    # Query 5: Detailed order analysis
    def detailed_orders(self):
        self._register_tables()

        result = self.conn.execute(
            """
            SELECT
                c.c_name,
                o.o_orderdate,
                l.l_extendedprice * (1 - l.l_discount) * (1 + l.l_tax) as line_total,
                c.c_mktsegment
            FROM lineitem l
            JOIN orders o ON l.l_orderkey = o.o_orderkey
            JOIN customer c ON o.o_custkey = c.c_custkey
            WHERE EXTRACT(YEAR FROM o.o_orderdate) = 1995
            ORDER BY line_total DESC
            LIMIT 1000
        """
        ).df()

        validate_query_result(result, "detailed_orders", "duckdb")
        return result

    # Query 6: Revenue trends with ranking
    def revenue_ranking(self):
        self._register_tables()

        result = self.conn.execute(
            """
            SELECT
                STRFTIME('%Y-%m', o_orderdate) as year_month,
                SUM(o_totalprice) as monthly_revenue,
                ROW_NUMBER() OVER (ORDER BY SUM(o_totalprice) DESC) as revenue_rank
            FROM orders
            GROUP BY STRFTIME('%Y-%m', o_orderdate)
            ORDER BY revenue_rank
        """
        ).df()

        validate_query_result(result, "revenue_ranking", "duckdb")
        return result

    def __del__(self):
        """Clean up connection"""
        if hasattr(self, "conn"):
            self.conn.close()
