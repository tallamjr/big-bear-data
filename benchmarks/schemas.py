"""Simple validation for benchmark result consistency"""


def validate_query_result(result_df, query_name: str, library: str):
    """
    Lightweight validation for benchmark results.
    Focuses on basic sanity checks rather than complex schema validation.
    """
    try:
        # Basic sanity checks
        if len(result_df) == 0:
            print(f"Warning: {library} - {query_name} returned no results")
            return False

        # Query-specific validations
        if query_name == "simple_aggregation":
            required_cols = ["o_orderstatus", "total_revenue", "order_count"]
        elif query_name == "customer_segments":
            required_cols = ["c_mktsegment", "avg_acctbal", "customer_count"]
        elif query_name == "order_customer_join":
            required_cols = ["o_orderkey", "c_name", "o_totalprice", "c_mktsegment"]
        elif query_name == "supplier_revenue":
            required_cols = ["s_name", "s_nationkey", "total_revenue"]
        elif query_name == "detailed_orders":
            required_cols = ["c_name", "o_orderdate", "line_total", "c_mktsegment"]
        elif query_name == "revenue_ranking":
            required_cols = ["year_month", "monthly_revenue", "revenue_rank"]
        else:
            # Unknown query, skip validation
            return True

        # Check for required columns
        df_cols = set(result_df.columns)
        missing_cols = set(required_cols) - df_cols
        if missing_cols:
            print(f"Warning: {library} - {query_name} missing columns: {missing_cols}")
            return False

        return True

    except Exception as e:
        print(f"Validation error for {library} - {query_name}: {e}")
        return False
