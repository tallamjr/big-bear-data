import polars as pl

from benchmarks.aggregate import aggregate_timings, write_results, RESULTS_SCHEMA

HEADER = "solution,version,query_number,duration[s],io_type,scale_factor\n"


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + "".join(rows))
    return path


def test_aggregate_takes_median_over_iterations(tmp_path):
    csv = _csv(
        tmp_path / "polars" / "timings.csv",
        [
            "polars,1.31.0,1,2.0,parquet,10\n",
            "polars,1.31.0,1,4.0,parquet,10\n",  # median 3.0
            "polars,1.31.0,2,1.0,parquet,10\n",
        ],
    )
    records = [
        {
            "engine_label": "polars-cpu-inmemory",
            "scale_factor": 10,
            "status": "ok",
            "error": None,
            "timings_csv": csv,
        }
    ]
    df = aggregate_timings(
        records, hardware="RTX 4090", timestamp="2026-06-19T00:00:00"
    )
    q1 = df.filter((pl.col("query_number") == 1))
    assert q1["duration_s"][0] == 3.0
    assert q1["iterations"][0] == 2
    assert q1["engine_label"][0] == "polars-cpu-inmemory"
    assert q1["hardware"][0] == "RTX 4090"


def test_aggregate_preserves_failures_with_null_duration(tmp_path):
    records = [
        {
            "engine_label": "polars-gpu-cuda-async",
            "scale_factor": 100,
            "status": "failed",
            "error": "OOM",
            "timings_csv": None,
        }
    ]
    df = aggregate_timings(records, hardware="RTX 4090", timestamp="t")
    assert df.height == 1
    assert df["status"][0] == "failed"
    assert df["duration_s"][0] is None
    assert df["error"][0] == "OOM"


def test_schema_columns(tmp_path):
    records = [
        {
            "engine_label": "pandas",
            "scale_factor": 10,
            "status": "failed",
            "error": "x",
            "timings_csv": None,
        }
    ]
    df = aggregate_timings(records, hardware="h", timestamp="t")
    assert set(df.columns) == set(RESULTS_SCHEMA.keys())


def test_write_results_roundtrip(tmp_path):
    records = [
        {
            "engine_label": "duckdb",
            "scale_factor": 10,
            "status": "failed",
            "error": "x",
            "timings_csv": None,
        }
    ]
    df = aggregate_timings(records, hardware="h", timestamp="t")
    out = tmp_path / "results.parquet"
    write_results(df, out)
    assert pl.read_parquet(out).height == 1


def test_ok_status_without_csv_is_marked_missing_timings():
    records = [
        {
            "engine_label": "polars-gpu-managed",
            "scale_factor": 100,
            "status": "ok",
            "error": None,
            "timings_csv": None,
        }
    ]
    df = aggregate_timings(records, hardware="h", timestamp="t")
    assert df.height == 1
    assert df["status"][0] == "missing_timings"
    assert df["duration_s"][0] is None
