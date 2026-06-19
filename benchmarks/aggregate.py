from __future__ import annotations

from pathlib import Path

import polars as pl

RESULTS_SCHEMA: dict[str, pl.DataType] = {
    "engine_label": pl.Utf8,
    "solution": pl.Utf8,
    "scale_factor": pl.Int64,
    "query_number": pl.Int64,
    "duration_s": pl.Float64,
    "iterations": pl.Int64,
    "version": pl.Utf8,
    "io_type": pl.Utf8,
    "status": pl.Utf8,
    "error": pl.Utf8,
    "hardware": pl.Utf8,
    "timestamp": pl.Utf8,
}


def _failure_row(rec, hardware, timestamp) -> dict:
    status = rec["status"] if rec["status"] != "ok" else "missing_timings"
    return {
        "engine_label": rec["engine_label"],
        "solution": None,
        "scale_factor": rec["scale_factor"],
        "query_number": None,
        "duration_s": None,
        "iterations": None,
        "version": None,
        "io_type": None,
        "status": status,
        "error": rec.get("error"),
        "hardware": hardware,
        "timestamp": timestamp,
    }


def aggregate_timings(run_records, *, hardware: str, timestamp: str) -> pl.DataFrame:
    rows: list[dict] = []
    for rec in run_records:
        if rec["status"] != "ok" or rec.get("timings_csv") is None:
            rows.append(_failure_row(rec, hardware, timestamp))
            continue
        raw = pl.read_csv(rec["timings_csv"])
        med = (
            raw.group_by("query_number")
            .agg(
                pl.col("duration[s]").median().alias("duration_s"),
                pl.len().alias("iterations"),
                pl.col("solution").first().alias("solution"),
                pl.col("version").first().alias("version"),
                pl.col("io_type").first().alias("io_type"),
            )
            .sort("query_number")
        )
        for r in med.iter_rows(named=True):
            rows.append(
                {
                    "engine_label": rec["engine_label"],
                    "solution": r["solution"],
                    "scale_factor": rec["scale_factor"],
                    "query_number": r["query_number"],
                    "duration_s": float(r["duration_s"]),
                    "iterations": int(r["iterations"]),
                    "version": r["version"],
                    "io_type": r["io_type"],
                    "status": "ok",
                    "error": None,
                    "hardware": hardware,
                    "timestamp": timestamp,
                }
            )
    return pl.DataFrame(rows, schema=RESULTS_SCHEMA)


def write_results(df: pl.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
