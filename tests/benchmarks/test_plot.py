import matplotlib

matplotlib.use("Agg")

import polars as pl

from benchmarks.plot import (
    set_tahoma_style,
    compute_speedup,
    scaling_table,
    plot_per_query_bars,
    plot_speedup,
    plot_scaling_curve,
    plot_uvm_panel,
)


def _df():
    return pl.DataFrame(
        {
            "engine_label": [
                "polars-cpu-inmemory",
                "polars-gpu-managed",
                "polars-cpu-inmemory",
                "polars-gpu-cuda-async",
            ],
            "solution": [
                "polars",
                "polars-gpu-managed",
                "polars",
                "polars-gpu-cuda-async",
            ],
            "scale_factor": [10, 10, 100, 100],
            "query_number": [1, 1, 1, 1],
            "duration_s": [4.0, 1.0, 40.0, None],
            "iterations": [3, 3, 3, None],
            "version": ["1.31.0"] * 4,
            "io_type": ["parquet"] * 4,
            "status": ["ok", "ok", "ok", "failed"],
            "error": [None, None, None, "OOM"],
            "hardware": ["RTX 4090"] * 4,
            "timestamp": ["t"] * 4,
        }
    )


def test_set_style_returns_font():
    fam = set_tahoma_style()
    assert isinstance(fam, str) and fam


def test_compute_speedup():
    out = compute_speedup(_df(), "polars-cpu-inmemory", 10)
    gpu = out.filter(pl.col("engine_label") == "polars-gpu-managed")
    assert gpu["speedup"][0] == 4.0  # 4.0 / 1.0


def test_scaling_table_sums_ok_durations():
    tbl = scaling_table(_df())
    row = tbl.filter(
        (pl.col("engine_label") == "polars-cpu-inmemory")
        & (pl.col("scale_factor") == 100)
    )
    assert row["total_duration_s"][0] == 40.0


def test_plots_write_files(tmp_path):
    df = _df()
    set_tahoma_style()
    for fn, name in [
        (lambda p: plot_per_query_bars(df, 10, p), "bars.png"),
        (lambda p: plot_speedup(df, "polars-cpu-inmemory", 10, p), "speedup.png"),
        (lambda p: plot_scaling_curve(df, p), "scaling.png"),
        (lambda p: plot_uvm_panel(df, 100, p), "uvm.png"),
    ]:
        out = tmp_path / name
        assert fn(out).exists()
        assert out.with_suffix(".svg").exists()
