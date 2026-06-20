from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
from pathlib import Path

import polars as pl

from benchmarks.config import ENGINE_MATRIX, SCALE_FACTORS
from benchmarks.data import ensure_tables
from benchmarks.sweep import run_sweep
from benchmarks.aggregate import aggregate_timings, write_results
from benchmarks import plot as plotmod

logger = logging.getLogger("benchmarks.run")


def resolve_paths(harness_dir, tables_dir, timings_root):
    """Resolve orchestration paths to absolute so the harness subprocess
    (which runs in a different cwd) reads/writes the same locations."""
    return (
        Path(harness_dir).resolve(),
        Path(tables_dir).resolve(),
        Path(timings_root).resolve(),
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the GPU Polars benchmark sweep")
    p.add_argument(
        "--scales",
        default=",".join(str(s) for s in SCALE_FACTORS),
        type=lambda s: [int(x) for x in s.split(",")],
    )
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--harness-dir", default="libs/polars-benchmark")
    p.add_argument("--tables-dir", default="libs/polars-benchmark/data/tables")
    p.add_argument("--results", default="results.parquet")
    p.add_argument("--plots-dir", default="benchmarks/plots")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--hardware", default="RTX 4090 / 24GB")
    p.add_argument("--python-exe", default=sys.executable)
    return p.parse_args(argv)


def _render_all(results_path: Path, plots_dir: Path) -> None:
    df = pl.read_parquet(results_path)
    plotmod.set_tahoma_style()
    plots_dir.mkdir(parents=True, exist_ok=True)
    baseline = "polars-cpu-inmemory"
    scales_present = sorted(df["scale_factor"].unique().to_list())
    for sf in scales_present:
        plotmod.plot_per_query_bars(df, sf, plots_dir / f"per_query_sf{sf}.png")
        plotmod.plot_speedup(df, baseline, sf, plots_dir / f"speedup_sf{sf}.png")
    plotmod.plot_scaling_curve(df, plots_dir / "scaling_curve.png")
    if 100 in scales_present:
        plotmod.plot_uvm_panel(df, 100, plots_dir / "uvm_panel_sf100.png")


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ns = parse_args(argv)
    plots_dir = Path(ns.plots_dir)
    if ns.plot_only:
        _render_all(Path(ns.results), plots_dir)
        logger.info("Plots written to %s", plots_dir)
        return 0

    harness_dir, tables_dir, timings_root = resolve_paths(
        ns.harness_dir, ns.tables_dir, "benchmarks/_timings"
    )
    for sf in ns.scales:
        ensure_tables(harness_dir, tables_dir, sf, python_exe=ns.python_exe)
    records = run_sweep(
        ENGINE_MATRIX,
        ns.scales,
        harness_dir=harness_dir,
        tables_dir=tables_dir,
        timings_root=timings_root,
        python_exe=ns.python_exe,
        iterations=ns.iterations,
    )
    timestamp = _dt.datetime.now(_dt.timezone.utc).isoformat()
    df = aggregate_timings(records, hardware=ns.hardware, timestamp=timestamp)
    write_results(df, Path(ns.results))
    logger.info("Wrote %s rows to %s", df.height, ns.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
