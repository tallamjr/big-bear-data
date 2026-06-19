from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import polars as pl

logger = logging.getLogger(__name__)

_TRANSPARENT = True


def set_tahoma_style() -> str:
    """Apply Tahoma + transparent-background rcParams; fall back if Tahoma absent."""
    available = {f.name for f in fm.fontManager.ttflist}
    family = "Tahoma" if "Tahoma" in available else "DejaVu Sans"
    if family != "Tahoma":
        logger.warning("Tahoma font not found; falling back to %s", family)
    plt.rcParams.update(
        {
            "font.family": family,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.transparent": _TRANSPARENT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.figsize": (10, 6),
        }
    )
    return family


def compute_speedup(
    df: pl.DataFrame, baseline_label: str, scale_factor: int
) -> pl.DataFrame:
    sub = df.filter(
        (pl.col("scale_factor") == scale_factor) & (pl.col("status") == "ok")
    )
    base = sub.filter(pl.col("engine_label") == baseline_label).select(
        ["query_number", pl.col("duration_s").alias("baseline_s")]
    )
    return sub.join(base, on="query_number").with_columns(
        (pl.col("baseline_s") / pl.col("duration_s")).alias("speedup")
    )


def scaling_table(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("status") == "ok")
        .group_by(["engine_label", "scale_factor"])
        .agg(pl.col("duration_s").sum().alias("total_duration_s"))
        .sort(["engine_label", "scale_factor"])
    )


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150, transparent=_TRANSPARENT)
    svg = out_path.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight", transparent=_TRANSPARENT)
    plt.close(fig)
    return out_path


def plot_per_query_bars(df, scale_factor, out_path) -> Path:
    sub = df.filter(
        (pl.col("scale_factor") == scale_factor) & (pl.col("status") == "ok")
    )
    pdf = sub.to_pandas()
    fig, ax = plt.subplots()
    if not pdf.empty:
        pivot = pdf.pivot_table(
            index="query_number", columns="engine_label", values="duration_s"
        )
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("duration (s)")
        ax.set_title(f"TPC-H per-query runtime (SF{scale_factor})")
    return _save(fig, out_path)


def plot_speedup(df, baseline_label, scale_factor, out_path) -> Path:
    sp = compute_speedup(df, baseline_label, scale_factor).to_pandas()
    fig, ax = plt.subplots()
    if not sp.empty:
        pivot = sp.pivot_table(
            index="query_number", columns="engine_label", values="speedup"
        )
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(1.0, color="grey", linewidth=0.8)
        ax.set_ylabel(f"speedup vs {baseline_label}")
        ax.set_title(f"Speedup over {baseline_label} (SF{scale_factor})")
    return _save(fig, out_path)


def plot_scaling_curve(df, out_path) -> Path:
    tbl = scaling_table(df).to_pandas()
    fig, ax = plt.subplots()
    for label, grp in tbl.groupby("engine_label"):
        ax.plot(grp["scale_factor"], grp["total_duration_s"], marker="o", label=label)
    ax.set_xlabel("scale factor")
    ax.set_ylabel("total runtime (s)")
    ax.set_yscale("log")
    ax.set_title("Total TPC-H runtime vs scale factor")
    ax.legend()
    return _save(fig, out_path)


def plot_uvm_panel(df, scale_factor, out_path) -> Path:
    sub = df.filter(pl.col("scale_factor") == scale_factor)
    gpu = sub.filter(pl.col("engine_label").str.contains("gpu"))
    pdf = gpu.to_pandas()
    fig, ax = plt.subplots()
    if not pdf.empty:
        agg = (
            pdf.groupby("engine_label")
            .agg(
                ok=("status", lambda s: (s == "ok").sum()),
                total_s=("duration_s", "sum"),
            )
            .reset_index()
        )
        ax.bar(agg["engine_label"], agg["total_s"].fillna(0))
        ax.set_ylabel("total runtime (s)")
        ax.set_title(
            f"GPU memory resource at SF{scale_factor}: cuda-async vs managed (UVM)"
        )
    return _save(fig, out_path)
