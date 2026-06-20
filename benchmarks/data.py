from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def _nearest_existing(path: Path) -> Path:
    """Resolve to the nearest existing ancestor directory.

    When a path doesn't exist yet, traverse up to find the first existing parent.
    This is used to check disk space on the filesystem where data will be written.
    """
    p = Path(path)
    while not p.exists():
        if p.parent == p:
            # Reached the root and nothing exists; return root
            break
        p = p.parent
    return p


def required_free_gb(scale_factor: int) -> float:
    """Rough parquet + intermediate headroom needed to generate a scale factor."""
    return max(5.0, scale_factor * 0.5)


def check_disk_space(
    path: Path, scale_factor: int, free_bytes_fn=shutil.disk_usage
) -> None:
    existing_path = _nearest_existing(Path(path))
    free_gb = free_bytes_fn(str(existing_path)).free / (1024**3)
    needed = required_free_gb(scale_factor)
    if free_gb < needed:
        raise RuntimeError(
            f"insufficient disk for SF{scale_factor}: need ~{needed:.0f}GB, have {free_gb:.0f}GB"
        )


def tpchgen_executable(python_exe: str) -> str:
    """Locate the tpchgen-cli binary next to the given venv python, falling
    back to PATH lookup when it is not found there.

    The python_exe is typically a symlink (e.g. .venv/bin/python -> the
    interpreter); resolving the file would follow it away from the venv bin
    directory where tpchgen-cli lives, so only the parent directory is
    resolved, not the symlinked file itself.
    """
    candidate = Path(python_exe).parent.resolve() / "tpchgen-cli"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("tpchgen-cli")
    if found:
        return found
    raise RuntimeError(f"tpchgen-cli not found next to {python_exe} or on PATH")


def cast_decimal_columns_to_float(sf_dir: Path) -> list[str]:
    """Cast any Decimal columns in every parquet table under sf_dir to Float64,
    rewriting the file in place. Returns the sorted list of table file names that
    were modified. Idempotent: tables with no Decimal columns are left untouched.
    Needed because cudf-polars (GPU) does not support the Decimal dtype.
    Uses streaming sink_parquet so it stays memory-bounded on very large tables
    (e.g. SF100 lineitem)."""
    modified = []
    for pq in sorted(Path(sf_dir).glob("*.parquet")):
        lf = pl.scan_parquet(pq)
        schema = lf.collect_schema()
        decimal_cols = [
            name for name, dt in schema.items() if isinstance(dt, pl.Decimal)
        ]
        if not decimal_cols:
            continue
        tmp = pq.with_suffix(".parquet.casting.tmp")
        lf.with_columns(
            [pl.col(c).cast(pl.Float64) for c in decimal_cols]
        ).sink_parquet(tmp)
        os.replace(tmp, pq)
        modified.append(pq.name)
    return modified


def tables_present(tables_dir: Path, scale_factor: int) -> bool:
    return (
        Path(tables_dir) / f"scale-{float(scale_factor)}" / "lineitem.parquet"
    ).exists()


def ensure_tables(
    harness_dir,
    tables_dir,
    scale_factor,
    *,
    python_exe,
    runner=subprocess.run,
) -> Path:
    """Generate TPC-H parquet tables for a scale factor if missing, with a disk guard."""
    tables_dir = Path(tables_dir)
    sf_dir = tables_dir / f"scale-{float(scale_factor)}"
    if tables_present(tables_dir, scale_factor):
        logger.info("Tables for SF%s already present", scale_factor)
        return sf_dir
    check_disk_space(tables_dir, scale_factor)
    sf_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating SF%s tables with tpchgen-cli", scale_factor)
    gen = runner(
        [
            tpchgen_executable(python_exe),
            "--output-dir",
            str(sf_dir),
            "--format",
            "parquet",
            "-s",
            str(scale_factor),
        ],
        cwd=str(harness_dir),
        check=False,
    )
    if getattr(gen, "returncode", 0) != 0:
        raise RuntimeError(f"tpchgen-cli failed for SF{scale_factor}")
    modified = cast_decimal_columns_to_float(sf_dir)
    if modified:
        logger.info("Cast Decimal columns to Float64 in %s", ", ".join(modified))
    return sf_dir
