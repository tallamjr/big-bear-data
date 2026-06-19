from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def required_free_gb(scale_factor: int) -> float:
    """Rough parquet + intermediate headroom needed to generate a scale factor."""
    return max(5.0, scale_factor * 0.5)


def check_disk_space(
    path: Path, scale_factor: int, free_bytes_fn=shutil.disk_usage
) -> None:
    free_gb = free_bytes_fn(str(path)).free / (1024**3)
    needed = required_free_gb(scale_factor)
    if free_gb < needed:
        raise RuntimeError(
            f"insufficient disk for SF{scale_factor}: need ~{needed:.0f}GB, have {free_gb:.0f}GB"
        )


def tables_present(tables_dir: Path, scale_factor: int) -> bool:
    return (Path(tables_dir) / f"scale-{scale_factor}" / "lineitem.parquet").exists()


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
    sf_dir = tables_dir / f"scale-{scale_factor}"
    if tables_present(tables_dir, scale_factor):
        logger.info("Tables for SF%s already present", scale_factor)
        return sf_dir
    check_disk_space(tables_dir, scale_factor)
    sf_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating SF%s tables with tpchgen-cli", scale_factor)
    gen = runner(
        [
            "tpchgen-cli",
            "--output-dir",
            str(sf_dir),
            "--format",
            "parquet",
            "-s",
            str(scale_factor),
        ],
        cwd=str(harness_dir),
        check=True,
    )
    if getattr(gen, "returncode", 0) != 0:
        raise RuntimeError(f"tpchgen-cli failed for SF{scale_factor}")
    return sf_dir
