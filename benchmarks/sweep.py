from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from benchmarks.config import EngineConfig

logger = logging.getLogger(__name__)


def build_run_env(
    engine: EngineConfig,
    scale_factor: int,
    *,
    iterations: int,
    timings_dir: Path,
    tables_dir: Path,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment for one harness run. Does not mutate base_env."""
    env: dict[str, str] = dict(base_env or {})
    env.update(engine.env)
    env["SCALE_FACTOR"] = str(scale_factor)
    env["RUN_LOG_TIMINGS"] = "true"
    env["RUN_ITERATIONS"] = str(iterations)
    env["PATH_TIMINGS"] = str(timings_dir)
    env["PATH_TABLES"] = str(tables_dir)
    if engine.env.get("RUN_POLARS_GPU") == "true":
        env["CUDA_MODULE_LOADING"] = "EAGER"
    return env


def build_command(engine: EngineConfig, python_exe: str = "python") -> list[str]:
    """Build the harness invocation command for an engine."""
    return [python_exe, "-m", f"queries.{engine.module}"]


def run_one(
    engine,
    scale_factor,
    *,
    harness_dir,
    tables_dir,
    timings_root,
    python_exe,
    iterations,
    runner=subprocess.run,
):
    """Run a single (engine, scale) harness invocation, capturing failures."""
    timings_dir = Path(timings_root) / engine.label / f"sf{scale_factor}"
    timings_dir.mkdir(parents=True, exist_ok=True)
    env = build_run_env(
        engine,
        scale_factor,
        iterations=iterations,
        timings_dir=timings_dir,
        tables_dir=Path(tables_dir),
        base_env=dict(os.environ),
    )
    cmd = build_command(engine, python_exe)
    logger.info("Running %s at SF%s", engine.label, scale_factor)
    try:
        proc = runner(
            cmd,
            env=env,
            cwd=str(harness_dir),
            timeout=engine.timeout_s,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        logger.warning("%s SF%s timed out", engine.label, scale_factor)
        return {
            "engine_label": engine.label,
            "scale_factor": scale_factor,
            "status": "timeout",
            "error": f"timeout after {engine.timeout_s}s",
            "timings_csv": None,
            "returncode": None,
        }
    if proc.returncode != 0:
        logger.warning(
            "%s SF%s failed rc=%s", engine.label, scale_factor, proc.returncode
        )
        return {
            "engine_label": engine.label,
            "scale_factor": scale_factor,
            "status": "failed",
            "error": (proc.stderr or "")[-2000:],
            "timings_csv": None,
            "returncode": proc.returncode,
        }
    return {
        "engine_label": engine.label,
        "scale_factor": scale_factor,
        "status": "ok",
        "error": None,
        "timings_csv": timings_dir / "timings.csv",
        "returncode": 0,
    }


def run_sweep(
    engines,
    scales,
    *,
    harness_dir,
    tables_dir,
    timings_root,
    python_exe,
    iterations=3,
    runner=subprocess.run,
):
    """Run every (engine, scale) where scale is in both `scales` and engine.scales."""
    records = []
    for engine in engines:
        for scale in scales:
            if scale not in engine.scales:
                continue
            records.append(
                run_one(
                    engine,
                    scale,
                    harness_dir=harness_dir,
                    tables_dir=tables_dir,
                    timings_root=timings_root,
                    python_exe=python_exe,
                    iterations=iterations,
                    runner=runner,
                )
            )
    return records
