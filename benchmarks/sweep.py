from __future__ import annotations

from pathlib import Path

from benchmarks.config import EngineConfig


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
    if engine.use_cudf_pandas:
        return [python_exe, "-m", "cudf.pandas", "-m", f"queries.{engine.module}"]
    return [python_exe, "-m", f"queries.{engine.module}"]
