from pathlib import Path

from benchmarks.config import ENGINE_MATRIX
from benchmarks.sweep import build_run_env, build_command


def _engine(label):
    return next(e for e in ENGINE_MATRIX if e.label == label)


def test_env_sets_core_variables():
    env = build_run_env(
        _engine("polars-cpu-inmemory"),
        10,
        iterations=3,
        timings_dir=Path("/t/run1"),
        tables_dir=Path("/d/tables"),
        base_env={},
    )
    assert env["SCALE_FACTOR"] == "10"
    assert env["RUN_LOG_TIMINGS"] == "true"
    assert env["RUN_ITERATIONS"] == "3"
    assert env["PATH_TIMINGS"] == "/t/run1"
    assert env["PATH_TABLES"] == "/d/tables"


def test_env_merges_engine_overrides():
    env = build_run_env(
        _engine("polars-cpu-streaming"),
        100,
        iterations=3,
        timings_dir=Path("/t"),
        tables_dir=Path("/d"),
        base_env={},
    )
    assert env["RUN_POLARS_STREAMING"] == "true"


def test_gpu_env_sets_eager_cuda_module_loading():
    env = build_run_env(
        _engine("polars-gpu-managed"),
        100,
        iterations=3,
        timings_dir=Path("/t"),
        tables_dir=Path("/d"),
        base_env={},
    )
    assert env["RUN_POLARS_GPU"] == "true"
    assert env["RUN_USE_RMM_MR"] == "managed"
    assert env["CUDA_MODULE_LOADING"] == "EAGER"


def test_cpu_env_has_no_cuda_module_loading():
    env = build_run_env(
        _engine("duckdb"),
        10,
        iterations=3,
        timings_dir=Path("/t"),
        tables_dir=Path("/d"),
        base_env={},
    )
    assert "CUDA_MODULE_LOADING" not in env


def test_command_default():
    assert build_command(_engine("polars-cpu-inmemory"), "/v/python") == [
        "/v/python",
        "-m",
        "queries.polars",
    ]


def test_command_cudf_uses_accelerator():
    assert build_command(_engine("cudf"), "/v/python") == [
        "/v/python",
        "-m",
        "cudf.pandas",
        "-m",
        "queries.pandas",
    ]


def test_base_env_is_not_mutated():
    base = {"HOME": "/home/tarek"}
    build_run_env(
        _engine("duckdb"),
        10,
        iterations=1,
        timings_dir=Path("/t"),
        tables_dir=Path("/d"),
        base_env=base,
    )
    assert base == {"HOME": "/home/tarek"}
