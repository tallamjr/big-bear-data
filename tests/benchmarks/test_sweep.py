from pathlib import Path
from types import SimpleNamespace

from benchmarks.config import ENGINE_MATRIX, EngineConfig
from benchmarks.sweep import build_run_env, build_command, run_one, run_sweep


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


def make_runner(returncode=0, raise_timeout=False):
    calls = []

    def fake_run(cmd, *, env, cwd, timeout, capture_output, text):
        calls.append(SimpleNamespace(cmd=cmd, env=env, cwd=cwd, timeout=timeout))
        if raise_timeout:
            import subprocess

            raise subprocess.TimeoutExpired(cmd, timeout)
        return SimpleNamespace(
            returncode=returncode, stdout="", stderr="boom" if returncode else ""
        )

    fake_run.calls = calls
    return fake_run


def test_run_one_success(tmp_path):
    eng = EngineConfig("duckdb", "duckdb", {}, (10,))
    rec = run_one(
        eng,
        10,
        harness_dir=tmp_path,
        tables_dir=tmp_path / "d",
        timings_root=tmp_path / "t",
        python_exe="py",
        iterations=3,
        runner=make_runner(0),
    )
    assert rec["status"] == "ok"
    assert rec["timings_csv"] == tmp_path / "t" / "duckdb" / "sf10" / "timings.csv"


def test_run_one_failure_is_captured_not_raised(tmp_path):
    eng = EngineConfig(
        "polars-gpu-cuda-async",
        "polars",
        {"RUN_POLARS_GPU": "true", "RUN_USE_RMM_MR": "cuda-async"},
        (100,),
    )
    rec = run_one(
        eng,
        100,
        harness_dir=tmp_path,
        tables_dir=tmp_path / "d",
        timings_root=tmp_path / "t",
        python_exe="py",
        iterations=3,
        runner=make_runner(1),
    )
    assert rec["status"] == "failed"
    assert rec["returncode"] == 1
    assert "boom" in rec["error"]
    assert rec["timings_csv"] is None


def test_run_one_timeout_is_captured(tmp_path):
    eng = EngineConfig("pandas", "pandas", {}, (10,))
    rec = run_one(
        eng,
        10,
        harness_dir=tmp_path,
        tables_dir=tmp_path / "d",
        timings_root=tmp_path / "t",
        python_exe="py",
        iterations=3,
        runner=make_runner(raise_timeout=True),
    )
    assert rec["status"] == "timeout"


def test_run_sweep_respects_engine_scales(tmp_path):
    engines = [
        EngineConfig("pandas", "pandas", {}, (10,)),
        EngineConfig("duckdb", "duckdb", {}, (10, 100)),
    ]
    runner = make_runner(0)
    records = run_sweep(
        engines,
        (10, 100),
        harness_dir=tmp_path,
        tables_dir=tmp_path / "d",
        timings_root=tmp_path / "t",
        python_exe="py",
        iterations=1,
        runner=runner,
    )
    # pandas only at sf10 -> 1 run; duckdb at both -> 2 runs; total 3
    assert len(records) == 3
    pandas_scales = sorted(
        r["scale_factor"] for r in records if r["engine_label"] == "pandas"
    )
    assert pandas_scales == [10]
