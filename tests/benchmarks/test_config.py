from benchmarks.config import ENGINE_MATRIX, SCALE_FACTORS


def test_scale_factors():
    assert SCALE_FACTORS == (10, 100)


def test_labels_unique():
    labels = [e.label for e in ENGINE_MATRIX]
    assert len(labels) == len(set(labels))


def test_expected_engines_present():
    labels = {e.label for e in ENGINE_MATRIX}
    assert labels == {
        "polars-cpu-inmemory",
        "polars-cpu-streaming",
        "polars-gpu-cuda-async",
        "polars-gpu-managed",
        "duckdb",
        "pandas",
        "cudf",
    }


def test_gpu_engines_carry_rmm_resource():
    gpu = {e.label: e for e in ENGINE_MATRIX if "gpu" in e.label}
    assert gpu["polars-gpu-cuda-async"].env["RUN_USE_RMM_MR"] == "cuda-async"
    assert gpu["polars-gpu-managed"].env["RUN_USE_RMM_MR"] == "managed"
    assert all(e.env["RUN_POLARS_GPU"] == "true" for e in gpu.values())


def test_cudf_uses_pandas_module_via_accelerator():
    cudf = next(e for e in ENGINE_MATRIX if e.label == "cudf")
    assert cudf.module == "pandas"
    assert cudf.use_cudf_pandas is True


def test_pandas_runs_only_at_sf10():
    pandas = next(e for e in ENGINE_MATRIX if e.label == "pandas")
    assert pandas.scales == (10,)


def test_scales_are_subset_of_sweep():
    for e in ENGINE_MATRIX:
        assert set(e.scales).issubset(set(SCALE_FACTORS))
