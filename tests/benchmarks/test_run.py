from benchmarks.run import parse_args, resolve_paths


def test_parse_scales_list():
    ns = parse_args(["--scales", "10,100"])
    assert ns.scales == [10, 100]


def test_plot_only_flag():
    ns = parse_args(["--plot-only"])
    assert ns.plot_only is True


def test_defaults():
    ns = parse_args([])
    assert ns.iterations == 3
    assert ns.results.endswith("results.parquet")
    assert ns.plot_only is False


def test_resolve_paths_returns_absolute():
    h, t, ti = resolve_paths(
        "libs/polars-benchmark",
        "libs/polars-benchmark/data/tables",
        "benchmarks/_timings",
    )
    assert h.is_absolute() and t.is_absolute() and ti.is_absolute()
    assert h.name == "polars-benchmark"
    assert ti.name == "_timings"
