from benchmarks.run import parse_args


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
