from benchmarks.remote import (
    build_rsync_push,
    build_rsync_pull,
    build_remote_sweep_command,
)


def test_push_includes_archive_and_paths():
    cmd = build_rsync_push(
        "libs/polars-benchmark/",
        "arg1",
        "/home/tarek/big-bear-data/libs/polars-benchmark/",
    )
    assert cmd[0] == "rsync"
    assert "-a" in cmd
    assert "--delete" in cmd
    assert cmd[-2] == "libs/polars-benchmark/"
    assert cmd[-1] == "arg1:/home/tarek/big-bear-data/libs/polars-benchmark/"


def test_pull_brings_results_back():
    cmd = build_rsync_pull(
        "arg1", "/home/tarek/big-bear-data/results.parquet", "results.parquet"
    )
    assert cmd[0] == "rsync"
    assert cmd[-2] == "arg1:/home/tarek/big-bear-data/results.parquet"
    assert cmd[-1] == "results.parquet"


def test_remote_sweep_command_is_ssh():
    cmd = build_remote_sweep_command("/home/tarek/big-bear-data", ".venv/bin/python")
    assert cmd[0] == "ssh"
    assert cmd[1] == "arg1"
    joined = " ".join(cmd)
    assert "/home/tarek/big-bear-data" in joined
    assert "-m benchmarks.run" in joined
