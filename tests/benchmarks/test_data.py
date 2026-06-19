from types import SimpleNamespace
import pytest

from benchmarks.data import (
    required_free_gb,
    check_disk_space,
    tables_present,
    ensure_tables,
)


def test_required_free_scales_with_sf():
    assert required_free_gb(100) >= required_free_gb(10)
    assert required_free_gb(1) >= 5


def test_check_disk_space_raises_when_low(tmp_path):
    def fake_usage(p):
        return SimpleNamespace(total=0, used=0, free=1 * 1024**3)  # 1 GB free

    with pytest.raises(RuntimeError, match="insufficient disk"):
        check_disk_space(tmp_path, 100, free_bytes_fn=fake_usage)


def test_check_disk_space_ok_when_high(tmp_path):
    def fake_usage(p):
        return SimpleNamespace(total=0, used=0, free=500 * 1024**3)

    check_disk_space(tmp_path, 100, free_bytes_fn=fake_usage)  # no raise


def test_tables_present_detects_lineitem(tmp_path):
    sf_dir = tmp_path / "scale-10.0"
    sf_dir.mkdir(parents=True)
    assert tables_present(tmp_path, 10) is False
    (sf_dir / "lineitem.parquet").write_bytes(b"x")
    assert tables_present(tmp_path, 10) is True


def test_ensure_tables_skips_generation_when_present(tmp_path):
    sf_dir = tmp_path / "scale-10.0"
    sf_dir.mkdir(parents=True)
    (sf_dir / "lineitem.parquet").write_bytes(b"x")
    called = []

    def runner(*a, **k):
        called.append(a)
        return SimpleNamespace(returncode=0)

    out = ensure_tables(tmp_path, tmp_path, 10, python_exe="py", runner=runner)
    assert out == sf_dir
    assert called == []  # no generation triggered


def test_check_disk_space_handles_missing_dir(tmp_path):
    """check_disk_space should work even when target dir doesn't exist yet.
    It should stat the nearest existing ancestor, not the missing path itself.
    """
    missing = tmp_path / "does" / "not" / "exist" / "yet"
    call_log = []

    def fake_usage(p):
        call_log.append(p)
        # Verify the function resolved to an existing ancestor
        import os

        assert os.path.exists(p), f"disk check stat'd a non-existent path: {p}"
        return SimpleNamespace(total=0, used=0, free=500 * 1024**3)

    check_disk_space(missing, 100, free_bytes_fn=fake_usage)  # must not raise
    assert len(call_log) == 1
    # The call should have been to an existing ancestor, not the missing path
    assert call_log[0] != str(missing)


def test_scale_dir_matches_harness_float_format(tmp_path):
    """Harness resolves scale dir using f'scale-{float(scale_factor)}' (e.g. scale-10.0).
    Test that tables_present and ensure_tables use the same float-formatted directory.
    """
    # Create the float-formatted dir that the harness expects
    sf_dir = tmp_path / "scale-10.0"
    sf_dir.mkdir(parents=True)
    (sf_dir / "lineitem.parquet").write_bytes(b"x")

    # tables_present must find tables in scale-10.0, not scale-10
    assert tables_present(tmp_path, 10) is True

    # Prove ensure_tables targets the float dir (returns scale-10.0, not scale-10)
    called = []

    def runner(*a, **k):
        called.append(a)
        return SimpleNamespace(returncode=0)

    out = ensure_tables(tmp_path, tmp_path, 10, python_exe="py", runner=runner)
    assert out == tmp_path / "scale-10.0"
    assert called == []  # no generation since tables already present
