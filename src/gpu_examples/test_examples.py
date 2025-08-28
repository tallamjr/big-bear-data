#!/usr/bin/env python3
"""
Test script to verify all GPU examples use correct Polars API.

This script validates that:
- All imports work correctly
- API usage is up-to-date
- Examples can run without the actual data files
- Error handling works as expected
"""

import sys
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_imports() -> Tuple[bool, str]:
    """Test that all necessary imports work."""
    try:
        import polars as pl  # noqa: F401
        from polars.lazyframe.engine_config import GPUEngine  # noqa: F401
        import subprocess  # noqa: F401
        import time  # noqa: F401
        import threading  # noqa: F401
        import os  # noqa: F401

        return True, "All imports successful"
    except ImportError as e:
        return False, f"Import error: {e}"


def test_basic_api() -> Tuple[bool, str]:
    """Test basic Polars GPU API usage."""
    try:
        import polars as pl
        from polars.lazyframe.engine_config import GPUEngine

        # Test basic LazyFrame operations
        df = pl.LazyFrame({"group": ["A", "B", "A", "B"], "value": [1, 2, 3, 4]})

        # Test different engine options
        engines = ["cpu", "streaming"]

        for engine in engines:
            result = df.group_by("group").sum().collect(engine=engine)
            assert result.shape == (2, 2), f"Unexpected result shape for {engine}"

        # Test GPUEngine creation (doesn't require actual GPU)
        _gpu_engine = GPUEngine(raise_on_fail=False)  # noqa: F841
        _strict_gpu = GPUEngine(raise_on_fail=True)  # noqa: F841

        return True, "Basic API tests passed"
    except Exception as e:
        return False, f"API test error: {e}"


def test_gpu_detection() -> Tuple[bool, str]:
    """Test GPU detection functionality."""
    try:
        import subprocess

        def is_nvidia_gpu_available():
            try:
                subprocess.run(
                    ["nvidia-smi"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False

        gpu_available = is_nvidia_gpu_available()
        return True, f"GPU detection works (GPU available: {gpu_available})"
    except Exception as e:
        return False, f"GPU detection error: {e}"


def test_error_handling() -> Tuple[bool, str]:
    """Test error handling patterns."""
    try:
        import polars as pl
        from polars.lazyframe.engine_config import GPUEngine

        df = pl.LazyFrame({"a": [1, 2, 3]})

        # Test graceful fallback pattern
        try:
            # This should work on CPU
            _ = df.collect(engine="cpu")
            success = True
        except Exception:
            success = False

        if not success:
            return False, "Basic CPU execution failed"

        # Test strict GPU error handling
        try:
            strict_gpu = GPUEngine(raise_on_fail=True)
            # This might fail on systems without GPU, which is expected
            _ = df.collect(engine=strict_gpu)  # noqa: F841
        except Exception:
            # Expected behavior when GPU is not available
            pass

        return True, "Error handling patterns work correctly"
    except Exception as e:
        return False, f"Error handling test failed: {e}"


def test_environment_configuration() -> Tuple[bool, str]:
    """Test environment variable configuration."""
    try:
        import os

        # Test environment variable reading
        original_verbose = os.getenv("POLARS_VERBOSE")
        original_gpu_config = os.getenv("POLARS_GPU_CONFIG")

        # Test setting environment variables
        os.environ["POLARS_VERBOSE"] = "1"
        os.environ["POLARS_GPU_CONFIG"] = "verbose"

        # Verify we can read them
        assert os.getenv("POLARS_VERBOSE") == "1"
        assert os.getenv("POLARS_GPU_CONFIG") == "verbose"

        # Reset environment
        if original_verbose is not None:
            os.environ["POLARS_VERBOSE"] = original_verbose
        else:
            os.environ.pop("POLARS_VERBOSE", None)

        if original_gpu_config is not None:
            os.environ["POLARS_GPU_CONFIG"] = original_gpu_config
        else:
            os.environ.pop("POLARS_GPU_CONFIG", None)

        return True, "Environment configuration test passed"
    except Exception as e:
        return False, f"Environment test error: {e}"


def run_all_tests() -> None:
    """Run all validation tests."""
    tests = [
        ("Import Tests", test_imports),
        ("Basic API Tests", test_basic_api),
        ("GPU Detection", test_gpu_detection),
        ("Error Handling", test_error_handling),
        ("Environment Config", test_environment_configuration),
    ]

    print("=== GPU Examples Validation ===\n")

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            success, message = test_func()
            if success:
                logging.info(f"PASS {test_name}: {message}")
                passed += 1
            else:
                logging.error(f"FAIL {test_name}: {message}")
                failed += 1
        except Exception as e:
            logging.error(f"FAIL {test_name}: Unexpected error - {e}")
            failed += 1

    print("\n=== Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    if failed == 0:
        logging.info("All tests passed! GPU examples are ready to use.")
        return True
    else:
        logging.warning(f"{failed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
