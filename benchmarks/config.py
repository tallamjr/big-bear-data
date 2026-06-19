from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EngineConfig:
    """A single benchmarkable engine configuration mapped to a harness module."""

    label: str
    module: str  # harness queries.<module>
    env: dict[str, str] = field(default_factory=dict)
    scales: tuple[int, ...] = (10, 100)
    use_cudf_pandas: bool = False
    timeout_s: int = 7200


SCALE_FACTORS: tuple[int, ...] = (10, 100)

ENGINE_MATRIX: tuple[EngineConfig, ...] = (
    EngineConfig("polars-cpu-inmemory", "polars", {}, (10, 100)),
    EngineConfig(
        "polars-cpu-streaming", "polars", {"RUN_POLARS_STREAMING": "true"}, (10, 100)
    ),
    EngineConfig(
        "polars-gpu-cuda-async",
        "polars",
        {"RUN_POLARS_GPU": "true", "RUN_USE_RMM_MR": "cuda-async"},
        (10, 100),
    ),
    EngineConfig(
        "polars-gpu-managed",
        "polars",
        {"RUN_POLARS_GPU": "true", "RUN_USE_RMM_MR": "managed"},
        (10, 100),
    ),
    EngineConfig("duckdb", "duckdb", {}, (10, 100)),
)
