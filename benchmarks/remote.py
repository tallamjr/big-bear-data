from __future__ import annotations

HOST = "arg1"


def build_rsync_push(local_dir: str, host: str, remote_dir: str) -> list[str]:
    return ["rsync", "-a", "--delete", local_dir, f"{host}:{remote_dir}"]


def build_rsync_pull(host: str, remote_path: str, local_path: str) -> list[str]:
    return ["rsync", "-a", f"{host}:{remote_path}", local_path]


def build_remote_sweep_command(
    remote_repo: str, python_exe: str, host: str = HOST
) -> list[str]:
    inner = f"cd {remote_repo} && {python_exe} -m benchmarks.run"
    return ["ssh", host, inner]
