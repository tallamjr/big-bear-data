#!/usr/bin/env bash
set -euo pipefail

HOST="arg1"
REMOTE_REPO="/home/tarek/big-bear-data"
PY=".venv/bin/python"

echo "Syncing repo state to ${HOST}"
ssh "${HOST}" "cd ${REMOTE_REPO} && git fetch && git checkout issues/1/benchmarks && git pull --ff-only"

echo "Pushing vendored harness (gitignored) to ${HOST}"
rsync -a --delete "libs/polars-benchmark/" "${HOST}:${REMOTE_REPO}/libs/polars-benchmark/"
rsync -a "benchmarks/" "${HOST}:${REMOTE_REPO}/benchmarks/"

echo "Installing harness runtime deps into remote venv"
ssh "${HOST}" "cd ${REMOTE_REPO} && uv pip install pydantic pydantic-settings linetimer"

echo "Running sweep on ${HOST} (this generates data and runs the matrix)"
ssh "${HOST}" "cd ${REMOTE_REPO} && ${PY} -m benchmarks.run"

echo "Pulling results.parquet back to mac"
rsync -a "${HOST}:${REMOTE_REPO}/results.parquet" "results.parquet"
echo "Done. Run: .venv/bin/python -m benchmarks.run --plot-only"
