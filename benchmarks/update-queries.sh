#!/usr/bin/env bash

PDS_DIR="../libs/polars-benchmark"

echo $PDS_DIR

pushd $PDS_DIR && git pull --all
rsync -ravh data/answers ./data/
rsync -ravh queries .
popd
