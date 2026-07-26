#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://d37ci6vzurychx.cloudfront.net/trip-data"
DEST_DIR="${1:-data/raw}"

mkdir -p "${DEST_DIR}"

for month in 01 04 08; do
  file="green_tripdata_2020-${month}.parquet"
  curl -L "${BASE_URL}/${file}" -o "${DEST_DIR}/${file}"
done
