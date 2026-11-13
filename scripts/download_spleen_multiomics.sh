#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-data/external/hubmap}"
mkdir -p "$out_dir"

download() {
  local url="$1"
  local output="$2"
  .venv-atlas/bin/python scripts/parallel_range_download.py "$url" "$output" --workers 12 --chunk-mib 1
}

download \
  "https://hubmap-data-products.s3.amazonaws.com/6135c7c8-b2f9-4eb2-bf0c-859fd0afdc6f/SP_raw.h5ad" \
  "$out_dir/hubmap_spleen_rnaseq_raw_2025-05-19.h5ad"

download \
  "https://hubmap-data-products.s3.amazonaws.com/f74d01ae-5e1e-482a-9a2c-c4480c43b1b2/SP_raw.h5ad" \
  "$out_dir/hubmap_spleen_codex_raw_2024-11-07.h5ad"

shasum -a 256 "$out_dir"/*.h5ad > "$out_dir/SHA256SUMS"
