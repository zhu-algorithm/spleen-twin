#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-data/external/models/geneformer-v1-10m}"
base="https://huggingface.co/ctheodoris/Geneformer/resolve/main"
mkdir -p "$out_dir"

download() {
  local relative="$1"
  local output="$2"
  curl --fail --location --retry 5 --continue-at - \
    --output "$out_dir/$output" "$base/$relative"
}

download "Geneformer-V1-10M/config.json" "config.json"
download "Geneformer-V1-10M/model.safetensors" "model.safetensors"
download "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl" \
  "token_dictionary_gc30M.pkl"
download "geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl" \
  "gene_median_dictionary_gc30M.pkl"

shasum -a 256 "$out_dir"/config.json "$out_dir"/model.safetensors \
  "$out_dir"/token_dictionary_gc30M.pkl \
  "$out_dir"/gene_median_dictionary_gc30M.pkl > "$out_dir/SHA256SUMS"

