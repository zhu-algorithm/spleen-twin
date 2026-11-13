#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-data/external/bone_marrow/10x_20k_bmmnc_5p_ht_v2}"
base_url="https://cf.10xgenomics.com/samples/cell-vdj/6.1.0/20k_BMMNC_5pv2_HT_nextgem_20k_BMMNC_5pv2_HT_nextgem"
prefix="20k_BMMNC_5pv2_HT_nextgem_20k_BMMNC_5pv2_HT_nextgem"
mkdir -p "$out_dir"

files=(
  "${prefix}_count_sample_feature_bc_matrix.h5"
  "${prefix}_count_feature_reference.csv"
  "${prefix}_vdj_b_filtered_contig_annotations.csv"
  "${prefix}_vdj_b_filtered_contig.fasta"
  "${prefix}_vdj_b_clonotypes.csv"
  "${prefix}_metrics_summary.csv"
)

for file in "${files[@]}"; do
  curl --fail --location --retry 5 --continue-at - \
    --output "$out_dir/$file" "$base_url/$file"
done

checksum_paths=()
for file in "${files[@]}"; do
  checksum_paths+=("$out_dir/$file")
done
shasum -a 256 "${checksum_paths[@]}" > "$out_dir/SHA256SUMS"

python_bin="python3"
if [[ -x .venv-atlas/bin/python ]]; then
  python_bin=".venv-atlas/bin/python"
fi
"$python_bin" scripts/validate_bone_marrow_vdj.py "$out_dir" \
  --output "$out_dir/validation_summary.json"

