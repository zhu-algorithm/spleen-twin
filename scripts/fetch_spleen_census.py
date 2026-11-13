"""Pull every human spleen cell in CELLxGENE Census and summarise it.

Census is the largest standardised aggregation of single-cell data in
existence; slicing it on ``tissue_general == 'spleen'`` gives the most
complete human spleen atlas currently assembled from public data.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import cellxgene_census
from scipy import sparse

CENSUS_VERSION = "2025-11-08"

# TileDB's S3 client does not read the shell proxy variables, and this
# sandbox has no direct DNS, so the proxy has to be configured explicitly.
import os as _os
import urllib.parse as _up

_proxy = _os.environ.get("https_proxy") or _os.environ.get("HTTPS_PROXY")
TILEDB_CONFIG = {
    "vfs.s3.region": "us-west-2",
    "vfs.s3.no_sign_request": "true",
    # Path-style requests avoid resolving a long bucket-specific hostname,
    # which is less reliable on restricted/corporate DNS resolvers.
    "vfs.s3.use_virtual_addressing": "false",
    # The public Census bucket occasionally pauses for more than TileDB's
    # short defaults.  Long reads are safe because every expression block is
    # checkpointed independently.
    "vfs.s3.connect_timeout_ms": "30000",
    "vfs.s3.request_timeout_ms": "30000",
    "vfs.s3.connect_max_tries": "4",
    "vfs.s3.retry_limit": "4",
}
if _proxy:
    _p = _up.urlparse(_proxy)
    TILEDB_CONFIG.update({
        "vfs.s3.proxy_scheme": _p.scheme or "http",
        "vfs.s3.proxy_host": _p.hostname or "localhost",
        "vfs.s3.proxy_port": str(_p.port or 80),
    })
VALUE_FILTER = "tissue_general == 'spleen' and is_primary_data == True"
# Scattered SOMA coordinates can make very large reads stall for many minutes.
# Keep blocks small enough to checkpoint regularly; allow tuning without edits.
BLOCK = int(_os.environ.get("SPLEEN_CENSUS_BLOCK", "10000"))
BLOCK_GROUP = int(_os.environ.get("SPLEEN_CENSUS_BLOCK_GROUP", "5"))
QUERY_SUBBLOCK = int(_os.environ.get("SPLEEN_CENSUS_QUERY_SUBBLOCK", str(BLOCK)))

GENES = [
    # germinal centre programme
    "AICDA", "BCL6", "GCSAM", "RGS13", "LMO2", "MEF2B", "S1PR2", "MKI67", "AURKB", "MYC", "BATF",
    # zone positioning
    "CXCR4", "CXCR5", "CCR7", "GPR183",
    # B lineage identity and stage
    "MS4A1", "CD19", "PAX5", "EBF1", "POU2AF1", "TCL1A", "CD27", "CD38", "FCRL4",
    "IGHM", "IGHD", "IGHG1", "IGHA1", "IGHE", "JCHAIN",
    # plasma cell
    "PRDM1", "XBP1", "IRF4", "SDC1",
    # bone-marrow style V(D)J machinery (should be near-absent in spleen)
    "RAG1", "RAG2", "DNTT", "VPREB1", "IGLL1",
    # T-cell help
    "CD4", "CD8A", "FOXP3", "ICOS", "IL21", "IL21R", "PDCD1", "CD40LG",
    # survival / co-stimulation received by B cells
    "CD40", "TNFSF13B", "TNFRSF13B", "TNFRSF13C", "ICOSLG", "CD80", "CD86", "CD83", "BCL2A1", "NR4A1",
    # antigen presentation
    "HLA-DRA", "HLA-DRB1", "CD74", "CIITA", "CTSS", "LAMP1",
    # follicular dendritic cells, stroma, macrophages, spleen architecture
    "CR1", "CR2", "FCER2", "FDCSP", "CXCL13", "CCL19", "CCL21", "VCAM1", "MADCAM1",
    "ITGAX", "CLEC9A", "LYZ", "CD68", "MARCO",
]

t0 = time.time()
prov = {
    "organ": "spleen",
    "census_version": CENSUS_VERSION,
    "value_filter": VALUE_FILTER,
    "genes_requested": sorted(GENES),
}

with cellxgene_census.open_soma(census_version=CENSUS_VERSION,
                                tiledb_config=TILEDB_CONFIG) as census:
    human = census["census_data"]["homo_sapiens"]

    obs_path = Path("data/spleen_obs.parquet")
    if obs_path.exists():
        # A resumable expression run should not repeatedly pull the large obs
        # enumeration metadata.  The parquet file is the exact Census slice
        # written by the first successful pass.
        obs = pd.read_parquet(obs_path)
        print(f"[{time.time()-t0:.0f}s] reused local spleen obs", flush=True)
    else:
        obs = human.obs.read(
            value_filter=VALUE_FILTER,
            column_names=[
                "soma_joinid", "cell_type", "tissue", "tissue_general", "assay",
                "disease", "sex", "development_stage", "donor_id",
                "dataset_id", "self_reported_ethnicity", "suspension_type",
            ],
        ).concat().to_pandas()
        obs.to_parquet(obs_path, index=False)
    print(f"[{time.time()-t0:.0f}s] spleen cells: {len(obs):,}", flush=True)

    prov.update(
        n_cells_total=int(len(obs)),
        n_donors=int(obs.donor_id.nunique()),
        n_datasets=int(obs.dataset_id.nunique()),
        n_cell_types=int(obs.cell_type.nunique()),
        assays=obs.assay.value_counts().to_dict(),
        diseases=obs.disease.value_counts().to_dict(),
        tissue_labels=obs.tissue.value_counts().to_dict(),
        sexes=obs.sex.value_counts().to_dict(),
    )
    normal = obs[obs.disease == "normal"]
    prov.update(
        n_cells_normal=int(len(normal)),
        n_donors_normal=int(normal.donor_id.nunique()),
        n_datasets_normal=int(normal.dataset_id.nunique()),
    )
    print(f"  normal: {len(normal):,} cells, {normal.donor_id.nunique()} donors, "
          f"{obs.cell_type.nunique()} cell types", flush=True)

    var = human.ms["RNA"].var.read(
        value_filter=f"feature_name in {GENES}",
        column_names=["soma_joinid", "feature_id", "feature_name"],
    ).concat().to_pandas()
    gene_ids = var.soma_joinid.to_numpy()
    prov["genes_resolved"] = sorted(var.feature_name.tolist())
    prov["genes_missing"] = sorted(set(GENES) - set(var.feature_name))
    print(f"  genes resolved {len(var)}/{len(GENES)}  missing={prov['genes_missing']}", flush=True)

    # Stream expression in blocks and accumulate a per-cell-type mean after
    # closure-normalising the selected 77-gene panel and applying log1p.
    joinids = obs.soma_joinid.to_numpy()
    ct_codes, ct_levels = pd.factorize(obs.cell_type)
    n_ct, n_g = len(ct_levels), len(var)
    sum_expr = np.zeros((n_ct, n_g))
    n_cells_ct = np.zeros(n_ct)
    pos_expr = np.zeros((n_ct, n_g))
    jid_to_row = pd.Series(np.arange(len(obs)), index=joinids)

    blocks = int(np.ceil(len(joinids) / BLOCK))
    failed = []
    checkpoint_path = Path("data/spleen_expression_checkpoint.npz")
    completed = set()
    if checkpoint_path.exists():
        cp = np.load(checkpoint_path, allow_pickle=False)
        if tuple(cp["shape"]) != sum_expr.shape:
            raise RuntimeError("expression checkpoint shape does not match current Census slice")
        sum_expr[:] = cp["sum_expr"]
        pos_expr[:] = cp["pos_expr"]
        n_cells_ct[:] = cp["n_cells_ct"]
        completed = set(int(x) for x in cp["completed"])
        print(f"  resumed {len(completed)}/{blocks} completed blocks", flush=True)
    group_starts = list(range(0, blocks, BLOCK_GROUP))
    if _os.environ.get("SPLEEN_CENSUS_REVERSE", "0") == "1":
        group_starts.reverse()
    for group_start in group_starts:
        pending = [b for b in range(group_start, min(blocks, group_start + BLOCK_GROUP))
                   if b not in completed]
        if not pending:
            continue
        chunk = np.concatenate([joinids[b * BLOCK:(b + 1) * BLOCK] for b in pending])
        try:
            local_sum = np.zeros_like(sum_expr)
            local_pos = np.zeros_like(pos_expr)
            local_n = np.zeros_like(n_cells_ct)
            for sub_start in range(0, len(chunk), QUERY_SUBBLOCK):
                sub = chunk[sub_start:sub_start + QUERY_SUBBLOCK]
                # Read only the sparse X slice.  Smaller queries avoid a few
                # pathological scattered-coordinate requests in the S3 array.
                table = human.ms["RNA"].X["raw"].read(
                    coords=(sub.tolist(), gene_ids.tolist())
                ).tables().concat()
                dim0 = table["soma_dim_0"].to_numpy()
                dim1 = table["soma_dim_1"].to_numpy()
                values = table["soma_data"].to_numpy()
                row_lookup = pd.Series(np.arange(len(sub)), index=sub)
                col_lookup = pd.Series(np.arange(len(gene_ids)), index=gene_ids)
                X = sparse.coo_matrix(
                    (values, (row_lookup.loc[dim0].to_numpy(),
                              col_lookup.loc[dim1].to_numpy())),
                    shape=(len(sub), len(gene_ids)),
                ).tocsr()
                tot = np.asarray(X.sum(axis=1)).ravel()
                tot[tot == 0] = 1.0
                Xn = X.multiply(1e4 / tot[:, None]).tocsr()
                Xn.data = np.log1p(Xn.data)
                dense = np.asarray(Xn.todense())
                rows = jid_to_row.loc[sub].to_numpy()
                codes = ct_codes[rows]
                for c in np.unique(codes):
                    sel = codes == c
                    local_sum[c] += dense[sel].sum(axis=0)
                    local_pos[c] += (dense[sel] > 0).sum(axis=0)
                    local_n[c] += sel.sum()
                print(f"    query {sub_start + len(sub):,}/{len(chunk):,}", flush=True)
            sum_expr += local_sum
            pos_expr += local_pos
            n_cells_ct += local_n
            completed.update(pending)
            temporary = checkpoint_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, sum_expr=sum_expr, pos_expr=pos_expr,
                                n_cells_ct=n_cells_ct,
                                completed=np.array(sorted(completed), dtype=int),
                                shape=np.array(sum_expr.shape, dtype=int))
            temporary.replace(checkpoint_path)
            print(f"  [{time.time()-t0:.0f}s] blocks {pending[0]+1}-{pending[-1]+1}/{blocks} "
                  f"({len(chunk):,} cells)", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.extend({"block": b, "error": repr(exc)[:300]} for b in pending)
            print(f"  blocks {pending[0]+1}-{pending[-1]+1} FAILED: {exc}", flush=True)

mean_expr = np.divide(sum_expr, np.maximum(n_cells_ct[:, None], 1))
frac_pos = np.divide(pos_expr, np.maximum(n_cells_ct[:, None], 1))

pd.DataFrame(mean_expr, index=ct_levels, columns=var.feature_name).to_csv("data/spleen_expression_mean.csv")
pd.DataFrame(frac_pos, index=ct_levels, columns=var.feature_name).to_csv("data/spleen_expression_fracpos.csv")
pd.DataFrame({"cell_type": ct_levels, "n_cells": n_cells_ct.astype(int)}) \
  .sort_values("n_cells", ascending=False).to_csv("data/spleen_celltype_counts.csv", index=False)

prov.update(blocks=blocks, blocks_failed=len(failed), failed=failed,
            wall_seconds=round(time.time() - t0, 1),
            normalization="selected-77-gene-panel closure to 10,000 then log1p; mean over all cells of the cell type (zeros included)")
json.dump(prov, open("data/spleen_census_provenance.json", "w"), indent=1)
if failed:
    raise RuntimeError(f"{len(failed)} expression blocks failed; rerun to resume")
print(f"\nDONE in {time.time()-t0:.0f}s — {prov['n_cells_total']:,} cells, "
      f"{prov['n_cell_types']} cell types, {prov['n_datasets']} datasets, blocks_failed={len(failed)}")
