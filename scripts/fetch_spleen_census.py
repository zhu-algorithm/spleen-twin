"""Pull every human spleen cell in CELLxGENE Census and summarise it.

Census is the largest standardised aggregation of single-cell data in
existence; slicing it on ``tissue_general == 'spleen'`` gives the most
complete human spleen atlas currently assembled from public data.
"""
import json
import time

import numpy as np
import pandas as pd
import cellxgene_census

CENSUS_VERSION = "2025-11-08"

# TileDB's S3 client does not read the shell proxy variables, and this
# sandbox has no direct DNS, so the proxy has to be configured explicitly.
import os as _os
import urllib.parse as _up

_proxy = _os.environ.get("https_proxy") or _os.environ.get("HTTPS_PROXY")
TILEDB_CONFIG = {"vfs.s3.region": "us-west-2", "vfs.s3.no_sign_request": "true"}
if _proxy:
    _p = _up.urlparse(_proxy)
    TILEDB_CONFIG.update({
        "vfs.s3.proxy_scheme": _p.scheme or "http",
        "vfs.s3.proxy_host": _p.hostname or "localhost",
        "vfs.s3.proxy_port": str(_p.port or 80),
    })
VALUE_FILTER = "tissue_general == 'spleen' and is_primary_data == True"
BLOCK = 150_000

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

    obs = human.obs.read(
        value_filter=VALUE_FILTER,
        column_names=[
            "soma_joinid", "cell_type", "tissue", "tissue_general", "assay",
            "disease", "sex", "development_stage", "donor_id",
            "dataset_id", "self_reported_ethnicity", "suspension_type",
        ],
    ).concat().to_pandas()
    print(f"[{time.time()-t0:.0f}s] spleen cells: {len(obs):,}", flush=True)

    obs.to_parquet("data/spleen_obs.parquet", index=False)

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

    # Stream expression in blocks and accumulate a per-cell-type mean of
    # CP10K-normalised, log1p-transformed counts.
    joinids = obs.soma_joinid.to_numpy()
    ct_codes, ct_levels = pd.factorize(obs.cell_type)
    n_ct, n_g = len(ct_levels), len(var)
    sum_expr = np.zeros((n_ct, n_g))
    n_cells_ct = np.zeros(n_ct)
    pos_expr = np.zeros((n_ct, n_g))
    jid_to_row = pd.Series(np.arange(len(obs)), index=joinids)

    blocks = int(np.ceil(len(joinids) / BLOCK))
    failed = []
    for b in range(blocks):
        chunk = joinids[b * BLOCK:(b + 1) * BLOCK]
        try:
            ad = cellxgene_census.get_anndata(
                census, "homo_sapiens",
                obs_coords=chunk.tolist(), var_coords=gene_ids.tolist(),
                column_names={"obs": ["soma_joinid"], "var": ["feature_name"]},
            )
            X = ad.X
            tot = np.asarray(X.sum(axis=1)).ravel()
            tot[tot == 0] = 1.0
            Xn = X.multiply(1e4 / tot[:, None]).tocsr()
            Xn.data = np.log1p(Xn.data)
            dense = np.asarray(Xn.todense())
            order = [list(ad.var.feature_name).index(g) for g in var.feature_name]
            dense = dense[:, order]
            rows = jid_to_row.loc[ad.obs.soma_joinid.to_numpy()].to_numpy()
            codes = ct_codes[rows]
            for c in np.unique(codes):
                sel = codes == c
                sum_expr[c] += dense[sel].sum(axis=0)
                pos_expr[c] += (dense[sel] > 0).sum(axis=0)
                n_cells_ct[c] += sel.sum()
            print(f"  [{time.time()-t0:.0f}s] block {b+1}/{blocks} ({len(chunk):,} cells)", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append({"block": b, "error": repr(exc)[:300]})
            print(f"  block {b+1} FAILED: {exc}", flush=True)

mean_expr = np.divide(sum_expr, np.maximum(n_cells_ct[:, None], 1))
frac_pos = np.divide(pos_expr, np.maximum(n_cells_ct[:, None], 1))

pd.DataFrame(mean_expr, index=ct_levels, columns=var.feature_name).to_csv("data/spleen_expression_mean.csv")
pd.DataFrame(frac_pos, index=ct_levels, columns=var.feature_name).to_csv("data/spleen_expression_fracpos.csv")
pd.DataFrame({"cell_type": ct_levels, "n_cells": n_cells_ct.astype(int)}) \
  .sort_values("n_cells", ascending=False).to_csv("data/spleen_celltype_counts.csv", index=False)

prov.update(blocks=blocks, blocks_failed=len(failed), failed=failed,
            wall_seconds=round(time.time() - t0, 1),
            normalization="CP10K then log1p; mean over all cells of the cell type (zeros included)")
json.dump(prov, open("data/spleen_census_provenance.json", "w"), indent=1)
print(f"\nDONE in {time.time()-t0:.0f}s — {prov['n_cells_total']:,} cells, "
      f"{prov['n_cell_types']} cell types, {prov['n_datasets']} datasets, blocks_failed={len(failed)}")
