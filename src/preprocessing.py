from __future__ import annotations

import scanpy as sc
import squidpy as sq


def preprocess_adata(adata, config):
    params = config["preprocessing"]
    adata = adata.copy()

    if adata.n_obs <= 10:
        raise ValueError(
            f"MuVi-Niche requires more than 10 observations per slice/sample; got n_obs={adata.n_obs}."
        )

    min_genes = params.get("min_genes")
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes=min_genes)

    if params.get("filter_genes", True):
        sc.pp.filter_genes(adata, min_cells=params["min_cells"])

    if params.get("normalize_total", True):
        sc.pp.normalize_total(adata, target_sum=params["target_sum"])

    if params.get("ensure_log1p", True) and not adata.uns.get("log1p", None):
        sc.pp.log1p(adata)

    if adata.n_vars > params["n_top_genes"]:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=params["n_top_genes"],
            flavor=params["hvg_flavor"],
        )
        adata = adata[:, adata.var["highly_variable"]].copy()

    if (
        params["use_existing_spatial_key"]
        and "spatial" not in adata.obsm
        and "X_spatial" in adata.obsm
    ):
        adata.obsm["spatial"] = adata.obsm["X_spatial"]

    sq.gr.spatial_neighbors(
        adata,
        coord_type="generic",
        delaunay=False,
        n_neighs=max(1, min(int(params["spatial_neighbors"]), adata.n_obs - 1)),
    )
    return adata
