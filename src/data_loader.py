import scanpy as sc


def load_adata(path: str):
    adata = sc.read_h5ad(path)
    if "spatial" not in adata.obsm and "X_spatial" in adata.obsm:
        adata.obsm["spatial"] = adata.obsm["X_spatial"].copy()
    adata.var_names_make_unique()
    return adata
