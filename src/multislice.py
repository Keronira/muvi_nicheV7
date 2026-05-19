from __future__ import annotations

from typing import Iterable


def detect_training_mode(adatas: Iterable, config: dict) -> str:
    mode_cfg = config.get("training_mode", {})
    requested = str(mode_cfg.get("mode", "single_slice")).lower()
    if requested in {"single_slice", "multi_slice"}:
        return requested
    return "single_slice"


def split_adata_by_obs_key(adata, key: str) -> dict[str, object]:
    if key not in adata.obs:
        raise KeyError(f"multi_slice mode requires adata.obs[{key!r}] when splitting a single h5ad.")
    groups = {}
    for value in adata.obs[key].astype(str).dropna().unique().tolist():
        groups[str(value)] = adata[adata.obs[key].astype(str) == str(value)].copy()
    if not groups:
        raise ValueError(f"No slices found in adata.obs[{key!r}].")
    return groups
