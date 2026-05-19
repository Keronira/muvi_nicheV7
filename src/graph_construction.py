from __future__ import annotations

import math

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import from_scipy_sparse_matrix


def _as_dense(matrix) -> np.ndarray:
    if issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _scaled_matrix(matrix) -> np.ndarray:
    arr = _as_dense(matrix).astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={arr.shape}.")
    return StandardScaler(with_mean=True, with_std=True).fit_transform(arr).astype(np.float32)


def _pca_or_pad(matrix: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    matrix = _scaled_matrix(matrix)
    n_components = int(n_components)
    usable = min(n_components, matrix.shape[0], matrix.shape[1])
    if usable <= 0:
        raise ValueError("Cannot build X_self with zero PCA components.")
    if matrix.shape[1] > usable:
        out = PCA(n_components=usable, random_state=seed).fit_transform(matrix)
    else:
        out = matrix[:, :usable]
    if usable < n_components:
        pad = np.zeros((out.shape[0], n_components - usable), dtype=out.dtype)
        out = np.hstack([out, pad])
    return out.astype(np.float32)


def _spatial_knn(spatial: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n_obs = spatial.shape[0]
    k = max(1, min(int(k), n_obs - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(spatial)
    dist, idx = nn.kneighbors(spatial)
    return idx[:, 1:].astype(np.int64), dist[:, 1:].astype(np.float32)


def _feature_knn_graph(features: np.ndarray, rareq_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_obs = features.shape[0]
    k = max(1, min(int(rareq_cfg.get("feature_knn_k", 20)), max(n_obs - 1, 1)))
    metric = str(rareq_cfg.get("metric", "cosine"))
    x = _scaled_matrix(features)
    if n_obs <= 1:
        index = np.zeros((n_obs, 1), dtype=np.int64)
        mask = np.zeros((n_obs, 1), dtype=np.float32)
        edge_attr = np.zeros((n_obs, 1, 2), dtype=np.float32)
        return index, mask, edge_attr

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(x)
    dist, idx = nn.kneighbors(x)
    idx = idx[:, 1:].astype(np.int64)
    dist = dist[:, 1:].astype(np.float32)
    if metric == "cosine":
        sim = np.clip(1.0 - dist, -1.0, 1.0)
        sim = ((sim + 1.0) * 0.5).astype(np.float32)
    else:
        scale = np.quantile(dist[dist > 0], 0.9) if np.any(dist > 0) else 1.0
        sim = np.exp(-dist / max(float(scale), 1e-6)).astype(np.float32)
    rank = np.linspace(0.0, 1.0, idx.shape[1], dtype=np.float32)
    rank = np.broadcast_to(rank.reshape(1, -1), idx.shape).astype(np.float32)
    edge_attr = np.stack([sim, rank], axis=2).astype(np.float32)
    mask = np.ones(idx.shape, dtype=np.float32)
    return idx, mask, edge_attr


def _radius_thresholds(spatial: np.ndarray, scales: list[int], graph_cfg: dict) -> dict[int, float]:
    _, nearest_dist = _spatial_knn(spatial, 1)
    positive = nearest_dist[:, 0][nearest_dist[:, 0] > 0]
    if positive.size == 0:
        base_spacing = 1.0
    else:
        base_spacing = float(np.quantile(positive, float(graph_cfg.get("radius_quantile", 0.75))))
    multipliers = graph_cfg.get("radius_multipliers", {}) or {}
    thresholds = {}
    for scale in scales:
        multiplier = multipliers.get(scale, multipliers.get(str(scale), None))
        if multiplier is None:
            multiplier = max(2.0, math.sqrt(float(scale)) * 1.35)
        thresholds[int(scale)] = float(base_spacing) * float(multiplier)
    return thresholds


def _neighbor_mean_radius(
    features: np.ndarray,
    indices: np.ndarray,
    distances: np.ndarray,
    radius: float,
    min_valid_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = distances <= float(radius)
    valid_count = valid.sum(axis=1).astype(np.float32)
    density = np.clip(valid_count / max(indices.shape[1], 1), 0.0, 1.0).astype(np.float32)
    out = np.zeros((indices.shape[0], features.shape[1]), dtype=np.float32)
    for i in range(indices.shape[0]):
        if valid_count[i] >= int(min_valid_neighbors):
            out[i] = features[indices[i, valid[i]]].mean(axis=0)
        else:
            out[i] = features[i]
    return out.astype(np.float32), valid.astype(np.float32), density


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return np.sum(a * b, axis=1) / denom


def _build_spatial_connectivities(
    indices: np.ndarray,
    distances: np.ndarray,
    features: np.ndarray,
    temperature: float,
    min_weight: float,
    max_radius: float | None = None,
) -> sparse.csr_matrix:
    n_obs = indices.shape[0]
    valid = np.ones_like(distances, dtype=bool) if max_radius is None else distances <= float(max_radius)
    rows = np.repeat(np.arange(n_obs), indices.shape[1])[valid.reshape(-1)]
    cols = indices.reshape(-1)[valid.reshape(-1)]
    dist_flat = distances.reshape(-1)[valid.reshape(-1)]
    if rows.size == 0:
        return sparse.csr_matrix((n_obs, n_obs), dtype=np.float32)
    src_feat = features[rows]
    dst_feat = features[cols]
    sim = (_cosine_rows(src_feat, dst_feat) + 1.0) * 0.5
    dist_scale = np.quantile(dist_flat[dist_flat > 0], 0.9) if np.any(dist_flat > 0) else 1.0
    spatial_w = np.exp(-dist_flat / max(float(dist_scale), 1e-6))
    weights = np.maximum(float(min_weight), spatial_w * (0.5 + 0.5 * sim / max(float(temperature), 1e-6)))
    adj = sparse.coo_matrix((weights.astype(np.float32), (rows, cols)), shape=(n_obs, n_obs)).tocsr()
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def _direction_bucket(dx: np.ndarray, dy: np.ndarray, num_directions: int) -> np.ndarray:
    angles = np.arctan2(dy, dx)
    shifted = (angles + 2.0 * math.pi) % (2.0 * math.pi)
    bucket_width = 2.0 * math.pi / int(num_directions)
    return np.floor((shifted + bucket_width / 2.0) / bucket_width).astype(np.int64) % int(num_directions)


def _directional_index(
    spatial: np.ndarray,
    features: np.ndarray,
    boundary_strength: np.ndarray,
    k: int,
    num_directions: int,
    neighbors_per_direction: int,
    radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx, dist = _spatial_knn(spatial, k)
    n_obs = spatial.shape[0]
    m = int(neighbors_per_direction)
    direction_index = np.zeros((n_obs, int(num_directions), m), dtype=np.int64)
    direction_mask = np.zeros((n_obs, int(num_directions), m), dtype=np.float32)
    edge_attr = np.zeros((n_obs, int(num_directions), m, 5), dtype=np.float32)
    global_dist_scale = np.quantile(dist[dist > 0], 0.9) if np.any(dist > 0) else 1.0

    for i in range(n_obs):
        nbr = idx[i]
        valid_radius = np.ones_like(dist[i], dtype=bool) if radius is None else dist[i] <= float(radius)
        delta = spatial[nbr] - spatial[i]
        buckets = _direction_bucket(delta[:, 0], delta[:, 1], num_directions)
        sim = (_cosine_rows(np.repeat(features[i][None, :], len(nbr), axis=0), features[nbr]) + 1.0) * 0.5
        boundary_pair = 0.5 * (boundary_strength[i] + boundary_strength[nbr])
        for d in range(int(num_directions)):
            candidates = np.where((buckets == d) & valid_radius)[0]
            if candidates.size == 0:
                continue
            order = candidates[np.argsort(dist[i, candidates])[:m]]
            for pos, cand in enumerate(order):
                j = int(nbr[cand])
                direction_index[i, d, pos] = j
                direction_mask[i, d, pos] = 1.0
                raw_angle = math.atan2(float(delta[cand, 1]), float(delta[cand, 0]))
                edge_attr[i, d, pos] = np.array(
                    [
                        float(dist[i, cand] / max(global_dist_scale, 1e-6)),
                        math.sin(raw_angle),
                        math.cos(raw_angle),
                        float(sim[cand]),
                        float(boundary_pair[cand]),
                    ],
                    dtype=np.float32,
                )
    return direction_index, direction_mask, edge_attr


def _boundary_features(features: np.ndarray, indices: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nbr_feat = features[indices]
    center = features[:, None, :]
    diff = nbr_feat - center
    local_var = diff.var(axis=1).mean(axis=1)
    local_shift = np.linalg.norm(diff.mean(axis=1), axis=1)
    grad = (np.linalg.norm(diff, axis=2) / (distances + 1e-6)).mean(axis=1)
    raw = np.vstack([local_var, local_shift, grad]).T.astype(np.float32)
    scaled = _scaled_matrix(raw)
    strength = raw.mean(axis=1)
    lo, hi = np.quantile(strength, [0.05, 0.95])
    norm_strength = np.clip((strength - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    return scaled.astype(np.float32), norm_strength


def build_graph_inputs(adata, config):
    graph_cfg = config["graph_construction"]
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError("muvi_niche_v7 requires adata.obsm['spatial'] with at least two columns.")

    scales = [int(k) for k in graph_cfg.get("scales", [8, 15, 30])]
    max_k = max(max(scales), int(graph_cfg.get("spatial_connectivity_k", max(scales))))
    radius_by_scale = _radius_thresholds(spatial, scales, graph_cfg)
    x_self = _pca_or_pad(adata.X, int(graph_cfg.get("self_pca_dim", 80)), int(config.get("seed", 1234)))
    base_idx, base_dist = _spatial_knn(spatial, max_k)
    x_boundary, boundary_strength = _boundary_features(x_self, base_idx, base_dist)

    adata.obsm["X_self"] = x_self
    adata.obsm["X_boundary"] = x_boundary
    adata.obs["X_v6_boundary_strength"] = boundary_strength

    rareq_cfg = config.get("model", {}).get("rareq", {})
    if rareq_cfg.get("enabled", False):
        r_index, r_mask, r_attr = _feature_knn_graph(x_self, rareq_cfg)
        adata.obsm["rareq_neighbor_index"] = r_index
        adata.obsm["rareq_neighbor_mask"] = r_mask
        adata.obsm["rareq_edge_attr"] = r_attr
        adata.uns["v7_rareq_graph"] = {
            "feature_key": str(rareq_cfg.get("input_key", "X_self")),
            "metric": str(rareq_cfg.get("metric", "cosine")),
            "feature_knn_k": int(rareq_cfg.get("feature_knn_k", 20)),
            "q_neighbors": int(rareq_cfg.get("q_neighbors", 6)),
        }

    for scale in scales:
        idx = base_idx[:, : min(scale, base_idx.shape[1])]
        dist = base_dist[:, : min(scale, base_dist.shape[1])]
        radius = radius_by_scale[int(scale)]
        x_nbr, valid_mask, density = _neighbor_mean_radius(
            x_self,
            idx,
            dist,
            radius=radius,
            min_valid_neighbors=int(graph_cfg.get("min_valid_neighbors_for_pool", 1)),
        )
        adata.obsm[f"X_nbr_k{scale}"] = x_nbr
        adata.obsm[f"spatial_valid_mask_k{scale}"] = valid_mask
        adata.obs[f"X_v6_density_k{scale}"] = density
        d_index, d_mask, d_attr = _directional_index(
            spatial=spatial,
            features=x_self,
            boundary_strength=boundary_strength,
            k=scale,
            num_directions=int(graph_cfg.get("num_directions", 8)),
            neighbors_per_direction=int(graph_cfg.get("neighbors_per_direction", 4)),
            radius=radius,
        )
        adata.obsm[f"direction_index_k{scale}"] = d_index
        adata.obsm[f"direction_mask_k{scale}"] = d_mask
        adata.obsm[f"direction_edge_attr_k{scale}"] = d_attr

    adj = _build_spatial_connectivities(
        indices=base_idx,
        distances=base_dist,
        features=x_self,
        temperature=float(graph_cfg.get("edge_similarity_temperature", 0.25)),
        min_weight=float(graph_cfg.get("min_edge_weight", 0.05)),
        max_radius=max(radius_by_scale.values()) if radius_by_scale else None,
    )
    adata.obsp["spatial_connectivities"] = adj
    adata.obsp["spatial_edge_weighted_connectivities"] = adj
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    edge_index = edge_index.long()
    edge_weight = edge_weight.float()
    adata.uns["v6_graph_construction"] = {
        "scales": scales,
        "num_directions": int(graph_cfg.get("num_directions", 8)),
        "neighbors_per_direction": int(graph_cfg.get("neighbors_per_direction", 4)),
        "self_pca_dim": int(graph_cfg.get("self_pca_dim", 80)),
        "radius_by_scale": {str(key): float(value) for key, value in radius_by_scale.items()},
        "edges": int(adj.nnz),
    }
    graphs_best = {
        "spatial": edge_index,
        "spatial_weight": edge_weight,
    }
    print(
        f"[muvi_niche_v7] graph inputs built: n_obs={adata.n_obs}, "
        f"self_dim={x_self.shape[1]}, scales={scales}, rareq={bool(rareq_cfg.get('enabled', False))}, "
        f"radii={{{', '.join(f'k{k}:{v:.3g}' for k, v in radius_by_scale.items())}}}, edges={adj.nnz}"
    )
    return adata, graphs_best
