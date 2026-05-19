from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from scipy import sparse
from sklearn.cluster import KMeans


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_numpy(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    if hasattr(matrix, "to_numpy"):
        return matrix.to_numpy()
    return np.asarray(matrix)


def to_tensor(matrix, device):
    return torch.tensor(to_numpy(matrix), dtype=torch.float32, device=device)


def sanitize_obsm_for_h5ad(adata) -> None:
    for key in list(adata.obsm.keys()):
        value = adata.obsm[key]
        if isinstance(value, pd.DataFrame):
            adata.uns[f"{key}_columns"] = [str(col) for col in value.columns]
            adata.obsm[key] = value.to_numpy()


def save_final_adata(adata, output_dir: Path, final_name: str) -> None:
    sanitize_obsm_for_h5ad(adata)
    adata.write_h5ad(output_dir / final_name)


def save_graphs(graphs_best, output_dir: Path, config: dict) -> None:
    torch.save(graphs_best, output_dir / config["outputs"]["best_graphs_name"])


def now_seconds() -> float:
    return time.perf_counter()


def get_cpu_memory_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def run_kmeans_on_embedding(adata, config) -> None:
    training_cfg = config["training"]
    embedding_key = training_cfg["embedding_key"]
    cluster_key = training_cfg["cluster_key"]
    label_key = config["data"]["label_key"]

    embedding = np.asarray(adata.obsm[embedding_key])
    if training_cfg["kmeans_clusters"] is not None:
        n_clusters = int(training_cfg["kmeans_clusters"])
    elif label_key in adata.obs:
        n_clusters = adata.obs[label_key].nunique()
    else:
        n_clusters = 10

    n_clusters = max(1, min(int(n_clusters), int(embedding.shape[0])))

    adata.obs[cluster_key] = KMeans(
        n_clusters=n_clusters,
        random_state=config["seed"],
    ).fit_predict(embedding).astype(str)


def assign_niche_prototypes(adata, config) -> None:
    prototype_cfg = config.get("prototype_learning", {})
    if not prototype_cfg.get("enabled", True):
        return
    training_cfg = config["training"]
    embedding_key = training_cfg["embedding_key"]
    label_key = config["data"]["label_key"]
    obs_key = prototype_cfg.get("obs_key", "common_niche")
    prob_key = prototype_cfg.get("prob_key", "X_common_niche_prob")
    distance_key = prototype_cfg.get("distance_key", "X_common_niche_distance")
    temperature = float(prototype_cfg.get("temperature", 0.2))

    embedding = np.asarray(adata.obsm[embedding_key])
    if prototype_cfg.get("num_prototypes") is not None:
        n_prototypes = int(prototype_cfg["num_prototypes"])
    elif training_cfg.get("kmeans_clusters") is not None:
        n_prototypes = int(training_cfg["kmeans_clusters"])
    elif label_key in adata.obs:
        n_prototypes = int(adata.obs[label_key].nunique())
    else:
        n_prototypes = 10

    n_prototypes = max(1, min(int(n_prototypes), int(embedding.shape[0])))

    model = KMeans(
        n_clusters=n_prototypes,
        random_state=config["seed"],
    ).fit(embedding)
    distances = model.transform(embedding)
    logits = -distances / max(temperature, 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)

    adata.obs[obs_key] = model.labels_.astype(str)
    adata.obsm[prob_key] = probs
    adata.obsm[distance_key] = distances


def fit_prototype_memory(adatas: list, config) -> dict:
    prototype_cfg = config.get("prototype_learning", {})
    training_cfg = config["training"]
    embedding_key = training_cfg["embedding_key"]
    label_key = config["data"]["label_key"]
    temperature = float(prototype_cfg.get("temperature", 0.2))

    embeddings = [np.asarray(adata.obsm[embedding_key]) for adata in adatas]
    combined_embedding = np.vstack(embeddings)
    if prototype_cfg.get("num_prototypes") is not None:
        n_prototypes = int(prototype_cfg["num_prototypes"])
    elif training_cfg.get("kmeans_clusters") is not None:
        n_prototypes = int(training_cfg["kmeans_clusters"])
    elif all(label_key in adata.obs for adata in adatas):
        labels = pd.concat([adata.obs[label_key] for adata in adatas], axis=0)
        n_prototypes = int(labels.nunique())
    else:
        n_prototypes = 10

    n_prototypes = max(1, min(int(n_prototypes), int(combined_embedding.shape[0])))

    model = KMeans(
        n_clusters=n_prototypes,
        random_state=config["seed"],
    ).fit(combined_embedding)
    return {
        "centers": model.cluster_centers_,
        "temperature": temperature,
    }


def apply_prototype_memory(adata, config, memory: dict) -> None:
    prototype_cfg = config.get("prototype_learning", {})
    if not prototype_cfg.get("enabled", True):
        return
    embedding = np.asarray(adata.obsm[config["training"]["embedding_key"]])
    centers = np.asarray(memory["centers"])
    temperature = float(memory.get("temperature", prototype_cfg.get("temperature", 0.2)))
    distances = np.linalg.norm(embedding[:, None, :] - centers[None, :, :], axis=2)
    logits = -distances / max(temperature, 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    adata.obs[prototype_cfg.get("obs_key", "common_niche")] = np.argmax(probs, axis=1).astype(str)
    adata.obsm[prototype_cfg.get("prob_key", "X_common_niche_prob")] = probs
    adata.obsm[prototype_cfg.get("distance_key", "X_common_niche_distance")] = distances


def build_profile(runtime_seconds: float, cpu_peak_mb: float) -> dict:
    return {
        "runtime_seconds": float(runtime_seconds),
        "cpu_rss_peak_mb": float(cpu_peak_mb),
    }
