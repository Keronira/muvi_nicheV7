from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.models import DirectionalMultiScaleLPEncoder
from src.utils import to_numpy


def _save_checkpoint(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _plot_training_curve(loss_history, lr_reductions, output_dir: Path, config: dict) -> None:
    if not config.get("plotting", {}).get("plot_loss_curve", True) or not loss_history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [item["epoch"] for item in loss_history]
    axes[0].plot(epochs, [item["loss"] for item in loss_history], label="Total", color="#2f5f9f")
    axes[0].plot(epochs, [item["best_loss"] for item in loss_history], label="Best", color="#888888", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)
    components = [
        ("reconstruction_loss", "Self recon"),
        ("directional_continuity_loss", "Directional"),
        ("boundary_preservation_loss", "Boundary"),
        ("distance_preservation_loss", "Distance"),
        ("small_direction_continuity_loss", "Small dir"),
    ]
    for key, label in components:
        axes[1].plot(epochs, [item[key] for item in loss_history], label=label)
    for event in lr_reductions:
        axes[0].axvline(event["epoch"], color="#cc6677", alpha=0.25)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Component loss")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path = output_dir / config.get("outputs", {}).get("loss_curve_name", "loss_epoch_curve.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[muvi_niche_v7] loss curve saved: {out_path}")


def _tensor_from_obsm(adata, key: str, device: torch.device, dtype=torch.float32):
    arr = np.asarray(adata.obsm[key])
    return torch.tensor(arr, dtype=dtype, device=device)


def _build_batch(adata, config: dict, device: torch.device) -> dict[str, object]:
    scales = [int(scale) for scale in config["model"].get("scales", [8, 15, 30])]
    batch = {
        "X_self": _tensor_from_obsm(adata, "X_self", device),
        "X_boundary": _tensor_from_obsm(adata, "X_boundary", device),
    }
    rareq_cfg = config.get("model", {}).get("rareq", {})
    if rareq_cfg.get("enabled", False):
        for key, dtype in [
            ("rareq_neighbor_index", torch.long),
            ("rareq_neighbor_mask", torch.float32),
            ("rareq_edge_attr", torch.float32),
        ]:
            if key not in adata.obsm:
                raise KeyError(f"Missing adata.obsm['{key}'] required by v7 learnable RareQ.")
            batch[key] = _tensor_from_obsm(adata, key, device, dtype=dtype)
    for scale in scales:
        batch[f"X_nbr_k{scale}"] = _tensor_from_obsm(adata, f"X_nbr_k{scale}", device)
        density_key = f"X_v6_density_k{scale}"
        if density_key in adata.obs:
            batch[f"density_k{scale}"] = torch.tensor(
                adata.obs[density_key].to_numpy(dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
        else:
            batch[f"density_k{scale}"] = torch.ones(batch["X_self"].shape[0], dtype=torch.float32, device=device)
        batch[f"direction_index_k{scale}"] = _tensor_from_obsm(
            adata, f"direction_index_k{scale}", device, dtype=torch.long
        )
        batch[f"direction_mask_k{scale}"] = _tensor_from_obsm(adata, f"direction_mask_k{scale}", device)
        batch[f"direction_edge_attr_k{scale}"] = _tensor_from_obsm(adata, f"direction_edge_attr_k{scale}", device)
    return batch


def _directional_pairs(
    batch: dict[str, object],
    scales: list[int],
    use_density: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src_all = []
    dst_all = []
    weight_all = []
    n_obs = batch["X_self"].shape[0]
    device = batch["X_self"].device
    for scale in scales:
        index = batch[f"direction_index_k{scale}"].reshape(n_obs, -1)
        mask = batch[f"direction_mask_k{scale}"].reshape(n_obs, -1)
        attr = batch[f"direction_edge_attr_k{scale}"].reshape(n_obs, -1, 5)
        src = torch.arange(n_obs, device=device).unsqueeze(1).expand_as(index).reshape(-1)
        dst = index.reshape(-1)
        valid = mask.reshape(-1) > 0
        sim = attr[..., 3].reshape(-1)
        boundary = attr[..., 4].reshape(-1)
        density = batch.get(f"density_k{scale}")
        if use_density and density is not None:
            edge_density = density[src] * density[dst]
        else:
            edge_density = 1.0
        weight = (sim * (1.0 - boundary) * edge_density).clamp_min(0.0)
        src_all.append(src[valid])
        dst_all.append(dst[valid])
        weight_all.append(weight[valid])
    if not src_all:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, torch.empty(0, dtype=torch.float32, device=device)
    return torch.cat(src_all), torch.cat(dst_all), torch.cat(weight_all)


def _weighted_pair_distance(z, src, dst, weight):
    if src.numel() == 0:
        return z.new_tensor(0.0)
    dist = (z[src] - z[dst]).pow(2).sum(dim=1)
    return (dist * weight).sum() / weight.sum().clamp_min(1e-6)


def _boundary_preservation_loss(z, batch, scales: list[int], threshold: float):
    src_all = []
    dst_all = []
    weight_all = []
    n_obs = batch["X_self"].shape[0]
    device = batch["X_self"].device
    for scale in scales:
        index = batch[f"direction_index_k{scale}"].reshape(n_obs, -1)
        mask = batch[f"direction_mask_k{scale}"].reshape(n_obs, -1)
        attr = batch[f"direction_edge_attr_k{scale}"].reshape(n_obs, -1, 5)
        src = torch.arange(n_obs, device=device).unsqueeze(1).expand_as(index).reshape(-1)
        valid = mask.reshape(-1) > 0
        sim = attr[..., 3].reshape(-1)
        boundary = attr[..., 4].reshape(-1)
        hard = valid & (boundary >= threshold) & (sim <= 0.55)
        if hard.any():
            src_all.append(src[hard])
            dst_all.append(index.reshape(-1)[hard])
            weight_all.append((boundary[hard] * (1.0 - sim[hard])).clamp_min(0.0))
    if not src_all:
        return z.new_tensor(0.0)
    src = torch.cat(src_all)
    dst = torch.cat(dst_all)
    weight = torch.cat(weight_all)
    dist = (z[src] - z[dst]).pow(2).sum(dim=1)
    margin = 1.0
    return (F.relu(margin - dist) * weight).sum() / weight.sum().clamp_min(1e-6)


def _normalize_vector(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    mean = values.mean()
    std = values.std().clamp_min(1e-6)
    return (values - mean) / std


def _boundary_aware_distance_preservation_loss(outputs, batch, scales: list[int], use_density: bool):
    src, dst, weight = _directional_pairs(batch, scales, use_density=use_density)
    if src.numel() == 0:
        return outputs["z_final"].new_tensor(0.0)
    z = outputs["z_final"]
    x_self = batch["X_self"]
    embed_dist = (z[src] - z[dst]).pow(2).sum(dim=1).sqrt()
    expr_dist = (x_self[src] - x_self[dst]).pow(2).sum(dim=1).sqrt()
    keep = weight > 0
    if not keep.any():
        return z.new_tensor(0.0)
    embed_norm = _normalize_vector(embed_dist[keep])
    expr_norm = _normalize_vector(expr_dist[keep])
    w = weight[keep].clamp_min(0.0)
    return ((embed_norm - expr_norm).pow(2) * w).sum() / w.sum().clamp_min(1e-6)


def _small_direction_continuity_loss(outputs, batch, scales: list[int], use_density: bool):
    if "small_gate" not in outputs:
        return outputs["z_final"].new_tensor(0.0)
    src, dst, weight = _directional_pairs(batch, scales, use_density=use_density)
    if src.numel() == 0:
        return outputs["z_final"].new_tensor(0.0)
    small_gate = outputs["small_gate"]
    small_weight = weight * small_gate[src] * small_gate[dst]
    keep = small_weight > 0
    if not keep.any():
        return outputs["z_final"].new_tensor(0.0)
    return _weighted_pair_distance(outputs["z_final"], src[keep], dst[keep], small_weight[keep])


def _rare_consistency_loss(outputs, batch, config):
    if "rare_score" not in outputs:
        return outputs["z_final"].new_tensor(0.0)
    z = outputs["z_final"]
    score = outputs["rare_score"]
    index = batch["rareq_neighbor_index"]
    mask = batch["rareq_neighbor_mask"]
    edge_attr = batch["rareq_edge_attr"]
    q_neighbors = int(config.get("model", {}).get("rareq", {}).get("q_neighbors", 6))
    q_neighbors = max(1, min(q_neighbors, index.shape[1]))
    index = index[:, :q_neighbors]
    mask = mask[:, :q_neighbors]
    sim = edge_attr[:, :q_neighbors, 0].clamp_min(0.0)
    n_obs = z.shape[0]
    src = torch.arange(n_obs, device=z.device).unsqueeze(1).expand_as(index)
    pair_weight = score[src] * score[index] * sim * mask
    valid = pair_weight.reshape(-1) > 0
    if not valid.any():
        return z.new_tensor(0.0)
    src_flat = src.reshape(-1)[valid]
    dst_flat = index.reshape(-1)[valid]
    weight = pair_weight.reshape(-1)[valid]
    return _weighted_pair_distance(z, src_flat, dst_flat, weight)


def compute_v7_loss(outputs, batch, config):
    training_cfg = config["training"]
    scales = [int(scale) for scale in config["model"].get("scales", [8, 15, 30])]
    use_density = bool(training_cfg.get("use_density_weighting", True))
    mask_nodes = outputs["mask_nodes"]
    if mask_nodes is not None and len(mask_nodes) > 0:
        recon_loss = F.mse_loss(outputs["x_self_recon"][mask_nodes], batch["X_self"][mask_nodes])
    else:
        recon_loss = F.mse_loss(outputs["x_self_recon"], batch["X_self"])

    src, dst, weight = _directional_pairs(batch, scales, use_density=use_density)
    pos_threshold = float(training_cfg.get("similarity_positive_threshold", 0.55))
    keep = weight >= pos_threshold
    directional_loss = _weighted_pair_distance(outputs["z_final"], src[keep], dst[keep], weight[keep])
    boundary_loss = _boundary_preservation_loss(
        outputs["z_final"],
        batch,
        scales,
        threshold=float(training_cfg.get("boundary_negative_threshold", 0.45)),
    )
    rare_loss = _rare_consistency_loss(outputs, batch, config)
    distance_loss = _boundary_aware_distance_preservation_loss(outputs, batch, scales, use_density=use_density)
    small_dir_loss = _small_direction_continuity_loss(outputs, batch, scales, use_density=use_density)
    total = (
        float(training_cfg.get("reconstruction_weight", 1.0)) * recon_loss
        + float(training_cfg.get("directional_continuity_weight", 0.3)) * directional_loss
        + float(training_cfg.get("boundary_preservation_weight", 0.3)) * boundary_loss
        + float(training_cfg.get("rare_consistency_weight", 0.0)) * rare_loss
        + float(training_cfg.get("distance_preservation_weight", 0.25)) * distance_loss
        + float(training_cfg.get("small_direction_continuity_weight", 0.20)) * small_dir_loss
    )
    return total, recon_loss, directional_loss, boundary_loss, rare_loss, distance_loss, small_dir_loss


def run_training_pipeline(adata, graphs_best, config, output_dir):
    print("[muvi_niche_v7] run_training_pipeline started")
    model_cfg = config["model"]
    training_cfg = config["training"]
    architecture = str(model_cfg.get("architecture", "directional_lp_learnable_rareq")).lower()
    if architecture != "directional_lp_learnable_rareq":
        raise ValueError("muvi_niche_v7 supports only model.architecture='directional_lp_learnable_rareq'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = _build_batch(adata, config, device)
    scales = [int(scale) for scale in model_cfg.get("scales", [8, 15, 30])]
    model = DirectionalMultiScaleLPEncoder(
        self_dim=batch["X_self"].shape[1],
        boundary_dim=batch["X_boundary"].shape[1],
        scales=scales,
        hidden_dim=int(model_cfg.get("hidden_dim", 96)),
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_directions=int(model_cfg.get("num_directions", 8)),
        edge_attr_dim=int(model_cfg.get("edge_attr_dim", 5)),
        dropout=float(model_cfg.get("dropout", 0.08)),
        rareq=model_cfg.get("rareq", {}),
        small_gate=model_cfg.get("small_gate", {}),
    ).to(device)
    print(
        f"[muvi_niche_v7] model initialized: encoder={architecture}, "
        f"self_dim={batch['X_self'].shape[1]}, hidden={model_cfg.get('hidden_dim', 96)}, device={device}"
    )

    output_dir = Path(output_dir)
    checkpoint_path = output_dir / config["outputs"]["checkpoint_name"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("lr", 0.001)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0001)),
    )
    scheduler_cfg = training_cfg.get("lr_scheduler", {})
    scheduler = None
    if scheduler_cfg.get("enabled", True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 12)),
            threshold=float(scheduler_cfg.get("threshold", 0.0001)),
            min_lr=float(scheduler_cfg.get("min_lr", 0.00005)),
        )

    best_loss = float("inf")
    min_delta = float(training_cfg.get("early_stopping_min_delta", 0.0001))
    patience = int(training_cfg.get("early_stopping_patience", 24))
    no_improve = 0
    print_interval = int(training_cfg.get("print_interval", 10))
    loss_history = []
    lr_reductions = []

    print(f"[muvi_niche_v7] training started: epochs={training_cfg['epochs']}")
    for epoch in range(int(training_cfg["epochs"])):
        model.train()
        optimizer.zero_grad()
        outputs = model(batch, mask_rate=float(training_cfg.get("mask_rate", 0.2)))
        (
            loss,
            loss_recon,
            loss_dir,
            loss_boundary,
            loss_rare,
            loss_distance,
            loss_small_dir,
        ) = compute_v7_loss(outputs, batch, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("grad_clip", 5.0)))
        optimizer.step()

        current_loss = float(loss.item())
        prev_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(current_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None and current_lr < prev_lr:
            lr_reductions.append({"epoch": epoch + 1, "old_lr": float(prev_lr), "new_lr": float(current_lr)})

        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            no_improve = 0
            _save_checkpoint(model, checkpoint_path)
        else:
            no_improve += 1

        loss_history.append(
            {
                "epoch": epoch + 1,
                "loss": current_loss,
                "best_loss": float(best_loss),
                "reconstruction_loss": float(loss_recon.item()),
                "directional_continuity_loss": float(loss_dir.item()),
                "boundary_preservation_loss": float(loss_boundary.item()),
                "rare_consistency_loss": float(loss_rare.item()),
                "distance_preservation_loss": float(loss_distance.item()),
                "small_direction_continuity_loss": float(loss_small_dir.item()),
                "lr": float(current_lr),
            }
        )
        if (epoch + 1 == 1) or ((epoch + 1) % print_interval == 0):
            print(
                f"[muvi_niche_v7] epoch {epoch + 1}/{training_cfg['epochs']}: "
                f"loss={current_loss:.6f}, best={best_loss:.6f}, "
                f"lr={current_lr:.6g}, no_improve={no_improve}/{patience}"
            )
        if no_improve >= patience:
            print(f"[muvi_niche_v7] early stopping at epoch {epoch + 1}: best_loss={best_loss:.6f}")
            break

    _plot_training_curve(loss_history, lr_reductions, output_dir, config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    with torch.no_grad():
        outputs = model(batch, mask_rate=0.0)
    adata.obsm[training_cfg["embedding_key"]] = outputs["z_final"].detach().cpu().numpy()
    adata.obsm["X_v7_global"] = outputs["z_global"].detach().cpu().numpy()
    adata.obsm["X_v7_local"] = outputs["z_local"].detach().cpu().numpy()
    adata.obsm["X_v7_view_weights"] = outputs["view_weights"].detach().cpu().numpy()
    direction_weights = outputs["direction_weights"].detach().cpu().numpy()
    adata.obsm["X_v7_direction_weights"] = direction_weights.reshape(direction_weights.shape[0], -1)
    scale_weights = outputs["scale_weights"].detach().cpu().numpy()
    adata.obsm["X_v7_scale_weights"] = scale_weights.reshape(scale_weights.shape[0], -1)
    adata.obsm["X_v7_boundary_gate"] = outputs["boundary_gate"].detach().cpu().numpy()
    adata.obsm["X_v7_small_gate"] = outputs["small_gate"].detach().cpu().numpy().reshape(-1, 1)
    adata.obs["X_v7_small_gate"] = outputs["small_gate"].detach().cpu().numpy()
    if "rare_score" in outputs:
        adata.obsm["X_v7_rare_score"] = outputs["rare_score"].detach().cpu().numpy().reshape(-1, 1)
        adata.obs["X_v7_rare_score"] = outputs["rare_score"].detach().cpu().numpy()
        adata.obsm["X_v7_rare_token_weight"] = outputs["rare_token_weight"].detach().cpu().numpy().reshape(-1, 1)
        adata.obsm["X_v7_rare_attention_entropy"] = (
            outputs["rare_attention_entropy"].detach().cpu().numpy().reshape(-1, 1)
        )
    adata.obsm["X_v6_global"] = adata.obsm["X_v7_global"]
    adata.obsm["X_v6_local"] = adata.obsm["X_v7_local"]
    adata.obsm["X_v6_view_weights"] = adata.obsm["X_v7_view_weights"]
    adata.obsm["X_v6_direction_weights"] = adata.obsm["X_v7_direction_weights"]
    adata.obsm["X_v6_scale_weights"] = adata.obsm["X_v7_scale_weights"]
    adata.obsm["X_v6_boundary_gate"] = adata.obsm["X_v7_boundary_gate"]
    adata.obsm["X_smvr_fused"] = adata.obsm[training_cfg["embedding_key"]]
    adata.obsm["X_smvr_domain"] = adata.obsm["X_v7_global"]
    if "spatial_edge_weighted_connectivities" in adata.obsp:
        adata.obsp[training_cfg["consensus_adj_key"]] = adata.obsp["spatial_edge_weighted_connectivities"].copy()
    adata.uns["training_loss_weights"] = {
        "strategy": architecture,
        "reconstruction": float(training_cfg.get("reconstruction_weight", 1.0)),
        "directional_continuity": float(training_cfg.get("directional_continuity_weight", 0.3)),
        "boundary_preservation": float(training_cfg.get("boundary_preservation_weight", 0.3)),
        "rare_consistency": float(training_cfg.get("rare_consistency_weight", 0.0)),
        "distance_preservation": float(training_cfg.get("distance_preservation_weight", 0.25)),
        "small_direction_continuity": float(training_cfg.get("small_direction_continuity_weight", 0.20)),
    }
    return {
        "adata": adata,
        "model": model,
        "edge_index_dict": graphs_best,
        "loss_history": loss_history,
    }
