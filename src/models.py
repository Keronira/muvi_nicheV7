from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
        nn.LayerNorm(out_dim),
    )


class GatedTokenFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_tokens: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.num_tokens = int(num_tokens)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return F.layer_norm(fused + self.refine(fused), fused.shape[1:]), weights


class LearnableRareQToken(nn.Module):
    def __init__(
        self,
        *,
        self_dim: int,
        hidden_dim: int,
        edge_attr_dim: int = 2,
        rare_dropout: float = 0.15,
        attention_dropout: float = 0.15,
        token_dropout: float = 0.20,
        score_temperature: float = 0.2,
    ):
        super().__init__()
        self.score_temperature = max(float(score_temperature), 1e-6)
        self.feature_projector = _mlp(self_dim, hidden_dim, hidden_dim, rare_dropout)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_score = nn.Sequential(
            nn.Linear(edge_attr_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(rare_dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.token_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(token_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(rare_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_self, neighbor_index, neighbor_mask, edge_attr):
        h = self.feature_projector(x_self)
        q = self.query(h).unsqueeze(1)
        nbr_h = h[neighbor_index]
        k = self.key(nbr_h)
        v = self.value(nbr_h)
        logits = (q * k).sum(dim=-1) / (h.shape[1] ** 0.5)
        logits = logits + self.edge_score(edge_attr).squeeze(-1)
        logits = logits / self.score_temperature
        logits = logits.masked_fill(neighbor_mask <= 0, -1e9)
        attn = torch.softmax(logits, dim=1)
        attn = torch.where(neighbor_mask > 0, attn, torch.zeros_like(attn))
        denom = attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        attn = attn / denom
        attn = self.attn_dropout(attn)
        attn = torch.where(neighbor_mask > 0, attn, torch.zeros_like(attn))
        denom = attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        attn = attn / denom
        rare_msg = torch.sum(v * attn.unsqueeze(-1), dim=1)
        token_in = torch.cat([h, rare_msg], dim=1)
        rare_token = self.token_head(token_in)
        rare_score = torch.sigmoid(self.score_head(token_in)).squeeze(-1)
        entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=1)
        return {
            "rare_token": rare_token,
            "rare_score": rare_score,
            "rare_attention": attn,
            "rare_attention_entropy": entropy,
        }


class DirectionalPoolingBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_attr_dim: int, num_directions: int, num_scales: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_score = nn.Sequential(
            nn.Linear(edge_attr_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.direction_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_directions),
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_scales),
        )
        self.boundary_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.self_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _pool_scale(self, h, direction_index, direction_mask, edge_attr):
        n_obs, num_directions, per_direction = direction_index.shape
        q = self.query(h).view(n_obs, 1, 1, -1)
        nbr_h = h[direction_index]
        k = self.key(nbr_h)
        v = self.value(nbr_h)
        score = (q * k).sum(dim=-1) / (h.shape[1] ** 0.5)
        score = score + self.edge_score(edge_attr).squeeze(-1)
        score = score.masked_fill(direction_mask <= 0, -1e9)
        attn = torch.softmax(score, dim=2)
        attn = torch.where(direction_mask > 0, attn, torch.zeros_like(attn))
        denom = attn.sum(dim=2, keepdim=True).clamp_min(1e-6)
        attn = attn / denom
        direction_msg = torch.sum(v * attn.unsqueeze(-1), dim=2)
        empty = (direction_mask.sum(dim=2) <= 0).unsqueeze(-1)
        direction_msg = torch.where(empty, torch.zeros_like(direction_msg), direction_msg)
        return direction_msg, attn

    def forward(self, h, boundary_h, scale_inputs):
        per_scale_msg = []
        per_scale_attn = []
        per_scale_dir_weights = []
        for item in scale_inputs:
            direction_msg, attn = self._pool_scale(
                h,
                item["direction_index"],
                item["direction_mask"],
                item["edge_attr"],
            )
            gate_in = torch.cat([h, boundary_h], dim=1)
            direction_weight = torch.sigmoid(self.direction_gate(gate_in))
            msg = torch.sum(direction_msg * direction_weight.unsqueeze(-1), dim=1)
            denom = direction_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
            per_scale_msg.append(msg / denom)
            per_scale_attn.append(attn)
            per_scale_dir_weights.append(direction_weight)

        stacked = torch.stack(per_scale_msg, dim=1)
        scale_weight = torch.softmax(self.scale_gate(torch.cat([h, boundary_h], dim=1)), dim=1)
        pooled = torch.sum(stacked * scale_weight.unsqueeze(-1), dim=1)
        boundary_gate = torch.sigmoid(self.boundary_gate(torch.cat([h, boundary_h], dim=1)))
        mixed_update = (1.0 - boundary_gate) * self.update(pooled) + boundary_gate * self.self_update(h)
        h_next = F.layer_norm(h + mixed_update, h.shape[1:])
        diagnostics = {
            "direction_weights": torch.stack(per_scale_dir_weights, dim=1),
            "scale_weights": scale_weight,
            "boundary_gate": boundary_gate,
        }
        return h_next, diagnostics


class DirectionalMultiScaleLPEncoder(nn.Module):
    def __init__(
        self,
        *,
        self_dim: int,
        boundary_dim: int,
        scales: list[int],
        hidden_dim: int = 96,
        embedding_dim: int = 64,
        num_layers: int = 4,
        num_directions: int = 8,
        edge_attr_dim: int = 5,
        dropout: float = 0.08,
        rareq: dict | None = None,
    ):
        super().__init__()
        self.scales = [int(scale) for scale in scales]
        self.rareq_cfg = rareq or {}
        self.use_rareq = bool(self.rareq_cfg.get("enabled", False))
        self.mask_token = nn.Parameter(torch.zeros(1, self_dim))
        self.self_projector = _mlp(self_dim, hidden_dim, hidden_dim, dropout)
        self.boundary_projector = _mlp(boundary_dim, hidden_dim, hidden_dim, dropout)
        self.neighbor_projectors = nn.ModuleDict(
            {str(scale): _mlp(self_dim, hidden_dim, hidden_dim, dropout) for scale in self.scales}
        )
        if self.use_rareq:
            self.rareq_token = LearnableRareQToken(
                self_dim=self_dim,
                hidden_dim=hidden_dim,
                edge_attr_dim=2,
                rare_dropout=float(self.rareq_cfg.get("rare_dropout", 0.15)),
                attention_dropout=float(self.rareq_cfg.get("attention_dropout", 0.15)),
                token_dropout=float(self.rareq_cfg.get("token_dropout", 0.20)),
                score_temperature=float(self.rareq_cfg.get("score_temperature", 0.2)),
            )
        else:
            self.rareq_token = None
        self.token_fusion = GatedTokenFusion(hidden_dim, len(self.scales) + 2 + int(self.use_rareq), dropout)
        self.blocks = nn.ModuleList(
            [
                DirectionalPoolingBlock(
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    num_directions=num_directions,
                    num_scales=len(self.scales),
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.global_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embedding_dim))
        self.local_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embedding_dim))
        self.final_head = nn.Sequential(
            nn.Linear(embedding_dim * 2 + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.self_decoder = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, self_dim))
        self.neighbor_decoders = nn.ModuleDict(
            {str(scale): nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, self_dim)) for scale in self.scales}
        )

    def encoding_mask_noise(self, x, mask_rate: float):
        if mask_rate <= 0:
            return x, None
        num_nodes = x.shape[0]
        mask_count = int(mask_rate * num_nodes)
        if mask_count <= 0:
            return x, None
        perm = torch.randperm(num_nodes, device=x.device)
        mask_nodes = perm[:mask_count]
        out = x.clone()
        out[mask_nodes] = self.mask_token.to(x.dtype)
        return out, mask_nodes

    def forward(self, batch: dict[str, object], mask_rate: float = 0.2):
        x_self = batch["X_self"]
        masked_self, mask_nodes = self.encoding_mask_noise(x_self, mask_rate)
        h_self = self.self_projector(masked_self)
        h_boundary = self.boundary_projector(batch["X_boundary"])
        tokens = [h_self]
        for scale in self.scales:
            tokens.append(self.neighbor_projectors[str(scale)](batch[f"X_nbr_k{scale}"]))
        tokens.append(h_boundary)
        rare_outputs = None
        if self.use_rareq:
            rare_outputs = self.rareq_token(
                x_self,
                batch["rareq_neighbor_index"],
                batch["rareq_neighbor_mask"],
                batch["rareq_edge_attr"],
            )
            tokens.append(rare_outputs["rare_token"])
        h, view_weights = self.token_fusion(torch.stack(tokens, dim=1))

        diag_history = []
        scale_inputs = [
            {
                "direction_index": batch[f"direction_index_k{scale}"],
                "direction_mask": batch[f"direction_mask_k{scale}"],
                "edge_attr": batch[f"direction_edge_attr_k{scale}"],
            }
            for scale in self.scales
        ]
        for block in self.blocks:
            h, diagnostics = block(h, h_boundary, scale_inputs)
            diag_history.append(diagnostics)

        z_global = self.global_head(h)
        z_local = self.local_head(h)
        z_final = F.layer_norm(self.final_head(torch.cat([z_global, z_local, h_self], dim=1)), (z_global.shape[1],))
        out = {
            "z_final": z_final,
            "z_global": z_global,
            "z_local": z_local,
            "x_self_recon": self.self_decoder(z_final),
            "x_nbr_recon": {scale: self.neighbor_decoders[str(scale)](z_final) for scale in self.scales},
            "mask_nodes": mask_nodes,
            "view_weights": view_weights,
            "direction_weights": torch.stack([item["direction_weights"] for item in diag_history], dim=1),
            "scale_weights": torch.stack([item["scale_weights"] for item in diag_history], dim=1),
            "boundary_gate": torch.stack([item["boundary_gate"] for item in diag_history], dim=1).mean(dim=1),
        }
        if rare_outputs is not None:
            out.update(
                {
                    "rare_score": rare_outputs["rare_score"],
                    "rare_attention": rare_outputs["rare_attention"],
                    "rare_attention_entropy": rare_outputs["rare_attention_entropy"],
                    "rare_token_weight": view_weights[:, -1],
                }
            )
        return out
