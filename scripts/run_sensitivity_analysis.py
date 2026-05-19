#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNNER = METHOD_ROOT / "scripts" / "run_muvi_niche_v7.py"


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    overrides: dict


def deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def loss_overrides(
    *,
    recon: float = 1.5,
    direction: float = 0.15,
    boundary: float = 0.7,
    rare: float = 0.0,
) -> dict:
    return {
        "training": {
            "reconstruction_weight": recon,
            "directional_continuity_weight": direction,
            "boundary_preservation_weight": boundary,
            "rare_consistency_weight": rare,
            "neighbor_reconstruction_weight": 0.0,
            "variance_weight": 0.0,
        }
    }


def rareq_overrides(
    *,
    rare_dropout: float = 0.20,
    attention_dropout: float = 0.20,
    token_dropout: float = 0.20,
    k: int = 20,
    q: int = 6,
) -> dict:
    return {
        "model": {
            "rareq": {
                "feature_knn_k": k,
                "q_neighbors": q,
                "rare_dropout": rare_dropout,
                "attention_dropout": attention_dropout,
                "token_dropout": token_dropout,
            }
        }
    }


ANCHOR_LOSS = loss_overrides(recon=1.5, direction=0.15, boundary=0.7, rare=0.0)
ANCHOR_RAREQ = rareq_overrides(rare_dropout=0.20, attention_dropout=0.20, token_dropout=0.20, k=20, q=6)
ANCHOR_OVERRIDES = deep_update(ANCHOR_LOSS, ANCHOR_RAREQ)


def anchored_loss(
    *,
    recon: float = 1.5,
    direction: float = 0.15,
    boundary: float = 0.7,
    rare: float = 0.0,
) -> dict:
    return deep_update(
        ANCHOR_RAREQ,
        loss_overrides(recon=recon, direction=direction, boundary=boundary, rare=rare),
    )


def anchored_rareq(
    *,
    rare_dropout: float = 0.20,
    attention_dropout: float = 0.20,
    token_dropout: float = 0.20,
    k: int = 20,
    q: int = 6,
) -> dict:
    return deep_update(
        ANCHOR_LOSS,
        rareq_overrides(
            rare_dropout=rare_dropout,
            attention_dropout=attention_dropout,
            token_dropout=token_dropout,
            k=k,
            q=q,
        ),
    )


def variant_catalog() -> dict[str, Variant]:
    anchor = Variant(
        "anchor",
        "New v7 anchor: recon=1.5, dir=0.15, boundary=0.7, rare=0, RareQ k=20 q=6 dropout=0.2.",
        ANCHOR_OVERRIDES,
    )
    variants = [
        anchor,
        Variant("rare_005", "Turn on very weak rare consistency from the new anchor.", anchored_loss(rare=0.05)),
        Variant("rare_010", "Turn on weak rare consistency from the new anchor.", anchored_loss(rare=0.10)),
        Variant("rare_020", "Turn on moderate rare consistency from the new anchor.", anchored_loss(rare=0.20)),
        Variant("rare_030", "Stress rare consistency while keeping the new anchor otherwise fixed.", anchored_loss(rare=0.30)),
        Variant("recon_13", "Slightly reduce self reconstruction from anchor.", anchored_loss(recon=1.30)),
        Variant("recon_17", "Slightly increase self reconstruction from anchor.", anchored_loss(recon=1.70)),
        Variant("dir_010", "Weaker directional continuity than anchor.", anchored_loss(direction=0.10)),
        Variant("dir_020", "Stronger directional continuity than anchor.", anchored_loss(direction=0.20)),
        Variant("bnd_05", "Weaker boundary preservation than anchor.", anchored_loss(boundary=0.50)),
        Variant("bnd_09", "Stronger boundary preservation than anchor.", anchored_loss(boundary=0.90)),
        Variant(
            "rare010_bnd09",
            "Weak rare consistency with stronger boundary preservation.",
            anchored_loss(boundary=0.90, rare=0.10),
        ),
        Variant(
            "rare010_dir010",
            "Weak rare consistency with weaker directional smoothing.",
            anchored_loss(direction=0.10, rare=0.10),
        ),
        Variant(
            "balanced_local",
            "Local best-guess around anchor: slightly lower direction, stronger boundary, weak rare.",
            anchored_loss(recon=1.50, direction=0.10, boundary=0.90, rare=0.10),
        ),
        Variant(
            "self_rare_local",
            "Higher self reconstruction plus weak rare consistency.",
            anchored_loss(recon=1.70, direction=0.10, boundary=0.70, rare=0.10),
        ),
        Variant(
            "dropout_low",
            "Lower RareQ dropout around the new anchor.",
            anchored_rareq(rare_dropout=0.10, attention_dropout=0.10, token_dropout=0.10),
        ),
        Variant(
            "dropout_high",
            "Higher RareQ dropout around the new anchor.",
            anchored_rareq(rare_dropout=0.30, attention_dropout=0.30, token_dropout=0.30),
        ),
        Variant(
            "dropout_rare010",
            "Anchor dropout with weak rare consistency enabled.",
            anchored_loss(rare=0.10),
        ),
        Variant(
            "rareq_k12_q4",
            "Smaller feature-space RareQ neighborhood around anchor.",
            anchored_rareq(k=12, q=4),
        ),
        Variant(
            "rareq_k30_q8",
            "Larger feature-space RareQ neighborhood around anchor.",
            anchored_rareq(k=30, q=8),
        ),
        Variant(
            "rareq_k12_q4_rare010",
            "Smaller RareQ neighborhood with weak rare consistency.",
            deep_update(anchored_rareq(k=12, q=4), {"training": {"rare_consistency_weight": 0.10}}),
        ),
        Variant(
            "rareq_k30_q8_rare010",
            "Larger RareQ neighborhood with weak rare consistency.",
            deep_update(anchored_rareq(k=30, q=8), {"training": {"rare_consistency_weight": 0.10}}),
        ),
    ]
    return {variant.name: variant for variant in variants}


def default_variant_names(variant_set: str) -> list[str]:
    core = [
        "anchor",
        "rare_005",
        "rare_010",
        "rare_020",
        "rare_030",
        "recon_13",
        "recon_17",
        "dir_010",
        "dir_020",
        "bnd_05",
        "bnd_09",
        "rare010_bnd09",
        "rare010_dir010",
        "balanced_local",
        "self_rare_local",
    ]
    dropout = ["dropout_low", "dropout_high", "dropout_rare010"]
    graph = ["rareq_k12_q4", "rareq_k30_q8", "rareq_k12_q4_rare010", "rareq_k30_q8_rare010"]
    if variant_set == "core":
        return core
    if variant_set == "dropout":
        return ["anchor", *dropout]
    if variant_set == "graph":
        return ["anchor", *graph]
    return [*core, *dropout, *graph]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run v7 learnable RareQ sensitivity.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=WORKSPACE_ROOT / "bench_results_v7_sensitivity")
    parser.add_argument("--dataset-name")
    parser.add_argument("--samples", nargs="+", help="Optional single sample or sample subset, without .h5ad suffix.")
    parser.add_argument("--ground-truth-key", default="annotation_final")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pca-dims", type=int, default=50)
    parser.add_argument("--point-size", type=float, default=1.5)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--spatial-neighbors", type=int, default=30)
    parser.add_argument("--config", type=Path, default=METHOD_ROOT / "configs" / "config.yaml")
    parser.add_argument(
        "--variant-set",
        choices=["core", "dropout", "graph", "all"],
        default="core",
        help="core scans loss weights; dropout scans RareQ dropout; graph scans RareQ feature-kNN size; all runs all variants.",
    )
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--method-prefix", default="muvi_niche_v7_sa")
    parser.add_argument("--python-exe", default="/root/autodl-tmp/muvi/bin/python")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def dataset_name_from_args(args: argparse.Namespace) -> str:
    return args.dataset_name or args.data_dir.name


def write_variant_config(base_config: dict, variant: Variant, config_dir: Path, epochs: int | None) -> Path:
    config = deep_update(base_config, variant.overrides)
    if epochs is not None and "epochs" not in variant.overrides.get("training", {}):
        config.setdefault("training", {})["epochs"] = int(epochs)
    config.setdefault("outputs", {})["save_final_adata"] = False
    config.setdefault("sensitivity_analysis", {})["variant"] = variant.name
    config["sensitivity_analysis"]["description"] = variant.description
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{variant.name}.yaml"
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=False)
    return path


def build_command(args: argparse.Namespace, variant: Variant, config_path: Path) -> list[str]:
    command = [
        args.python_exe,
        str(RUNNER),
        "--data-dir",
        str(args.data_dir),
        "--output-root",
        str(args.output_root),
        "--dataset-name",
        dataset_name_from_args(args),
        "--method-name",
        f"{args.method_prefix}_{variant.name}",
        "--ground-truth-key",
        args.ground_truth_key,
        "--seed",
        str(args.seed),
        "--pca-dims",
        str(args.pca_dims),
        "--point-size",
        str(args.point_size),
        "--min-genes",
        str(args.min_genes),
        "--spatial-neighbors",
        str(args.spatial_neighbors),
        "--config",
        str(config_path),
    ]
    if args.epochs is not None and "epochs" not in variant.overrides.get("training", {}):
        command.extend(["--epochs", str(args.epochs)])
    if args.samples:
        command.extend(["--samples", *args.samples])
    return command


def _contingency_metrics(valid: pd.DataFrame) -> dict:
    table = pd.crosstab(valid["ground_truth"].astype(str), valid["pred"].astype(str))
    total = float(table.to_numpy().sum())
    pred_counts = table.sum(axis=0).to_numpy(dtype=float)
    truth_counts = table.sum(axis=1).to_numpy(dtype=float)
    largest_pred_fraction = float(pred_counts.max() / total) if total else np.nan
    p = pred_counts / max(total, 1.0)
    cluster_size_entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(max(len(p), 2))) if len(p) else np.nan
    pred_purity = float((table.max(axis=0).sum()) / total) if total else np.nan
    truth_coverage = float((table.max(axis=1).sum()) / total) if total else np.nan
    small_cutoff = np.quantile(truth_counts, 0.35) if len(truth_counts) else 0
    small_truth = table.index[truth_counts <= small_cutoff].tolist()
    small_fragmentation = float(np.mean([(table.loc[label] > 0).sum() for label in small_truth])) if small_truth else np.nan
    large_cutoff = np.quantile(truth_counts, 0.65) if len(truth_counts) else 0
    large_truth = table.index[truth_counts >= large_cutoff].tolist()
    merge_scores = []
    for pred in table.columns:
        col = table[pred]
        covered_large = [label for label in large_truth if col.loc[label] / max(table.loc[label].sum(), 1) >= 0.2]
        if len(covered_large) > 1:
            merge_scores.append(len(covered_large) - 1)
    return {
        "largest_pred_cluster_fraction": largest_pred_fraction,
        "cluster_size_entropy": cluster_size_entropy,
        "pred_to_truth_purity": pred_purity,
        "truth_to_pred_coverage": truth_coverage,
        "small_niche_fragmentation": small_fragmentation,
        "large_niche_merge_score": float(np.sum(merge_scores)) if merge_scores else 0.0,
    }


def collect_variant_rows(args: argparse.Namespace, variants: list[Variant]) -> list[dict]:
    rows = []
    dataset_name = dataset_name_from_args(args)
    for variant in variants:
        method_name = f"{args.method_prefix}_{variant.name}"
        result_dir = args.output_root / dataset_name / method_name / "results"
        if not result_dir.is_dir():
            rows.append({"variant": variant.name, "method": method_name, "status": "missing_results"})
            continue
        files = sorted(path for path in result_dir.glob("*.csv") if not path.name.endswith("_embedding.csv"))
        if args.samples:
            wanted = {f"{sample}.csv" for sample in args.samples}
            files = [path for path in files if path.name in wanted]
        for result_file in files:
            row = {"variant": variant.name, "method": method_name, "sample": result_file.stem, "status": "ok"}
            try:
                df = pd.read_csv(result_file)
                valid = df[["ground_truth", "pred"]].dropna()
                row["ARI"] = adjusted_rand_score(valid["ground_truth"].astype(str), valid["pred"].astype(str))
                row["NMI"] = normalized_mutual_info_score(valid["ground_truth"].astype(str), valid["pred"].astype(str))
                row["n_pred_clusters"] = int(valid["pred"].nunique())
                row["n_truth_labels"] = int(valid["ground_truth"].nunique())
                row.update(_contingency_metrics(valid))
            except Exception as exc:
                row["status"] = "metric_error"
                row["error"] = str(exc)
            profile_path = result_dir / f"{result_file.stem}_profile.json"
            if profile_path.is_file():
                with profile_path.open("r", encoding="utf-8") as fh:
                    profile = json.load(fh)
                row["runtime_seconds"] = profile.get("runtime_seconds")
                row["cpu_rss_peak_mb"] = profile.get("cpu_rss_peak_mb")
            rows.append(row)
    return rows


def save_summary(args: argparse.Namespace, variants: list[Variant], rows: list[dict]) -> None:
    out_dir = args.output_root / dataset_name_from_args(args) / f"{args.method_prefix}_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "sensitivity_detail.csv", index=False)
    metric_cols = [
        "ARI",
        "NMI",
        "largest_pred_cluster_fraction",
        "cluster_size_entropy",
        "pred_to_truth_purity",
        "truth_to_pred_coverage",
        "small_niche_fragmentation",
        "large_niche_merge_score",
        "runtime_seconds",
        "cpu_rss_peak_mb",
    ]
    numeric_cols = [col for col in metric_cols if col in detail.columns]
    if not detail.empty and numeric_cols:
        summary = detail.groupby("variant", dropna=False)[numeric_cols].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        order = {variant.name: idx for idx, variant in enumerate(variants)}
        summary = summary.reset_index()
        summary["_order"] = summary["variant"].map(order).fillna(999).astype(int)
        summary = summary.sort_values(["_order", "variant"]).drop(columns=["_order"])
        summary.to_csv(out_dir / "sensitivity_summary.csv", index=False)
        report_cols = [
            col
            for col in [
                "variant",
                "ARI_mean",
                "NMI_mean",
                "large_niche_merge_score_mean",
                "small_niche_fragmentation_mean",
                "largest_pred_cluster_fraction_mean",
            ]
            if col in summary.columns
        ]
        summary[report_cols].to_csv(out_dir / "ari_nmi_merge_report.csv", index=False)
        print(summary[report_cols].to_string(index=False))
    print(f"[muvi_niche_v7] sensitivity detail saved: {out_dir / 'sensitivity_detail.csv'}")
    print(f"[muvi_niche_v7] sensitivity summary saved: {out_dir / 'sensitivity_summary.csv'}")
    print(f"[muvi_niche_v7] ARI/NMI merge report saved: {out_dir / 'ari_nmi_merge_report.csv'}")


def main() -> int:
    args = parse_args()
    catalog = variant_catalog()
    names = args.variants or default_variant_names(args.variant_set)
    unknown = [name for name in names if name not in catalog]
    if unknown:
        raise ValueError(f"Unknown variant(s): {', '.join(unknown)}. Available: {', '.join(catalog)}")
    variants = [catalog[name] for name in names]
    with args.config.resolve().open("r", encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh)
    config_dir = args.output_root / dataset_name_from_args(args) / f"{args.method_prefix}_configs"
    failures = []
    for variant in variants:
        config_path = write_variant_config(base_config, variant, config_dir, args.epochs)
        command = build_command(args, variant, config_path)
        print(f"[RUN ] {variant.name}: {variant.description}")
        print("       " + subprocess.list2cmdline(command))
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=WORKSPACE_ROOT, check=False)
        if result.returncode != 0:
            failures.append((variant.name, result.returncode))
            print(f"[FAIL] {variant.name}: exit_code={result.returncode}")
            if not args.continue_on_error:
                break
    rows = collect_variant_rows(args, variants)
    save_summary(args, variants, rows)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
