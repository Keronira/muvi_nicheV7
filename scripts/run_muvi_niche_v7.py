#!/usr/bin/env python3
from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = Path(__file__).resolve().parents[1]
for path in (WORKSPACE_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_common import (
    ResourceMonitor,
    build_parser,
    dataset_name_from_args,
    format_profile_metrics,
    iter_h5ad_files,
    load_spatial_adata,
    method_output_dirs,
    print_sample_header,
    resolve_preprocess_options,
    save_profile_metrics,
    save_standard_outputs,
    seed_from_precedence,
)
from src.graph_construction import build_graph_inputs
from src.multislice import detect_training_mode, split_adata_by_obs_key
from src.preprocessing import preprocess_adata
from src.trainer import run_training_pipeline
from src.utils import (
    apply_prototype_memory,
    assign_niche_prototypes,
    fit_prototype_memory,
    load_config,
    run_kmeans_on_embedding,
    save_final_adata,
    save_graphs,
    set_random_seed,
)


def ensure_scanpy_spatial_plot_metadata(adata) -> None:
    if "spatial" in adata.uns:
        return
    adata.uns["spatial"] = {
        "muvi_niche_v7": {
            "images": {},
            "scalefactors": {
                "spot_diameter_fullres": 1.0,
                "tissue_hires_scalef": 1.0,
                "tissue_lowres_scalef": 1.0,
            },
        }
    }


def parse_args():
    parser = build_parser(
        "muvi_niche_v7_learnable_rareq",
        "Run muvi_niche_v7 learnable RareQ directional local pooling on .h5ad files.",
        WORKSPACE_ROOT,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=METHOD_ROOT / "configs" / "config.yaml",
        help="Base YAML config for muvi_niche_v7.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--spatial-neighbors", type=int, default=30)
    return parser.parse_args()


def build_runtime_config(args, sample_name: str, method_root: Path, dataset_name: str) -> tuple[dict, dict, Path]:
    config = copy.deepcopy(load_config(str(args.config.resolve())))
    preprocess_opts = resolve_preprocess_options(
        args,
        args.method_name,
        {
            "spatial_obsm_key": "spatial",
            "target_spatial_key": "spatial",
            "filter_genes": False,
            "min_cells": 3,
            "min_genes": args.min_genes,
            "normalize_total": True,
            "target_sum": 1e4,
            "ensure_log1p": True,
            "n_top_genes": 5000,
            "hvg_flavor": "seurat",
            "spatial_neighbors": args.spatial_neighbors,
        },
    )

    sample_work_dir = method_root / "artifacts" / sample_name
    sample_work_dir.mkdir(parents=True, exist_ok=True)

    config["seed"] = seed_from_precedence(args, int(config.get("seed", 1234)))
    config["paths"]["output_dir"] = str(sample_work_dir)
    config["bench"]["dataset_name"] = dataset_name
    config["bench"]["sample_name"] = sample_name
    config["bench"]["method_name"] = args.method_name
    config["bench"]["bench_root"] = str(args.output_root.resolve())
    config["data"]["label_key"] = args.ground_truth_key

    config["preprocessing"]["filter_genes"] = preprocess_opts["filter_genes"]
    config["preprocessing"]["min_cells"] = preprocess_opts["min_cells"]
    config["preprocessing"]["min_genes"] = preprocess_opts["min_genes"]
    config["preprocessing"]["normalize_total"] = preprocess_opts["normalize_total"]
    config["preprocessing"]["target_sum"] = preprocess_opts["target_sum"]
    config["preprocessing"]["ensure_log1p"] = preprocess_opts["ensure_log1p"]
    config["preprocessing"]["n_top_genes"] = preprocess_opts["n_top_genes"]
    config["preprocessing"]["hvg_flavor"] = preprocess_opts["hvg_flavor"]
    config["preprocessing"]["spatial_neighbors"] = preprocess_opts["spatial_neighbors"]
    config["preprocessing"]["use_existing_spatial_key"] = (
        preprocess_opts["spatial_obsm_key"] != preprocess_opts["target_spatial_key"]
    )
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    return config, preprocess_opts, sample_work_dir


def prepare_slice(input_source, config: dict, preprocess_opts: dict):
    if isinstance(input_source, Path):
        adata = load_spatial_adata(
            input_source,
            spatial_obsm_key=preprocess_opts["spatial_obsm_key"],
            target_spatial_key=preprocess_opts["target_spatial_key"],
        )
        adata.uns["sample_name"] = config.get("bench", {}).get("sample_name", input_source.stem)
    else:
        adata = input_source
        adata.uns["sample_name"] = config.get("bench", {}).get(
            "sample_name",
            adata.uns.get("sample_name", "sample"),
        )
    adata_processed = preprocess_adata(adata, config)
    ensure_scanpy_spatial_plot_metadata(adata_processed)
    adata_graph, graphs_best = build_graph_inputs(adata_processed, config)
    print(
        f"[muvi_niche_v7] build_graph_inputs returned: "
        f"n_obs={adata_graph.n_obs}, graph_keys={list(graphs_best.keys())}"
    )
    return adata_graph, graphs_best


def format_loss_weight_annotation(config: dict) -> str:
    training = config.get("training", {})
    rareq = config.get("model", {}).get("rareq", {})
    keys = [
        ("recon", "reconstruction_weight"),
        ("dir", "directional_continuity_weight"),
        ("bnd", "boundary_preservation_weight"),
        ("rare", "rare_consistency_weight"),
    ]
    loss_text = "Loss weights: " + ", ".join(
        f"{label}={float(training.get(key, 0.0)):.3g}" for label, key in keys
    )
    if rareq.get("enabled", False):
        rare_text = (
            f"RareQ token: on, k={int(rareq.get('feature_knn_k', 20))}, "
            f"q={int(rareq.get('q_neighbors', 6))}, "
            f"drop={float(rareq.get('token_dropout', 0.2)):.2g}"
        )
    else:
        rare_text = "RareQ token: off"
    return f"{loss_text}; {rare_text}"


def run_single_slice(
    *,
    args,
    input_path: Path,
    dataset_name: str,
    method_root: Path,
    plots_dir: Path,
    results_dir: Path,
):
    sample_name = input_path.stem
    print_sample_header(args.method_name, sample_name)
    config, preprocess_opts, sample_work_dir = build_runtime_config(args, sample_name, method_root, dataset_name)

    set_random_seed(config["seed"])
    monitor = ResourceMonitor().start()
    try:
        adata_graph, graphs_best = prepare_slice(input_path, config, preprocess_opts)
        training_outputs = run_training_pipeline(
            adata=adata_graph,
            graphs_best=graphs_best,
            config=config,
            output_dir=sample_work_dir,
        )
        run_kmeans_on_embedding(training_outputs["adata"], config)
        assign_niche_prototypes(training_outputs["adata"], config)

        if config.get("outputs", {}).get("save_final_adata", False):
            save_final_adata(
                adata=training_outputs["adata"],
                output_dir=sample_work_dir,
                final_name=f"{sample_name}_{config['outputs']['final_adata_name']}",
            )
        save_graphs(
            graphs_best=training_outputs.get("edge_index_dict", graphs_best),
            output_dir=sample_work_dir,
            config=config,
        )
        save_standard_outputs(
            adata=training_outputs["adata"],
            embedding=training_outputs["adata"].obsm[config["training"]["embedding_key"]],
            sample_name=sample_name,
            method_title=args.method_name,
            cluster_key=config["training"]["cluster_key"],
            plots_dir=plots_dir,
            results_dir=results_dir,
            ground_truth_key=args.ground_truth_key,
            point_size=args.point_size,
            plot_annotation=format_loss_weight_annotation(config),
        )
    finally:
        profile = monitor.stop()

    save_profile_metrics(results_dir, sample_name, args.method_name, profile)
    print(f"{sample_name} done.")
    print(f"[{args.method_name}] {sample_name} profile: {format_profile_metrics(profile)}")


def run_multi_slice(
    *,
    args,
    input_paths: list[Path],
    dataset_name: str,
    method_root: Path,
    plots_dir: Path,
    results_dir: Path,
):
    run_name = f"{dataset_name}_multi_slice"
    print_sample_header(args.method_name, run_name)
    config, preprocess_opts, _sample_work_dir = build_runtime_config(args, run_name, method_root, dataset_name)
    config.setdefault("training_mode", {})["resolved_mode"] = "multi_slice"
    set_random_seed(config["seed"])
    run_multi_slice_with_slice_batches(
        args=args,
        input_paths=input_paths,
        dataset_name=dataset_name,
        method_root=method_root,
        plots_dir=plots_dir,
        results_dir=results_dir,
        base_config=config,
        preprocess_opts=preprocess_opts,
    )


def run_multi_slice_with_slice_batches(
    *,
    args,
    input_paths: list[Path],
    dataset_name: str,
    method_root: Path,
    plots_dir: Path,
    results_dir: Path,
    base_config: dict,
    preprocess_opts: dict,
):
    trained_adatas = []
    profiles = {}
    slice_inputs = expand_multi_slice_inputs(input_paths, base_config, preprocess_opts)
    for sample_name, input_source in slice_inputs:
        print_sample_header(args.method_name, f"{sample_name} slice-mini-batch")
        sample_work_dir = method_root / "artifacts" / f"{sample_name}_slice_batch"
        sample_work_dir.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(base_config)
        config["paths"]["output_dir"] = str(sample_work_dir)
        config["bench"]["sample_name"] = sample_name

        monitor = ResourceMonitor().start()
        try:
            adata_graph, graphs_best = prepare_slice(input_source, config, preprocess_opts)
            training_outputs = run_training_pipeline(
                adata=adata_graph,
                graphs_best=graphs_best,
                config=config,
                output_dir=sample_work_dir,
            )
            run_kmeans_on_embedding(training_outputs["adata"], config)
            save_graphs(
                graphs_best=training_outputs.get("edge_index_dict", graphs_best),
                output_dir=sample_work_dir,
                config=config,
            )
        finally:
            profile = monitor.stop()
        profiles[sample_name] = profile
        trained_adatas.append((sample_name, training_outputs["adata"], config, sample_work_dir))

    memory = fit_prototype_memory([adata for _name, adata, _config, _dir in trained_adatas], base_config)
    for sample_name, adata, config, sample_work_dir in trained_adatas:
        apply_prototype_memory(adata, config, memory)
        if config.get("outputs", {}).get("save_final_adata", False):
            save_final_adata(
                adata=adata,
                output_dir=sample_work_dir,
                final_name=f"{sample_name}_{config['outputs']['final_adata_name']}",
            )
        save_standard_outputs(
            adata=adata,
            embedding=adata.obsm[config["training"]["embedding_key"]],
            sample_name=sample_name,
            method_title=args.method_name,
            cluster_key=config["training"]["cluster_key"],
            plots_dir=plots_dir,
            results_dir=results_dir,
            ground_truth_key=args.ground_truth_key,
            point_size=args.point_size,
            plot_annotation=format_loss_weight_annotation(config),
        )
        save_profile_metrics(results_dir, sample_name, args.method_name, profiles[sample_name])
        print(f"{sample_name} done.")
        print(f"[{args.method_name}] {sample_name} profile: {format_profile_metrics(profiles[sample_name])}")


def expand_multi_slice_inputs(input_paths: list[Path], config: dict, preprocess_opts: dict):
    batch_key = config.get("training_mode", {}).get("batch_key", "slice_id")
    if len(input_paths) != 1:
        return [(path.stem, path) for path in input_paths]

    input_path = input_paths[0]
    adata = load_spatial_adata(
        input_path,
        spatial_obsm_key=preprocess_opts["spatial_obsm_key"],
        target_spatial_key=preprocess_opts["target_spatial_key"],
    )
    if batch_key not in adata.obs:
        adata.uns["sample_name"] = input_path.stem
        return [(input_path.stem, adata)]
    slice_inputs = []
    for slice_name, slice_adata in split_adata_by_obs_key(adata, batch_key).items():
        sample_name = f"{input_path.stem}_{slice_name}"
        slice_adata.uns["sample_name"] = sample_name
        slice_inputs.append((sample_name, slice_adata))
    return slice_inputs


def main() -> int:
    args = parse_args()
    dataset_name = dataset_name_from_args(args)
    method_root, plots_dir, results_dir = method_output_dirs(
        args.output_root.resolve(), dataset_name, args.method_name
    )
    input_paths = iter_h5ad_files(args.data_dir.resolve(), args.samples)

    base_config = load_config(str((args.config or (METHOD_ROOT / "configs" / "config.yaml")).resolve()))
    mode = detect_training_mode(input_paths, base_config)
    print(f"[{args.method_name}] training_mode={mode}")

    if mode == "multi_slice":
        run_multi_slice(
            args=args,
            input_paths=input_paths,
            dataset_name=dataset_name,
            method_root=method_root,
            plots_dir=plots_dir,
            results_dir=results_dir,
        )
    else:
        for input_path in input_paths:
            run_single_slice(
                args=args,
                input_path=input_path,
                dataset_name=dataset_name,
                method_root=method_root,
                plots_dir=plots_dir,
                results_dir=results_dir,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
