from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

try:
    from .dataset_utils import default_eval_dataset_order, matched_records_for_preset
    from .protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        filter_records_for_task,
        get_dataset_preset,
    )
    from .evaluate_jepa import format_pct, run_grouped_cv, summarize_runs
    from .evaluate_fusion_subject_vote import (
        build_sample_metadata,
        load_or_build_fc_feature_cache,
        load_or_build_jepa_embedding_cache,
        load_checkpoint,
        maybe_subsample_balanced,
        parse_method_list,
        parse_seed_list,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.dataset_utils import default_eval_dataset_order, matched_records_for_preset
    from src.jepa_fusion.protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        filter_records_for_task,
        get_dataset_preset,
    )
    from src.jepa_fusion.evaluate_jepa import format_pct, run_grouped_cv, summarize_runs
    from src.jepa_fusion.evaluate_fusion_subject_vote import (
        build_sample_metadata,
        load_or_build_fc_feature_cache,
        load_or_build_jepa_embedding_cache,
        load_checkpoint,
        maybe_subsample_balanced,
        parse_method_list,
        parse_seed_list,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate grouped-CV site probes for FC+JEPA fusion representations built "
            "from concatenated subject-level FC features and mean JEPA embeddings."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["abide", "abide1", "abide2", "srpbs", "all"], default="all")
    parser.add_argument("--task", choices=["default", "hc_asd", "hc_disorder"], default="default")
    parser.add_argument("--all-sites", action="store_true")
    parser.add_argument("--triangle", choices=["upper", "lower"], default="upper")
    parser.add_argument("--reorder", choices=["none", "hemi", "hemi_network"], default="hemi_network")
    parser.add_argument("--fc-methods", default="fc,fc_combat,fc_covbat")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-feature", choices=["mean", "mean_std", "mean_cov"], default="mean")
    parser.add_argument("--covariance-dim", type=int, default=32)
    parser.add_argument("--covariance-seed", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=30.0)
    parser.add_argument("--target-timepoints", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", default="results/cache/jepa_fusion")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--healthy-control-label", default=HEALTHY_CONTROL_LABEL)
    parser.add_argument("--autism-label", default=AUTISM_LABEL)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def default_output_paths(checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    stem = f"{checkpoint.parent.name}_{checkpoint.stem}_fusion_site_probe"
    base_dir = Path("results/reports/tables/jepa_fusion")
    return base_dir / f"{stem}.json", base_dir / f"{stem}.md"


def aggregate_subject_feature_map(features: np.ndarray, metadata: list[dict]):
    buckets = defaultdict(list)
    sites_by_subject: dict[str, str] = {}
    for feature, meta in zip(features, metadata):
        subject_id = str(meta["subject_id"])
        buckets[subject_id].append(np.asarray(feature, dtype=np.float32))
        sites_by_subject[subject_id] = str(meta.get("site_batch", meta["site"]))

    aggregated = {}
    for subject_id, rows in buckets.items():
        aggregated[subject_id] = {
            "feature": np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32),
            "site": sites_by_subject[subject_id],
        }
    return aggregated


def build_fusion_subject_matrix(
    *,
    fc_features: np.ndarray,
    fc_metadata: list[dict],
    jepa_embeddings: np.ndarray,
    jepa_metadata: list[dict],
):
    fc_map = aggregate_subject_feature_map(fc_features, fc_metadata)
    jepa_map = aggregate_subject_feature_map(jepa_embeddings, jepa_metadata)
    common_subjects = sorted(set(fc_map) & set(jepa_map))
    if not common_subjects:
        raise RuntimeError("No common subjects between FC and JEPA features for fusion site probe.")

    fused = np.stack(
        [
            np.concatenate(
                [
                    np.asarray(fc_map[subject_id]["feature"], dtype=np.float32),
                    np.asarray(jepa_map[subject_id]["feature"], dtype=np.float32),
                ],
                axis=0,
            )
            for subject_id in common_subjects
        ],
        axis=0,
    ).astype(np.float32)
    sites = np.asarray([fc_map[subject_id]["site"] for subject_id in common_subjects], dtype=object)
    groups = np.asarray(common_subjects, dtype=object)
    return fused, sites, groups


def build_method_labels(fc_methods: list[str]):
    labels = {}
    for method_key in fc_methods:
        if method_key == "fc":
            labels[method_key] = "Static FC + JEPA (fusion site probe)"
        elif method_key == "fc_combat":
            labels[method_key] = "Static FC + ComBat + JEPA (fusion site probe)"
        elif method_key == "fc_covbat":
            labels[method_key] = "Static FC + CovBat + JEPA (fusion site probe)"
    return labels


def build_markdown(payload: dict):
    lines = [
        f"# {payload['checkpoint']['label']} Fusion Site Probe",
        "",
        f"Checkpoint: `{payload['checkpoint']['path']}`",
        "",
        "Site probe uses grouped subject-level cross-validation with the same downstream probe family",
        "(`StandardScaler -> PCA -> LogisticRegression`).",
        "",
        "Fusion features are formed by concatenating subject-level static FC features with mean JEPA subject embeddings.",
        "",
        "| Dataset | Method | Site bAcc | Site ROC-AUC | Subjects |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for dataset_key in payload["order"]:
        dataset_payload = payload["datasets"][dataset_key]
        for method_key in dataset_payload["method_order"]:
            label = dataset_payload["method_labels"][method_key]
            summary = dataset_payload["summary"][method_key]
            lines.append(
                f"| {dataset_payload['config']['dataset_label']} | {label} | "
                f"{format_pct(summary['balanced_accuracy'])} | {format_pct(summary['roc_auc'])} | "
                f"{summary['n_samples']} |"
            )
    return "\n".join(lines) + "\n"


def evaluate_dataset(
    *,
    model,
    checkpoint_cfg: dict,
    checkpoint_path: str,
    dataset_key: str,
    task_arg: str,
    all_sites: bool,
    fc_methods: list[str],
    batch_size: int,
    seeds: list[int],
    folds: int,
    max_samples: int | None,
    device: str,
    window_feature: str,
    covariance_dim: int,
    covariance_seed: int,
    window_seconds: float | None,
    window_stride_seconds: float,
    target_timepoints: int | None,
    triangle: str,
    reorder: str,
    healthy_control_label: str,
    autism_label: str,
    cache_dir: Path,
    use_cache: bool,
    refresh_cache: bool,
):
    preset = get_dataset_preset(dataset_key)
    task_key = preset.default_task if task_arg == "default" else task_arg
    mixed_sites_only = False if all_sites else preset.default_mixed_sites_only

    records = matched_records_for_preset(preset)
    task_result = filter_records_for_task(
        records,
        task_key,
        mixed_sites_only=mixed_sites_only,
        healthy_control_label=healthy_control_label,
        autism_label=autism_label,
    )
    task_records, task_y = maybe_subsample_balanced(
        task_result.records,
        task_result.y,
        max_samples=max_samples,
        seed=seeds[0],
    )

    fc_metadata = build_sample_metadata(task_records, dataset_key)
    fc_feature_map, fc_cache_errors = load_or_build_fc_feature_cache(
        cache_dir=cache_dir,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        dataset_key=dataset_key,
        task_key=task_result.task_key,
        mixed_sites_only=mixed_sites_only,
        triangle=triangle,
        reorder=reorder,
        max_samples=max_samples,
        subsample_seed=seeds[0],
        task_records=task_records,
    )

    jepa_embeddings, jepa_metadata, _ = load_or_build_jepa_embedding_cache(
        cache_dir=cache_dir,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        model=model,
        preset=preset,
        checkpoint_cfg=checkpoint_cfg,
        checkpoint_path=checkpoint_path,
        task_records=task_records,
        dataset_key=dataset_key,
        task_key=task_result.task_key,
        mixed_sites_only=mixed_sites_only,
        task_y=task_y,
        device=device,
        batch_size=batch_size,
        window_feature=window_feature,
        covariance_dim=covariance_dim,
        covariance_seed=covariance_seed,
        window_seconds=window_seconds,
        window_stride_seconds=window_stride_seconds,
        target_timepoints=target_timepoints,
        max_samples=max_samples,
        subsample_seed=seeds[0],
    )

    method_labels = build_method_labels(fc_methods)
    runs = defaultdict(list)
    errors: dict[str, list[str]] = defaultdict(list)

    for method_key in fc_methods:
        if method_key in fc_cache_errors:
            errors[method_key].append(fc_cache_errors[method_key])
            continue
        fused_features, fused_sites, fused_groups = build_fusion_subject_matrix(
            fc_features=fc_feature_map[method_key],
            fc_metadata=fc_metadata,
            jepa_embeddings=jepa_embeddings,
            jepa_metadata=jepa_metadata,
        )
        site_encoder = LabelEncoder()
        y_site = site_encoder.fit_transform(fused_sites)
        site_class_names = [str(name) for name in site_encoder.classes_.tolist()]

        for seed in seeds:
            print(f"[fusion-site] dataset={dataset_key} method={method_key} seed={seed}", flush=True)
            try:
                runs[method_key].append(
                    run_grouped_cv(
                        fused_features,
                        y_site,
                        site_class_names,
                        fused_groups,
                        seed=seed,
                        folds=folds,
                    )
                )
            except Exception as exc:
                errors[method_key].append(f"{type(exc).__name__}: {exc}")
                break

    return {
        "config": {
            "dataset": preset.key,
            "dataset_label": preset.label,
            "task_key": task_result.task_key,
            "task_label": task_result.task_label,
            "mixed_sites_only": bool(mixed_sites_only),
            "triangle": triangle,
            "reorder": reorder,
            "fc_methods": list(fc_methods),
            "seeds": list(seeds),
            "folds": int(folds),
        },
        "method_order": list(fc_methods),
        "method_labels": {key: method_labels[key] for key in fc_methods},
        "summary": {
            method_key: summarize_runs(runs[method_key]) for method_key in fc_methods if runs[method_key]
        },
        "errors": {method_key: error_list for method_key, error_list in errors.items() if error_list},
    }


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    fc_methods = parse_method_list(args.fc_methods)
    cache_dir = Path(args.cache_dir)
    use_cache = not args.no_cache

    model, checkpoint, checkpoint_cfg = load_checkpoint(args.checkpoint)
    device = args.device or checkpoint_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset_order = default_eval_dataset_order(args.dataset)
    payload = {
        "checkpoint": {
            "path": args.checkpoint,
            "label": f"{Path(args.checkpoint).parent.name}/{Path(args.checkpoint).stem}",
        },
        "config": {
            "fc_methods": list(fc_methods),
            "seeds": list(seeds),
            "folds": int(args.folds),
        },
        "order": dataset_order,
        "datasets": {},
    }

    for dataset_key in dataset_order:
        payload["datasets"][dataset_key] = evaluate_dataset(
            model=model,
            checkpoint_cfg=checkpoint_cfg,
            checkpoint_path=args.checkpoint,
            dataset_key=dataset_key,
            task_arg=args.task,
            all_sites=args.all_sites,
            fc_methods=fc_methods,
            batch_size=args.batch_size,
            seeds=seeds,
            folds=args.folds,
            max_samples=args.max_samples,
            device=device,
            window_feature=args.window_feature,
            covariance_dim=args.covariance_dim,
            covariance_seed=args.covariance_seed,
            window_seconds=args.window_seconds,
            window_stride_seconds=args.window_stride_seconds,
            target_timepoints=args.target_timepoints,
            triangle=args.triangle,
            reorder=args.reorder,
            healthy_control_label=args.healthy_control_label,
            autism_label=args.autism_label,
            cache_dir=cache_dir,
            use_cache=use_cache,
            refresh_cache=args.refresh_cache,
        )

    output_json = Path(args.output_json) if args.output_json else default_output_paths(args.checkpoint)[0]
    output_md = Path(args.output_md) if args.output_md else default_output_paths(args.checkpoint)[1]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w") as f:
        json.dump(payload, f, indent=2)
    output_md.write_text(build_markdown(payload))


if __name__ == "__main__":
    main()
