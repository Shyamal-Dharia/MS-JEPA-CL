from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

try:
    from .model import build_model_from_config
    from .protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        batch_site_label,
        center_site_label,
        class_counts,
        dataset_order_for_choice,
        filter_records_for_task,
        get_dataset_preset,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.model import build_model_from_config
    from src.jepa_fusion.protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        batch_site_label,
        center_site_label,
        class_counts,
        dataset_order_for_choice,
        filter_records_for_task,
        get_dataset_preset,
    )
    from src.jepa_clean.data import WindowedFMRIDataset
    from src.jepa_clean.pooling import pool_sequence_features
    from src.jepa_clean.repro import seed_everything
else:
    from ..jepa_clean.data import WindowedFMRIDataset
    from ..jepa_clean.pooling import pool_sequence_features
    from ..jepa_clean.repro import seed_everything


@dataclass(frozen=True)
class EvalRecord:
    index: int
    subject: str
    site: str
    diagnosis: str


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate jepa_fusion checkpoints with grouped CV.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["abide", "abide1", "abide2", "srpbs", "all"], default="all")
    parser.add_argument("--task", choices=["default", "hc_asd", "hc_disorder"], default="default")
    parser.add_argument("--all-sites", action="store_true")
    parser.add_argument("--exclude-sites", default=None, help="Comma-separated center site labels to exclude from eval")
    parser.add_argument("--window-feature", choices=["mean", "mean_std", "mean_cov"], default="mean")
    parser.add_argument("--covariance-dim", type=int, default=32)
    parser.add_argument("--covariance-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--healthy-control-label", default=HEALTHY_CONTROL_LABEL)
    parser.add_argument("--autism-label", default=AUTISM_LABEL)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def parse_seed_list(seed_text: str):
    seeds = []
    for item in seed_text.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def mean_std(values: list[float]):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def _make_linear_probe(feature_dim: int, train_count: int, seed: int):
    n_components = min(50, feature_dim, max(1, train_count - 1))
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=seed),
        LogisticRegression(
            max_iter=10000,
            random_state=seed,
            class_weight="balanced",
        ),
    )


def _effective_group_folds(y: np.ndarray, groups: np.ndarray, requested_folds: int):
    per_class_group_counts = []
    for cls in np.unique(y):
        cls_groups = np.unique(groups[y == cls])
        per_class_group_counts.append(len(cls_groups))
    return min(requested_folds, min(per_class_group_counts))


def _balanced_accuracy_present(y_true: np.ndarray, y_pred: np.ndarray):
    recalls = []
    for cls in np.unique(y_true):
        cls_mask = y_true == cls
        recalls.append(float((y_pred[cls_mask] == cls).mean()))
    return float(np.mean(recalls))


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if scores.ndim == 1:
        return float(roc_auc_score(y_true, scores))
    return float(
        roc_auc_score(
            y_true,
            scores,
            labels=np.arange(scores.shape[1]),
            multi_class="ovr",
            average="macro",
        )
    )


def load_checkpoint(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    model = build_model_from_config(cfg)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing = {
        "global_predictor.0.weight",
        "global_predictor.0.bias",
        "global_predictor.1.weight",
        "global_predictor.1.bias",
        "global_predictor.1.running_mean",
        "global_predictor.1.running_var",
        "global_predictor.1.num_batches_tracked",
        "global_predictor.3.weight",
        "global_predictor.3.bias",
    }
    missing = set(incompatible.missing_keys)
    if missing - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint incompatibility: "
            f"missing={sorted(missing)} unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return model, checkpoint, cfg


def collate_fn(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), list(labels)


@torch.no_grad()
def extract_scan_embeddings(
    model,
    dataset,
    device,
    *,
    batch_size: int,
    window_feature: str,
    covariance_dim: int,
    covariance_seed: int,
):
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    embeddings = []
    metadata = []
    for x, labels in loader:
        x = x.to(device)
        sequence = model.encode_sequence(x)
        pooled = pool_sequence_features(
            sequence,
            mode=window_feature,
            covariance_dim=covariance_dim,
            covariance_seed=covariance_seed,
        )
        embeddings.append(pooled.cpu().numpy())
        metadata.extend(labels)
    return np.concatenate(embeddings, axis=0), metadata


def run_grouped_cv(
    features: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    groups: np.ndarray,
    *,
    seed: int,
    folds: int,
):
    effective_folds = _effective_group_folds(y, groups, folds)
    if effective_folds < 2:
        raise RuntimeError(f"Need at least 2 effective folds, got {effective_folds}.")

    cv = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    preds = np.empty_like(y)
    n_classes = len(class_names)
    oof_scores = np.full((len(y), n_classes), 0.0, dtype=np.float64)
    fold_rows = []

    sample_indices = np.arange(len(y), dtype=np.int64)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(sample_indices, y, groups=groups), start=1):
        x_train = features[train_idx]
        x_test = features[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        clf = _make_linear_probe(feature_dim=x_train.shape[1], train_count=len(train_idx), seed=seed)
        clf.fit(x_train, y_train)
        fold_pred = clf.predict(x_test)
        preds[test_idx] = fold_pred

        clf_classes = clf.named_steps["logisticregression"].classes_
        fold_scores = clf.predict_proba(x_test)
        aligned_scores = np.zeros((len(test_idx), n_classes), dtype=np.float64)
        aligned_scores[:, clf_classes] = fold_scores
        oof_scores[test_idx] = aligned_scores

        fold_row = {
            "fold": int(fold_idx),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "accuracy": float(accuracy_score(y_test, fold_pred)),
            "balanced_accuracy": _balanced_accuracy_present(y_test, fold_pred),
        }
        if n_classes == 2:
            fold_row["roc_auc"] = _safe_roc_auc(y_test, aligned_scores[:, 1])
        else:
            fold_row["roc_auc"] = _safe_roc_auc(y_test, aligned_scores)
        fold_rows.append(fold_row)

    result = {
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": _balanced_accuracy_present(y, preds),
        "folds": int(effective_folds),
        "fold_rows": fold_rows,
        "n_samples": int(len(y)),
        "classes": list(class_names),
    }
    if n_classes == 2:
        result["roc_auc"] = _safe_roc_auc(y, oof_scores[:, 1])
    else:
        result["roc_auc"] = _safe_roc_auc(y, oof_scores)
    return result


def summarize_runs(runs: list[dict]):
    return {
        "accuracy": mean_std([run["accuracy"] for run in runs]),
        "balanced_accuracy": mean_std([run["balanced_accuracy"] for run in runs]),
        "roc_auc": mean_std([run["roc_auc"] for run in runs]),
        "n_runs": int(len(runs)),
        "n_samples": int(runs[0]["n_samples"]),
        "folds": int(runs[0]["folds"]),
        "classes": list(runs[0]["classes"]),
    }


def format_pct(metric: dict):
    return f"{100.0 * metric['mean']:.1f} ± {100.0 * metric['std']:.1f}"


def build_markdown(payload: dict):
    lines = [
        f"# {payload['checkpoint']['label']} JEPA Evaluation",
        "",
        f"Checkpoint: `{payload['checkpoint']['path']}`",
        "",
        "| Dataset | Site Probe bAcc | Site Probe ROC-AUC | Diagnosis bAcc | Diagnosis ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset_key in payload["order"]:
        summary = payload["datasets"][dataset_key]["summary"]
        config = payload["datasets"][dataset_key]["config"]
        lines.append(
            f"| {config['dataset_label']} | "
            f"{format_pct(summary['site']['balanced_accuracy'])} | "
            f"{format_pct(summary['site']['roc_auc'])} | "
            f"{format_pct(summary['diagnosis']['balanced_accuracy'])} | "
            f"{format_pct(summary['diagnosis']['roc_auc'])} |"
        )
    return "\n".join(lines) + "\n"


def default_output_paths(checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    stem = f"{checkpoint.parent.name}_{checkpoint.stem}"
    base_dir = Path("results/reports/tables/jepa_fusion")
    return base_dir / f"{stem}_jepa_eval.json", base_dir / f"{stem}_jepa_eval.md"


def evaluate_dataset(
    *,
    model,
    checkpoint_cfg: dict,
    dataset_key: str,
    task_arg: str,
    all_sites: bool,
    batch_size: int,
    folds: int,
    seeds: list[int],
    max_scans: int | None,
    device: str,
    window_feature: str,
    covariance_dim: int,
    covariance_seed: int,
    healthy_control_label: str,
    autism_label: str,
    exclude_sites: set[str] | None = None,
):
    preset = get_dataset_preset(dataset_key)
    task_key = preset.default_task if task_arg == "default" else task_arg
    mixed_sites_only = False if all_sites else preset.default_mixed_sites_only

    dataset = WindowedFMRIDataset(
        preset.raw_dir,
        window_seconds=float(checkpoint_cfg.get("window_seconds", 90.0)),
        stride_seconds=float(checkpoint_cfg.get("window_stride_seconds", 30.0)),
        target_timepoints=int(checkpoint_cfg.get("target_timepoints", 45)),
        sampling="center",
        max_scans=max_scans,
        exclude_unknown_diagnoses=False,
    )
    embeddings, metadata = extract_scan_embeddings(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        window_feature=window_feature,
        covariance_dim=covariance_dim,
        covariance_seed=covariance_seed,
    )

    eval_records = [
        EvalRecord(
            index=idx,
            subject=str(meta["subject_id"]),
            site=center_site_label(str(meta.get("site_raw", meta["site"]))),
            diagnosis=str(meta["diag_name"]),
        )
        for idx, meta in enumerate(metadata)
    ]
    task_result = filter_records_for_task(
        eval_records,
        task_key,
        mixed_sites_only=mixed_sites_only,
        healthy_control_label=healthy_control_label,
        autism_label=autism_label,
    )
    # Optionally exclude sites (for transfer experiments)
    if exclude_sites:
        keep_mask = [record.site not in exclude_sites for record in task_result.records]
        task_result_records = [r for r, k in zip(task_result.records, keep_mask) if k]
        task_result_y = task_result.y[keep_mask]
        n_excluded = sum(1 for k in keep_mask if not k)
        if n_excluded > 0:
            print(f"  Excluded {n_excluded} records from sites: {sorted(exclude_sites)}")
    else:
        task_result_records = list(task_result.records)
        task_result_y = task_result.y

    keep_indices = np.asarray([record.index for record in task_result_records], dtype=np.int64)
    task_embeddings = embeddings[keep_indices]
    task_y = task_result_y
    subjects = np.asarray([task_result_records[idx].subject for idx in range(len(task_result_records))], dtype=object)
    batch_sites = np.asarray(
        [
            batch_site_label(
                str(metadata[idx].get("site_raw", metadata[idx]["site"])),
                source_dataset=str(metadata[idx].get("dataset", "")),
            )
            for idx in keep_indices.tolist()
        ],
        dtype=object,
    )

    # Track exclusion metadata
    n_excluded_sites = len(exclude_sites) if exclude_sites else 0

    site_encoder = LabelEncoder()
    y_site = site_encoder.fit_transform(batch_sites)
    site_class_names = [str(name) for name in site_encoder.classes_.tolist()]

    site_runs = []
    diagnosis_runs = []
    for seed in seeds:
        site_runs.append(
            run_grouped_cv(
                task_embeddings,
                y_site,
                site_class_names,
                subjects,
                seed=seed,
                folds=folds,
            )
        )
        diagnosis_runs.append(
            run_grouped_cv(
                task_embeddings,
                task_y,
                task_result.class_names,
                subjects,
                seed=seed,
                folds=folds,
            )
        )

    return {
        "config": {
            "dataset": preset.key,
            "dataset_label": preset.label,
            "raw_dir": preset.raw_dir,
            "task_key": task_result.task_key,
            "task_label": task_result.task_label,
            "mixed_sites_only": bool(mixed_sites_only),
            "folds": int(folds),
            "seeds": list(seeds),
            "window_feature": window_feature,
            "covariance_dim": int(covariance_dim),
            "covariance_seed": int(covariance_seed),
        },
        "meta": {
            "n_records_total": int(len(dataset.records)),
            "n_records_post_task_filter": int(len(task_result.records)),
            "removed_unknown": int(task_result.removed_unknown),
            "removed_task_mismatch": int(task_result.removed_task_mismatch),
            "removed_single_class_sites": int(task_result.removed_single_class_sites),
            "n_sites": int(len(site_class_names)),
            "site_classes": list(site_class_names),
            "diagnosis_class_counts": class_counts(task_y, task_result.class_names),
            "excluded_sites": sorted(exclude_sites) if exclude_sites else [],
        },
        "runs": {
            "site": site_runs,
            "diagnosis": diagnosis_runs,
        },
        "summary": {
            "site": summarize_runs(site_runs),
            "diagnosis": summarize_runs(diagnosis_runs),
        },
    }


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    seed_everything(seeds[0], deterministic=True)
    model, checkpoint, cfg = load_checkpoint(args.checkpoint)
    device = args.device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset_order = dataset_order_for_choice(args.dataset)
    payload = {
        "checkpoint": {
            "path": str(Path(args.checkpoint)),
            "label": f"{Path(args.checkpoint).parent.name}/{Path(args.checkpoint).stem}",
            "config": cfg,
        },
        "order": list(dataset_order),
        "datasets": {},
    }

    for dataset_key in dataset_order:
        payload["datasets"][dataset_key] = evaluate_dataset(
            model=model,
            checkpoint_cfg=cfg,
            dataset_key=dataset_key,
            task_arg=args.task,
            all_sites=args.all_sites,
            batch_size=args.batch_size,
            folds=args.folds,
            seeds=seeds,
            max_scans=args.max_scans,
            device=device,
            window_feature=args.window_feature,
            covariance_dim=args.covariance_dim,
            covariance_seed=args.covariance_seed,
            healthy_control_label=args.healthy_control_label,
            autism_label=args.autism_label,
            exclude_sites=set(args.exclude_sites.split(",")) if args.exclude_sites else None,
        )

    output_json, output_md = default_output_paths(args.checkpoint)
    if args.output_json is not None:
        output_json = Path(args.output_json)
    if args.output_md is not None:
        output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump(payload, handle, indent=2)
    output_md.write_text(build_markdown(payload))
    print(f"Saved JSON: {output_json}")
    print(f"Saved Markdown: {output_md}")


if __name__ == "__main__":
    main()
