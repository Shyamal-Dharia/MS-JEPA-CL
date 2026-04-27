from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import json
import sys

import numpy as np
import torch
from torch.utils.data import Subset

try:
    from .dataset_utils import default_eval_dataset_order, matched_records_for_preset
    from .protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        batch_site_label,
        center_site_label,
        filter_records_for_task,
        get_dataset_preset,
    )
    from .evaluate_jepa_subject_vote import (
        add_site_std_to_summary,
        aggregate_subject_predictions,
        attach_labels,
        build_probe_classifier,
        compute_binary_metrics,
        compute_site_level_metrics,
        extract_window_embeddings,
        get_probe_scores,
        load_checkpoint,
    )
    from .evaluate_baselines import format_pct
    from ..jepa_clean.data import WindowedFMRIDataset
    from ..jepa_clean.plot_tsne_fc_vs_raw import (
        apply_combat_harmonization,
        apply_covbat_harmonization,
        build_fc_features,
        get_roi_permutation,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.dataset_utils import default_eval_dataset_order, matched_records_for_preset
    from src.jepa_fusion.protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        batch_site_label,
        center_site_label,
        filter_records_for_task,
        get_dataset_preset,
    )
    from src.jepa_fusion.evaluate_jepa_subject_vote import (
        add_site_std_to_summary,
        aggregate_subject_predictions,
        attach_labels,
        build_probe_classifier,
        compute_binary_metrics,
        compute_site_level_metrics,
        extract_window_embeddings,
        get_probe_scores,
        load_checkpoint,
    )
    from src.jepa_fusion.evaluate_baselines import format_pct
    from src.jepa_clean.data import WindowedFMRIDataset
    from src.jepa_clean.plot_tsne_fc_vs_raw import (
        apply_combat_harmonization,
        apply_covbat_harmonization,
        build_fc_features,
        get_roi_permutation,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate subject-level LOSO late fusion between static FC branches and a "
            "JEPA checkpoint on matched FC/raw scans."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["abide", "abide1", "abide2", "srpbs", "all"], default="all")
    parser.add_argument("--task", choices=["default", "hc_asd", "hc_disorder"], default="default")
    parser.add_argument("--all-sites", action="store_true")
    parser.add_argument("--triangle", choices=["upper", "lower"], default="upper")
    parser.add_argument("--reorder", choices=["none", "hemi", "hemi_network"], default="hemi_network")
    parser.add_argument("--fc-methods", default="fc,fc_combat,fc_covbat")
    parser.add_argument(
        "--fc-weight",
        type=float,
        default=0.5,
        help="Late-fusion probability weight on the FC branch. JEPA gets (1 - fc_weight).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-feature", choices=["mean", "mean_std", "mean_cov"], default="mean")
    parser.add_argument("--covariance-dim", type=int, default=32)
    parser.add_argument("--covariance-seed", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=30.0)
    parser.add_argument("--target-timepoints", type=int, default=None)
    parser.add_argument("--seeds", default="42")
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


def parse_seed_list(seed_text: str):
    seeds = []
    for item in seed_text.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_method_list(method_text: str):
    allowed = {"fc", "fc_combat", "fc_covbat"}
    methods = []
    for item in method_text.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"Unsupported FC method: {item}")
        methods.append(item)
    if not methods:
        raise ValueError("At least one FC method is required.")
    return methods


def mean_std(values: list[float]):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def format_scalar_pct(value: float):
    return f"{100.0 * value:.1f}"


def cache_key(prefix: str, payload: dict):
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def dataset_subject_prefix(dataset_key: str):
    mapping = {
        "abide1": "abide_i",
        "abide2": "abide_ii",
        "srpbs": "srpbs_1600",
    }
    if dataset_key not in mapping:
        raise KeyError(f"Unknown dataset key: {dataset_key}")
    return mapping[dataset_key]


def maybe_subsample_balanced(records: list, y: np.ndarray, max_samples: int | None, seed: int):
    if max_samples is None or len(records) <= max_samples:
        return records, y

    rng = np.random.default_rng(seed)
    indices_by_class = {int(cls): np.flatnonzero(y == cls) for cls in np.unique(y)}
    target_total = min(max_samples, len(records))

    selected = []
    selected_set: set[int] = set()
    for cls in sorted(indices_by_class):
        cls_indices = np.array(indices_by_class[cls], copy=True)
        rng.shuffle(cls_indices)
        quota = max(1, int(round(target_total * len(cls_indices) / len(records))))
        cls_take = cls_indices[: min(quota, len(cls_indices))]
        selected.extend(cls_take.tolist())
        selected_set.update(int(idx) for idx in cls_take.tolist())

    if len(selected) < target_total:
        remaining = [idx for idx in range(len(records)) if idx not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_total - len(selected)])
    elif len(selected) > target_total:
        rng.shuffle(selected)
        selected = selected[:target_total]

    selected = sorted(set(int(idx) for idx in selected))
    return [records[idx] for idx in selected], y[np.asarray(selected, dtype=np.int64)]


def summarize_subject_rows(subject_rows: list[dict]):
    if not subject_rows:
        return {
            "bacc": float("nan"),
            "auc": float("nan"),
            "site_bacc_mean": float("nan"),
            "site_bacc_std": float("nan"),
            "site_auc_mean": float("nan"),
            "site_auc_std": float("nan"),
            "n_subjects": 0,
            "n_valid_sites": 0,
        }

    y_true = np.asarray([row["y"] for row in subject_rows], dtype=int)
    y_pred = np.asarray([row["pred"] for row in subject_rows], dtype=int)
    y_prob = np.asarray([row["prob"] for row in subject_rows], dtype=float)
    summary = compute_binary_metrics(y_true, y_pred, y_prob)
    summary["n_subjects"] = int(len(subject_rows))
    site_metrics = compute_site_level_metrics(subject_rows)
    summary["n_valid_sites"] = int(len(site_metrics))
    add_site_std_to_summary(summary, site_metrics)
    return summary


def summarize_runs(runs: list[dict]):
    return {
        "bacc": mean_std([run["bacc"] for run in runs]),
        "auc": mean_std([run["auc"] for run in runs]),
        "site_bacc_mean": mean_std([run["site_bacc_mean"] for run in runs]),
        "n_runs": int(len(runs)),
        "n_subjects": int(runs[0]["n_subjects"]),
        "n_valid_sites": int(runs[0]["n_valid_sites"]),
    }


def build_sample_metadata(records: list, dataset_key: str):
    prefix = dataset_subject_prefix(dataset_key)
    return [
        {
            "subject_id": f"{prefix}:{record.subject}",
            "site": str(record.site),
            "site_batch": str(record.site_batch),
            "scan_id": str(record.sample_id),
        }
        for record in records
    ]


def compute_fc_subject_rows_for_site(
    *,
    features: np.ndarray,
    metadata: list[dict],
    y: np.ndarray,
    held_out_site: str,
    seed: int,
    method_key: str,
):
    sites = np.asarray([str(meta["site"]) for meta in metadata], dtype=object)
    train_mask = sites != held_out_site
    test_mask = sites == held_out_site
    if not np.any(test_mask):
        return []

    y_train = y[train_mask]
    y_test = y[test_mask]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return []

    x_train = features[train_mask]
    x_test = features[test_mask]
    clf = build_probe_classifier(seed, y_train)
    clf.fit(x_train, y_train)
    test_scores = get_probe_scores(clf, x_test)
    test_metadata = attach_labels([metadata[idx] for idx in np.flatnonzero(test_mask)], y_test)
    return aggregate_subject_predictions(test_metadata, test_scores, classifier="logreg", threshold=0.5)


def compute_jepa_subject_rows_for_site(
    *,
    embeddings: np.ndarray,
    metadata: list[dict],
    y: np.ndarray,
    held_out_site: str,
    seed: int,
):
    sites = np.asarray([str(meta["site"]) for meta in metadata], dtype=object)
    train_mask = sites != held_out_site
    test_mask = sites == held_out_site
    if not np.any(test_mask):
        return []

    y_train = y[train_mask]
    y_test = y[test_mask]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return []

    clf = build_probe_classifier(seed, y_train)
    clf.fit(embeddings[train_mask], y_train)
    test_scores = get_probe_scores(clf, embeddings[test_mask])
    test_metadata = attach_labels([metadata[idx] for idx in np.flatnonzero(test_mask)], y_test)
    return aggregate_subject_predictions(test_metadata, test_scores, classifier="logreg", threshold=0.5)


def fuse_subject_rows(fc_rows: list[dict], jepa_rows: list[dict], fc_weight: float):
    fc_by_subject = {str(row["subject_id"]): row for row in fc_rows}
    jepa_by_subject = {str(row["subject_id"]): row for row in jepa_rows}
    common_subjects = sorted(set(fc_by_subject) & set(jepa_by_subject))

    fused_rows = []
    for subject_id in common_subjects:
        fc_row = fc_by_subject[subject_id]
        jepa_row = jepa_by_subject[subject_id]
        fused_prob = float(fc_weight) * float(fc_row["prob"]) + (1.0 - float(fc_weight)) * float(jepa_row["prob"])
        fused_rows.append(
            {
                "subject_id": subject_id,
                "site": fc_row["site"],
                "y": int(fc_row["y"]),
                "pred": int(fused_prob >= 0.5),
                "prob": fused_prob,
                "n_windows": int(jepa_row.get("n_windows", 0)),
                "fc_prob": float(fc_row["prob"]),
                "jepa_prob": float(jepa_row["prob"]),
            }
        )
    return fused_rows


def extract_matched_jepa_windows(
    *,
    model,
    preset,
    checkpoint_cfg: dict,
    task_records: list,
    dataset_key: str,
    task_y: np.ndarray,
    device: str,
    batch_size: int,
    window_feature: str,
    covariance_dim: int,
    covariance_seed: int,
    window_seconds: float | None,
    window_stride_seconds: float,
    target_timepoints: int | None,
):
    dataset = WindowedFMRIDataset(
        preset.raw_dir,
        window_seconds=float(window_seconds or checkpoint_cfg.get("window_seconds", 90.0)),
        stride_seconds=float(window_stride_seconds),
        target_timepoints=int(target_timepoints or checkpoint_cfg.get("target_timepoints", 45)),
        sampling="sliding",
        exclude_unknown_diagnoses=False,
    )
    allowed_scan_ids = {str(record.sample_id) for record in task_records}
    keep_indices = [
        idx
        for idx, window_record in enumerate(dataset.window_records or [])
        if str(window_record.record.scan_id) in allowed_scan_ids
    ]
    if not keep_indices:
        raise RuntimeError(f"No matched windows found for dataset {dataset_key}.")

    subset = Subset(dataset, keep_indices)
    embeddings, metadata = extract_window_embeddings(
        model,
        subset,
        device=device,
        batch_size=batch_size,
        window_feature=window_feature,
        covariance_dim=covariance_dim,
        covariance_seed=covariance_seed,
    )
    labels_by_subject = {str(record.subject_id): int(label) for record, label in zip(task_records, task_y.tolist())}

    filtered_embeddings = []
    filtered_metadata = []
    filtered_y = []
    for emb, meta in zip(embeddings, metadata):
        subject_id = str(meta["subject_id"])
        if subject_id not in labels_by_subject:
            continue
        filtered_embeddings.append(emb)
        filtered_metadata.append(
            {
                "subject_id": subject_id,
                "site": center_site_label(str(meta.get("site_raw", meta["site"]))),
                "site_batch": batch_site_label(
                    str(meta.get("site_raw", meta["site"])),
                    source_dataset=str(meta.get("dataset", "")),
                ),
                "scan_id": str(meta["scan_id"]),
            }
        )
        filtered_y.append(labels_by_subject[subject_id])

    return (
        np.asarray(filtered_embeddings, dtype=np.float32),
        filtered_metadata,
        np.asarray(filtered_y, dtype=np.int64),
    )


def transductive_harmonize_fc(features: np.ndarray, sites: list[str], method_key: str):
    if method_key == "fc":
        return np.asarray(features, dtype=np.float32)

    features = np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite_var = np.var(features, axis=0)
    varying_mask = np.isfinite(finite_var) & (finite_var > 1e-8)
    if not np.any(varying_mask):
        return np.asarray(features, dtype=np.float32)

    harmonized = np.array(features, copy=True)
    if method_key == "fc_combat":
        harmonized[:, varying_mask] = apply_combat_harmonization(features[:, varying_mask], sites)
    elif method_key == "fc_covbat":
        harmonized[:, varying_mask] = apply_covbat_harmonization(
            features[:, varying_mask],
            sites,
            n_pc=50,
            pct_var=0.95,
        )
    else:
        raise ValueError(f"Unsupported FC method: {method_key}")
    return np.nan_to_num(np.asarray(harmonized, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def load_or_build_fc_feature_cache(
    *,
    cache_dir: Path,
    use_cache: bool,
    refresh_cache: bool,
    dataset_key: str,
    task_key: str,
    mixed_sites_only: bool,
    triangle: str,
    reorder: str,
    max_samples: int | None,
    subsample_seed: int,
    task_records: list,
):
    cache_payload = {
        "dataset": dataset_key,
        "task": task_key,
        "mixed_sites_only": bool(mixed_sites_only),
        "triangle": triangle,
        "reorder": reorder,
        "max_samples": None if max_samples is None else int(max_samples),
        "subsample_seed": int(subsample_seed),
        "n_records": int(len(task_records)),
    }
    stem = cache_key("fc_subject_vote", cache_payload)
    npz_path = cache_dir / f"{stem}.npz"
    meta_path = cache_dir / f"{stem}.json"

    if use_cache and not refresh_cache and npz_path.exists() and meta_path.exists():
        print(f"[fusion] loading FC cache: {npz_path}", flush=True)
        with np.load(npz_path) as data:
            feature_map = {
                "fc": np.asarray(data["fc"], dtype=np.float32),
                "fc_combat": np.asarray(data["fc_combat"], dtype=np.float32),
            }
            if "fc_covbat" in data.files:
                feature_map["fc_covbat"] = np.asarray(data["fc_covbat"], dtype=np.float32)
        with meta_path.open() as f:
            meta = json.load(f)
        return feature_map, meta.get("errors", {})

    print(f"[fusion] building FC cache for dataset={dataset_key} task={task_key}", flush=True)
    permutation, _ = get_roi_permutation(reorder)
    fc_features = build_fc_features(task_records, permutation=permutation, triangle=triangle)
    sites = [str(record.site_batch) for record in task_records]
    feature_map = {"fc": np.asarray(fc_features, dtype=np.float32)}
    errors: dict[str, str] = {}

    for method_key in ("fc_combat", "fc_covbat"):
        try:
            feature_map[method_key] = transductive_harmonize_fc(fc_features, sites, method_key)
        except Exception as exc:
            errors[method_key] = f"{type(exc).__name__}: {exc}"

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_kwargs = {key: value for key, value in feature_map.items()}
        np.savez_compressed(npz_path, **save_kwargs)
        with meta_path.open("w") as f:
            json.dump({"cache_payload": cache_payload, "errors": errors}, f, indent=2)
        print(f"[fusion] saved FC cache: {npz_path}", flush=True)

    return feature_map, errors


def load_or_build_jepa_embedding_cache(
    *,
    cache_dir: Path,
    use_cache: bool,
    refresh_cache: bool,
    model,
    preset,
    checkpoint_cfg: dict,
    checkpoint_path: str,
    task_records: list,
    dataset_key: str,
    task_key: str,
    mixed_sites_only: bool,
    task_y: np.ndarray,
    device: str,
    batch_size: int,
    window_feature: str,
    covariance_dim: int,
    covariance_seed: int,
    window_seconds: float | None,
    window_stride_seconds: float,
    target_timepoints: int | None,
    max_samples: int | None,
    subsample_seed: int,
):
    cache_payload = {
        "checkpoint": str(Path(checkpoint_path)),
        "dataset": dataset_key,
        "task": task_key,
        "mixed_sites_only": bool(mixed_sites_only),
        "window_feature": window_feature,
        "covariance_dim": int(covariance_dim),
        "covariance_seed": int(covariance_seed),
        "window_seconds": float(window_seconds or checkpoint_cfg.get("window_seconds", 90.0)),
        "window_stride_seconds": float(window_stride_seconds),
        "target_timepoints": int(target_timepoints or checkpoint_cfg.get("target_timepoints", 45)),
        "max_samples": None if max_samples is None else int(max_samples),
        "subsample_seed": int(subsample_seed),
        "n_records": int(len(task_records)),
    }
    stem = cache_key("jepa_subject_vote", cache_payload)
    npz_path = cache_dir / f"{stem}.npz"
    meta_path = cache_dir / f"{stem}.json"

    if use_cache and not refresh_cache and npz_path.exists() and meta_path.exists():
        print(f"[fusion] loading JEPA cache: {npz_path}", flush=True)
        with np.load(npz_path) as data:
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.int64)
            metadata = [
                {
                    "subject_id": str(subject_id),
                    "site": str(site),
                    "site_batch": str(site_batch),
                    "scan_id": str(scan_id),
                }
                for subject_id, site, site_batch, scan_id in zip(
                    data["subject_id"].astype(str).tolist(),
                    data["site"].astype(str).tolist(),
                    (data["site_batch"].astype(str).tolist() if "site_batch" in data.files else data["site"].astype(str).tolist()),
                    data["scan_id"].astype(str).tolist(),
                )
            ]
        return embeddings, metadata, y

    print(f"[fusion] extracting JEPA embeddings for dataset={dataset_key} task={task_key}", flush=True)
    embeddings, metadata, y = extract_matched_jepa_windows(
        model=model,
        preset=preset,
        checkpoint_cfg=checkpoint_cfg,
        task_records=task_records,
        dataset_key=dataset_key,
        task_y=task_y,
        device=device,
        batch_size=batch_size,
        window_feature=window_feature,
        covariance_dim=covariance_dim,
        covariance_seed=covariance_seed,
        window_seconds=window_seconds,
        window_stride_seconds=window_stride_seconds,
        target_timepoints=target_timepoints,
    )

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            y=np.asarray(y, dtype=np.int64),
            subject_id=np.asarray([meta["subject_id"] for meta in metadata], dtype=str),
            site=np.asarray([meta["site"] for meta in metadata], dtype=str),
            site_batch=np.asarray([meta.get("site_batch", meta["site"]) for meta in metadata], dtype=str),
            scan_id=np.asarray([meta["scan_id"] for meta in metadata], dtype=str),
        )
        with meta_path.open("w") as f:
            json.dump({"cache_payload": cache_payload}, f, indent=2)
        print(f"[fusion] saved JEPA cache: {npz_path}", flush=True)

    return embeddings, metadata, y


def build_method_labels(fc_methods: list[str], fc_weight: float, checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    jepa_label = f"JEPA ({checkpoint.parent.name}/{checkpoint.stem})"
    labels = {
        "jepa": jepa_label,
        "fc": "Static FC",
        "fc_combat": "Static FC + ComBat (transductive LOSO)",
        "fc_covbat": "Static FC + CovBat (transductive LOSO)",
    }
    for method_key in fc_methods:
        if method_key == "fc":
            labels[f"fusion::{method_key}"] = f"Static FC + JEPA (late fusion, fc_weight={fc_weight:.2f})"
        elif method_key == "fc_combat":
            labels[f"fusion::{method_key}"] = (
                f"Static FC + ComBat + JEPA (late fusion, fc_weight={fc_weight:.2f}; transductive LOSO)"
            )
        elif method_key == "fc_covbat":
            labels[f"fusion::{method_key}"] = (
                f"Static FC + CovBat + JEPA (late fusion, fc_weight={fc_weight:.2f}; transductive LOSO)"
            )
    return labels


def default_output_paths(checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    stem = f"{checkpoint.parent.name}_{checkpoint.stem}_fusion_subject_vote"
    base_dir = Path("results/reports/tables/jepa_fusion")
    return base_dir / f"{stem}.json", base_dir / f"{stem}.md"


def build_markdown(payload: dict):
    lines = [
        f"# {payload['checkpoint']['label']} FC+JEPA Subject-Vote Fusion",
        "",
        f"Checkpoint: `{payload['checkpoint']['path']}`",
        "",
        "Diagnosis uses leave-one-center-out subject-level evaluation on matched FC/raw scans.",
        "FC-only and FC+JEPA rows use the same subject-level LOSO protocol.",
        "",
        f"Late fusion uses fixed probabilities: `p = {payload['config']['fc_weight']:.2f} * p_fc + "
        f"{1.0 - payload['config']['fc_weight']:.2f} * p_jepa`.",
        "",
        "| Method | Dx LOSO bAcc | Dx LOSO ROC-AUC | Mean Per-Site Dx bAcc | Valid Sites |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset_key in payload["order"]:
        dataset_payload = payload["datasets"][dataset_key]
        lines.append("")
        lines.append(f"## {dataset_payload['config']['dataset_label']}")
        lines.append("")
        for method_key in dataset_payload["method_order"]:
            label = dataset_payload["method_labels"][method_key]
            summary = dataset_payload["summary"][method_key]
            lines.append(
                f"| {label} | {format_pct(summary['bacc'])} | {format_pct(summary['auc'])} | "
                f"{format_pct(summary['site_bacc_mean'])} | {summary['n_valid_sites']} |"
            )
        lines.append(
            f"| Chance | 50.0 | 50.0 | 50.0 | {dataset_payload['meta']['n_sites']} |"
        )
    return "\n".join(lines) + "\n"


def evaluate_dataset(
    *,
    model,
    checkpoint_cfg: dict,
    dataset_key: str,
    task_arg: str,
    all_sites: bool,
    fc_methods: list[str],
    fc_weight: float,
    batch_size: int,
    seeds: list[int],
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
    checkpoint_path: str,
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
    sites = sorted({str(meta["site"]) for meta in fc_metadata})
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

    jepa_embeddings, jepa_metadata, jepa_y = load_or_build_jepa_embedding_cache(
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

    runs = defaultdict(list)
    run_rows = defaultdict(list)
    method_labels = build_method_labels(fc_methods, fc_weight, checkpoint_path)
    method_order = ["jepa"] + fc_methods + [f"fusion::{method_key}" for method_key in fc_methods]
    method_errors: dict[str, list[str]] = defaultdict(list)
    for method_key, error_text in fc_cache_errors.items():
        method_errors[method_key].append(error_text)
        method_errors[f"fusion::{method_key}"].append(error_text)

    for seed in seeds:
        print(f"[fusion] dataset={dataset_key} seed={seed} sites={len(sites)}", flush=True)
        seed_subject_rows = {key: [] for key in method_order}
        seed_method_errors: dict[str, str] = dict(fc_cache_errors)
        for held_out_site in sites:
            print(f"[fusion] dataset={dataset_key} seed={seed} held_out_site={held_out_site}", flush=True)
            jepa_rows = compute_jepa_subject_rows_for_site(
                embeddings=jepa_embeddings,
                metadata=jepa_metadata,
                y=jepa_y,
                held_out_site=held_out_site,
                seed=seed,
            )
            if jepa_rows:
                seed_subject_rows["jepa"].extend(jepa_rows)

            for method_key in fc_methods:
                if method_key in seed_method_errors:
                    continue
                try:
                    fc_rows = compute_fc_subject_rows_for_site(
                        features=fc_feature_map[method_key],
                        metadata=fc_metadata,
                        y=task_y,
                        held_out_site=held_out_site,
                        seed=seed,
                        method_key=method_key,
                    )
                except Exception as exc:
                    seed_method_errors[method_key] = f"{type(exc).__name__}: {exc}"
                    continue
                if fc_rows:
                    seed_subject_rows[method_key].extend(fc_rows)
                if fc_rows and jepa_rows:
                    fused_rows = fuse_subject_rows(fc_rows, jepa_rows, fc_weight=fc_weight)
                    seed_subject_rows[f"fusion::{method_key}"].extend(fused_rows)

        for method_key in method_order:
            summary = summarize_subject_rows(seed_subject_rows[method_key])
            if method_key in seed_method_errors:
                summary["error"] = seed_method_errors[method_key]
                method_errors[method_key].append(seed_method_errors[method_key])
            elif method_key.startswith("fusion::"):
                source_method = method_key.split("::", 1)[1]
                if source_method in seed_method_errors:
                    summary["error"] = seed_method_errors[source_method]
                    method_errors[method_key].append(seed_method_errors[source_method])
            runs[method_key].append(summary)
            run_rows[method_key].append(
                {
                    "seed": int(seed),
                    "summary": summary,
                    "subject_rows": seed_subject_rows[method_key],
                }
            )

    return {
        "config": {
            "dataset": preset.key,
            "dataset_label": preset.label,
            "task_key": task_result.task_key,
            "task_label": task_result.task_label,
            "mixed_sites_only": bool(mixed_sites_only),
            "triangle": triangle,
            "reorder": reorder,
            "fc_weight": float(fc_weight),
            "fc_methods": list(fc_methods),
            "seeds": list(seeds),
            "window_feature": window_feature,
        },
        "meta": {
            "n_records_used": int(len(task_records)),
            "n_sites": int(len(sites)),
            "n_subjects_fc": int(len({meta["subject_id"] for meta in fc_metadata})),
            "n_subjects_jepa": int(len({meta["subject_id"] for meta in jepa_metadata})),
        },
        "method_labels": {key: method_labels[key] for key in method_order},
        "method_order": method_order,
        "summary": {method_key: summarize_runs(runs[method_key]) for method_key in method_order},
        "errors": {method_key: errors for method_key, errors in method_errors.items() if errors},
        "runs": dict(run_rows),
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
            "fc_weight": float(args.fc_weight),
            "fc_methods": list(fc_methods),
            "seeds": list(seeds),
        },
        "order": dataset_order,
        "datasets": {},
    }

    for dataset_key in dataset_order:
        payload["datasets"][dataset_key] = evaluate_dataset(
            model=model,
            checkpoint_cfg=checkpoint_cfg,
            dataset_key=dataset_key,
            task_arg=args.task,
            all_sites=args.all_sites,
            fc_methods=fc_methods,
            fc_weight=args.fc_weight,
            batch_size=args.batch_size,
            seeds=seeds,
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
            checkpoint_path=args.checkpoint,
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
