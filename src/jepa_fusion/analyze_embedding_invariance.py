from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from .dataset_utils import matched_records_for_preset
    from .evaluate_baselines import (
        build_harmonization_covariates,
        mean_std,
        parse_covariate_list,
        run_grouped_cv,
        run_loso_diagnosis,
        transductive_transform_features,
    )
    from .evaluate_jepa import extract_scan_embeddings, load_checkpoint
    from .protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        batch_site_label,
        class_counts,
        filter_records_for_task,
        get_dataset_preset,
        center_site_label,
    )
    from ..jepa_clean.data import WindowedFMRIDataset
    from ..jepa_clean.plot_tsne_fc_vs_raw import build_fc_features, get_roi_permutation
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.dataset_utils import matched_records_for_preset
    from src.jepa_fusion.evaluate_baselines import (
        build_harmonization_covariates,
        mean_std,
        parse_covariate_list,
        run_grouped_cv,
        run_loso_diagnosis,
        transductive_transform_features,
    )
    from src.jepa_fusion.evaluate_jepa import extract_scan_embeddings, load_checkpoint
    from src.jepa_fusion.protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        batch_site_label,
        class_counts,
        filter_records_for_task,
        get_dataset_preset,
        center_site_label,
    )
    from src.jepa_clean.data import WindowedFMRIDataset
    from src.jepa_clean.plot_tsne_fc_vs_raw import build_fc_features, get_roi_permutation


@dataclass(frozen=True)
class IndexedEvalRecord:
    index: int
    subject: str
    site: str
    diagnosis: str


@dataclass
class SubjectBundle:
    features: np.ndarray
    subject_ids: np.ndarray
    site_centers: np.ndarray
    site_batches: np.ndarray
    diagnoses: np.ndarray

    def subset_to_subjects(self, subject_ids: list[str]) -> "SubjectBundle":
        index_map = {str(subject): idx for idx, subject in enumerate(self.subject_ids.tolist())}
        keep = np.asarray([index_map[str(subject)] for subject in subject_ids], dtype=np.int64)
        return SubjectBundle(
            features=self.features[keep],
            subject_ids=self.subject_ids[keep],
            site_centers=self.site_centers[keep],
            site_batches=self.site_batches[keep],
            diagnoses=self.diagnoses[keep],
        )


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    family: str
    kind: str
    run_template: str | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit site invariance of FC and JEPA representations. "
            "FC + ComBat/CovBat use the new-baseline controls-only + age/sex harmonization setting."
        )
    )
    parser.add_argument("--dataset", choices=["abide", "srpbs", "all"], default="all")
    parser.add_argument("--run-tag", default="final_run_v1")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--seeds", default="0,8,16,32,128")
    parser.add_argument("--include-curriculum-nce", action="store_true")
    parser.add_argument("--triangle", choices=["upper", "lower"], default="upper")
    parser.add_argument("--reorder", choices=["none", "hemi", "hemi_network"], default="hemi_network")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--harmonization-covariates", default="age,sex")
    parser.add_argument("--harmonization-fit-controls-only", action="store_true", default=True)
    parser.add_argument("--harmonization-age-min", type=float, default=0.0)
    parser.add_argument("--harmonization-age-max", type=float, default=100.0)
    parser.add_argument("--harmonization-seed", type=int, default=0)
    parser.add_argument("--window-feature", choices=["mean", "mean_std", "mean_cov"], default="mean")
    parser.add_argument("--covariance-dim", type=int, default=32)
    parser.add_argument("--covariance-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--distribution-pca-dim", type=int, default=32)
    parser.add_argument("--distribution-hist-bins", type=int, default=24)
    parser.add_argument("--output-prefix", default="embedding_invariance")
    parser.add_argument("--cache-dir", default="results/cache/jepa_embedding_invariance")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> list[int]:
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def maybe_json(value):
    if isinstance(value, dict):
        return {str(k): maybe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [maybe_json(v) for v in value]
    if isinstance(value, tuple):
        return [maybe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [maybe_json(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    return value


def fmt_pct(metric: dict) -> str:
    return f"{100.0 * metric['mean']:.1f} ± {100.0 * metric['std']:.1f}"


def fmt_small(metric: dict) -> str:
    return f"{metric['mean']:.3f} ± {metric['std']:.3f}"


def encode_task_labels(diagnoses: np.ndarray, task_key: str, class_names: list[str]) -> np.ndarray:
    if task_key == "hc_asd":
        mapping = {
            str(class_names[0]): 0,
            str(class_names[1]): 1,
        }
        return np.asarray([mapping[str(name)] for name in diagnoses.tolist()], dtype=np.int64)
    if task_key == "hc_disorder":
        return np.asarray(
            [0 if str(name) == HEALTHY_CONTROL_LABEL else 1 for name in diagnoses.tolist()],
            dtype=np.int64,
        )
    raise ValueError(f"Unsupported task_key: {task_key}")


def sanitize_method_label(label: str) -> str:
    return (
        label.replace(" + ", " + ")
        .replace("(", "")
        .replace(")", "")
    )


def method_specs(run_tag: str, include_curriculum_nce: bool) -> list[MethodSpec]:
    specs = [
        MethodSpec("fc", "Static FC", "fc", "fc"),
        MethodSpec("fc_combat", "FC + ComBat", "fc", "fc"),
        MethodSpec("fc_covbat", "FC + CovBat", "fc", "fc"),
        MethodSpec(
            "jepa_rotary_pooled",
            "JEPA Rotary (Pooled)",
            "jepa",
            "jepa",
            run_template=f"{run_tag}_pilot_temporal_rotary_conv_net_pooled_seed{{seed}}",
        ),
        MethodSpec(
            "jepa_rotary_curriculum",
            "JEPA Rotary (Curriculum + Replay)",
            "jepa",
            "jepa",
            run_template=f"{run_tag}_rotary_curr_replay_seed{{seed}}",
        ),
    ]
    if include_curriculum_nce:
        specs.append(
            MethodSpec(
                "jepa_rotary_curriculum_nce",
                "JEPA Rotary (Curriculum + Replay + InfoNCE)",
                "jepa",
                "jepa",
                run_template=f"{run_tag}_rotary_curr_replay_nce_seed{{seed}}",
            )
        )
    return specs


def output_paths(output_prefix: str, dataset_choice: str):
    base = Path("results/reports/tables/jepa_fusion") / output_prefix
    stem = f"{output_prefix}_{dataset_choice}"
    return {
        "json": base / f"{stem}.json",
        "md": base / f"{stem}.md",
        "tex": base / f"{stem}.tex",
        "fig_base": Path("results/figures/jepa_fusion") / output_prefix,
    }


def aggregate_subject_bundle(
    features: np.ndarray,
    subject_ids: np.ndarray,
    site_centers: np.ndarray,
    site_batches: np.ndarray,
    diagnoses: np.ndarray,
) -> SubjectBundle:
    acc: dict[str, dict] = {}
    for idx, subject_id in enumerate(subject_ids.tolist()):
        key = str(subject_id)
        row = acc.setdefault(
            key,
            {
                "features": [],
                "site_center": str(site_centers[idx]),
                "site_batch": str(site_batches[idx]),
                "diagnosis": str(diagnoses[idx]),
            },
        )
        row["features"].append(np.asarray(features[idx], dtype=np.float32))
        if row["site_center"] != str(site_centers[idx]):
            raise RuntimeError(f"Inconsistent center site for subject {key}.")
        if row["site_batch"] != str(site_batches[idx]):
            raise RuntimeError(f"Inconsistent batch site for subject {key}.")
        if row["diagnosis"] != str(diagnoses[idx]):
            raise RuntimeError(f"Inconsistent diagnosis for subject {key}.")

    ordered = sorted(acc.keys())
    agg_features = np.stack(
        [np.mean(np.stack(acc[subject]["features"], axis=0), axis=0) for subject in ordered],
        axis=0,
    )
    return SubjectBundle(
        features=agg_features.astype(np.float32, copy=False),
        subject_ids=np.asarray(ordered, dtype=object),
        site_centers=np.asarray([acc[subject]["site_center"] for subject in ordered], dtype=object),
        site_batches=np.asarray([acc[subject]["site_batch"] for subject in ordered], dtype=object),
        diagnoses=np.asarray([acc[subject]["diagnosis"] for subject in ordered], dtype=object),
    )


def prepare_fc_bundles(dataset_key: str, args) -> tuple[dict[str, SubjectBundle], dict]:
    preset = get_dataset_preset(dataset_key)
    records = matched_records_for_preset(preset)
    task_result = filter_records_for_task(
        records,
        preset.default_task,
        mixed_sites_only=preset.default_mixed_sites_only,
        healthy_control_label=HEALTHY_CONTROL_LABEL,
        autism_label=AUTISM_LABEL,
    )

    task_records = task_result.records
    permutation, _ = get_roi_permutation(args.reorder)
    fc_features = build_fc_features(task_records, permutation=permutation, triangle=args.triangle)
    batch_sites = np.asarray([str(record.site_batch) for record in task_records], dtype=object)
    fit_controls_only = bool(args.harmonization_fit_controls_only)
    fit_mask = (
        np.asarray(
            [str(record.diagnosis) == HEALTHY_CONTROL_LABEL for record in task_records],
            dtype=bool,
        )
        if fit_controls_only
        else np.ones(len(task_records), dtype=bool)
    )
    covariate_names = parse_covariate_list(args.harmonization_covariates)
    covariates, covariate_imputations = build_harmonization_covariates(
        task_records,
        covariate_names,
        fit_mask,
        age_min=args.harmonization_age_min,
        age_max=args.harmonization_age_max,
    )

    method_features = {
        "fc": fc_features,
        "fc_combat": transductive_transform_features(
            "fc_combat",
            fc_features,
            batch_sites,
            seed=args.harmonization_seed,
            covbat_n_pc=50,
            covbat_pct_var=0.95,
            covbat_max_components=256,
            covariates=covariates,
            fit_mask=fit_mask,
        ),
        "fc_covbat": transductive_transform_features(
            "fc_covbat",
            fc_features,
            batch_sites,
            seed=args.harmonization_seed,
            covbat_n_pc=50,
            covbat_pct_var=0.95,
            covbat_max_components=256,
            covariates=covariates,
            fit_mask=fit_mask,
        ),
    }

    subject_ids = np.asarray([str(record.subject_id) for record in task_records], dtype=object)
    site_centers = np.asarray([str(record.site) for record in task_records], dtype=object)
    site_batches = np.asarray([str(record.site_batch) for record in task_records], dtype=object)
    diagnoses = np.asarray([str(record.diagnosis) for record in task_records], dtype=object)

    bundles = {
        method_key: aggregate_subject_bundle(
            features,
            subject_ids=subject_ids,
            site_centers=site_centers,
            site_batches=site_batches,
            diagnoses=diagnoses,
        )
        for method_key, features in method_features.items()
    }

    meta = {
        "dataset": preset.key,
        "dataset_label": preset.label,
        "task_key": task_result.task_key,
        "task_label": task_result.task_label,
        "class_names": list(task_result.class_names),
        "class_counts_subject_level": class_counts(
            encode_task_labels(bundles["fc"].diagnoses, task_result.task_key, task_result.class_names),
            task_result.class_names,
        ),
        "n_records_post_task_filter": int(len(task_records)),
        "n_subjects": int(len(bundles["fc"].subject_ids)),
        "n_sites": int(len(np.unique(bundles["fc"].site_batches))),
        "removed_unknown": int(task_result.removed_unknown),
        "removed_task_mismatch": int(task_result.removed_task_mismatch),
        "removed_single_class_sites": int(task_result.removed_single_class_sites),
        "mixed_sites_kept": list(task_result.mixed_sites_kept),
        "harmonization": {
            "fit_controls_only": bool(fit_controls_only),
            "fit_count": int(fit_mask.sum()),
            "covariates": list(covariate_names),
            "age_min": float(args.harmonization_age_min),
            "age_max": float(args.harmonization_age_max),
            "covariate_imputations": covariate_imputations,
        },
    }
    return bundles, meta


def jepa_cache_path(cache_dir: Path, run_name: str, dataset_key: str) -> Path:
    return cache_dir / f"{run_name}_{dataset_key}_subject_embeddings.npz"


def load_jepa_subject_bundle_from_cache(cache_path: Path) -> SubjectBundle:
    data = np.load(cache_path, allow_pickle=True)
    return SubjectBundle(
        features=np.asarray(data["features"], dtype=np.float32),
        subject_ids=np.asarray(data["subject_ids"], dtype=object),
        site_centers=np.asarray(data["site_centers"], dtype=object),
        site_batches=np.asarray(data["site_batches"], dtype=object),
        diagnoses=np.asarray(data["diagnoses"], dtype=object),
    )


def save_jepa_subject_bundle_to_cache(cache_path: Path, bundle: SubjectBundle):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=bundle.features.astype(np.float32),
        subject_ids=bundle.subject_ids,
        site_centers=bundle.site_centers,
        site_batches=bundle.site_batches,
        diagnoses=bundle.diagnoses,
    )


def load_jepa_subject_bundle(
    dataset_key: str,
    checkpoint_path: Path,
    checkpoint_cfg: dict,
    *,
    batch_size: int,
    device: str | None,
    window_feature: str,
    covariance_dim: int,
    covariance_seed: int,
) -> SubjectBundle:
    preset = get_dataset_preset(dataset_key)
    model, _checkpoint, cfg = load_checkpoint(str(checkpoint_path))
    eval_device = device or ("cuda" if getattr(np, "__name__", "numpy") else "cpu")
    import torch

    eval_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(eval_device).eval()

    dataset = WindowedFMRIDataset(
        preset.raw_dir,
        window_seconds=float(checkpoint_cfg.get("window_seconds", cfg.get("window_seconds", 90.0))),
        stride_seconds=float(checkpoint_cfg.get("window_stride_seconds", cfg.get("window_stride_seconds", 30.0))),
        target_timepoints=int(checkpoint_cfg.get("target_timepoints", cfg.get("target_timepoints", 45))),
        sampling="center",
        exclude_unknown_diagnoses=False,
    )
    embeddings, metadata = extract_scan_embeddings(
        model,
        dataset,
        device=eval_device,
        batch_size=batch_size,
        window_feature=window_feature,
        covariance_dim=covariance_dim,
        covariance_seed=covariance_seed,
    )
    eval_records = [
        IndexedEvalRecord(
            index=idx,
            subject=str(meta["subject_id"]),
            site=center_site_label(str(meta.get("site_raw", meta["site"]))),
            diagnosis=str(meta["diag_name"]),
        )
        for idx, meta in enumerate(metadata)
    ]
    task_result = filter_records_for_task(
        eval_records,
        preset.default_task,
        mixed_sites_only=preset.default_mixed_sites_only,
        healthy_control_label=HEALTHY_CONTROL_LABEL,
        autism_label=AUTISM_LABEL,
    )
    keep_indices = np.asarray([record.index for record in task_result.records], dtype=np.int64)
    task_embeddings = embeddings[keep_indices]
    task_subjects = np.asarray([eval_records[idx].subject for idx in keep_indices.tolist()], dtype=object)
    task_sites = np.asarray([eval_records[idx].site for idx in keep_indices.tolist()], dtype=object)
    task_site_batches = np.asarray(
        [
            batch_site_label(
                str(metadata[idx].get("site_raw", metadata[idx]["site"])),
                source_dataset=str(metadata[idx].get("dataset", "")),
            )
            for idx in keep_indices.tolist()
        ],
        dtype=object,
    )
    task_diagnoses = np.asarray([eval_records[idx].diagnosis for idx in keep_indices.tolist()], dtype=object)
    return aggregate_subject_bundle(
        task_embeddings,
        subject_ids=task_subjects,
        site_centers=task_sites,
        site_batches=task_site_batches,
        diagnoses=task_diagnoses,
    )


def prepare_jepa_bundles(dataset_key: str, method: MethodSpec, seeds: list[int], results_dir: Path, args) -> dict[int, SubjectBundle]:
    cache_dir = Path(args.cache_dir)
    bundles: dict[int, SubjectBundle] = {}
    for seed in seeds:
        run_name = str(method.run_template).format(seed=seed)
        groupedcv_json = results_dir / f"{run_name}_groupedcv.json"
        if not groupedcv_json.exists():
            raise FileNotFoundError(f"Missing JEPA grouped-CV JSON: {groupedcv_json}")
        groupedcv = json.loads(groupedcv_json.read_text())
        checkpoint_path = Path(groupedcv["checkpoint"]["path"])
        cache_path = jepa_cache_path(cache_dir, run_name, dataset_key)
        if cache_path.exists():
            bundles[seed] = load_jepa_subject_bundle_from_cache(cache_path)
            continue
        bundle = load_jepa_subject_bundle(
            dataset_key,
            checkpoint_path=checkpoint_path,
            checkpoint_cfg=groupedcv["checkpoint"]["config"],
            batch_size=args.batch_size,
            device=args.device,
            window_feature=args.window_feature,
            covariance_dim=args.covariance_dim,
            covariance_seed=args.covariance_seed,
        )
        save_jepa_subject_bundle_to_cache(cache_path, bundle)
        bundles[seed] = bundle
    return bundles


def compute_common_subjects(method_seed_bundles: dict[str, dict[int, SubjectBundle]]) -> list[str]:
    subject_sets = []
    for by_seed in method_seed_bundles.values():
        for bundle in by_seed.values():
            subject_sets.append(set(str(subject) for subject in bundle.subject_ids.tolist()))
    if not subject_sets:
        return []
    return sorted(set.intersection(*subject_sets))


def align_bundles_to_common_subjects(method_seed_bundles: dict[str, dict[int, SubjectBundle]], common_subjects: list[str]):
    for method_key, by_seed in method_seed_bundles.items():
        for seed, bundle in list(by_seed.items()):
            by_seed[seed] = bundle.subset_to_subjects(common_subjects)


def reduce_features(features: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(np.asarray(features, dtype=np.float64))
    max_components = min(int(n_components), scaled.shape[0] - 1, scaled.shape[1])
    max_components = max(2, max_components)
    return PCA(n_components=max_components, random_state=seed).fit_transform(scaled)


def rbf_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.concatenate([x, y], axis=0)
    if z.shape[0] < 4:
        return float("nan")
    sample = z
    if len(sample) > 512:
        sample = sample[np.linspace(0, len(sample) - 1, 512).astype(int)]
    dists = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2.0, axis=2)
    dists = dists[np.triu_indices_from(dists, k=1)]
    sigma2 = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
    gamma = 1.0 / max(2.0 * sigma2, 1e-8)

    def kernel(a, b):
        sq = np.sum((a[:, None, :] - b[None, :, :]) ** 2.0, axis=2)
        return np.exp(-gamma * sq)

    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)
    return float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))


def js_divergence_from_probs(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-12))))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def pairwise_distribution_metrics(features: np.ndarray, site_labels: np.ndarray, seed: int, pca_dim: int, bins: int) -> dict:
    reduced = reduce_features(features, pca_dim, seed=seed)
    reduced2 = reduced[:, :2]
    unique_sites = sorted(str(site) for site in np.unique(site_labels))
    site_arrays = {site: reduced[np.asarray(site_labels, dtype=object) == site] for site in unique_sites}
    mins = reduced2.min(axis=0)
    maxs = reduced2.max(axis=0)
    pad = 0.05 * np.maximum(maxs - mins, 1e-6)
    hist_range = [(mins[0] - pad[0], maxs[0] + pad[0]), (mins[1] - pad[1], maxs[1] + pad[1])]

    mmd_values = []
    js_values = []
    for site_a, site_b in combinations(unique_sites, 2):
        x = site_arrays[site_a]
        y = site_arrays[site_b]
        if len(x) < 2 or len(y) < 2:
            continue
        mmd_values.append(rbf_mmd2(x, y))

        hist_a, _ = np.histogramdd(reduced2[np.asarray(site_labels, dtype=object) == site_a], bins=bins, range=hist_range)
        hist_b, _ = np.histogramdd(reduced2[np.asarray(site_labels, dtype=object) == site_b], bins=bins, range=hist_range)
        hist_a = hist_a.astype(np.float64).reshape(-1) + 1e-8
        hist_b = hist_b.astype(np.float64).reshape(-1) + 1e-8
        js_values.append(js_divergence_from_probs(hist_a, hist_b))

    return {
        "pairwise_mmd": float(np.mean(mmd_values)) if mmd_values else float("nan"),
        "pairwise_js": float(np.mean(js_values)) if js_values else float("nan"),
        "n_site_pairs": int(len(mmd_values)),
    }


def run_conditional_site_probe(
    features: np.ndarray,
    diagnosis_y: np.ndarray,
    class_names: list[str],
    site_batches: np.ndarray,
    subject_ids: np.ndarray,
    *,
    seed: int,
    folds: int,
) -> dict:
    per_class = []
    for class_idx, class_name in enumerate(class_names):
        class_mask = diagnosis_y == class_idx
        if int(class_mask.sum()) < 8:
            continue
        class_sites = site_batches[class_mask]
        unique_sites, counts = np.unique(class_sites, return_counts=True)
        valid_sites = set(unique_sites[counts >= 2].tolist())
        valid_mask = class_mask & np.asarray([site in valid_sites for site in site_batches], dtype=bool)
        if int(valid_mask.sum()) < 8 or len(valid_sites) < 2:
            continue
        site_encoder = LabelEncoder()
        y_site = site_encoder.fit_transform(site_batches[valid_mask])
        try:
            result = run_grouped_cv(
                features[valid_mask],
                y_site,
                class_names=[str(name) for name in site_encoder.classes_.tolist()],
                groups=subject_ids[valid_mask],
                seed=seed,
                folds=folds,
            )
        except RuntimeError:
            continue
        per_class.append(
            {
                "class_name": str(class_name),
                "balanced_accuracy": float(result["balanced_accuracy"]),
                "roc_auc": float(result["roc_auc"]),
                "n_subjects": int(valid_mask.sum()),
                "n_sites": int(len(site_encoder.classes_)),
            }
        )

    if not per_class:
        return {
            "balanced_accuracy": float("nan"),
            "roc_auc": float("nan"),
            "n_classes_evaluated": 0,
            "per_class": [],
        }
    return {
        "balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in per_class])),
        "roc_auc": float(np.mean([row["roc_auc"] for row in per_class])),
        "n_classes_evaluated": int(len(per_class)),
        "per_class": per_class,
    }


def evaluate_bundle(
    bundle: SubjectBundle,
    task_key: str,
    class_names: list[str],
    *,
    seed: int,
    folds: int,
    distribution_pca_dim: int,
    distribution_hist_bins: int,
) -> dict:
    diagnosis_y = encode_task_labels(bundle.diagnoses, task_key, class_names)
    site_encoder = LabelEncoder()
    y_site = site_encoder.fit_transform(bundle.site_batches)

    site_probe = run_grouped_cv(
        bundle.features,
        y_site,
        class_names=[str(name) for name in site_encoder.classes_.tolist()],
        groups=bundle.subject_ids,
        seed=seed,
        folds=folds,
    )
    diagnosis_probe = run_loso_diagnosis(
        bundle.features,
        diagnosis_y,
        class_names=class_names,
        sites=bundle.site_centers,
        seed=seed,
    )
    conditional_site_probe = run_conditional_site_probe(
        bundle.features,
        diagnosis_y,
        class_names,
        bundle.site_batches,
        bundle.subject_ids,
        seed=seed,
        folds=folds,
    )
    distribution = pairwise_distribution_metrics(
        bundle.features,
        bundle.site_batches,
        seed=seed,
        pca_dim=distribution_pca_dim,
        bins=distribution_hist_bins,
    )
    return {
        "site_probe": {
            "balanced_accuracy": float(site_probe["balanced_accuracy"]),
            "roc_auc": float(site_probe["roc_auc"]),
            "n_sites": int(len(site_encoder.classes_)),
        },
        "conditional_site_probe": conditional_site_probe,
        "diagnosis_probe": {
            "balanced_accuracy": float(diagnosis_probe["balanced_accuracy"]),
            "roc_auc": float(diagnosis_probe["roc_auc"]),
            "n_sites": int(len(np.unique(bundle.site_centers))),
        },
        "distribution": distribution,
    }


def summarize_metric(values: list[float]) -> dict:
    return mean_std([float(v) for v in values if np.isfinite(float(v))]) if values else {"mean": float("nan"), "std": float("nan")}


def summarize_runs(runs: list[dict]) -> dict:
    return {
        "site_probe_bacc": summarize_metric([run["site_probe"]["balanced_accuracy"] for run in runs]),
        "site_probe_auc": summarize_metric([run["site_probe"]["roc_auc"] for run in runs]),
        "conditional_site_probe_bacc": summarize_metric([run["conditional_site_probe"]["balanced_accuracy"] for run in runs]),
        "conditional_site_probe_auc": summarize_metric([run["conditional_site_probe"]["roc_auc"] for run in runs]),
        "diagnosis_bacc": summarize_metric([run["diagnosis_probe"]["balanced_accuracy"] for run in runs]),
        "diagnosis_auc": summarize_metric([run["diagnosis_probe"]["roc_auc"] for run in runs]),
        "pairwise_mmd": summarize_metric([run["distribution"]["pairwise_mmd"] for run in runs]),
        "pairwise_js": summarize_metric([run["distribution"]["pairwise_js"] for run in runs]),
        "n_runs": int(len(runs)),
    }


def build_markdown(payload: dict, specs: list[MethodSpec]) -> str:
    lines = [
        "# Embedding Invariance Audit",
        "",
        "Static FC + ComBat/CovBat use controls-only harmonization with age and sex covariates.",
        "Site probe uses grouped subject-level CV on subject-aggregated representations.",
        "Conditional site probe averages site-probe performance within each diagnosis class.",
        "Diagnosis probe uses leave-one-site-out evaluation on subject-aggregated representations.",
        "Pairwise MMD is computed on standardized PCA-reduced features; Pairwise JS is computed from shared 2D PCA histograms.",
        "",
    ]
    for dataset_key, dataset_payload in payload["datasets"].items():
        lines.extend(
            [
                f"## {dataset_payload['meta']['dataset_label']}",
                "",
                f"Task: `{dataset_payload['meta']['task_label']}`",
                "",
                "| Method | Site bAcc (%) | Cond. Site bAcc (%) | Dx bAcc (%) | Dx AUC (%) | Pairwise MMD | Pairwise JS |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        summary = dataset_payload["summary"]
        for spec in specs:
            row = summary[spec.key]
            lines.append(
                f"| {spec.label} | {fmt_pct(row['site_probe_bacc'])} | {fmt_pct(row['conditional_site_probe_bacc'])} | "
                f"{fmt_pct(row['diagnosis_bacc'])} | {fmt_pct(row['diagnosis_auc'])} | "
                f"{fmt_small(row['pairwise_mmd'])} | {fmt_small(row['pairwise_js'])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def build_latex(payload: dict, specs: list[MethodSpec]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Embedding-level site-invariance audit for FC baselines and Rotary JEPA models.",
        r"Static FC + ComBat/CovBat use controls-only harmonization with age and sex covariates.",
        r"Site probe uses grouped subject-level cross-validation on subject-aggregated representations; conditional site probe averages site-probe performance within each diagnosis class.",
        r"Diagnosis balanced accuracy and ROC-AUC are computed with leave-one-site-out evaluation.",
        r"Pairwise MMD is computed on standardized PCA-reduced features and pairwise JS is computed from shared 2D PCA histograms.",
        r"Values are mean $\pm$ s.d.\ across 5 seeds. Lower is better for site/MMD/JS; higher is better for diagnosis.}",
        r"\label{tab:embedding-invariance}",
        r"\scriptsize",
        r"\begin{tabular}{l rr rr rr}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Site} & \textbf{Cond. Site} & \textbf{Dx bAcc} & \textbf{Dx AUC} & \textbf{MMD} & \textbf{JS} \\",
        r"\midrule",
    ]
    dataset_keys = list(payload["datasets"].keys())
    for dataset_idx, dataset_key in enumerate(dataset_keys):
        dataset_payload = payload["datasets"][dataset_key]
        chance = 100.0 / max(1, int(dataset_payload["meta"]["n_sites"]))
        lines.append(
            rf"\multicolumn{{7}}{{l}}{{\textit{{{dataset_payload['meta']['dataset_label']} \quad [chance site bAcc = {chance:.1f}\%]}}}} \\[2pt]"
        )
        for spec in specs:
            row = dataset_payload["summary"][spec.key]
            lines.append(
                f"{sanitize_method_label(spec.label)} & {fmt_pct(row['site_probe_bacc'])} & {fmt_pct(row['conditional_site_probe_bacc'])} & "
                f"{fmt_pct(row['diagnosis_bacc'])} & {fmt_pct(row['diagnosis_auc'])} & {fmt_small(row['pairwise_mmd'])} & {fmt_small(row['pairwise_js'])} \\\\"
            )
        if dataset_idx != len(dataset_keys) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def method_color(spec: MethodSpec) -> str:
    color_map = {
        "fc": "#6b7280",
        "fc_combat": "#c2410c",
        "fc_covbat": "#2563eb",
        "jepa_rotary_pooled": "#b45309",
        "jepa_rotary_curriculum": "#15803d",
        "jepa_rotary_curriculum_nce": "#7c3aed",
    }
    return color_map.get(spec.key, "#111827")


def plot_tradeoff(dataset_payload: dict, specs: list[MethodSpec], output_path: Path):
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.set_facecolor("#fbfbfa")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    points = []
    for spec in specs:
        row = dataset_payload["summary"][spec.key]
        x = 100.0 * row["site_probe_bacc"]["mean"]
        y = 100.0 * row["diagnosis_bacc"]["mean"]
        points.append((spec, x, y))
        ax.scatter(
            x,
            y,
            s=150,
            color=method_color(spec),
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        ax.text(x + 0.15, y + 0.08, spec.label.replace("JEPA ", ""), fontsize=9, color="#111827")

    by_key = {spec.key: (x, y) for spec, x, y in points}
    if "jepa_rotary_pooled" in by_key and "jepa_rotary_curriculum" in by_key:
        ax.plot(
            [by_key["jepa_rotary_pooled"][0], by_key["jepa_rotary_curriculum"][0]],
            [by_key["jepa_rotary_pooled"][1], by_key["jepa_rotary_curriculum"][1]],
            linestyle="--",
            color="#6b7280",
            linewidth=1.2,
            zorder=2,
        )

    ax.annotate(
        "Better",
        xy=(ax.get_xlim()[0], ax.get_ylim()[1]),
        xytext=(0.03, 0.95),
        textcoords="axes fraction",
        fontsize=10,
        fontweight="bold",
        color="#111827",
    )
    ax.annotate(
        "",
        xy=(0.08, 0.90),
        xytext=(0.20, 0.78),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#111827", linewidth=1.5),
    )
    ax.set_xlabel("Site Probe bAcc (%)")
    ax.set_ylabel("Diagnosis LOSO bAcc (%)")
    ax.set_title(f"Site-Invariance Tradeoff — {dataset_payload['meta']['dataset_label']}", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.18, linewidth=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap(dataset_payload: dict, specs: list[MethodSpec], output_path: Path):
    metric_defs = [
        ("site_probe_bacc", "Site", True),
        ("conditional_site_probe_bacc", "Cond. Site", True),
        ("diagnosis_bacc", "Dx bAcc", False),
        ("diagnosis_auc", "Dx AUC", False),
        ("pairwise_mmd", "MMD", True),
        ("pairwise_js", "JS", True),
    ]
    values = np.zeros((len(specs), len(metric_defs)), dtype=np.float64)
    labels = np.empty_like(values, dtype=object)
    for row_idx, spec in enumerate(specs):
        summary = dataset_payload["summary"][spec.key]
        for col_idx, (metric_key, _label, low_is_better) in enumerate(metric_defs):
            metric = summary[metric_key]
            values[row_idx, col_idx] = float(metric["mean"])
            if "mmd" in metric_key or metric_key.endswith("js"):
                labels[row_idx, col_idx] = f"{metric['mean']:.3f}"
            else:
                labels[row_idx, col_idx] = f"{100.0 * metric['mean']:.1f}"

    normalized = np.zeros_like(values)
    for col_idx, (_metric_key, _label, low_is_better) in enumerate(metric_defs):
        col = values[:, col_idx]
        lo = np.nanmin(col)
        hi = np.nanmax(col)
        if hi - lo < 1e-12:
            normalized[:, col_idx] = 0.5
        else:
            normalized[:, col_idx] = (hi - col) / (hi - lo) if low_is_better else (col - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    im = ax.imshow(normalized, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(metric_defs)))
    ax.set_xticklabels([label for _key, label, _ in metric_defs], fontsize=10)
    ax.set_yticks(np.arange(len(specs)))
    ax.set_yticklabels([spec.label.replace("JEPA ", "") for spec in specs], fontsize=10)
    ax.set_title(f"Representation Scorecard — {dataset_payload['meta']['dataset_label']}", fontsize=13, fontweight="bold")

    for row_idx in range(len(specs)):
        for col_idx in range(len(metric_defs)):
            ax.text(
                col_idx,
                row_idx,
                labels[row_idx, col_idx],
                ha="center",
                va="center",
                fontsize=9,
                color="#111827",
                fontweight="bold" if normalized[row_idx, col_idx] > 0.75 else None,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Relative goodness", rotation=90)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    dataset_keys = ["abide", "srpbs"] if args.dataset == "all" else [args.dataset]
    specs = method_specs(args.run_tag, args.include_curriculum_nce)
    results_dir = Path(args.results_dir) if args.results_dir else Path("results/reports/tables/jepa_fusion") / args.run_tag
    paths = output_paths(args.output_prefix, args.dataset)

    payload = {
        "config": {
            "dataset_choice": args.dataset,
            "run_tag": args.run_tag,
            "results_dir": str(results_dir),
            "seeds": list(seeds),
            "fc_harmonization": {
                "fit_controls_only": bool(args.harmonization_fit_controls_only),
                "covariates": parse_covariate_list(args.harmonization_covariates),
                "age_min": float(args.harmonization_age_min),
                "age_max": float(args.harmonization_age_max),
            },
            "representation_protocol": "subject_aggregated",
            "jepa_window_sampling": "center",
            "distribution_metrics": {
                "pairwise_mmd": f"rbf_kernel_pca{int(args.distribution_pca_dim)}",
                "pairwise_js": f"shared_histogram_pca2_bins{int(args.distribution_hist_bins)}",
            },
        },
        "datasets": {},
    }

    for dataset_key in dataset_keys:
        fc_bundles, fc_meta = prepare_fc_bundles(dataset_key, args)

        method_seed_bundles: dict[str, dict[int, SubjectBundle]] = {}
        for spec in specs:
            if spec.kind == "fc":
                method_seed_bundles[spec.key] = {
                    seed: fc_bundles[spec.key]
                    for seed in seeds
                }
            else:
                method_seed_bundles[spec.key] = prepare_jepa_bundles(dataset_key, spec, seeds, results_dir, args)

        common_subjects = compute_common_subjects(method_seed_bundles)
        if not common_subjects:
            raise RuntimeError(f"No common subjects across methods for dataset {dataset_key}.")
        align_bundles_to_common_subjects(method_seed_bundles, common_subjects)

        aligned_example = method_seed_bundles["fc"][seeds[0]]
        y_subject = encode_task_labels(
            aligned_example.diagnoses,
            fc_meta["task_key"],
            fc_meta["class_names"],
        )
        dataset_payload = {
            "meta": {
                **fc_meta,
                "n_common_subjects": int(len(common_subjects)),
                "common_class_counts": class_counts(y_subject, fc_meta["class_names"]),
            },
            "per_method": {},
            "summary": {},
        }

        for spec in specs:
            runs = []
            representative_seed = seeds[0]
            for seed in seeds:
                bundle = method_seed_bundles[spec.key][seed]
                run = evaluate_bundle(
                    bundle,
                    fc_meta["task_key"],
                    fc_meta["class_names"],
                    seed=seed,
                    folds=args.folds,
                    distribution_pca_dim=args.distribution_pca_dim,
                    distribution_hist_bins=args.distribution_hist_bins,
                )
                run["seed"] = int(seed)
                runs.append(run)
            dataset_payload["per_method"][spec.key] = {
                "label": spec.label,
                "family": spec.family,
                "representative_seed": int(representative_seed),
                "runs": runs,
            }
            dataset_payload["summary"][spec.key] = summarize_runs(runs)

        payload["datasets"][dataset_key] = dataset_payload

        if not args.skip_figures:
            fig_base = paths["fig_base"]
            plot_tradeoff(
                dataset_payload,
                specs,
                fig_base / f"{args.output_prefix}_{dataset_key}_tradeoff.png",
            )
            plot_metric_heatmap(
                dataset_payload,
                specs,
                fig_base / f"{args.output_prefix}_{dataset_key}_heatmap.png",
            )

    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    paths["md"].parent.mkdir(parents=True, exist_ok=True)
    paths["tex"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(maybe_json(payload), indent=2))
    paths["md"].write_text(build_markdown(payload, specs))
    paths["tex"].write_text(build_latex(payload, specs))
    print(f"Saved JSON: {paths['json']}")
    print(f"Saved Markdown: {paths['md']}")
    print(f"Saved LaTeX: {paths['tex']}")


if __name__ == "__main__":
    main()
