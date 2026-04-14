from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from .protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        class_counts,
        filter_records_for_task,
        get_dataset_preset,
    )
    from ..jepa_clean.plot_tsne_fc_vs_raw import (
        build_fc_features,
        build_raw_features,
        get_roi_permutation,
        load_matched_records,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        class_counts,
        filter_records_for_task,
        get_dataset_preset,
    )
    from src.jepa_clean.plot_tsne_fc_vs_raw import (
        build_fc_features,
        build_raw_features,
        get_roi_permutation,
        load_matched_records,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the first clean baseline table for a single dataset: "
            "ROI time series, Static FC, Static FC + ComBat, and Static FC + CovBat."
        )
    )
    parser.add_argument("--dataset", choices=["abide1", "abide2", "srpbs"], required=True)
    parser.add_argument(
        "--task",
        choices=["default", "hc_asd", "hc_disorder"],
        default="default",
        help="Diagnosis task. Dataset defaults are ABIDE-I/II -> hc_asd, SRPBS -> hc_disorder.",
    )
    parser.add_argument(
        "--all-sites",
        action="store_true",
        help="Disable mixed-site filtering. By default, dataset presets use mixed-site-only binary evaluation.",
    )
    parser.add_argument("--triangle", choices=["upper", "lower"], default="upper")
    parser.add_argument(
        "--reorder",
        choices=["none", "hemi", "hemi_network"],
        default="hemi_network",
    )
    parser.add_argument("--raw-target-timepoints", type=int, default=45)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional dev-only cap applied after task filtering. Records are sampled in a class-balanced way.",
    )
    parser.add_argument(
        "--covbat-n-pc",
        type=int,
        default=50,
        help="Number of leading PCs to score-adjust in CovBat.",
    )
    parser.add_argument(
        "--covbat-pct-var",
        type=float,
        default=0.95,
        help="Variance threshold for CovBat when --covbat-n-pc <= 0.",
    )
    parser.add_argument(
        "--covbat-max-components",
        type=int,
        default=256,
        help="Maximum PCA components retained inside the approximate fold-wise CovBat transform.",
    )
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


def maybe_subsample_balanced(records: list, y: np.ndarray, max_samples: int | None, seed: int):
    if max_samples is None or len(records) <= max_samples:
        return records, y

    rng = np.random.default_rng(seed)
    indices_by_class = {
        int(cls): np.flatnonzero(y == cls)
        for cls in np.unique(y)
    }
    target_total = min(max_samples, len(records))

    selected = []
    for cls in sorted(indices_by_class):
        cls_indices = np.array(indices_by_class[cls], copy=True)
        rng.shuffle(cls_indices)
        quota = max(1, int(round(target_total * len(cls_indices) / len(records))))
        selected.extend(cls_indices[: min(quota, len(cls_indices))].tolist())

    selected = sorted(set(selected))
    if len(selected) < target_total:
        remaining = [idx for idx in range(len(records)) if idx not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_total - len(selected)])
    elif len(selected) > target_total:
        rng.shuffle(selected)
        selected = selected[:target_total]

    selected = sorted(selected)
    subsampled_records = [records[idx] for idx in selected]
    subsampled_y = y[np.asarray(selected, dtype=np.int64)]
    return subsampled_records, subsampled_y


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


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if np.asarray(y_score).ndim == 1:
        return float(roc_auc_score(y_true, y_score))
    return float(
        roc_auc_score(
            y_true,
            y_score,
            labels=np.arange(np.asarray(y_score).shape[1]),
            multi_class="ovr",
            average="macro",
        )
    )


class CombatTransformer:
    def __init__(self):
        try:
            import pandas as pd
            from neuroCombat import neuroCombat, neuroCombatFromTraining
        except ImportError as exc:
            raise ImportError(
                "Fold-wise ComBat requires neuroCombat. Install it with `pip install neuroCombat`."
            ) from exc

        self.pd = pd
        self.neuroCombat = neuroCombat
        self.neuroCombatFromTraining = neuroCombatFromTraining

    def fit_transform(self, x_train: np.ndarray, sites_train: np.ndarray, x_test: np.ndarray, sites_test: np.ndarray):
        covars = self.pd.DataFrame({"site": [str(site) for site in sites_train.tolist()]})
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            train_result = self.neuroCombat(
                dat=np.asarray(x_train, dtype=np.float32).T.copy(),
                covars=covars,
                batch_col="site",
            )

        seen_batches = {str(batch) for batch in train_result["estimates"]["batches"]}
        missing_batches = sorted(set(str(site) for site in sites_test.tolist()) - seen_batches)
        if missing_batches:
            raise ValueError(
                "Fold-wise ComBat cannot transform held-out samples from unseen sites: "
                f"{missing_batches}"
            )

        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            test_result = self.neuroCombatFromTraining(
                dat=np.asarray(x_test, dtype=np.float32).T.copy(),
                batch=np.asarray([str(site) for site in sites_test.tolist()], dtype=object),
                estimates=train_result["estimates"],
            )

        x_train_h = np.asarray(train_result["data"], dtype=np.float32).T
        x_test_h = np.asarray(test_result["data"], dtype=np.float32).T
        return (
            np.nan_to_num(x_train_h, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(x_test_h, nan=0.0, posinf=0.0, neginf=0.0),
        )


def _fit_batch_location_scale(features: np.ndarray, batches: np.ndarray):
    global_mean = np.asarray(features.mean(axis=0), dtype=np.float64)
    pooled_ss = np.zeros(features.shape[1], dtype=np.float64)
    batch_means = {}
    batch_vars = {}

    for batch in np.unique(batches):
        batch_mask = batches == batch
        batch_features = np.asarray(features[batch_mask], dtype=np.float64)
        batch_mean = batch_features.mean(axis=0)
        batch_var = batch_features.var(axis=0)
        batch_means[str(batch)] = batch_mean
        batch_vars[str(batch)] = batch_var
        pooled_ss += ((batch_features - batch_mean) ** 2.0).sum(axis=0)

    pooled_var = pooled_ss / max(1, features.shape[0])
    return {
        "global_mean": global_mean,
        "pooled_var": pooled_var,
        "batch_means": batch_means,
        "batch_vars": batch_vars,
    }


def _apply_batch_location_scale(features: np.ndarray, batches: np.ndarray, model: dict):
    adjusted = np.empty_like(features, dtype=np.float64)
    pooled_std = np.sqrt(np.maximum(model["pooled_var"], 1e-8))

    for batch in np.unique(batches):
        batch_key = str(batch)
        if batch_key not in model["batch_means"]:
            raise ValueError(
                "Fold-wise CovBat cannot transform held-out samples from unseen sites: "
                f"{batch_key}"
            )
        batch_mask = batches == batch
        batch_features = np.asarray(features[batch_mask], dtype=np.float64)
        batch_mean = model["batch_means"][batch_key]
        batch_std = np.sqrt(np.maximum(model["batch_vars"][batch_key], 1e-8))
        adjusted[batch_mask] = ((batch_features - batch_mean) / batch_std) * pooled_std + model["global_mean"]

    return adjusted.astype(np.float32, copy=False)


class CovBatTransformer:
    def __init__(self, n_pc: int, pct_var: float, max_components: int, seed: int):
        self.n_pc = int(n_pc)
        self.pct_var = float(pct_var)
        self.max_components = int(max_components)
        self.seed = int(seed)
        self.combat = CombatTransformer()

    def _fit_pca(self, x_train: np.ndarray):
        max_components = min(self.max_components, x_train.shape[0] - 1, x_train.shape[1])
        max_components = max(1, max_components)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)
        pca = PCA(
            n_components=max_components,
            svd_solver="randomized",
            random_state=self.seed,
        )
        pca.fit(x_scaled)
        return scaler, pca

    def _effective_n_pc(self, pca: PCA):
        if self.n_pc > 0:
            return min(self.n_pc, pca.n_components_)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        return int(min(pca.n_components_, np.searchsorted(cumulative, self.pct_var, side="left") + 1))

    def fit_transform(self, x_train: np.ndarray, sites_train: np.ndarray, x_test: np.ndarray, sites_test: np.ndarray):
        # Fold-wise CovBat is implemented as train-only ComBat followed by train-only
        # PC-score batch matching on the leading components so held-out samples are
        # never used to estimate harmonization parameters.
        x_train_combat, x_test_combat = self.combat.fit_transform(x_train, sites_train, x_test, sites_test)

        scaler, pca = self._fit_pca(x_train_combat)
        n_pc = self._effective_n_pc(pca)

        z_train = scaler.transform(x_train_combat)
        z_test = scaler.transform(x_test_combat)
        scores_train = pca.transform(z_train)
        scores_test = pca.transform(z_test)

        score_model = _fit_batch_location_scale(scores_train[:, :n_pc], sites_train)
        adjusted_train_scores = np.array(scores_train, copy=True)
        adjusted_test_scores = np.array(scores_test, copy=True)
        adjusted_train_scores[:, :n_pc] = _apply_batch_location_scale(scores_train[:, :n_pc], sites_train, score_model)
        adjusted_test_scores[:, :n_pc] = _apply_batch_location_scale(scores_test[:, :n_pc], sites_test, score_model)

        z_train_covbat = pca.inverse_transform(adjusted_train_scores)
        z_test_covbat = pca.inverse_transform(adjusted_test_scores)
        x_train_covbat = scaler.inverse_transform(z_train_covbat)
        x_test_covbat = scaler.inverse_transform(z_test_covbat)

        return (
            np.nan_to_num(np.asarray(x_train_covbat, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(np.asarray(x_test_covbat, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        )


def transform_fold(
    method_key: str,
    x_train: np.ndarray,
    train_sites: np.ndarray,
    x_test: np.ndarray,
    test_sites: np.ndarray,
    *,
    seed: int,
    covbat_n_pc: int,
    covbat_pct_var: float,
    covbat_max_components: int,
):
    if method_key in {"raw", "fc"}:
        return x_train, x_test
    if method_key == "fc_combat":
        return CombatTransformer().fit_transform(x_train, train_sites, x_test, test_sites)
    if method_key == "fc_covbat":
        return CovBatTransformer(
            n_pc=covbat_n_pc,
            pct_var=covbat_pct_var,
            max_components=covbat_max_components,
            seed=seed,
        ).fit_transform(x_train, train_sites, x_test, test_sites)
    raise ValueError(f"Unsupported method: {method_key}")


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
    summary = {
        "accuracy": mean_std([run["accuracy"] for run in runs]),
        "balanced_accuracy": mean_std([run["balanced_accuracy"] for run in runs]),
        "roc_auc": mean_std([run["roc_auc"] for run in runs]),
        "n_runs": int(len(runs)),
        "n_samples": int(runs[0]["n_samples"]),
        "folds": int(runs[0]["folds"]),
        "classes": list(runs[0]["classes"]),
    }
    return summary


def transductive_transform_features(
    method_key: str,
    features: np.ndarray,
    sites: np.ndarray,
    *,
    seed: int,
    covbat_n_pc: int,
    covbat_pct_var: float,
    covbat_max_components: int,
):
    if method_key in {"raw", "fc"}:
        return features
    if method_key == "fc_combat":
        transformed, _ = CombatTransformer().fit_transform(features, sites, features, sites)
        return transformed
    if method_key == "fc_covbat":
        transformed, _ = CovBatTransformer(
            n_pc=covbat_n_pc,
            pct_var=covbat_pct_var,
            max_components=covbat_max_components,
            seed=seed,
        ).fit_transform(features, sites, features, sites)
        return transformed
    raise ValueError(f"Unsupported method: {method_key}")


def run_loso_diagnosis(
    features: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    sites: np.ndarray,
    *,
    seed: int,
):
    unique_sites = np.unique(sites)
    preds = np.full(len(y), -1, dtype=np.int64)
    n_classes = len(class_names)
    oof_scores = np.full((len(y), n_classes), 0.0, dtype=np.float64)
    evaluated = np.zeros(len(y), dtype=bool)
    fold_rows = []

    for fold_idx, held_out_site in enumerate(unique_sites.tolist(), start=1):
        train_mask = sites != held_out_site
        test_mask = sites == held_out_site
        if not np.any(test_mask):
            continue

        x_train = features[train_mask]
        x_test = features[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        clf = _make_linear_probe(feature_dim=x_train.shape[1], train_count=int(train_mask.sum()), seed=seed)
        clf.fit(x_train, y_train)
        fold_pred = clf.predict(x_test)
        test_indices = np.flatnonzero(test_mask)
        preds[test_indices] = fold_pred
        evaluated[test_indices] = True

        clf_classes = clf.named_steps["logisticregression"].classes_
        fold_scores = clf.predict_proba(x_test)
        aligned_scores = np.zeros((len(test_indices), n_classes), dtype=np.float64)
        aligned_scores[:, clf_classes] = fold_scores
        oof_scores[test_indices] = aligned_scores

        fold_row = {
            "fold": int(fold_idx),
            "held_out_site": str(held_out_site),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "accuracy": float(accuracy_score(y_test, fold_pred)),
            "balanced_accuracy": _balanced_accuracy_present(y_test, fold_pred),
        }
        if n_classes == 2:
            fold_row["roc_auc"] = _safe_roc_auc(y_test, aligned_scores[:, 1])
        else:
            fold_row["roc_auc"] = _safe_roc_auc(y_test, aligned_scores)
        fold_rows.append(fold_row)

    if not np.any(evaluated):
        raise RuntimeError("LOSO diagnosis could not evaluate any held-out site.")

    y_eval = y[evaluated]
    preds_eval = preds[evaluated]
    scores_eval = oof_scores[evaluated]
    result = {
        "accuracy": float(accuracy_score(y_eval, preds_eval)),
        "balanced_accuracy": _balanced_accuracy_present(y_eval, preds_eval),
        "folds": int(len(fold_rows)),
        "fold_rows": fold_rows,
        "n_samples": int(evaluated.sum()),
        "classes": list(class_names),
    }
    if n_classes == 2:
        result["roc_auc"] = _safe_roc_auc(y_eval, scores_eval[:, 1])
    else:
        result["roc_auc"] = _safe_roc_auc(y_eval, scores_eval)
    return result


def format_pct(metric: dict):
    return f"{100.0 * metric['mean']:.1f} ± {100.0 * metric['std']:.1f}"


def format_scalar_pct(value: float):
    return f"{100.0 * value:.1f}"


def build_markdown_table(payload: dict):
    rows = [
        ("raw", "ROI time series"),
        ("fc", "Static FC"),
        ("fc_combat", "Static FC + ComBat"),
        ("fc_covbat", "Static FC + CovBat"),
    ]
    site_chance = 1.0 / max(1, payload["meta"]["n_sites"])
    diagnosis_chance = 1.0 / max(1, len(payload["meta"]["diagnosis_classes"]))
    lines = [
        f"# {payload['config']['dataset_label']} Baselines",
        "",
        f"Task: `{payload['config']['task_label']}`",
        "",
        "Site probe uses grouped subject-level CV. Diagnosis uses leave-one-site-out evaluation.",
        "",
        "| Method | Site bAcc | Site ROC-AUC | Dx bAcc | Dx ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method_key, label in rows:
        site_metric = payload["summary"]["site"][method_key]["balanced_accuracy"]
        site_auc_metric = payload["summary"]["site"][method_key]["roc_auc"]
        dx_metric = payload["summary"]["diagnosis"][method_key]["balanced_accuracy"]
        auc_metric = payload["summary"]["diagnosis"][method_key]["roc_auc"]
        lines.append(
            f"| {label} | {format_pct(site_metric)} | {format_pct(site_auc_metric)} | "
            f"{format_pct(dx_metric)} | {format_pct(auc_metric)} |"
        )
    lines.append(
        f"| Chance | {format_scalar_pct(site_chance)} | 50.0 | {format_scalar_pct(diagnosis_chance)} | 50.0 |"
    )
    return "\n".join(lines) + "\n"


def default_output_paths(dataset_key: str, task_key: str, mixed_sites_only: bool):
    site_scope = "mixedsites" if mixed_sites_only else "allsites"
    stem = f"{dataset_key}_{task_key}_{site_scope}_baseline_table"
    base_dir = Path("results/reports/tables/jepa_fusion")
    return base_dir / f"{stem}.json", base_dir / f"{stem}.md"


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    preset = get_dataset_preset(args.dataset)
    task_key = preset.default_task if args.task == "default" else args.task
    mixed_sites_only = False if args.all_sites else preset.default_mixed_sites_only

    records = load_matched_records(Path(preset.fc_dir), Path(preset.raw_dir))
    if not records:
        raise FileNotFoundError(
            f"No matched FC/raw records found for dataset {preset.label} "
            f"under {preset.fc_dir!r} and {preset.raw_dir!r}."
        )

    task_result = filter_records_for_task(
        records,
        task_key,
        mixed_sites_only=mixed_sites_only,
        healthy_control_label=args.healthy_control_label,
        autism_label=args.autism_label,
    )
    task_records, task_y = maybe_subsample_balanced(
        task_result.records,
        task_result.y,
        max_samples=args.max_samples,
        seed=seeds[0],
    )

    permutation, _ = get_roi_permutation(args.reorder)
    fc_features = build_fc_features(task_records, permutation=permutation, triangle=args.triangle)
    raw_features = build_raw_features(
        task_records,
        permutation=permutation,
        raw_target_timepoints=args.raw_target_timepoints,
    )

    subjects = np.asarray([str(record.subject) for record in task_records], dtype=object)
    sites = np.asarray([str(record.site) for record in task_records], dtype=object)
    site_encoder = LabelEncoder()
    y_site = site_encoder.fit_transform(sites)

    method_features = {
        "raw": raw_features,
        "fc": fc_features,
        "fc_combat": transductive_transform_features(
            "fc_combat",
            fc_features,
            sites,
            seed=seeds[0],
            covbat_n_pc=args.covbat_n_pc,
            covbat_pct_var=args.covbat_pct_var,
            covbat_max_components=args.covbat_max_components,
        ),
        "fc_covbat": transductive_transform_features(
            "fc_covbat",
            fc_features,
            sites,
            seed=seeds[0],
            covbat_n_pc=args.covbat_n_pc,
            covbat_pct_var=args.covbat_pct_var,
            covbat_max_components=args.covbat_max_components,
        ),
    }

    site_runs = {method_key: [] for method_key in method_features}
    diagnosis_runs = {method_key: [] for method_key in method_features}

    for seed in seeds:
        for method_key, features in method_features.items():
            site_runs[method_key].append(
                run_grouped_cv(
                    features,
                    y_site,
                    class_names=[str(name) for name in site_encoder.classes_.tolist()],
                    groups=subjects,
                    seed=seed,
                    folds=args.folds,
                )
            )
            diagnosis_runs[method_key].append(
                run_loso_diagnosis(
                    features,
                    task_y,
                    class_names=task_result.class_names,
                    sites=sites,
                    seed=seed,
                )
            )

    payload = {
        "config": {
            "dataset": preset.key,
            "dataset_label": preset.label,
            "fc_dir": preset.fc_dir,
            "raw_dir": preset.raw_dir,
            "task_key": task_result.task_key,
            "task_label": task_result.task_label,
            "mixed_sites_only": bool(mixed_sites_only),
            "triangle": args.triangle,
            "reorder": args.reorder,
            "raw_target_timepoints": int(args.raw_target_timepoints),
            "folds": int(args.folds),
            "seeds": list(seeds),
            "max_samples": None if args.max_samples is None else int(args.max_samples),
            "site_probe_protocol": "grouped_subject_cv",
            "diagnosis_protocol": "leave_one_site_out",
            "covbat": {
                "n_pc": int(args.covbat_n_pc),
                "pct_var": float(args.covbat_pct_var),
                "max_components": int(args.covbat_max_components),
                "harmonization_mode": "transductive_full_dataset_before_loso",
            },
        },
        "meta": {
            "n_records_total": int(len(records)),
            "n_records_post_task_filter": int(len(task_result.records)),
            "n_records_used": int(len(task_records)),
            "n_subjects": int(len(set(subjects.tolist()))),
            "n_sites": int(len(set(sites.tolist()))),
            "site_classes": [str(name) for name in site_encoder.classes_.tolist()],
            "diagnosis_classes": list(task_result.class_names),
            "diagnosis_class_counts": class_counts(task_y, task_result.class_names),
            "removed_unknown": int(task_result.removed_unknown),
            "removed_task_mismatch": int(task_result.removed_task_mismatch),
            "removed_single_class_sites": int(task_result.removed_single_class_sites),
            "mixed_sites_kept": list(task_result.mixed_sites_kept),
        },
        "per_seed": {
            "site": site_runs,
            "diagnosis": diagnosis_runs,
        },
        "summary": {
            "site": {
                method_key: summarize_runs(runs)
                for method_key, runs in site_runs.items()
            },
            "diagnosis": {
                method_key: summarize_runs(runs)
                for method_key, runs in diagnosis_runs.items()
            },
        },
    }

    output_json, output_md = default_output_paths(preset.key, task_result.task_key, mixed_sites_only)
    if args.output_json is not None:
        output_json = Path(args.output_json)
    if args.output_md is not None:
        output_md = Path(args.output_md)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))

    markdown = build_markdown_table(payload)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown)

    print(markdown)
    print(f"Saved JSON: {output_json}")
    print(f"Saved Markdown: {output_md}")


if __name__ == "__main__":
    main()
