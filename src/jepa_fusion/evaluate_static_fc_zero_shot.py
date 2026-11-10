from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

try:
    from .dataset_utils import matched_records_for_dataset
    from .evaluate_baselines import (
        ExplicitTransductiveCombat,
        _apply_batch_location_scale,
        _fit_batch_location_scale,
        mean_std,
        parse_seed_list,
        run_grouped_cv,
    )
    from .protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        class_counts,
        filter_records_for_task,
        get_dataset_preset,
    )
    from ..jepa_clean.plot_tsne_fc_vs_raw import build_fc_features, get_roi_permutation
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.dataset_utils import matched_records_for_dataset
    from src.jepa_fusion.evaluate_baselines import (
        ExplicitTransductiveCombat,
        _apply_batch_location_scale,
        _fit_batch_location_scale,
        mean_std,
        parse_seed_list,
        run_grouped_cv,
    )
    from src.jepa_fusion.protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        class_counts,
        filter_records_for_task,
        get_dataset_preset,
    )
    from src.jepa_clean.plot_tsne_fc_vs_raw import build_fc_features, get_roi_permutation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a source-only Static-FC classifier and evaluate zero-shot on a target dataset."
    )
    parser.add_argument(
        "--source-datasets",
        required=True,
        help="Comma-separated source dataset keys, e.g. srpbs,abide1.",
    )
    parser.add_argument("--target-dataset", choices=["abide", "abide1", "abide2", "srpbs"], required=True)
    parser.add_argument(
        "--task",
        choices=["default", "hc_asd", "hc_disorder"],
        default="default",
        help=(
            "Diagnosis task. Default uses HC-vs-disorder when SRPBS participates, "
            "otherwise the target dataset default."
        ),
    )
    parser.add_argument("--all-sites", action="store_true", help="Disable mixed-site filtering.")
    parser.add_argument(
        "--exclude-target-sites",
        default="",
        help="Comma-separated target center or batch site labels to exclude before evaluation.",
    )
    parser.add_argument("--triangle", choices=["upper", "lower"], default="upper")
    parser.add_argument("--reorder", choices=["none", "hemi", "hemi_network"], default="hemi_network")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument(
        "--methods",
        default="fc,fc_combat,fc_covbat",
        help="Comma-separated methods: fc,fc_combat,fc_covbat.",
    )
    parser.add_argument("--covbat-n-pc", type=int, default=50)
    parser.add_argument("--covbat-pct-var", type=float, default=0.95)
    parser.add_argument("--covbat-max-components", type=int, default=256)
    parser.add_argument("--healthy-control-label", default=HEALTHY_CONTROL_LABEL)
    parser.add_argument("--autism-label", default=AUTISM_LABEL)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def parse_dataset_list(text: str) -> list[str]:
    datasets = []
    for item in text.split(","):
        key = item.strip()
        if key:
            get_dataset_preset(key)
            datasets.append(key)
    if not datasets:
        raise ValueError("At least one source dataset is required.")
    return datasets


def parse_methods(text: str) -> list[str]:
    methods = []
    valid = {"fc", "fc_combat", "fc_covbat"}
    for item in text.split(","):
        method = item.strip()
        if method:
            if method not in valid:
                raise ValueError(f"Unsupported method {method!r}; expected one of {sorted(valid)}.")
            methods.append(method)
    if not methods:
        raise ValueError("At least one method is required.")
    return methods


def resolve_task(task_arg: str, source_datasets: list[str], target_dataset: str) -> str:
    if task_arg != "default":
        return task_arg
    if "srpbs" in set(source_datasets + [target_dataset]):
        return "hc_disorder"
    return get_dataset_preset(target_dataset).default_task


def load_records(dataset_keys: list[str]):
    records = []
    for key in dataset_keys:
        records.extend(matched_records_for_dataset(key))
    if not records:
        raise FileNotFoundError(f"No records found for dataset keys: {dataset_keys}")
    return records


def exclude_records(records: list, y: np.ndarray, exclude_sites: set[str]):
    if not exclude_sites:
        return records, y, 0
    kept_records = []
    kept_y = []
    removed = 0
    for record, label in zip(records, y.tolist()):
        labels = {str(record.site)}
        if hasattr(record, "site_batch"):
            labels.add(str(record.site_batch))
        if labels & exclude_sites:
            removed += 1
            continue
        kept_records.append(record)
        kept_y.append(label)
    if not kept_records:
        raise RuntimeError("Target-site exclusion removed every evaluation record.")
    kept_y_arr = np.asarray(kept_y, dtype=np.int64)
    if len(np.unique(kept_y_arr)) < 2:
        raise RuntimeError("Target-site exclusion left fewer than two diagnosis classes.")
    return kept_records, kept_y_arr, removed


def make_probe(feature_dim: int, train_count: int, seed: int, pca_components: int):
    n_components = min(int(pca_components), int(feature_dim), max(1, int(train_count) - 1))
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=seed),
        LogisticRegression(max_iter=10000, random_state=seed, class_weight="balanced"),
    )


def transform_source_target(
    method: str,
    x_source: np.ndarray,
    x_target: np.ndarray,
    source_sites: np.ndarray,
    target_sites: np.ndarray,
    *,
    seed: int,
    covbat_n_pc: int,
    covbat_pct_var: float,
    covbat_max_components: int,
):
    if method == "fc":
        return x_source, x_target

    x_all = np.concatenate([x_source, x_target], axis=0)
    sites_all = np.concatenate([source_sites, target_sites], axis=0).astype(object)
    covariates = np.zeros((len(x_all), 0), dtype=np.float64)

    combat = ExplicitTransductiveCombat().fit(x_all, sites_all, covariates)
    combat_features = combat.transform(x_all, sites_all, covariates)
    if method == "fc_combat":
        return combat_features[: len(x_source)], combat_features[len(x_source):]
    if method != "fc_covbat":
        raise ValueError(f"Unsupported method: {method}")

    scaler = StandardScaler()
    z_all = scaler.fit_transform(combat_features)
    max_components = min(int(covbat_max_components), z_all.shape[0] - 1, z_all.shape[1])
    max_components = max(1, max_components)
    pca = PCA(n_components=max_components, svd_solver="randomized", random_state=seed)
    pca.fit(z_all)
    if covbat_n_pc > 0:
        n_pc = min(int(covbat_n_pc), pca.n_components_)
    else:
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        n_pc = int(min(pca.n_components_, np.searchsorted(cumulative, covbat_pct_var, side="left") + 1))

    scores_all = pca.transform(z_all)
    score_model = _fit_batch_location_scale(scores_all[:, :n_pc], sites_all)
    adjusted_scores = np.array(scores_all, copy=True)
    adjusted_scores[:, :n_pc] = _apply_batch_location_scale(scores_all[:, :n_pc], sites_all, score_model)
    z_covbat = pca.inverse_transform(adjusted_scores)
    covbat_features = scaler.inverse_transform(z_covbat)
    covbat_features = np.nan_to_num(
        np.asarray(covbat_features, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return covbat_features[: len(x_source)], covbat_features[len(x_source):]


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray):
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


def align_scores(clf, raw_scores: np.ndarray, n_classes: int):
    classes = clf.named_steps["logisticregression"].classes_
    aligned = np.zeros((raw_scores.shape[0], n_classes), dtype=np.float64)
    aligned[:, classes] = raw_scores
    return aligned


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, class_names: list[str]):
    row = {
        "n_samples": int(len(y_true)),
        "classes": list(class_names),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    row["roc_auc"] = safe_roc_auc(y_true, scores[:, 1] if scores.shape[1] == 2 else scores)
    return row


def aggregate_subject_scores(records: list, y: np.ndarray, scores: np.ndarray):
    grouped = defaultdict(lambda: {"scores": [], "labels": [], "site": None})
    for record, label, score in zip(records, y.tolist(), scores):
        key = str(record.subject_id)
        grouped[key]["scores"].append(np.asarray(score, dtype=np.float64))
        grouped[key]["labels"].append(int(label))
        grouped[key]["site"] = str(record.site)

    subject_ids = []
    subject_sites = []
    subject_y = []
    subject_scores = []
    for subject_id in sorted(grouped):
        labels = set(grouped[subject_id]["labels"])
        if len(labels) != 1:
            raise RuntimeError(f"Inconsistent labels for subject {subject_id}.")
        subject_ids.append(subject_id)
        subject_sites.append(grouped[subject_id]["site"])
        subject_y.append(next(iter(labels)))
        subject_scores.append(np.mean(np.stack(grouped[subject_id]["scores"], axis=0), axis=0))
    return (
        np.asarray(subject_ids, dtype=object),
        np.asarray(subject_sites, dtype=object),
        np.asarray(subject_y, dtype=np.int64),
        np.stack(subject_scores, axis=0),
    )


def summarize_runs(runs: list[dict]):
    return {
        "accuracy": mean_std([run["accuracy"] for run in runs]),
        "balanced_accuracy": mean_std([run["balanced_accuracy"] for run in runs]),
        "roc_auc": mean_std([run["roc_auc"] for run in runs]),
        "f1_macro": mean_std([run["f1_macro"] for run in runs]),
        "n_runs": int(len(runs)),
        "n_samples": int(runs[0]["n_samples"]),
        "classes": list(runs[0]["classes"]),
    }


def summarize_site_runs(runs: list[dict]):
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


METHOD_LABELS = {
    "fc": "Static FC",
    "fc_combat": "FC + ComBat",
    "fc_covbat": "FC + CovBat",
}


def default_output_paths(source_datasets: list[str], target_dataset: str, task_key: str):
    source_key = "_".join(source_datasets)
    stem = f"static_fc_zero_shot_{source_key}_to_{target_dataset}_{task_key}"
    base = Path("results/reports/tables/jepa_fusion/static_fc_zero_shot")
    return base / f"{stem}.json", base / f"{stem}.md"


def build_markdown(payload: dict):
    lines = [
        "# Static-FC Source-Only Zero-Shot",
        "",
        f"Source datasets: `{', '.join(payload['config']['source_datasets'])}`",
        f"Target dataset: `{payload['config']['target_dataset']}`",
        f"Task: `{payload['config']['task_key']}`",
        "",
        (
            "Diagnosis classifiers are fit on source labels only. ComBat/CovBat rows use "
            "unsupervised transductive harmonization on source+target Static-FC features; "
            "target diagnosis labels are not used."
        ),
        "",
        "| Method | Site bAcc | Site ROC-AUC | Subject Dx bAcc | Subject Dx ROC-AUC | Subject F1 | N subjects |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in payload["config"]["methods"]:
        method_summary = payload["summary"][method]
        site_summary = method_summary["site"]
        subject_summary = method_summary["subject"]
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{format_pct(site_summary['balanced_accuracy'])} | "
            f"{format_pct(site_summary['roc_auc'])} | "
            f"{format_pct(subject_summary['balanced_accuracy'])} | "
            f"{format_pct(subject_summary['roc_auc'])} | "
            f"{format_pct(subject_summary['f1_macro'])} | "
            f"{subject_summary['n_samples']} |"
        )
    if payload["config"]["exclude_target_sites"]:
        lines.extend(
            [
                "",
                f"Excluded target sites: `{', '.join(payload['config']['exclude_target_sites'])}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    source_datasets = parse_dataset_list(args.source_datasets)
    target_dataset = args.target_dataset
    seeds = parse_seed_list(args.seeds)
    methods = parse_methods(args.methods)
    task_key = resolve_task(args.task, source_datasets, target_dataset)
    exclude_sites = {item.strip() for item in args.exclude_target_sites.split(",") if item.strip()}
    mixed_sites_only = not bool(args.all_sites)

    source_records_all = load_records(source_datasets)
    target_records_all = load_records([target_dataset])

    source_task = filter_records_for_task(
        source_records_all,
        task_key,
        mixed_sites_only=mixed_sites_only,
        healthy_control_label=args.healthy_control_label,
        autism_label=args.autism_label,
    )
    target_task = filter_records_for_task(
        target_records_all,
        task_key,
        mixed_sites_only=mixed_sites_only,
        healthy_control_label=args.healthy_control_label,
        autism_label=args.autism_label,
    )
    target_records, target_y, removed_by_exclusion = exclude_records(
        target_task.records,
        target_task.y,
        exclude_sites,
    )

    permutation, _ = get_roi_permutation(args.reorder)
    x_source = build_fc_features(source_task.records, permutation=permutation, triangle=args.triangle)
    x_target = build_fc_features(target_records, permutation=permutation, triangle=args.triangle)
    source_batch_sites = np.asarray([str(record.site_batch) for record in source_task.records], dtype=object)
    target_batch_sites = np.asarray([str(record.site_batch) for record in target_records], dtype=object)

    if len(np.unique(source_task.y)) < 2:
        raise RuntimeError("Source data has fewer than two diagnosis classes.")
    if len(np.unique(target_y)) < 2:
        raise RuntimeError("Target data has fewer than two diagnosis classes.")

    target_subjects = np.asarray([str(record.subject_id) for record in target_records], dtype=object)
    site_encoder = LabelEncoder()
    target_site_y = site_encoder.fit_transform(target_batch_sites)
    site_class_names = [str(name) for name in site_encoder.classes_.tolist()]

    n_classes = len(source_task.class_names)
    runs_by_method = {}
    summary_by_method = {}
    for method in methods:
        transformed_source, transformed_target = transform_source_target(
            method,
            x_source,
            x_target,
            source_batch_sites,
            target_batch_sites,
            seed=seeds[0],
            covbat_n_pc=args.covbat_n_pc,
            covbat_pct_var=args.covbat_pct_var,
            covbat_max_components=args.covbat_max_components,
        )
        site_runs = []
        record_runs = []
        subject_runs = []
        run_rows = []
        for seed in seeds:
            site_metrics = run_grouped_cv(
                transformed_target,
                target_site_y,
                site_class_names,
                target_subjects,
                seed=seed,
                folds=args.folds,
            )
            clf = make_probe(
                feature_dim=transformed_source.shape[1],
                train_count=transformed_source.shape[0],
                seed=seed,
                pca_components=args.pca_components,
            )
            clf.fit(transformed_source, source_task.y)
            record_pred = clf.predict(transformed_target)
            record_scores = align_scores(clf, clf.predict_proba(transformed_target), n_classes=n_classes)
            record_metrics = metric_row(target_y, record_pred, record_scores, source_task.class_names)

            subject_ids, subject_sites, subject_y, subject_scores = aggregate_subject_scores(
                target_records,
                target_y,
                record_scores,
            )
            subject_pred = np.argmax(subject_scores, axis=1)
            subject_metrics = metric_row(subject_y, subject_pred, subject_scores, source_task.class_names)
            subject_metrics["n_subjects"] = int(len(subject_ids))
            subject_metrics["sites"] = sorted(set(str(site) for site in subject_sites.tolist()))

            site_runs.append(site_metrics)
            record_runs.append(record_metrics)
            subject_runs.append(subject_metrics)
            run_rows.append(
                {
                    "seed": int(seed),
                    "site": site_metrics,
                    "record": record_metrics,
                    "subject": subject_metrics,
                }
            )
        runs_by_method[method] = run_rows
        summary_by_method[method] = {
            "site": summarize_site_runs(site_runs),
            "record": summarize_runs(record_runs),
            "subject": summarize_runs(subject_runs),
        }

    payload = {
        "config": {
            "source_datasets": list(source_datasets),
            "target_dataset": target_dataset,
            "task_key": task_key,
            "task_label": source_task.task_label,
            "mixed_sites_only": bool(mixed_sites_only),
            "exclude_target_sites": sorted(exclude_sites),
            "triangle": args.triangle,
            "reorder": args.reorder,
            "folds": int(args.folds),
            "seeds": list(seeds),
            "pca_components": int(args.pca_components),
            "methods": list(methods),
            "method": "static_fc_source_only_zero_shot",
            "harmonization_protocol": "source_plus_target_unsupervised_transductive_for_combat_covbat",
            "covbat": {
                "n_pc": int(args.covbat_n_pc),
                "pct_var": float(args.covbat_pct_var),
                "max_components": int(args.covbat_max_components),
            },
        },
        "meta": {
            "source": {
                "n_records_total": int(len(source_records_all)),
                "n_records_used": int(len(source_task.records)),
                "n_subjects": int(len(set(str(r.subject_id) for r in source_task.records))),
                "n_centers": int(len(set(str(r.site) for r in source_task.records))),
                "diagnosis_class_counts": class_counts(source_task.y, source_task.class_names),
                "removed_unknown": int(source_task.removed_unknown),
                "removed_task_mismatch": int(source_task.removed_task_mismatch),
                "removed_single_class_sites": int(source_task.removed_single_class_sites),
            },
            "target": {
                "n_records_total": int(len(target_records_all)),
                "n_records_post_task_filter": int(len(target_task.records)),
                "n_records_used": int(len(target_records)),
                "n_subjects": int(len(set(str(r.subject_id) for r in target_records))),
                "n_centers": int(len(set(str(r.site) for r in target_records))),
                "n_site_probe_classes": int(len(site_class_names)),
                "site_probe_classes": list(site_class_names),
                "diagnosis_class_counts": class_counts(target_y, target_task.class_names),
                "removed_unknown": int(target_task.removed_unknown),
                "removed_task_mismatch": int(target_task.removed_task_mismatch),
                "removed_single_class_sites": int(target_task.removed_single_class_sites),
                "removed_by_excluded_target_sites": int(removed_by_exclusion),
                "target_centers": sorted(set(str(r.site) for r in target_records)),
            },
        },
        "runs": runs_by_method,
        "summary": summary_by_method,
    }

    output_json, output_md = default_output_paths(source_datasets, target_dataset, task_key)
    if args.output_json is not None:
        output_json = Path(args.output_json)
    if args.output_md is not None:
        output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    output_md.write_text(build_markdown(payload))
    print(build_markdown(payload))
    print(f"Saved JSON: {output_json}")
    print(f"Saved Markdown: {output_md}")


if __name__ == "__main__":
    main()
