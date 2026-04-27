import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate_fc_vs_raw_loso import run_loso_probe
from .plot_tsne_fc_vs_raw import (
    DEFAULT_FC_DIR,
    DEFAULT_RAW_DIR,
    apply_combat_harmonization,
    apply_covbat_harmonization,
    build_fc_features,
    build_raw_features,
    build_raw_window_bags,
    get_roi_permutation,
    load_matched_records,
    run_grouped_probe,
    run_window_bag_grouped_probe,
    run_window_bag_loso_probe,
)


DEFAULT_JSON = "results/reports/tables/abide_fc_vs_raw_site_diagnosis_probe_multiseed.json"
DEFAULT_LATEX = "results/reports/tables/abide_fc_vs_raw_site_diagnosis_probe.tex"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run matched FC-vs-raw probes with grouped-CV site prediction and LOSO diagnosis evaluation."
    )
    parser.add_argument("--fc-dir", default=DEFAULT_FC_DIR)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--dataset-label",
        default="ABIDE-I",
        help="Short dataset label used in the LaTeX caption/footnote.",
    )
    parser.add_argument("--triangle", choices=["upper", "lower"], default="upper")
    parser.add_argument(
        "--reorder",
        choices=["none", "hemi", "hemi_network"],
        default="hemi_network",
    )
    parser.add_argument(
        "--raw-mode",
        choices=["window_bag", "resample_full_scan"],
        default="window_bag",
        help="How to featurize raw ROI time series before the linear probe.",
    )
    parser.add_argument("--raw-target-timepoints", type=int, default=45)
    parser.add_argument(
        "--raw-window-definition",
        choices=["seconds", "volumes"],
        default="seconds",
        help="Define raw sliding windows in seconds (JEPA-like) or native volumes.",
    )
    parser.add_argument(
        "--raw-window-size",
        type=int,
        default=45,
        help="Window length in native volumes when --raw-window-definition=volumes.",
    )
    parser.add_argument(
        "--raw-window-stride",
        type=int,
        default=15,
        help="Window stride in native volumes when --raw-window-definition=volumes.",
    )
    parser.add_argument(
        "--raw-window-seconds",
        type=float,
        default=90.0,
        help="Window length in seconds when --raw-window-definition=seconds.",
    )
    parser.add_argument(
        "--raw-window-stride-seconds",
        type=float,
        default=30.0,
        help="Window stride in seconds when --raw-window-definition=seconds.",
    )
    parser.add_argument(
        "--raw-window-aggregation",
        choices=["mean_proba", "vote"],
        default="vote",
        help="How per-window raw predictions are aggregated to the matched sample level.",
    )
    parser.add_argument(
        "--keep-diagnosis-labels",
        default="",
        help="Optional comma-separated diagnosis labels to keep before building FC/raw features.",
    )
    parser.add_argument(
        "--binarize-diagnosis-hc-vs-disorder",
        action="store_true",
        help="Map diagnosis labels to Healthy Control vs Disorder, pooling every non-HC label as Disorder.",
    )
    parser.add_argument(
        "--healthy-control-label",
        default="Healthy Control",
        help="Diagnosis label treated as Healthy Control when --binarize-diagnosis-hc-vs-disorder is used.",
    )
    parser.add_argument(
        "--pooled-disorder-label",
        default="Disorder",
        help="Output label used for non-HC diagnoses when --binarize-diagnosis-hc-vs-disorder is used.",
    )
    parser.add_argument(
        "--include-fc-combat",
        action="store_true",
        help=(
            "Add a Static FC + ComBat row. ComBat is fit transductively on all matched samples "
            "using site as the batch variable."
        ),
    )
    parser.add_argument(
        "--combat-preserve-diagnosis",
        action="store_true",
        help=(
            "Pass diagnosis as a categorical covariate to preserve in ComBat. "
            "This uses diagnosis labels during harmonization and is label-leaky for prediction baselines."
        ),
    )
    parser.add_argument(
        "--combat-mean-only",
        action="store_true",
        help="Use mean-only ComBat harmonization instead of location-and-scale adjustment.",
    )
    parser.add_argument(
        "--combat-verbose",
        action="store_true",
        help="Do not suppress neuroCombat progress messages.",
    )
    parser.add_argument(
        "--include-fc-covbat",
        action="store_true",
        help=(
            "Add a Static FC + CovBat row using the local covbat.py implementation. "
            "CovBat is fit transductively on all matched samples using site as the batch variable."
        ),
    )
    parser.add_argument(
        "--covbat-n-pc",
        type=int,
        default=50,
        help="Number of leading PCs harmonized by CovBat. Set <=0 to use --covbat-pct-var.",
    )
    parser.add_argument(
        "--covbat-pct-var",
        type=float,
        default=0.95,
        help="Cumulative variance threshold used by CovBat when --covbat-n-pc <= 0.",
    )
    parser.add_argument(
        "--covbat-verbose",
        action="store_true",
        help="Do not suppress CovBat progress messages.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of grouped CV folds used for the site probe.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated seed list used for CV/PCA randomness.",
    )
    parser.add_argument("--output-json", default=DEFAULT_JSON)
    parser.add_argument("--output-tex", default=DEFAULT_LATEX)
    return parser.parse_args()


def parse_seed_list(seed_text: str):
    seeds = []
    for item in seed_text.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_label_list(label_text: str):
    labels = []
    for item in label_text.split(","):
        item = item.strip()
        if item:
            labels.append(item)
    return labels


def mean_and_std(values: list[float]):
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def summarize_probe_runs(
    runs: list[dict],
    metric_sources: dict[str, str] | None = None,
    passthrough_keys: list[str] | None = None,
):
    metric_sources = metric_sources or {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
    }
    summary = {}
    for out_key, metric_key in metric_sources.items():
        mean, std = mean_and_std([run[metric_key] for run in runs])
        summary[out_key] = {"mean": mean, "std": std}

    for key in (passthrough_keys or [
        "n_classes",
        "n_samples",
        "folds",
        "classes",
        "aggregation",
        "n_windows_total",
        "mean_windows_per_sample",
        "median_windows_per_sample",
        "cv",
        "n_groups",
        "n_valid_sites",
        "n_skipped_sites",
        "n_samples_total",
        "n_samples_valid",
        "valid_sites",
    ]):
        if key in runs[0]:
            value = runs[0][key]
            summary[key] = list(value) if isinstance(value, list) else value
    return summary


def format_pct(mean: float, std: float):
    return f"{100.0 * mean:.1f} $\\pm$ {100.0 * std:.1f}"


def maybe_bold(text: str, value: float, best_value: float):
    if np.isclose(value, best_value, rtol=0.0, atol=5e-4):
        return f"\\textbf{{{text}}}"
    return text


def mean_loso_fold_chance(runs: list[dict]):
    chances = []
    for run in runs:
        fold_chances = [1.0 / len(row["class_counts"]) for row in run["folds"]]
        chances.append(float(np.mean(fold_chances)))
    return float(np.mean(chances))


def filter_records_by_diagnosis(records, keep_labels: list[str]):
    if not keep_labels:
        return records
    keep = set(keep_labels)
    filtered = [record for record in records if record.diagnosis in keep]
    if not filtered:
        raise ValueError(
            "Diagnosis filtering removed every matched record. "
            f"Requested labels: {sorted(keep)}"
        )
    return filtered


def binarize_hc_vs_disorder(
    diagnoses: list[str],
    healthy_control_label: str,
    pooled_disorder_label: str,
):
    binary = [
        healthy_control_label if diagnosis == healthy_control_label else pooled_disorder_label
        for diagnosis in diagnoses
    ]
    if len(set(binary)) < 2:
        raise ValueError(
            "HC-vs-disorder binarization produced fewer than two classes. "
            f"Healthy control label was {healthy_control_label!r}."
        )
    return binary


def write_latex_table(path: Path, payload: dict):
    method_rows = [
        ("raw", "ROI time series"),
        ("fc", "Static FC"),
    ]
    if "fc_combat" in payload["summary"]["site"]:
        method_rows.append(("fc_combat", "Static FC + ComBat"))
    if "fc_covbat" in payload["summary"]["site"]:
        method_rows.append(("fc_covbat", "Static FC + CovBat"))

    site_chance = 1.0 / payload["summary"]["site"]["fc"]["n_classes"]
    if payload["summary"]["diagnosis"]["fc"].get("cv") == "leave_one_site_out":
        dx_chance = mean_loso_fold_chance(payload["per_seed"]["diagnosis"]["fc"])
    else:
        dx_chance = 1.0 / payload["summary"]["diagnosis"]["fc"]["n_classes"]

    best_site = min(
        payload["summary"]["site"][method_key]["balanced_accuracy"]["mean"]
        for method_key, _ in method_rows
    )
    best_dx = max(
        payload["summary"]["diagnosis"][method_key]["balanced_accuracy"]["mean"]
        for method_key, _ in method_rows
    )

    representation_lines = []
    for method_key, method_label in method_rows:
        site_metric = payload["summary"]["site"][method_key]["balanced_accuracy"]
        dx_metric = payload["summary"]["diagnosis"][method_key]["balanced_accuracy"]
        site_text = maybe_bold(
            format_pct(site_metric["mean"], site_metric["std"]),
            site_metric["mean"],
            best_site,
        )
        dx_text = maybe_bold(
            format_pct(dx_metric["mean"], dx_metric["std"]),
            dx_metric["mean"],
            best_dx,
        )
        representation_lines.append(f"{method_label} & {site_text} & {dx_text} \\\\")

    if payload["config"]["raw_mode"] == "window_bag":
        if payload["config"]["raw_window_definition"] == "seconds":
            raw_window_desc = (
                "%g-second sliding windows (stride %g seconds), each resampled to %d timepoints"
                % (
                    payload["config"]["raw_window_seconds"],
                    payload["config"]["raw_window_stride_seconds"],
                    payload["config"]["raw_target_timepoints"],
                )
            )
        else:
            raw_window_desc = (
                "%d native-volume sliding windows (stride %d), resampled to %d timepoints"
                % (
                    payload["config"]["raw_window_size"],
                    payload["config"]["raw_window_stride"],
                    payload["config"]["raw_target_timepoints"],
                )
            )

        raw_note = (
            "Raw ROI time series used matched sample-level %s. Each window was flattened and passed through "
            "(\\texttt{StandardScaler + PCA(50) + LogisticRegression}), then aggregated back to the sample level via %s. "
        ) % (
            raw_window_desc,
            "\\textbf{mean class probability}" if payload["config"]["raw_window_aggregation"] == "mean_proba" else "\\textbf{majority vote}",
        )
    else:
        raw_note = (
            "Raw ROI time series were resampled to %d timepoints and flattened before the linear probe. "
            % payload["config"]["raw_target_timepoints"]
        )

    diagnosis_filter_note = ""
    if payload["config"]["keep_diagnosis_labels"]:
        diagnosis_filter_note = (
            " Diagnosis tasks retained labels \\{%s\\}; all other diagnosis labels were excluded before FC/raw feature construction."
            % ", ".join(payload["config"]["keep_diagnosis_labels"])
        )
    diagnosis_binarize_note = ""
    if payload["config"].get("diagnosis_binarization", {}).get("enabled"):
        bin_config = payload["config"]["diagnosis_binarization"]
        diagnosis_binarize_note = (
            " Diagnosis labels were binarized as %s vs %s, with every non-%s diagnosis pooled as %s."
            % (
                bin_config["healthy_control_label"],
                bin_config["pooled_disorder_label"],
                bin_config["healthy_control_label"],
                bin_config["pooled_disorder_label"],
            )
        )

    loso_note = ""
    n_skipped_sites = max(
        int(payload["summary"]["diagnosis"]["fc"].get("n_skipped_sites", 0)),
        int(payload["summary"]["diagnosis"]["raw"].get("n_skipped_sites", 0)),
    )
    if n_skipped_sites > 0:
        loso_note = (
            " %d site(s) were excluded from diagnosis LOSO scoring because the held-out site contained fewer than two diagnosis classes."
            % n_skipped_sites
        )

    combat_note = ""
    combat_config = payload["config"].get("combat", {})
    if combat_config.get("include_fc_combat"):
        combat_note = (
            " The ComBat row harmonized static FC features with \\texttt{neuroCombat} using site as the batch variable before probing. "
            "This is a \\textbf{transductive} harmonization baseline because ComBat batch-effect estimates are fit on all matched samples."
        )
        if combat_config.get("preserve_diagnosis"):
            combat_note += (
                " Diagnosis was also passed as a categorical covariate to preserve, so that row uses diagnosis labels during harmonization."
            )
        else:
            combat_note += " Diagnosis labels were not passed to ComBat."

    covbat_note = ""
    covbat_config = payload["config"].get("covbat", {})
    if covbat_config.get("include_fc_covbat"):
        covbat_note = (
            " The CovBat row harmonized static FC features using the local \\texttt{covbat.py} implementation with site as the batch variable before probing. "
            "This is also a \\textbf{transductive} harmonization baseline because CovBat batch-effect estimates are fit on all matched samples. "
            "Diagnosis labels were not passed to CovBat."
        )

    lines = [
        "\\begin{table}[t]",
        "\\vspace{0.5cm}",
        "\\centering",
        "\\caption{%s linear-probe comparison for ROI time-series inputs and static FC features (mean $\\pm$ standard deviation across %d random seed%s).}"
        % (
            payload["config"]["dataset_label"],
            len(payload["config"]["seeds"]),
            "" if len(payload["config"]["seeds"]) == 1 else "s",
        ),
        "\\label{tab:abide_fc_vs_raw_site_diagnosis_probe}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "& Site probe (%d-way; grouped CV) & Diagnosis probe (%d-way; LOSO) \\\\"
        % (payload["meta"]["n_sites"], payload["meta"]["n_diagnosis_classes"]),
        "\\cmidrule(lr){2-2}\\cmidrule(lr){3-3}",
        "Representation & Balanced Acc. ($\\downarrow$) & Balanced Acc. ($\\uparrow$) \\\\",
        "\\midrule",
        *representation_lines,
        f"Chance & {100.0 * site_chance:.1f} & {100.0 * dx_chance:.1f} \\\\",
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\vspace{0.2cm}",
        "\\begin{minipage}{0.96\\columnwidth}",
        (
            "\\footnotesize Matched %s scans only ($n=%d$; %d sites; %d diagnosis classes). %s"
            "Static FC used the %s triangle of the $400 \\times 400$ matrix with "
            "(\\texttt{StandardScaler + PCA(50) + LogisticRegression}). "
            "The site probe used grouped %d-fold cross-validation with subject-level grouping. "
            "The diagnosis probe used leave-one-site-out evaluation.%s%s%s "
            "Reported values are mean $\\pm$ standard deviation across seeds \\{%s\\}.%s%s"
        )
        % (
            payload["config"]["dataset_label"],
            payload["meta"]["n_samples"],
            payload["meta"]["n_sites"],
            payload["meta"]["n_diagnosis_classes"],
            raw_note,
            payload["config"]["triangle"],
            payload["config"]["folds"],
            diagnosis_filter_note,
            diagnosis_binarize_note,
            loso_note,
            ", ".join(str(seed) for seed in payload["config"]["seeds"]),
            combat_note,
            covbat_note,
        ),
        "\\end{minipage}",
        "\\end{table}",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    keep_diagnosis_labels = parse_label_list(args.keep_diagnosis_labels)

    records = load_matched_records(Path(args.fc_dir), Path(args.raw_dir))
    if not records:
        raise FileNotFoundError("No matched FC/raw ABIDE records found.")
    records = filter_records_by_diagnosis(records, keep_diagnosis_labels)

    permutation, _ = get_roi_permutation(args.reorder)
    raw_features = None
    raw_window_bags = None

    if args.raw_mode == "window_bag":
        if args.raw_window_definition == "seconds":
            records, raw_window_bags = build_raw_window_bags(
                records,
                permutation=permutation,
                target_timepoints=args.raw_target_timepoints,
                window_seconds=args.raw_window_seconds,
                stride_seconds=args.raw_window_stride_seconds,
            )
        else:
            records, raw_window_bags = build_raw_window_bags(
                records,
                permutation=permutation,
                target_timepoints=args.raw_target_timepoints,
                window_size=args.raw_window_size,
                stride=args.raw_window_stride,
            )
    else:
        raw_features = build_raw_features(
            records,
            permutation=permutation,
            raw_target_timepoints=args.raw_target_timepoints,
        )

    fc_features = build_fc_features(records, permutation=permutation, triangle=args.triangle)
    sites = [record.site for record in records]
    diagnoses = [record.diagnosis for record in records]
    if args.binarize_diagnosis_hc_vs_disorder:
        diagnoses = binarize_hc_vs_disorder(
            diagnoses,
            healthy_control_label=args.healthy_control_label,
            pooled_disorder_label=args.pooled_disorder_label,
        )
    subjects = [record.subject for record in records]
    fc_combat_features = None
    if args.include_fc_combat:
        combat_covariates = {}
        if args.combat_preserve_diagnosis:
            combat_covariates["diagnosis"] = diagnoses
        fc_combat_features = apply_combat_harmonization(
            fc_features,
            sites,
            categorical_covariates=combat_covariates,
            mean_only=args.combat_mean_only,
            verbose=args.combat_verbose,
        )
    fc_covbat_features = None
    if args.include_fc_covbat:
        fc_covbat_features = apply_covbat_harmonization(
            fc_features,
            sites,
            n_pc=args.covbat_n_pc,
            pct_var=args.covbat_pct_var,
            verbose=args.covbat_verbose,
        )

    site_fc_runs = []
    site_raw_runs = []
    site_fc_combat_runs = []
    site_fc_covbat_runs = []
    dx_fc_runs = []
    dx_raw_runs = []
    dx_fc_combat_runs = []
    dx_fc_covbat_runs = []

    for seed in seeds:
        site_fc_runs.append(
            run_grouped_probe(
                fc_features,
                sites,
                groups=subjects,
                seed=seed,
                folds=args.folds,
                task_name="site",
            )
        )
        dx_fc_runs.append(run_loso_probe(fc_features, diagnoses, sites, seed=seed))
        if fc_combat_features is not None:
            site_fc_combat_runs.append(
                run_grouped_probe(
                    fc_combat_features,
                    sites,
                    groups=subjects,
                    seed=seed,
                    folds=args.folds,
                    task_name="site",
                )
            )
            dx_fc_combat_runs.append(run_loso_probe(fc_combat_features, diagnoses, sites, seed=seed))
        if fc_covbat_features is not None:
            site_fc_covbat_runs.append(
                run_grouped_probe(
                    fc_covbat_features,
                    sites,
                    groups=subjects,
                    seed=seed,
                    folds=args.folds,
                    task_name="site",
                )
            )
            dx_fc_covbat_runs.append(run_loso_probe(fc_covbat_features, diagnoses, sites, seed=seed))
        if args.raw_mode == "window_bag":
            site_raw_runs.append(
                run_window_bag_grouped_probe(
                    raw_window_bags,
                    sites,
                    groups=subjects,
                    seed=seed,
                    folds=args.folds,
                    task_name="site",
                    aggregation=args.raw_window_aggregation,
                )
            )
            dx_raw_runs.append(
                run_window_bag_loso_probe(
                    raw_window_bags,
                    diagnoses,
                    sites=sites,
                    seed=seed,
                    aggregation=args.raw_window_aggregation,
                )
            )
        else:
            site_raw_runs.append(
                run_grouped_probe(
                    raw_features,
                    sites,
                    groups=subjects,
                    seed=seed,
                    folds=args.folds,
                    task_name="site",
                )
            )
            dx_raw_runs.append(run_loso_probe(raw_features, diagnoses, sites, seed=seed))

    payload = {
        "config": {
            "fc_dir": args.fc_dir,
            "raw_dir": args.raw_dir,
            "dataset_label": args.dataset_label,
            "triangle": args.triangle,
            "reorder": args.reorder,
            "raw_mode": args.raw_mode,
            "raw_target_timepoints": args.raw_target_timepoints,
            "raw_window_definition": args.raw_window_definition,
            "raw_window_size": args.raw_window_size,
            "raw_window_stride": args.raw_window_stride,
            "raw_window_seconds": args.raw_window_seconds,
            "raw_window_stride_seconds": args.raw_window_stride_seconds,
            "raw_window_aggregation": args.raw_window_aggregation,
            "keep_diagnosis_labels": keep_diagnosis_labels,
            "diagnosis_binarization": {
                "enabled": bool(args.binarize_diagnosis_hc_vs_disorder),
                "healthy_control_label": args.healthy_control_label,
                "pooled_disorder_label": args.pooled_disorder_label,
            },
            "folds": args.folds,
            "seeds": seeds,
            "combat": {
                "include_fc_combat": bool(args.include_fc_combat),
                "target": "static_fc" if args.include_fc_combat else None,
                "protocol": "transductive_global",
                "batch_variable": "site",
                "preserve_diagnosis": bool(args.combat_preserve_diagnosis),
                "mean_only": bool(args.combat_mean_only),
                "eb": True,
                "parametric": True,
            },
            "covbat": {
                "include_fc_covbat": bool(args.include_fc_covbat),
                "target": "static_fc" if args.include_fc_covbat else None,
                "protocol": "transductive_global",
                "batch_variable": "site",
                "preserve_diagnosis": False,
                "n_pc": int(args.covbat_n_pc),
                "pct_var": float(args.covbat_pct_var),
            },
        },
        "meta": {
            "n_samples": int(len(records)),
            "n_subjects": int(len(set(subjects))),
            "n_sites": int(len(set(sites))),
            "n_diagnosis_classes": int(len(set(diagnoses))),
        },
        "per_seed": {
            "site": {"fc": site_fc_runs, "raw": site_raw_runs},
            "diagnosis": {"fc": dx_fc_runs, "raw": dx_raw_runs},
        },
        "summary": {
            "site": {
                "fc": summarize_probe_runs(site_fc_runs),
                "raw": summarize_probe_runs(site_raw_runs),
            },
            "diagnosis": {
                "fc": summarize_probe_runs(
                    dx_fc_runs,
                    metric_sources={
                        "accuracy": "mean_fold_accuracy",
                        "balanced_accuracy": "mean_fold_balanced_accuracy",
                        "pooled_accuracy": "pooled_accuracy",
                        "pooled_balanced_accuracy": "pooled_balanced_accuracy",
                    },
                    passthrough_keys=[
                        "n_classes",
                        "classes",
                        "n_samples_total",
                        "n_samples_valid",
                        "n_valid_sites",
                        "n_skipped_sites",
                        "valid_sites",
                        "cv",
                    ],
                ),
                "raw": summarize_probe_runs(
                    dx_raw_runs,
                    metric_sources={
                        "accuracy": "mean_fold_accuracy",
                        "balanced_accuracy": "mean_fold_balanced_accuracy",
                        "pooled_accuracy": "pooled_accuracy",
                        "pooled_balanced_accuracy": "pooled_balanced_accuracy",
                    },
                    passthrough_keys=[
                        "n_classes",
                        "classes",
                        "n_samples_total",
                        "n_samples_valid",
                        "n_valid_sites",
                        "n_skipped_sites",
                        "valid_sites",
                        "aggregation",
                        "n_windows_total",
                        "mean_windows_per_sample",
                        "median_windows_per_sample",
                        "cv",
                    ],
                ),
            },
        },
    }

    if args.include_fc_combat:
        payload["per_seed"]["site"]["fc_combat"] = site_fc_combat_runs
        payload["per_seed"]["diagnosis"]["fc_combat"] = dx_fc_combat_runs
        payload["summary"]["site"]["fc_combat"] = summarize_probe_runs(site_fc_combat_runs)
        payload["summary"]["diagnosis"]["fc_combat"] = summarize_probe_runs(
            dx_fc_combat_runs,
            metric_sources={
                "accuracy": "mean_fold_accuracy",
                "balanced_accuracy": "mean_fold_balanced_accuracy",
                "pooled_accuracy": "pooled_accuracy",
                "pooled_balanced_accuracy": "pooled_balanced_accuracy",
            },
            passthrough_keys=[
                "n_classes",
                "classes",
                "n_samples_total",
                "n_samples_valid",
                "n_valid_sites",
                "n_skipped_sites",
                "valid_sites",
                "cv",
            ],
        )
    if args.include_fc_covbat:
        payload["per_seed"]["site"]["fc_covbat"] = site_fc_covbat_runs
        payload["per_seed"]["diagnosis"]["fc_covbat"] = dx_fc_covbat_runs
        payload["summary"]["site"]["fc_covbat"] = summarize_probe_runs(site_fc_covbat_runs)
        payload["summary"]["diagnosis"]["fc_covbat"] = summarize_probe_runs(
            dx_fc_covbat_runs,
            metric_sources={
                "accuracy": "mean_fold_accuracy",
                "balanced_accuracy": "mean_fold_balanced_accuracy",
                "pooled_accuracy": "pooled_accuracy",
                "pooled_balanced_accuracy": "pooled_balanced_accuracy",
            },
            passthrough_keys=[
                "n_classes",
                "classes",
                "n_samples_total",
                "n_samples_valid",
                "n_valid_sites",
                "n_skipped_sites",
                "valid_sites",
                "cv",
            ],
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))

    output_tex = Path(args.output_tex)
    write_latex_table(output_tex, payload)

    site_fc_bacc = payload["summary"]["site"]["fc"]["balanced_accuracy"]
    site_raw_bacc = payload["summary"]["site"]["raw"]["balanced_accuracy"]
    dx_fc_bacc = payload["summary"]["diagnosis"]["fc"]["balanced_accuracy"]
    dx_raw_bacc = payload["summary"]["diagnosis"]["raw"]["balanced_accuracy"]

    print(f"Saved JSON: {output_json}")
    print(f"Saved LaTeX: {output_tex}")
    print(
        "Site bAcc (grouped CV) | RAW vs FC | "
        f"{100.0 * site_raw_bacc['mean']:.1f}±{100.0 * site_raw_bacc['std']:.1f} vs "
        f"{100.0 * site_fc_bacc['mean']:.1f}±{100.0 * site_fc_bacc['std']:.1f}"
    )
    print(
        "Dx bAcc (LOSO) | RAW vs FC | "
        f"{100.0 * dx_raw_bacc['mean']:.1f}±{100.0 * dx_raw_bacc['std']:.1f} vs "
        f"{100.0 * dx_fc_bacc['mean']:.1f}±{100.0 * dx_fc_bacc['std']:.1f}"
    )
    if args.include_fc_combat:
        site_fc_combat_bacc = payload["summary"]["site"]["fc_combat"]["balanced_accuracy"]
        dx_fc_combat_bacc = payload["summary"]["diagnosis"]["fc_combat"]["balanced_accuracy"]
        print(
            "ComBat bAcc | site grouped CV / dx LOSO | "
            f"{100.0 * site_fc_combat_bacc['mean']:.1f}±{100.0 * site_fc_combat_bacc['std']:.1f} / "
            f"{100.0 * dx_fc_combat_bacc['mean']:.1f}±{100.0 * dx_fc_combat_bacc['std']:.1f}"
        )
    if args.include_fc_covbat:
        site_fc_covbat_bacc = payload["summary"]["site"]["fc_covbat"]["balanced_accuracy"]
        dx_fc_covbat_bacc = payload["summary"]["diagnosis"]["fc_covbat"]["balanced_accuracy"]
        print(
            "CovBat bAcc | site grouped CV / dx LOSO | "
            f"{100.0 * site_fc_covbat_bacc['mean']:.1f}±{100.0 * site_fc_covbat_bacc['std']:.1f} / "
            f"{100.0 * dx_fc_covbat_bacc['mean']:.1f}±{100.0 * dx_fc_covbat_bacc['std']:.1f}"
        )


if __name__ == "__main__":
    main()
