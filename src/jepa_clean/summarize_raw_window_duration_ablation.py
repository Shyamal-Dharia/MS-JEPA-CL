import argparse
import json
from pathlib import Path
import re

import numpy as np

from .plot_tsne_fc_vs_raw import (
    DEFAULT_FC_DIR,
    DEFAULT_RAW_DIR,
    build_raw_window_bags,
    get_roi_permutation,
    load_matched_records,
    maybe_subsample_records,
    run_window_bag_grouped_probe,
    run_window_bag_loso_probe,
)


DEFAULT_JSON = "results/reports/tables/raw_roi_window_duration_ablation_multiseed.json"
DEFAULT_TEX = "results/reports/tables/raw_roi_window_duration_ablation.tex"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run raw ROI window-duration ablation with grouped-CV site and LOSO diagnosis probes."
    )
    parser.add_argument("--fc-dir", default=DEFAULT_FC_DIR)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--dataset-label", default="ABIDE-I")
    parser.add_argument(
        "--window-seconds",
        default="60,90,120,150,180",
        help="Comma-separated raw window durations in seconds.",
    )
    parser.add_argument("--stride-seconds", type=float, default=30.0)
    parser.add_argument("--target-timepoints", default="45", help="Comma-separated target timepoints, or a single value to use for all windows.")
    parser.add_argument(
        "--raw-window-aggregation",
        choices=["mean_proba", "vote"],
        default="vote",
        help="How per-window predictions are aggregated to matched-sample level.",
    )
    parser.add_argument(
        "--keep-diagnosis-labels",
        default="",
        help="Optional comma-separated diagnosis labels to keep before feature construction.",
    )
    parser.add_argument(
        "--binarize-diagnosis-hc-vs-disorder",
        action="store_true",
        help="Map diagnosis labels to Healthy Control vs Disorder, pooling every non-HC label as Disorder.",
    )
    parser.add_argument("--healthy-control-label", default="Healthy Control")
    parser.add_argument("--pooled-disorder-label", default="Disorder")
    parser.add_argument(
        "--reorder",
        choices=["none", "hemi", "hemi_network"],
        default="hemi_network",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample cap for smoke tests. Omit for full-dataset ablations.",
    )
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument(
        "--table-label",
        default=None,
        help="Optional LaTeX label. Defaults to a sanitized version of the output .tex stem.",
    )
    parser.add_argument("--output-json", default=DEFAULT_JSON)
    parser.add_argument("--output-tex", default=DEFAULT_TEX)
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


def parse_float_list(text: str):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("At least one window duration is required.")
    return values


def parse_int_list(text: str):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("At least one target timepoint is required.")
    return values


def parse_label_list(label_text: str):
    labels = []
    for item in label_text.split(","):
        item = item.strip()
        if item:
            labels.append(item)
    return labels


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


def mean_and_std(values: list[float]):
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def summarize_metric(runs: list[dict], metric_key: str):
    mean, std = mean_and_std([run[metric_key] for run in runs])
    return {"mean": mean, "std": std}


def summarize_runs(
    runs: list[dict],
    metric_sources: dict[str, str],
    passthrough_keys: list[str],
):
    summary = {}
    for out_key, metric_key in metric_sources.items():
        summary[out_key] = summarize_metric(runs, metric_key)
    for key in passthrough_keys:
        if key in runs[0]:
            value = runs[0][key]
            summary[key] = list(value) if isinstance(value, list) else value
    return summary


def mean_loso_fold_chance(runs: list[dict]):
    chances = []
    for run in runs:
        fold_chances = [1.0 / len(row["class_counts"]) for row in run["folds"]]
        chances.append(float(np.mean(fold_chances)))
    return float(np.mean(chances))


def format_pct(mean: float, std: float):
    return f"{100.0 * mean:.1f} $\\pm$ {100.0 * std:.1f}"


def maybe_bold(text: str, value: float, best_value: float):
    if np.isclose(value, best_value, rtol=0.0, atol=5e-4):
        return f"\\textbf{{{text}}}"
    return text


def format_seconds(value: float):
    return f"{int(value)}" if float(value).is_integer() else f"{value:g}"


def sanitize_latex_label(text: str):
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return cleaned or "raw_roi_window_duration_ablation"


def write_latex_table(path: Path, payload: dict):
    rows = payload["summary"]["durations"]
    best_site = min(row["site"]["balanced_accuracy"]["mean"] for row in rows)
    best_dx = max(row["diagnosis"]["balanced_accuracy"]["mean"] for row in rows)

    site_chance = 1.0 / payload["meta"]["n_sites"]
    dx_chance = mean_loso_fold_chance(payload["per_duration"][0]["diagnosis_runs"])
    table_label = payload["config"]["table_label"] or sanitize_latex_label(path.stem)

    row_lines = []
    for row in rows:
        site_metric = row["site"]["balanced_accuracy"]
        dx_metric = row["diagnosis"]["balanced_accuracy"]
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
        row_lines.append(
            "%ss ($T$=%d) & %.1f & %s & %s \\\\"
            % (
                format_seconds(row["window_seconds"]),
                row["target_timepoints"],
                row["site"]["mean_windows_per_sample"],
                site_text,
                dx_text,
            )
        )

    diagnosis_note = ""
    if payload["config"]["keep_diagnosis_labels"]:
        diagnosis_note += (
            " Diagnosis tasks retained labels \\{%s\\}; all other diagnosis labels were excluded before feature construction."
            % ", ".join(payload["config"]["keep_diagnosis_labels"])
        )
    if payload["config"]["diagnosis_binarization"]["enabled"]:
        bin_config = payload["config"]["diagnosis_binarization"]
        diagnosis_note += (
            " Diagnosis labels were binarized as %s vs %s, with every non-%s diagnosis pooled as %s."
            % (
                bin_config["healthy_control_label"],
                bin_config["pooled_disorder_label"],
                bin_config["healthy_control_label"],
                bin_config["pooled_disorder_label"],
            )
        )

    skipped_sites = max(row["diagnosis"].get("n_skipped_sites", 0) for row in rows)
    skipped_note = ""
    if skipped_sites:
        skipped_note = (
            " %d site(s) were excluded from diagnosis LOSO scoring because the held-out site contained fewer than two diagnosis classes."
            % skipped_sites
        )

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Raw ROI window-duration ablation for %s (mean $\\pm$ standard deviation across %d random seed%s).}"
        % (
            payload["config"]["dataset_label"],
            len(payload["config"]["seeds"]),
            "" if len(payload["config"]["seeds"]) == 1 else "s",
        ),
        f"\\label{{tab:{table_label}}}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Window ($T$) & Mean windows/sample & Site probe bAcc. ($\\downarrow$) & Diagnosis LOSO bAcc. ($\\uparrow$) \\\\",
        "\\midrule",
        *row_lines,
        f"Chance & -- & {100.0 * site_chance:.1f} & {100.0 * dx_chance:.1f} \\\\",
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\vspace{0.2cm}",
        "\\begin{minipage}{0.96\\columnwidth}",
        (
            "\\footnotesize Matched %s scans only ($n=%d$; %d sites; %d diagnosis classes). "
            "All rows used raw ROI time series with sliding windows of the listed duration and %.0f-second stride; each window was resampled to the specific number of target timepoints ($T$), flattened, and passed through "
            "\\texttt{StandardScaler + PCA(50) + LogisticRegression}. Window predictions were aggregated to matched-sample level via %s. "
            "The site probe used grouped %d-fold cross-validation with subject-level grouping; the diagnosis probe used leave-one-site-out evaluation.%s%s "
            "Reported values are mean $\\pm$ standard deviation across seeds \\{%s\\}."
        )
        % (
            payload["config"]["dataset_label"],
            payload["meta"]["n_samples"],
            payload["meta"]["n_sites"],
            payload["meta"]["n_diagnosis_classes"],
            payload["config"]["stride_seconds"],
            "\\textbf{mean class probability}" if payload["config"]["raw_window_aggregation"] == "mean_proba" else "\\textbf{majority vote}",
            payload["config"]["folds"],
            diagnosis_note,
            skipped_note,
            ", ".join(str(seed) for seed in payload["config"]["seeds"]),
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
    window_seconds_values = parse_float_list(args.window_seconds)
    target_timepoints_values = parse_int_list(args.target_timepoints)

    if len(target_timepoints_values) == 1:
        target_timepoints_values = target_timepoints_values * len(window_seconds_values)
    elif len(target_timepoints_values) != len(window_seconds_values):
        raise ValueError("--target-timepoints list must be length 1 or match length of --window-seconds")

    keep_diagnosis_labels = parse_label_list(args.keep_diagnosis_labels)

    records = load_matched_records(Path(args.fc_dir), Path(args.raw_dir))
    if not records:
        raise FileNotFoundError("No matched FC/raw records found.")
    records = filter_records_by_diagnosis(records, keep_diagnosis_labels)
    records = maybe_subsample_records(records, max_samples=args.max_samples, seed=args.subsample_seed)

    permutation, _ = get_roi_permutation(args.reorder)
    per_duration = []
    summary_rows = []

    for window_seconds, target_timepoints in zip(window_seconds_values, target_timepoints_values):
        duration_records, raw_window_bags = build_raw_window_bags(
            records,
            permutation=permutation,
            target_timepoints=target_timepoints,
            window_seconds=window_seconds,
            stride_seconds=args.stride_seconds,
        )
        sites = [record.site for record in duration_records]
        diagnoses = [record.diagnosis for record in duration_records]
        if args.binarize_diagnosis_hc_vs_disorder:
            diagnoses = binarize_hc_vs_disorder(
                diagnoses,
                healthy_control_label=args.healthy_control_label,
                pooled_disorder_label=args.pooled_disorder_label,
            )
        subjects = [record.subject for record in duration_records]

        site_runs = []
        dx_runs = []
        for seed in seeds:
            site_runs.append(
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
            dx_runs.append(
                run_window_bag_loso_probe(
                    raw_window_bags,
                    diagnoses,
                    sites=sites,
                    seed=seed,
                    aggregation=args.raw_window_aggregation,
                )
            )

        site_summary = summarize_runs(
            site_runs,
            metric_sources={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
            },
            passthrough_keys=[
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
            ],
        )
        dx_summary = summarize_runs(
            dx_runs,
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
        )

        per_duration.append(
            {
                "window_seconds": float(window_seconds),
                "target_timepoints": int(target_timepoints),
                "site_runs": site_runs,
                "diagnosis_runs": dx_runs,
            }
        )
        summary_rows.append(
            {
                "window_seconds": float(window_seconds),
                "target_timepoints": int(target_timepoints),
                "site": site_summary,
                "diagnosis": dx_summary,
            }
        )

    final_diagnoses = [record.diagnosis for record in records]
    if args.binarize_diagnosis_hc_vs_disorder:
        final_diagnoses = binarize_hc_vs_disorder(
            final_diagnoses,
            healthy_control_label=args.healthy_control_label,
            pooled_disorder_label=args.pooled_disorder_label,
        )

    payload = {
        "config": {
            "fc_dir": args.fc_dir,
            "raw_dir": args.raw_dir,
            "dataset_label": args.dataset_label,
            "window_seconds": window_seconds_values,
            "target_timepoints": target_timepoints_values,
            "stride_seconds": float(args.stride_seconds),
            "raw_window_aggregation": args.raw_window_aggregation,
            "keep_diagnosis_labels": keep_diagnosis_labels,
            "diagnosis_binarization": {
                "enabled": bool(args.binarize_diagnosis_hc_vs_disorder),
                "healthy_control_label": args.healthy_control_label,
                "pooled_disorder_label": args.pooled_disorder_label,
            },
            "reorder": args.reorder,
            "folds": args.folds,
            "seeds": seeds,
            "max_samples": args.max_samples,
            "subsample_seed": args.subsample_seed,
            "table_label": args.table_label,
        },
        "meta": {
            "n_samples": int(len(records)),
            "n_subjects": int(len(set(record.subject for record in records))),
            "n_sites": int(len(set(record.site for record in records))),
            "n_diagnosis_classes": int(len(set(final_diagnoses))),
            "diagnosis_classes": sorted(set(final_diagnoses)),
        },
        "per_duration": per_duration,
        "summary": {
            "durations": summary_rows,
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))

    output_tex = Path(args.output_tex)
    write_latex_table(output_tex, payload)

    print(f"Saved JSON: {output_json}")
    print(f"Saved LaTeX: {output_tex}")
    for row in summary_rows:
        site = row["site"]["balanced_accuracy"]
        dx = row["diagnosis"]["balanced_accuracy"]
        print(
            "Window %ss (tp=%d) | site bAcc %.1f±%.1f | dx LOSO bAcc %.1f±%.1f | mean windows/sample %.1f"
            % (
                format_seconds(row["window_seconds"]),
                row["target_timepoints"],
                100.0 * site["mean"],
                100.0 * site["std"],
                100.0 * dx["mean"],
                100.0 * dx["std"],
                row["site"]["mean_windows_per_sample"],
            )
        )


if __name__ == "__main__":
    main()
