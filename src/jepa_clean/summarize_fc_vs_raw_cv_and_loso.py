import argparse
import json
from pathlib import Path
import re

import numpy as np

from .evaluate_fc_vs_raw_loso import run_loso_probe
from .plot_tsne_fc_vs_raw import (
    DEFAULT_FC_DIR,
    DEFAULT_RAW_DIR,
    build_fc_features,
    build_raw_features,
    get_roi_permutation,
    load_matched_records,
    run_probe,
)


DEFAULT_JSON = "results/reports/tables/abide_fc_vs_raw_cv_and_loso_multiseed.json"
DEFAULT_TEX = "results/reports/tables/abide_fc_vs_raw_cv_and_loso_multiseed.tex"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize matched FC-vs-raw 5-fold and LOSO probes across one or more seeds."
    )
    parser.add_argument("--fc-dir", default=DEFAULT_FC_DIR)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--dataset-label",
        default="ABIDE-I",
        help="Short dataset name used in captions and printed summaries.",
    )
    parser.add_argument("--triangle", choices=["upper", "lower"], default="upper")
    parser.add_argument(
        "--reorder",
        choices=["none", "hemi", "hemi_network"],
        default="hemi_network",
    )
    parser.add_argument(
        "--keep-diagnosis-labels",
        default="",
        help="Optional comma-separated diagnosis labels to keep before running any probes.",
    )
    parser.add_argument("--raw-target-timepoints", type=int, default=45)
    parser.add_argument("--fivefold-folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2,3,4")
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


def summarize_metric(runs: list[dict], metric_key: str):
    mean, std = mean_and_std([run[metric_key] for run in runs])
    return {"mean": mean, "std": std}


def summarize_protocol(runs: list[dict], metric_key: str):
    return {
        "balanced_accuracy": summarize_metric(runs, metric_key),
        "n_runs": int(len(runs)),
    }


def format_pct(mean: float, std: float):
    return f"{100.0 * mean:.1f} $\\pm$ {100.0 * std:.1f}"


def sanitize_latex_label(text: str):
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return cleaned or "fc_vs_raw_cv_and_loso"


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


def describe_diagnosis_task(classes: list[str]):
    class_set = set(classes)
    if class_set == {"Healthy Control", "Autism Spectrum Disorder"} and len(classes) == 2:
        return "ASD vs HC"
    return f"{len(classes)}-way"


def write_latex_table(path: Path, payload: dict):
    site_raw = payload["summary"]["fivefold_site"]["raw"]["balanced_accuracy"]
    site_fc = payload["summary"]["fivefold_site"]["fc"]["balanced_accuracy"]
    dx5_raw = payload["summary"]["fivefold_diagnosis"]["raw"]["balanced_accuracy"]
    dx5_fc = payload["summary"]["fivefold_diagnosis"]["fc"]["balanced_accuracy"]
    dxloso_raw = payload["summary"]["loso_diagnosis"]["raw"]["balanced_accuracy"]
    dxloso_fc = payload["summary"]["loso_diagnosis"]["fc"]["balanced_accuracy"]

    site_chance = 1.0 / payload["meta"]["n_sites"]
    dx_chance = 1.0 / payload["meta"]["n_diagnosis_classes"]
    dataset_label = payload["config"]["dataset_label"]
    table_label = payload["config"]["table_label"] or sanitize_latex_label(path.stem)
    diagnosis_task = describe_diagnosis_task(payload["meta"]["diagnosis_classes"])
    loso_valid_sites = payload["per_seed"]["loso_diagnosis"]["raw"][0]["n_valid_sites"]
    loso_total_sites = payload["meta"]["n_sites"]

    footnote_bits = [
        (
            "\\footnotesize Matched %s scans only ($n=%d$; %d sites; %d diagnosis classes). "
            "Raw ROI time series were resampled to %d timepoints and flattened. "
            "Static FC used the %s triangle of the $400 \\times 400$ matrix. All probes used the same linear model "
            "(\\texttt{StandardScaler + PCA(50) + LogisticRegression}). The 5-fold results used stratified cross-validation. "
            "The LOSO diagnosis results used leave-one-site-out evaluation."
        )
        % (
            dataset_label,
            payload["meta"]["n_samples"],
            payload["meta"]["n_sites"],
            payload["meta"]["n_diagnosis_classes"],
            payload["config"]["raw_target_timepoints"],
            payload["config"]["triangle"],
        )
    ]
    if payload["config"]["keep_diagnosis_labels"]:
        footnote_bits.append(
            " Diagnosis tasks retained labels \\{%s\\}; all other diagnosis labels were excluded before feature construction."
            % ", ".join(payload["config"]["keep_diagnosis_labels"])
        )
    if loso_valid_sites < loso_total_sites:
        footnote_bits.append(
            " %d site(s) were excluded from LOSO diagnosis scoring because the held-out site contained fewer than two diagnosis classes."
            % (loso_total_sites - loso_valid_sites)
        )
    footnote_bits.append(
        " Reported values are mean $\\pm$ standard deviation across seeds \\{%s\\}."
        % ", ".join(str(seed) for seed in payload["config"]["seeds"])
    )

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Matched %s probe comparison for ROI time-series inputs and static FC features. Values are mean $\\pm$ standard deviation across %d random seed%s.}"
        % (
            dataset_label,
            len(payload["config"]["seeds"]),
            "" if len(payload["config"]["seeds"]) == 1 else "s",
        ),
        f"\\label{{tab:{table_label}}}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "& Site probe & Diagnosis probe & Diagnosis probe \\\\",
        f"& (5-fold, {payload['meta']['n_sites']}-way) & (5-fold, {diagnosis_task}) & (LOSO, {diagnosis_task}) \\\\",
        "\\cmidrule(lr){2-2}\\cmidrule(lr){3-3}\\cmidrule(lr){4-4}",
        "Representation & bAcc. ($\\downarrow$) & bAcc. ($\\uparrow$) & bAcc. ($\\uparrow$) \\\\",
        "\\midrule",
        f"ROI time series & \\textbf{{{format_pct(site_raw['mean'], site_raw['std'])}}} & {format_pct(dx5_raw['mean'], dx5_raw['std'])} & {format_pct(dxloso_raw['mean'], dxloso_raw['std'])} \\\\",
        f"Static FC & {format_pct(site_fc['mean'], site_fc['std'])} & \\textbf{{{format_pct(dx5_fc['mean'], dx5_fc['std'])}}} & \\textbf{{{format_pct(dxloso_fc['mean'], dxloso_fc['std'])}}} \\\\",
        f"Chance & {100.0 * site_chance:.1f} & {100.0 * dx_chance:.1f} & {100.0 * dx_chance:.1f} \\\\",
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\vspace{0.2cm}",
        "\\begin{minipage}{0.96\\columnwidth}",
        "".join(footnote_bits),
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
        raise FileNotFoundError("No matched FC/raw records found.")
    records = filter_records_by_diagnosis(records, keep_diagnosis_labels)

    permutation, _ = get_roi_permutation(args.reorder)
    fc_features = build_fc_features(records, permutation=permutation, triangle=args.triangle)
    raw_features = build_raw_features(
        records,
        permutation=permutation,
        raw_target_timepoints=args.raw_target_timepoints,
    )
    sites = [record.site for record in records]
    diagnoses = [record.diagnosis for record in records]

    fivefold_site_fc = []
    fivefold_site_raw = []
    fivefold_dx_fc = []
    fivefold_dx_raw = []
    loso_dx_fc = []
    loso_dx_raw = []

    for seed in seeds:
        fivefold_site_fc.append(run_probe(fc_features, sites, seed=seed, folds=args.fivefold_folds, task_name="site"))
        fivefold_site_raw.append(run_probe(raw_features, sites, seed=seed, folds=args.fivefold_folds, task_name="site"))
        fivefold_dx_fc.append(run_probe(fc_features, diagnoses, seed=seed, folds=args.fivefold_folds, task_name="diagnosis"))
        fivefold_dx_raw.append(run_probe(raw_features, diagnoses, seed=seed, folds=args.fivefold_folds, task_name="diagnosis"))
        loso_dx_fc.append(run_loso_probe(fc_features, diagnoses, sites, seed=seed))
        loso_dx_raw.append(run_loso_probe(raw_features, diagnoses, sites, seed=seed))

    payload = {
        "config": {
            "dataset_label": args.dataset_label,
            "fc_dir": args.fc_dir,
            "raw_dir": args.raw_dir,
            "triangle": args.triangle,
            "reorder": args.reorder,
            "keep_diagnosis_labels": keep_diagnosis_labels,
            "raw_target_timepoints": args.raw_target_timepoints,
            "fivefold_folds": args.fivefold_folds,
            "seeds": seeds,
            "table_label": args.table_label,
        },
        "meta": {
            "n_samples": int(len(records)),
            "n_sites": int(len(set(sites))),
            "n_diagnosis_classes": int(len(set(diagnoses))),
            "diagnosis_classes": [str(label) for label in sorted(set(diagnoses))],
        },
        "per_seed": {
            "fivefold_site": {"fc": fivefold_site_fc, "raw": fivefold_site_raw},
            "fivefold_diagnosis": {"fc": fivefold_dx_fc, "raw": fivefold_dx_raw},
            "loso_diagnosis": {"fc": loso_dx_fc, "raw": loso_dx_raw},
        },
        "summary": {
            "fivefold_site": {
                "fc": summarize_protocol(fivefold_site_fc, "balanced_accuracy"),
                "raw": summarize_protocol(fivefold_site_raw, "balanced_accuracy"),
            },
            "fivefold_diagnosis": {
                "fc": summarize_protocol(fivefold_dx_fc, "balanced_accuracy"),
                "raw": summarize_protocol(fivefold_dx_raw, "balanced_accuracy"),
            },
            "loso_diagnosis": {
                "fc": summarize_protocol(loso_dx_fc, "mean_fold_balanced_accuracy"),
                "raw": summarize_protocol(loso_dx_raw, "mean_fold_balanced_accuracy"),
                "fc_pooled": summarize_protocol(loso_dx_fc, "pooled_balanced_accuracy"),
                "raw_pooled": summarize_protocol(loso_dx_raw, "pooled_balanced_accuracy"),
            },
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))

    output_tex = Path(args.output_tex)
    write_latex_table(output_tex, payload)

    print(f"Saved JSON: {output_json}")
    print(f"Saved LaTeX: {output_tex}")
    print(
        f"{args.dataset_label} | 5-fold site bAcc | RAW vs FC | "
        f"{100.0 * payload['summary']['fivefold_site']['raw']['balanced_accuracy']['mean']:.1f}±"
        f"{100.0 * payload['summary']['fivefold_site']['raw']['balanced_accuracy']['std']:.1f} vs "
        f"{100.0 * payload['summary']['fivefold_site']['fc']['balanced_accuracy']['mean']:.1f}±"
        f"{100.0 * payload['summary']['fivefold_site']['fc']['balanced_accuracy']['std']:.1f}"
    )
    print(
        f"{args.dataset_label} | 5-fold dx bAcc | RAW vs FC | "
        f"{100.0 * payload['summary']['fivefold_diagnosis']['raw']['balanced_accuracy']['mean']:.1f}±"
        f"{100.0 * payload['summary']['fivefold_diagnosis']['raw']['balanced_accuracy']['std']:.1f} vs "
        f"{100.0 * payload['summary']['fivefold_diagnosis']['fc']['balanced_accuracy']['mean']:.1f}±"
        f"{100.0 * payload['summary']['fivefold_diagnosis']['fc']['balanced_accuracy']['std']:.1f}"
    )
    print(
        f"{args.dataset_label} | LOSO dx bAcc | RAW vs FC | "
        f"{100.0 * payload['summary']['loso_diagnosis']['raw']['balanced_accuracy']['mean']:.1f}±"
        f"{100.0 * payload['summary']['loso_diagnosis']['raw']['balanced_accuracy']['std']:.1f} vs "
        f"{100.0 * payload['summary']['loso_diagnosis']['fc']['balanced_accuracy']['mean']:.1f}±"
        f"{100.0 * payload['summary']['loso_diagnosis']['fc']['balanced_accuracy']['std']:.1f}"
    )


if __name__ == "__main__":
    main()
