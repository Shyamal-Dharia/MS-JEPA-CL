from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD_ORDER = [
    ("raw", "ROI time series"),
    ("fc", "Static FC"),
    ("fc_combat", "Static FC + ComBat"),
    ("fc_covbat", "Static FC + CovBat"),
]

DATASET_FILES = [
    "abide1_hc_asd_mixedsites_baseline_table.json",
    "abide2_hc_asd_mixedsites_baseline_table.json",
    "srpbs_hc_disorder_mixedsites_baseline_table.json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the completed phase-1 baselines as a publication-style LaTeX table."
    )
    parser.add_argument(
        "--results-dir",
        default="results/reports/tables/final_version_jepa_fusion",
        help="Directory containing the baseline JSON files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .tex path. Defaults to <results-dir>/phase1_baseline_table.tex.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def fmt_pm(metric: dict[str, float]) -> str:
    return f"{100.0 * metric['mean']:.1f} $\\pm$ {100.0 * metric['std']:.1f}"


def fmt_scalar(value: float) -> str:
    return f"{100.0 * value:.1f}"


def highlight(text: str, enabled: bool) -> str:
    return f"\\textbf{{{text}}}" if enabled else text


def dataset_header(payload: dict) -> str:
    label = payload["config"]["dataset_label"]
    n_sites = payload["meta"]["n_sites"]
    task_label = payload["config"]["task_label"]
    return f"\\textbf{{{label}}} ({n_sites} mixed-label sites; {task_label})"


def best_keys(payload: dict):
    site_summary = payload["summary"]["site"]
    dx_summary = payload["summary"]["diagnosis"]
    return {
        "site_bacc": min(METHOD_ORDER, key=lambda item: site_summary[item[0]]["balanced_accuracy"]["mean"])[0],
        "site_auc": min(METHOD_ORDER, key=lambda item: site_summary[item[0]]["roc_auc"]["mean"])[0],
        "dx_bacc": max(METHOD_ORDER, key=lambda item: dx_summary[item[0]]["balanced_accuracy"]["mean"])[0],
        "dx_auc": max(METHOD_ORDER, key=lambda item: dx_summary[item[0]]["roc_auc"]["mean"])[0],
    }


def build_table(payloads: list[dict]) -> str:
    lines = [
        "% Requires \\usepackage{booktabs}",
        "% Requires \\usepackage{graphicx}",
        "\\begin{table*}[t]",
        "\\centering",
        (
            "\\caption{Phase-1 classical baseline benchmark under the final baseline-first protocol. "
            "Rows report mean $\\pm$ standard deviation across 5 random seeds. Site leakage is measured "
            "with grouped subject-level cross-validation, while diagnosis is measured with leave-one-site-out "
            "(LOSO) evaluation. Lower site-probe values are better; higher diagnosis values are better.}"
        ),
        "\\label{tab:phase1_baselines_final}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        (
            "Representation & Site probe bAcc. ($\\downarrow$) & Site probe ROC-AUC ($\\downarrow$) & "
            "Diagnosis LOSO bAcc. ($\\uparrow$) & Diagnosis LOSO ROC-AUC ($\\uparrow$) \\\\"
        ),
        "\\midrule",
    ]

    for dataset_idx, payload in enumerate(payloads):
        best = best_keys(payload)
        site_summary = payload["summary"]["site"]
        dx_summary = payload["summary"]["diagnosis"]
        site_chance = 1.0 / max(1, payload["meta"]["n_sites"])
        dx_chance = 1.0 / max(1, len(payload["meta"]["diagnosis_classes"]))

        lines.append(f"\\multicolumn{{5}}{{l}}{{{dataset_header(payload)}}} \\\\")
        for method_key, label in METHOD_ORDER:
            site_bacc = fmt_pm(site_summary[method_key]["balanced_accuracy"])
            site_auc = fmt_pm(site_summary[method_key]["roc_auc"])
            dx_bacc = fmt_pm(dx_summary[method_key]["balanced_accuracy"])
            dx_auc = fmt_pm(dx_summary[method_key]["roc_auc"])
            lines.append(
                f"{label} & "
                f"{highlight(site_bacc, method_key == best['site_bacc'])} & "
                f"{highlight(site_auc, method_key == best['site_auc'])} & "
                f"{highlight(dx_bacc, method_key == best['dx_bacc'])} & "
                f"{highlight(dx_auc, method_key == best['dx_auc'])} \\\\"
            )
        lines.append(
            f"Chance & {fmt_scalar(site_chance)} & 50.0 & {fmt_scalar(dx_chance)} & 50.0 \\\\"
        )
        if dataset_idx != len(payloads) - 1:
            lines.append("\\midrule")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\vspace{0.2cm}",
            "\\begin{minipage}{0.97\\textwidth}",
            (
                "\\footnotesize \\textbf{Notes:} Mixed-site-only filtering is active for these binary tasks. "
                "The baseline rows are ROI time series, Static FC, Static FC + ComBat, and "
                "Static FC + CovBat. Because classical harmonization cannot be fit on a completely unseen site "
                "without additional adaptation machinery, the ComBat/CovBat feature transforms used for the LOSO "
                "diagnosis rows are the current branch's transductive full-dataset harmonized baselines; this "
                "should be stated explicitly in the paper. The grouped site probe and the LOSO diagnosis columns "
                "therefore answer different questions by design."
            ),
            "\\end{minipage}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "phase1_baseline_table.tex"

    payloads = [load_json(results_dir / filename) for filename in DATASET_FILES]
    table_text = build_table(payloads)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_text)
    print(f"Saved LaTeX: {output_path}")


if __name__ == "__main__":
    main()
