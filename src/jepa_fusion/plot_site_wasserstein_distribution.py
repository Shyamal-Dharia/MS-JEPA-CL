from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "Static FC",
    "FC + ComBat",
    "FC + CovBat",
    "JEPA Rotary (Pooled)",
    "JEPA Rotary (Curriculum + Replay)",
]
METHOD_SHORT = {
    "Static FC": "Static FC",
    "FC + ComBat": "FC + ComBat",
    "FC + CovBat": "FC + CovBat",
    "JEPA Rotary (Pooled)": "JEPA Pooled",
    "JEPA Rotary (Curriculum + Replay)": "JEPA Curr.+Replay",
}
METHOD_COLORS = {
    "Static FC": "#6B7280",
    "FC + ComBat": "#D94841",
    "FC + CovBat": "#2A6FBB",
    "JEPA Rotary (Pooled)": "#0F766E",
    "JEPA Rotary (Curriculum + Replay)": "#7C3AED",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create publication-style site-Wasserstein distribution plots."
    )
    parser.add_argument(
        "--report-json",
        default="results/reports/tables/jepa_fusion/site_wasserstein_pca50/site_wasserstein_pca50_all.json",
    )
    parser.add_argument(
        "--outdir",
        default="results/figures/site_wasserstein_pca50",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--trim-iqr-multiplier",
        type=float,
        default=3.0,
        help="Plot-only Tukey outlier trimming multiplier. Set <=0 to disable.",
    )
    parser.add_argument(
        "--metric",
        choices=["overall", "conditional", "both"],
        default="both",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def save_figure(fig, outbase: Path, formats: list[str], dpi: int):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(outbase.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")


def configure_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#0F172A",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "font.family": "DejaVu Sans",
            "font.size": 12.0,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def overall_values(method_row: dict) -> list[float]:
    values = []
    for run in method_row["runs"]:
        for pair in run["overall"]["pairwise"]:
            if pair["swd"] is not None:
                values.append(float(pair["swd"]))
    return values


def conditional_values(method_row: dict) -> list[float]:
    values = []
    for run in method_row["runs"]:
        for diagnosis_payload in run["conditional"]["per_diagnosis"].values():
            for pair in diagnosis_payload["pairwise"]:
                if pair["swd"] is not None:
                    values.append(float(pair["swd"]))
    return values


def dataset_method_rows(dataset_payload: dict) -> list[dict]:
    by_label = {row["label"]: row for row in dataset_payload["methods"]}
    return [by_label[label] for label in METHOD_ORDER if label in by_label]


def trim_outliers(values: list[float], iqr_multiplier: float) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 4 or iqr_multiplier <= 0:
        return arr.tolist()
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = float(q3 - q1)
    if iqr <= 0:
        return arr.tolist()
    lower = float(q1 - iqr_multiplier * iqr)
    upper = float(q3 + iqr_multiplier * iqr)
    trimmed = arr[(arr >= lower) & (arr <= upper)]
    return trimmed.tolist() if len(trimmed) else arr.tolist()


def box_scatter(ax, labels: list[str], values: list[list[float]]):
    positions = np.arange(1, len(labels) + 1)
    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.5},
        whiskerprops={"color": "#475569", "linewidth": 1.0},
        capprops={"color": "#475569", "linewidth": 1.0},
        boxprops={"linewidth": 1.0, "color": "#475569"},
    )
    rng = np.random.default_rng(0)
    for patch, label, vals, pos in zip(bp["boxes"], labels, values, positions):
        color = METHOD_COLORS[label]
        patch.set_facecolor(color)
        patch.set_alpha(0.24)
        x = pos + rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(x, vals, s=10, alpha=0.22, color=color, edgecolors="none", rasterized=True)

    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_SHORT[label] for label in labels], rotation=16, ha="right")
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def draw_metric(ax, method_rows: list[dict], metric: str, title: str, trim_iqr_multiplier: float):
    labels = [row["label"] for row in method_rows]
    raw_values = [overall_values(row) if metric == "overall" else conditional_values(row) for row in method_rows]
    values = [trim_outliers(v, trim_iqr_multiplier) for v in raw_values]
    box_scatter(ax, labels, values)
    ax.set_ylabel("Sliced Wasserstein Distance")


def main():
    args = parse_args()
    configure_style()
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    outdir = Path(args.outdir)
    payload = load_json(args.report_json)

    metrics = ["overall", "conditional"] if args.metric == "both" else [args.metric]
    title_lookup = {
        "overall": "Overall Site Distance",
        "conditional": "Conditional Site Distance",
    }

    for dataset_key, dataset_payload in payload["datasets"].items():
        method_rows = dataset_method_rows(dataset_payload)
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(7.6, 5.4))
            draw_metric(ax, method_rows, metric, title_lookup[metric], args.trim_iqr_multiplier)
            fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.96)
            save_figure(fig, outdir / f"site_wasserstein_{dataset_key}_{metric}", formats=formats, dpi=args.dpi)
            plt.close(fig)


if __name__ == "__main__":
    main()
