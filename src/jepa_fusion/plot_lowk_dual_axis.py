from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLES = {
    "fc_combat": {
        "label": "FC + ComBat",
        "color": "#D94841",
        "site_marker": "o",
        "dx_marker": "s",
    },
    "fc_covbat": {
        "label": "FC + CovBat",
        "color": "#2A6FBB",
        "site_marker": "o",
        "dx_marker": "s",
    },
}
PLOT_METHODS = ("fc_combat", "fc_covbat")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot low-K stress-test curves with site bAcc on the left axis and diagnosis bAcc on the right axis."
    )
    parser.add_argument(
        "--abide-json",
        default="results/reports/tables/jepa_fusion/abide_lowk_newbaseline_stress_test.json",
    )
    parser.add_argument(
        "--srpbs-json",
        default="results/reports/tables/jepa_fusion/srpbs_lowk_newbaseline_stress_test.json",
    )
    parser.add_argument(
        "--abide-baseline-json",
        default="results/reports/tables/jepa_fusion/new_baseline/new_baseline_abide_hc_asd_mixedsites_baseline_table.json",
    )
    parser.add_argument(
        "--srpbs-baseline-json",
        default="results/reports/tables/jepa_fusion/new_baseline/new_baseline_srpbs_hc_disorder_mixedsites_baseline_table.json",
    )
    parser.add_argument(
        "--datasets",
        choices=["abide", "srpbs", "both"],
        default="both",
    )
    parser.add_argument(
        "--outdir",
        default="results/figures/lowk_newbaseline_dual_axis",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf,svg",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--include-error-bands",
        action="store_true",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def configure_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#0F172A",
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 13.5,
            "font.family": "DejaVu Sans",
            "font.size": 12.5,
            "legend.fontsize": 11,
            "legend.title_fontsize": 11.5,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": True,
            "axes.grid": False,
        }
    )


def to_percent(metric: dict) -> tuple[np.ndarray, np.ndarray]:
    return 100.0 * np.asarray(metric["mean"], dtype=float), 100.0 * np.asarray(metric["std"], dtype=float)


def build_payload(lowk_payload: dict, baseline_payload: dict) -> dict:
    k_values = sorted(int(k) for k in lowk_payload["results"].keys())
    dataset_label = str(lowk_payload["config"].get("dataset_label", lowk_payload["config"].get("dataset", "")))
    n_sites = int(lowk_payload["config"].get("n_sites", 0))
    payload = {
        "dataset_label": dataset_label,
        "k_values": np.asarray(k_values, dtype=float),
        "site_chance": 100.0 / max(1, n_sites),
        "diagnosis_static_fc_baseline": 100.0
        * float(baseline_payload["summary"]["diagnosis"]["fc"]["balanced_accuracy"]["mean"]),
        "series": {},
    }
    for method_key in PLOT_METHODS:
        site_means = []
        site_stds = []
        dx_means = []
        dx_stds = []
        for k in k_values:
            row = lowk_payload["results"][str(k)][method_key]
            site_means.append(100.0 * float(row["site"]["balanced_accuracy"]["mean"]))
            site_stds.append(100.0 * float(row["site"]["balanced_accuracy"]["std"]))
            dx_means.append(100.0 * float(row["diagnosis"]["balanced_accuracy"]["mean"]))
            dx_stds.append(100.0 * float(row["diagnosis"]["balanced_accuracy"]["std"]))
        payload["series"][method_key] = {
            "site_mean": np.asarray(site_means, dtype=float),
            "site_std": np.asarray(site_stds, dtype=float),
            "dx_mean": np.asarray(dx_means, dtype=float),
            "dx_std": np.asarray(dx_stds, dtype=float),
        }
    return payload


def nice_limits(values: list[np.ndarray], pad: float, lower_floor: float | None = None) -> tuple[float, float]:
    arr = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    lower = lo - pad
    upper = hi + pad
    if lower_floor is not None:
        lower = max(lower_floor, lower)
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def save_figure(fig, outbase: Path, formats: list[str], dpi: int):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(outbase.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")


def plot_dataset(ax, payload: dict, include_error_bands: bool):
    ax2 = ax.twinx()
    method_handles = []

    for method_key, style in METHOD_STYLES.items():
        if method_key not in payload["series"]:
            continue
        series = payload["series"][method_key]
        ax.plot(
            payload["k_values"],
            series["site_mean"],
            color=style["color"],
            marker=style["site_marker"],
            linewidth=2.4,
            markersize=7,
            label=style["label"],
        )
        ax2.plot(
            payload["k_values"],
            series["dx_mean"],
            color=style["color"],
            marker=style["dx_marker"],
            linewidth=2.2,
            markersize=6.5,
            linestyle="--",
            alpha=0.95,
        )
        if include_error_bands:
            ax.fill_between(
                payload["k_values"],
                series["site_mean"] - series["site_std"],
                series["site_mean"] + series["site_std"],
                color=style["color"],
                alpha=0.10,
                linewidth=0,
            )
            ax2.fill_between(
                payload["k_values"],
                series["dx_mean"] - series["dx_std"],
                series["dx_mean"] + series["dx_std"],
                color=style["color"],
                alpha=0.06,
                linewidth=0,
            )
        method_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=2.2,
                label=style["label"],
            )
        )

    ax.axhline(
        payload["site_chance"],
        color="#64748B",
        linestyle="-",
        linewidth=1.7,
        alpha=0.95,
    )
    ax2.axhline(
        payload["diagnosis_static_fc_baseline"],
        color="#111827",
        linestyle=(0, (5, 3)),
        linewidth=1.9,
        alpha=0.95,
    )

    metric_handles = [
        plt.Line2D([0], [0], color="#0F172A", linewidth=2.2, marker="o", label="Site bAcc"),
        plt.Line2D([0], [0], color="#0F172A", linewidth=2.0, linestyle="--", marker="s", label="Diagnosis bAcc"),
        plt.Line2D([0], [0], color="#64748B", linewidth=1.7, linestyle="-", label="Site chance"),
        plt.Line2D(
            [0],
            [0],
            color="#111827",
            linewidth=1.9,
            linestyle=(0, (5, 3)),
            label="Static FC baseline",
        ),
    ]

    ax.set_xlabel("Calibration subjects per site (K)")
    ax.set_ylabel("Site bAcc (%)", color="#0F172A")
    ax2.set_ylabel("Diagnosis bAcc (%)", color="#0F172A")
    ax.set_xticks(payload["k_values"])
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, top=True, labeltop=False)
    ax.tick_params(axis="y", colors="#334155")
    ax2.tick_params(axis="y", colors="#334155")
    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_color("#334155")
    ax.spines["top"].set_linewidth(1.0)
    ax2.spines["top"].set_visible(False)

    ax.set_ylim(
        *nice_limits(
            [payload["series"][k]["site_mean"] for k in payload["series"]] + [np.array([payload["site_chance"]])],
            pad=3.0,
            lower_floor=0.0,
        )
    )
    ax2.set_ylim(
        *nice_limits(
            [payload["series"][k]["dx_mean"] for k in payload["series"]]
            + [np.array([payload["diagnosis_static_fc_baseline"]])],
            pad=1.5,
            lower_floor=45.0,
        )
    )

    legend1 = ax.legend(
        handles=method_handles,
        loc="lower left",
        bbox_to_anchor=(-0.14, 1.02),
        ncol=1,
        frameon=False,
        title="Method",
        handlelength=1.8,
        borderaxespad=0.0,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=metric_handles,
        loc="lower right",
        bbox_to_anchor=(1.17, 1.02),
        ncol=2,
        frameon=False,
        title="Metric",
        handlelength=1.8,
        columnspacing=1.0,
        borderaxespad=0.0,
    )


def main():
    args = parse_args()
    configure_style()
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    outdir = Path(args.outdir)

    dataset_inputs = {
        "abide": build_payload(load_json(args.abide_json), load_json(args.abide_baseline_json)),
        "srpbs": build_payload(load_json(args.srpbs_json), load_json(args.srpbs_baseline_json)),
    }
    dataset_keys = ["abide", "srpbs"] if args.datasets == "both" else [args.datasets]

    if len(dataset_keys) == 1:
        fig, ax = plt.subplots(figsize=(6.2, 4.9))
        plot_dataset(ax, dataset_inputs[dataset_keys[0]], include_error_bands=args.include_error_bands)
        fig.tight_layout(rect=[0, 0, 1, 0.79])
        save_figure(fig, outdir / f"lowk_dual_axis_{dataset_keys[0]}", formats=formats, dpi=args.dpi)
        plt.close(fig)
        return

    for dataset_key in dataset_keys:
        fig, ax = plt.subplots(figsize=(6.2, 4.9))
        plot_dataset(ax, dataset_inputs[dataset_key], include_error_bands=args.include_error_bands)
        fig.tight_layout(rect=[0, 0, 1, 0.79])
        save_figure(fig, outdir / f"lowk_dual_axis_{dataset_key}", formats=formats, dpi=args.dpi)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharex=False)
    for ax, dataset_key in zip(axes, dataset_keys):
        plot_dataset(ax, dataset_inputs[dataset_key], include_error_bands=args.include_error_bands)
    fig.tight_layout(rect=[0, 0, 1, 0.79], w_pad=4.0)
    save_figure(fig, outdir / "lowk_dual_axis_both", formats=formats, dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
