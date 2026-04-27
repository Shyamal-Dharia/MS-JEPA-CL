import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLES = {
    "fc": {
        "label": "Static FC",
        "color": "#6B7280",
        "linestyle": "--",
        "marker": None,
        "band_alpha": 0.10,
        "line_width": 1.8,
        "zorder": 1,
    },
    "fc_combat": {
        "label": "FC + ComBat",
        "color": "#D94841",
        "linestyle": "-",
        "marker": "o",
        "band_alpha": 0.14,
        "line_width": 2.2,
        "zorder": 3,
    },
    "fc_covbat": {
        "label": "FC + CovBat",
        "color": "#2A6FBB",
        "linestyle": "-",
        "marker": "s",
        "band_alpha": 0.14,
        "line_width": 2.2,
        "zorder": 3,
    },
}

DATASET_INFO = {
    "abide": {
        "label": "ABIDE-I/II",
        "panel_prefix": "abide",
    },
    "srpbs": {
        "label": "SRPBS-1600",
        "panel_prefix": "srpbs",
    },
}

PANEL_SPECS = [
    ("srpbs", "site", "balanced_accuracy", "Site bAcc (%)"),
    ("srpbs", "diagnosis", "balanced_accuracy", "Diagnosis bAcc (%)"),
    ("abide", "site", "balanced_accuracy", "Site bAcc (%)"),
    ("abide", "diagnosis", "balanced_accuracy", "Diagnosis bAcc (%)"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create publication-ready low-K calibration panels."
    )
    parser.add_argument(
        "--abide-lowk",
        default="final_results/phase6_lowk_stress_test/abide_lowk_stress_test.json",
    )
    parser.add_argument(
        "--srpbs-lowk",
        default="final_results/phase6_lowk_stress_test/srpbs_lowk_stress_test.json",
    )
    parser.add_argument(
        "--abide-baseline",
        default="final_results/phase1_baselines/abide_hc_asd_mixedsites_baseline_table.json",
    )
    parser.add_argument(
        "--srpbs-baseline",
        default="final_results/phase1_baselines/srpbs_hc_disorder_mixedsites_baseline_table.json",
    )
    parser.add_argument(
        "--outdir",
        default="results/figures/lowk_publication",
        help="Directory where panel assets are written.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated list of output formats, e.g. png,pdf.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--save-pairs",
        action="store_true",
        help="Also save one 2-panel figure per dataset.",
    )
    parser.add_argument(
        "--save-grid",
        action="store_true",
        help="Also save a 2x2 grid for quick review.",
    )
    return parser.parse_args()


def load_json(path: str | Path):
    with Path(path).open() as f:
        return json.load(f)


def configure_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2F3640",
            "axes.labelcolor": "#1F2937",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def percent(value: float) -> float:
    return 100.0 * float(value)


def build_dataset_payload(lowk_payload: dict, baseline_payload: dict, dataset_key: str):
    results = lowk_payload["results"]
    k_values = sorted(int(k) for k in results.keys())
    x_numeric = list(k_values) + [60]
    x_labels = [str(k) for k in k_values] + ["All"]
    n_site_classes = int(lowk_payload.get("config", {}).get("n_sites", 0))
    if n_site_classes <= 0:
        classes = baseline_payload["summary"]["site"]["fc"].get("classes", [])
        n_site_classes = len(classes)

    payload = {
        "dataset_key": dataset_key,
        "dataset_label": DATASET_INFO[dataset_key]["label"],
        "chance": 100.0 / max(1, n_site_classes),
        "x_numeric": np.array(x_numeric, dtype=float),
        "x_labels": x_labels,
        "series": {},
    }

    for method in ("fc", "fc_combat", "fc_covbat"):
        payload["series"][method] = {}
        for metric_group in ("site", "diagnosis"):
            payload["series"][method][metric_group] = {}
            for metric_name in ("balanced_accuracy", "roc_auc"):
                means = []
                stds = []
                for k in k_values:
                    metric = results[str(k)][method][metric_group][metric_name]
                    means.append(percent(metric["mean"]))
                    stds.append(percent(metric["std"]))

                baseline_metric = baseline_payload["summary"][metric_group][method][metric_name]
                means.append(percent(baseline_metric["mean"]))
                stds.append(percent(baseline_metric["std"]))

                payload["series"][method][metric_group][metric_name] = {
                    "x": np.array(x_numeric, dtype=float),
                    "mean": np.array(means, dtype=float),
                    "std": np.array(stds, dtype=float),
                }

    return payload


def _nice_limits(values: list[float], is_site_panel: bool, chance: float | None = None):
    values = np.asarray(values, dtype=float)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if chance is not None:
        lo = min(lo, chance)
        hi = max(hi, chance)

    if is_site_panel:
        lower = max(0.0, lo - 4.0)
        upper = hi + 6.0
    else:
        lower = max(45.0, lo - 1.4)
        upper = hi + 1.8

    span = max(upper - lower, 1.0)
    pad = 0.02 * span
    return lower - pad, upper + pad


def _format_panel_axes(ax, ylabel: str, x_numeric: np.ndarray, x_labels: list[str]):
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_numeric)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Calibration subjects per site ($K$)")
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", colors="#334155")
    ax.axvspan(55.5, 61.0, color="#F8FAFC", zorder=0)
    ax.axvline(55.5, color="#CBD5E1", linewidth=1.0, linestyle=":")


def draw_lowk_panel(
    ax,
    dataset_payload: dict,
    metric_group: str,
    metric_name: str,
    ylabel: str,
    show_legend: bool = True,
):
    x_numeric = dataset_payload["x_numeric"]
    x_labels = dataset_payload["x_labels"]
    is_site_panel = metric_group == "site"

    _format_panel_axes(ax, ylabel, x_numeric, x_labels)

    all_values = []
    legend_handles = []
    legend_labels = []

    for method in ("fc", "fc_combat", "fc_covbat"):
        style = METHOD_STYLES[method]
        series = dataset_payload["series"][method][metric_group][metric_name]
        mean = series["mean"]
        std = series["std"]
        all_values.extend((mean - std).tolist())
        all_values.extend((mean + std).tolist())

        if method == "fc":
            center = mean[0]
            spread = std[0]
            ax.fill_between(
                x_numeric,
                center - spread,
                center + spread,
                color=style["color"],
                alpha=style["band_alpha"],
                zorder=style["zorder"],
            )
            line = ax.axhline(
                center,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["line_width"],
                zorder=style["zorder"] + 0.1,
            )
            legend_handles.append(line)
            legend_labels.append(style["label"])
            continue

        ax.fill_between(
            x_numeric,
            mean - std,
            mean + std,
            color=style["color"],
            alpha=style["band_alpha"],
            zorder=style["zorder"] - 0.2,
        )
        line, = ax.plot(
            x_numeric,
            mean,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["line_width"],
            marker=style["marker"],
            markersize=5.8,
            markerfacecolor="white",
            markeredgewidth=1.5,
            zorder=style["zorder"],
        )
        ax.scatter(
            [x_numeric[-1]],
            [mean[-1]],
            s=48,
            marker="D",
            color=style["color"],
            edgecolors="white",
            linewidth=0.8,
            zorder=style["zorder"] + 0.4,
        )
        legend_handles.append(line)
        legend_labels.append(style["label"])

    chance = dataset_payload["chance"] if is_site_panel else None
    if chance is not None:
        all_values.append(chance)
        chance_line = ax.axhline(
            chance,
            color="#2E8B57",
            linestyle=(0, (2, 2)),
            linewidth=1.6,
            zorder=1.5,
        )
        legend_handles.append(chance_line)
        legend_labels.append(f"Chance ({chance:.1f}%)")

    lower, upper = _nice_limits(all_values, is_site_panel=is_site_panel, chance=chance)
    ax.set_ylim(lower, upper)

    if show_legend:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.03),
            frameon=False,
            handlelength=2.6,
            ncol=2,
            columnspacing=1.2,
        )

    return legend_handles, legend_labels


def save_figure(fig, output_base: Path, formats: list[str], dpi: int):
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")


def make_single_panel(dataset_payload: dict, metric_group: str, metric_name: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    draw_lowk_panel(
        ax=ax,
        dataset_payload=dataset_payload,
        metric_group=metric_group,
        metric_name=metric_name,
        ylabel=ylabel,
        show_legend=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def make_dataset_pair(dataset_payload: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=False)
    handles, labels = draw_lowk_panel(
        ax=axes[0],
        dataset_payload=dataset_payload,
        metric_group="site",
        metric_name="balanced_accuracy",
        ylabel="Site bAcc (%)",
        show_legend=False,
    )
    draw_lowk_panel(
        ax=axes[1],
        dataset_payload=dataset_payload,
        metric_group="diagnosis",
        metric_name="balanced_accuracy",
        ylabel="Diagnosis bAcc (%)",
        show_legend=False,
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def make_grid(abide_payload: dict, srpbs_payload: dict):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2))
    panel_map = [
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
        axes[1, 1],
    ]
    legend_handles = None
    legend_labels = None
    for idx, (ax, (dataset_key, metric_group, metric_name, ylabel)) in enumerate(zip(panel_map, PANEL_SPECS)):
        payload = srpbs_payload if dataset_key == "srpbs" else abide_payload
        handles, labels = draw_lowk_panel(
            ax=ax,
            dataset_payload=payload,
            metric_group=metric_group,
            metric_name=metric_name,
            ylabel=ylabel,
            show_legend=False,
        )
        if idx == 0:
            legend_handles = handles
            legend_labels = labels
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main():
    args = parse_args()
    configure_style()

    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    outdir = Path(args.outdir)

    abide_payload = build_dataset_payload(
        load_json(args.abide_lowk),
        load_json(args.abide_baseline),
        dataset_key="abide",
    )
    srpbs_payload = build_dataset_payload(
        load_json(args.srpbs_lowk),
        load_json(args.srpbs_baseline),
        dataset_key="srpbs",
    )

    dataset_lookup = {
        "abide": abide_payload,
        "srpbs": srpbs_payload,
    }

    for dataset_key, metric_group, metric_name, ylabel in PANEL_SPECS:
        payload = dataset_lookup[dataset_key]
        output_base = outdir / f"lowk_panel_{dataset_key}_{metric_group}"
        fig = make_single_panel(payload, metric_group, metric_name, ylabel)
        save_figure(fig, output_base, formats=formats, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved panel: {output_base}")

    if args.save_pairs:
        srpbs_pair = make_dataset_pair(srpbs_payload)
        save_figure(srpbs_pair, outdir / "lowk_srpbs_pair", formats=formats, dpi=args.dpi)
        plt.close(srpbs_pair)
        print(f"Saved pair: {outdir / 'lowk_srpbs_pair'}")

        abide_pair = make_dataset_pair(abide_payload)
        save_figure(abide_pair, outdir / "lowk_abide_pair", formats=formats, dpi=args.dpi)
        plt.close(abide_pair)
        print(f"Saved pair: {outdir / 'lowk_abide_pair'}")

    if args.save_grid:
        grid = make_grid(abide_payload, srpbs_payload)
        save_figure(grid, outdir / "lowk_grid", formats=formats, dpi=args.dpi)
        plt.close(grid)
        print(f"Saved grid: {outdir / 'lowk_grid'}")


if __name__ == "__main__":
    main()
