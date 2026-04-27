import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = "results/figures/abide_fc_vs_raw_tsne_by_site.json"
DEFAULT_FIG_PNG = "results/figures/abide_fc_vs_raw_site_publication.png"
DEFAULT_FIG_PDF = "results/figures/abide_fc_vs_raw_site_publication.pdf"
DEFAULT_TABLE_MD = "results/reports/tables/abide_fc_vs_raw_site_probe.md"
DEFAULT_TABLE_TEX = "results/reports/tables/abide_fc_vs_raw_site_probe.tex"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create publication-ready figure and tables for ABIDE FC-vs-raw site-information comparison."
    )
    parser.add_argument("--input-json", default=DEFAULT_INPUT)
    parser.add_argument("--figure-png", default=DEFAULT_FIG_PNG)
    parser.add_argument("--figure-pdf", default=DEFAULT_FIG_PDF)
    parser.add_argument("--table-md", default=DEFAULT_TABLE_MD)
    parser.add_argument("--table-tex", default=DEFAULT_TABLE_TEX)
    return parser.parse_args()


def load_payload(path: Path):
    with path.open() as f:
        return json.load(f)


def make_site_palette(sites: list[str]):
    unique_sites = sorted(set(sites))
    cmap = plt.get_cmap("tab20", len(unique_sites))
    return {site: cmap(i) for i, site in enumerate(unique_sites)}


def get_n_classes(result: dict):
    return int(result.get("n_classes", result.get("n_sites", 0)))


def add_metric_bars(ax, payload: dict):
    fc = payload["site_probe"]["fc"]
    raw = payload["site_probe"]["raw"]
    n_sites = get_n_classes(fc)
    chance = 1.0 / max(1, n_sites)

    labels = ["Raw ROI", "Static FC"]
    bacc = [raw["balanced_accuracy"], fc["balanced_accuracy"]]
    acc = [raw["accuracy"], fc["accuracy"]]
    colors = ["#4C78A8", "#E45756"]
    x = np.arange(len(labels))

    bars = ax.bar(x, np.array(bacc) * 100.0, color=colors, width=0.62, edgecolor="black", linewidth=0.8)
    ax.axhline(chance * 100.0, color="black", linestyle="--", linewidth=1.1)
    ax.text(
        1.02,
        chance * 100.0,
        f"Chance = {chance * 100.0:.1f}%",
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="left",
        fontsize=10,
    )

    for rect, bacc_value, acc_value in zip(bars, bacc, acc):
        xmid = rect.get_x() + rect.get_width() / 2.0
        ax.text(
            xmid,
            rect.get_height() + 1.5,
            f"bAcc {bacc_value * 100.0:.1f}%\nAcc {acc_value * 100.0:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Site Probe Balanced Accuracy (%)")
    ax.set_title("Matched Site-Decodability Comparison")
    ax.set_ylim(0, max(60.0, max(np.array(bacc) * 100.0) + 12.0))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)


def add_tsne_panel(ax, samples: list[dict], key: str, title: str, palette: dict):
    sites = np.array([sample["site"] for sample in samples])
    coords = np.array([sample[key] for sample in samples], dtype=np.float32)
    for site in sorted(palette):
        mask = sites == site
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.78,
            c=[palette[site]],
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.15, linewidth=0.5)


def make_figure(payload: dict, png_path: Path, pdf_path: Path):
    samples = payload["samples"]
    palette = make_site_palette([sample["site"] for sample in samples])
    embedding_method = payload.get("config", {}).get("embedding_method", "tsne")
    method_label = "t-SNE" if embedding_method == "tsne" else "PCA"

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.5, 5.8),
        gridspec_kw={"width_ratios": [1.0, 1.15, 1.15]},
        constrained_layout=True,
    )

    add_metric_bars(axes[0], payload)
    add_tsne_panel(
        axes[1],
        samples=samples,
        key="fc_embedding",
        title=f"Static FC {method_label}\n(colors denote site)",
        palette=palette,
    )
    add_tsne_panel(
        axes[2],
        samples=samples,
        key="raw_embedding",
        title=f"Raw ROI Time-Series {method_label}\n(colors denote site)",
        palette=palette,
    )

    fig.suptitle(
        "Site Information Is Much More Recoverable From Static FC Than From Raw ROI Time Series",
        fontsize=15,
        y=1.02,
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_rows(payload: dict):
    fc = payload["site_probe"]["fc"]
    raw = payload["site_probe"]["raw"]
    n_sites = get_n_classes(fc)
    chance = 1.0 / max(1, n_sites)
    raw_target_timepoints = int(payload.get("config", {}).get("raw_target_timepoints", 45))

    return [
        {
            "representation": "Raw ROI time series",
            "accuracy": raw["accuracy"],
            "balanced_accuracy": raw["balanced_accuracy"],
            "n_samples": raw["n_samples"],
            "n_sites": get_n_classes(raw),
            "folds": raw["folds"],
            "featureization": f"Resample to {raw_target_timepoints} timepoints, flatten, StandardScaler + PCA(50)",
        },
        {
            "representation": "Static FC (upper triangle)",
            "accuracy": fc["accuracy"],
            "balanced_accuracy": fc["balanced_accuracy"],
            "n_samples": fc["n_samples"],
            "n_sites": get_n_classes(fc),
            "folds": fc["folds"],
            "featureization": "400x400 FC upper triangle, StandardScaler + PCA(50)",
        },
        {
            "representation": "Random chance",
            "accuracy": chance,
            "balanced_accuracy": chance,
            "n_samples": fc["n_samples"],
            "n_sites": get_n_classes(fc),
            "folds": fc["folds"],
            "featureization": "1 / number of sites",
        },
    ]


def write_markdown_table(path: Path, rows: list[dict]):
    lines = [
        "# ABIDE FC vs Raw Site-Probe Table",
        "",
        "| Representation | Site Accuracy (%) | Site Balanced Accuracy (%) | Samples | Sites | CV Folds | Featureization |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['representation']} | "
            f"{100.0 * row['accuracy']:.1f} | "
            f"{100.0 * row['balanced_accuracy']:.1f} | "
            f"{row['n_samples']} | {row['n_sites']} | {row['folds']} | {row['featureization']} |"
        )

    lines += [
        "",
        "Note: matched ABIDE-I scans only. Both probes used the same sample set (`n=1153`) and the same 5-fold stratified cross-validation protocol. "
        "The site probe classifier was logistic regression on top of `StandardScaler + PCA(50)` features. "
        "Static FC retained substantially more site information than raw ROI time-series inputs.",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def escape_latex(text: str):
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def write_latex_table(path: Path, rows: list[dict]):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Matched ABIDE site-information comparison between raw ROI time-series inputs and static FC features.}",
        "\\label{tab:abide_fc_vs_raw_site_probe}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Representation & Accuracy (\\%) & Balanced accuracy (\\%) & Samples & Sites & CV folds\\\\",
        "\\hline",
    ]

    for row in rows:
        lines.append(
            f"{escape_latex(row['representation'])} & "
            f"{100.0 * row['accuracy']:.1f} & "
            f"{100.0 * row['balanced_accuracy']:.1f} & "
            f"{row['n_samples']} & {row['n_sites']} & {row['folds']}\\\\"
        )

    lines += [
        "\\hline",
        "\\end{tabular}",
        "",
        "\\vspace{0.4em}",
        "\\begin{minipage}{0.96\\linewidth}",
        "\\footnotesize Matched ABIDE-I scans only. Both site probes used the same sample set and the same 5-fold stratified cross-validation protocol. "
        "Each representation was standardized, reduced with PCA to 50 dimensions, and classified with logistic regression. "
        "Static FC retained substantially stronger site information than raw ROI time-series features.",
        "\\end{minipage}",
        "\\end{table}",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    args = parse_args()
    payload = load_payload(Path(args.input_json))
    rows = build_rows(payload)

    make_figure(
        payload=payload,
        png_path=Path(args.figure_png),
        pdf_path=Path(args.figure_pdf),
    )
    write_markdown_table(Path(args.table_md), rows)
    write_latex_table(Path(args.table_tex), rows)

    print(f"Saved figure PNG: {args.figure_png}")
    print(f"Saved figure PDF: {args.figure_pdf}")
    print(f"Saved Markdown table: {args.table_md}")
    print(f"Saved LaTeX table: {args.table_tex}")


if __name__ == "__main__":
    main()
