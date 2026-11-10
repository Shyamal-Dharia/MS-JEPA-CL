"""Figure 1 input panel: ROI time-series heatmaps sized for slides and the paper figure.

Draws the actual model input (one 90-s window resampled to 45 time points x 400
Schaefer ROIs, per-ROI z-scored), plus one small window from each of three SRPBS
sites. Writes transparent PNG (600 dpi) and SVG files that can be dropped into
PowerPoint:

  fig1_input_heatmap      main heatmap with axes and colorbar
  fig1_site_{A,B,C}       one thumbnail per site (colored frame + label)
  fig1_sites_row          the three thumbnails side by side
  fig1_input_panel        main heatmap above the site row (drop-in replacement)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

try:
    from ..jepa_clean.data import compute_sliding_windows, resample_time_window
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.jepa_clean.data import compute_sliding_windows, resample_time_window


DATA_DIR = "preprocessed_data_fir/full/roi_timeseries_1600_full"
SITES = [  # (subject, label, frame color)
    ("sub-0001", "Site A", "#3E6FB0"),
    ("sub-0236", "Site B", "#E08A4F"),
    ("sub-0360", "Site C", "#2E7D4F"),
]
# Teal <-> coral diverging map with a neutral gray midpoint (same poles as Figure 5).
BOLD_CMAP = LinearSegmentedColormap.from_list(
    "bold_teal_coral",
    ["#1f6f8b", "#5aa5bd", "#b5d6df", "#f0efec", "#ecc0b6", "#d9806f", "#b0463a"],
)
INK = "#222222"
FRAME = "#555555"
VLIM = 2.5


def load_window(data_dir: Path, subject: str, window_index: int, n_timepoints: int = 45):
    meta = json.loads((data_dir / f"{subject}_meta.json").read_text())
    ts = np.load(data_dir / f"{subject}_roi_ts.npy").astype(np.float32)
    windows = compute_sliding_windows(ts.shape[0], float(meta["tr"]), 90.0, 30.0)
    start, end = windows[min(window_index, len(windows) - 1)]
    return resample_time_window(ts[start:end], n_timepoints)  # (45, 400)


def save(fig, out_dir: Path, name: str, dpi: int):
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def style_frame(ax, color=FRAME, width=1.2):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(width)


def draw_main(ax, window, cax=None, fontsize=20):
    im = ax.imshow(window.T, aspect="auto", cmap=BOLD_CMAP, vmin=-VLIM, vmax=VLIM, interpolation="nearest")
    ax.set_xticks([0, 14, 29, 44], ["1", "15", "30", "45"])
    ax.set_yticks([0, 199, 399], ["1", "200", "400"])
    ax.set_xlabel("Time point", fontsize=fontsize, color=INK, labelpad=4)
    ax.set_ylabel("ROI", fontsize=fontsize, color=INK, labelpad=4)
    ax.tick_params(labelsize=fontsize - 3, colors=INK, length=4, width=1.0)
    style_frame(ax)
    if cax is not None:
        cbar = plt.colorbar(im, cax=cax, ticks=[-2, 0, 2])
        cbar.ax.tick_params(labelsize=fontsize - 3, colors=INK, length=3, width=1.0)
        cbar.set_label("BOLD (z)", fontsize=fontsize - 2, color=INK, labelpad=6)
        cbar.outline.set_edgecolor(FRAME)
        cbar.outline.set_linewidth(1.0)
    return im


def draw_thumb(ax, window, label, color, fontsize=20):
    ax.imshow(window.T, aspect="auto", cmap=BOLD_CMAP, vmin=-VLIM, vmax=VLIM, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    style_frame(ax, color=color, width=4.0)
    ax.set_xlabel(label, fontsize=fontsize, color=INK, fontweight="bold", labelpad=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--window-index", type=int, default=1, help="Which 90-s sliding window to show.")
    parser.add_argument("--output-dir", default="results/figures/figure1_ppt")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = [load_window(data_dir, subject, args.window_index) for subject, _label, _color in SITES]

    # Main heatmap with colorbar.
    fig = plt.figure(figsize=(4.0, 3.0))
    ax = fig.add_axes([0.0, 0.0, 0.78, 1.0])
    cax = fig.add_axes([0.83, 0.10, 0.04, 0.80])
    draw_main(ax, windows[0], cax=cax)
    save(fig, out_dir, "fig1_input_heatmap", args.dpi)

    # One thumbnail per site.
    for window, (_subject, label, color) in zip(windows, SITES):
        fig, ax = plt.subplots(figsize=(1.7, 1.4))
        draw_thumb(ax, window, label, color)
        save(fig, out_dir, f"fig1_{label.replace(' ', '_')}", args.dpi)

    # Thumbnails side by side.
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.7), gridspec_kw={"wspace": 0.22})
    for ax, window, (_subject, label, color) in zip(axes, windows, SITES):
        draw_thumb(ax, window, label, color)
    save(fig, out_dir, "fig1_sites_row", args.dpi)

    # Combined panel: main heatmap above the site row.
    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_axes([0.14, 0.45, 0.66, 0.53])
    cax = fig.add_axes([0.83, 0.49, 0.03, 0.45])
    draw_main(ax, windows[0], cax=cax)
    for i, (window, (_subject, label, color)) in enumerate(zip(windows, SITES)):
        thumb = fig.add_axes([0.06 + i * 0.32, 0.07, 0.25, 0.19])
        draw_thumb(thumb, window, label, color, fontsize=19)
    save(fig, out_dir, "fig1_input_panel", args.dpi)

    print(f"Wrote Figure 1 input heatmaps to {out_dir}")
    for path in sorted(out_dir.glob("fig1_*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
