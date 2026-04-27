#!/usr/bin/env python3
"""
Plot a Schaefer-400 functional connectivity matrix reordered by Yeo 7 networks.

The script produces:
  1. A reordered 400 x 400 FC heatmap with network block boundaries
  2. A compact 7 x 7 network-level mean correlation matrix
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist


@dataclass(frozen=True)
class RoiMeta:
    index: int
    label: str
    hemisphere: str
    network: str
    parcel_index: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a Schaefer FC matrix reordered by 7-network structure."
    )
    parser.add_argument(
        "--fc-path",
        required=True,
        help="Path to the square FC matrix stored as .npy",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="PNG output path. Defaults to results/figures/<fc-stem>_schaefer7_networks.png",
    )
    parser.add_argument(
        "--network-csv",
        default=None,
        help="Optional CSV path for the 7 x 7 network-level mean correlation matrix.",
    )
    parser.add_argument("--n-rois", type=int, default=400)
    parser.add_argument("--yeo-networks", type=int, default=7)
    parser.add_argument("--resolution-mm", type=int, default=2)
    parser.add_argument(
        "--roi-order",
        choices=["network_hemi", "network_cluster"],
        default="network_cluster",
        help=(
            "How ROIs are ordered inside the fixed 7-network blocks. "
            "'network_cluster' usually yields cleaner diagonal structure."
        ),
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Symmetric colorbar limit. Defaults to the 99th percentile of |off-diagonal FC|.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--mask-diagonal",
        action="store_true",
        help="Hide self-correlations in the ROI-level heatmap to emphasize block structure.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def decode_label(label) -> str:
    return label.decode("utf-8") if isinstance(label, bytes) else str(label)


def parse_schaefer_label(index: int, label: str) -> RoiMeta:
    parts = label.split("_")
    hemisphere = parts[1] if len(parts) > 1 else "UNK"
    network = parts[2] if len(parts) > 2 else "UNK"
    parcel_token = parts[-1] if parts and parts[-1].isdigit() else "0"
    return RoiMeta(
        index=index,
        label=label,
        hemisphere=hemisphere,
        network=network,
        parcel_index=int(parcel_token),
    )


def load_roi_metadata(n_rois: int, yeo_networks: int, resolution_mm: int):
    atlas = fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        resolution_mm=resolution_mm,
    )
    labels = [decode_label(label) for label in atlas["labels"]]
    roi_meta = [parse_schaefer_label(i, label) for i, label in enumerate(labels)]
    network_order = list(dict.fromkeys(meta.network for meta in roi_meta))
    return roi_meta, network_order


def build_network_first_permutation(roi_meta: list[RoiMeta], network_order: list[str]) -> np.ndarray:
    network_rank = {network: rank for rank, network in enumerate(network_order)}
    hemisphere_rank = {"LH": 0, "RH": 1}
    order = sorted(
        range(len(roi_meta)),
        key=lambda idx: (
            network_rank[roi_meta[idx].network],
            hemisphere_rank.get(roi_meta[idx].hemisphere, 2),
            roi_meta[idx].parcel_index,
            idx,
        ),
    )
    return np.asarray(order, dtype=np.int64)


def cluster_indices_by_connectivity(fc: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if len(indices) <= 2:
        return np.asarray(indices, dtype=np.int64)

    profiles = fc[np.ix_(indices, indices)].copy()
    np.fill_diagonal(profiles, 0.0)
    distances = pdist(profiles, metric="correlation")
    distances = np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)

    if np.allclose(distances, 0.0):
        return np.asarray(indices, dtype=np.int64)

    tree = linkage(distances, method="average")
    tree = optimal_leaf_ordering(tree, distances)
    return np.asarray(indices[np.asarray(leaves_list(tree), dtype=np.int64)], dtype=np.int64)


def build_network_cluster_permutation(
    fc: np.ndarray,
    roi_meta: list[RoiMeta],
    network_order: list[str],
) -> np.ndarray:
    permutation = []
    for network in network_order:
        network_indices = np.asarray(
            [meta.index for meta in roi_meta if meta.network == network],
            dtype=np.int64,
        )
        permutation.extend(cluster_indices_by_connectivity(fc=fc, indices=network_indices).tolist())
    return np.asarray(permutation, dtype=np.int64)


def contiguous_groups(ordered_meta: list[RoiMeta], key_fn):
    if not ordered_meta:
        return []

    groups = []
    start = 0
    current_key = key_fn(ordered_meta[0])
    for idx, meta in enumerate(ordered_meta[1:], start=1):
        next_key = key_fn(meta)
        if next_key != current_key:
            groups.append((current_key, start, idx))
            start = idx
            current_key = next_key
    groups.append((current_key, start, len(ordered_meta)))
    return groups


def load_fc_matrix(fc_path: Path, n_rois: int) -> np.ndarray:
    fc = np.load(fc_path).astype(np.float32, copy=False)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError(f"Expected a square FC matrix, got shape {fc.shape}")
    if fc.shape[0] != n_rois:
        raise ValueError(f"Expected {n_rois} ROIs, found matrix with shape {fc.shape}")
    return 0.5 * (fc + fc.T)


def infer_vmax(fc: np.ndarray) -> float:
    off_diagonal = fc[~np.eye(fc.shape[0], dtype=bool)]
    vmax = float(np.percentile(np.abs(off_diagonal), 99))
    return min(max(vmax, 0.3), 1.0)


def compute_network_summary(
    fc_reordered: np.ndarray,
    ordered_meta: list[RoiMeta],
    network_order: list[str],
):
    network_groups = contiguous_groups(ordered_meta, key_fn=lambda meta: meta.network)
    group_lookup = {network: np.arange(start, end) for network, start, end in network_groups}

    network_sizes = {network: int(len(indices)) for network, indices in group_lookup.items()}
    summary = np.zeros((len(network_order), len(network_order)), dtype=np.float32)

    for i, network_i in enumerate(network_order):
        indices_i = group_lookup[network_i]
        for j, network_j in enumerate(network_order):
            indices_j = group_lookup[network_j]
            if i == j:
                block = fc_reordered[np.ix_(indices_i, indices_i)]
                values = block[np.triu_indices_from(block, k=1)]
            else:
                values = fc_reordered[np.ix_(indices_i, indices_j)].ravel()
            summary[i, j] = float(values.mean()) if values.size else np.nan

    return summary, network_sizes


def save_network_summary_csv(
    csv_path: Path,
    network_order: list[str],
    network_sizes: dict[str, int],
    summary: np.ndarray,
):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["network"] + [f"{name} (n={network_sizes[name]})" for name in network_order]
        writer.writerow(header)
        for row_idx, network_name in enumerate(network_order):
            row = [f"{network_name} (n={network_sizes[network_name]})"]
            row.extend(f"{summary[row_idx, col_idx]:.6f}" for col_idx in range(len(network_order)))
            writer.writerow(row)


def plot_fc(
    fc_reordered: np.ndarray,
    fc_display: np.ndarray,
    network_summary: np.ndarray,
    ordered_meta: list[RoiMeta],
    network_order: list[str],
    network_sizes: dict[str, int],
    output_path: Path,
    title: str,
    roi_panel_title: str,
    vmax: float,
    dpi: int,
    close_figure: bool = True,
):
    network_groups = contiguous_groups(ordered_meta, key_fn=lambda meta: meta.network)
    hemisphere_groups = contiguous_groups(ordered_meta, key_fn=lambda meta: (meta.network, meta.hemisphere))

    network_centers = [(start + end - 1) / 2.0 for _, start, end in network_groups]
    network_boundaries = [end - 0.5 for _, _, end in network_groups[:-1]]
    hemisphere_boundaries = [end - 0.5 for _, _, end in hemisphere_groups[:-1]]
    tick_labels = [f"{network}\n(n={network_sizes[network]})" for network in network_order]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [4.8, 2.2]},
        constrained_layout=True,
    )

    ax_matrix, ax_summary = axes
    im = ax_matrix.imshow(
        fc_display,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    for boundary in hemisphere_boundaries:
        ax_matrix.axhline(boundary, color="black", linewidth=0.5, linestyle=":", alpha=0.45)
        ax_matrix.axvline(boundary, color="black", linewidth=0.5, linestyle=":", alpha=0.45)
    for boundary in network_boundaries:
        ax_matrix.axhline(boundary, color="black", linewidth=1.1, alpha=0.8)
        ax_matrix.axvline(boundary, color="black", linewidth=1.1, alpha=0.8)

    ax_matrix.set_xticks(network_centers)
    ax_matrix.set_xticklabels(network_order, rotation=45, ha="right")
    ax_matrix.set_yticks(network_centers)
    ax_matrix.set_yticklabels(network_order)
    ax_matrix.set_title(roi_panel_title)

    ax_summary.imshow(
        network_summary,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_summary.set_xticks(np.arange(len(network_order)))
    ax_summary.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax_summary.set_yticks(np.arange(len(network_order)))
    ax_summary.set_yticklabels(tick_labels)
    ax_summary.set_title("Mean network correlation")

    for row_idx in range(network_summary.shape[0]):
        for col_idx in range(network_summary.shape[1]):
            value = network_summary[row_idx, col_idx]
            text_color = "white" if abs(value) > (0.55 * vmax) else "black"
            ax_summary.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label("Pearson r")

    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if close_figure:
        plt.close(fig)
    return fig


def main():
    args = parse_args()

    fc_path = Path(args.fc_path)
    if args.output is None:
        output_path = Path("results/figures") / f"{fc_path.stem}_schaefer7_networks.png"
    else:
        output_path = Path(args.output)

    if args.network_csv is None:
        network_csv_path = output_path.with_name(output_path.stem + "_network_summary.csv")
    else:
        network_csv_path = Path(args.network_csv)

    roi_meta, network_order = load_roi_metadata(
        n_rois=args.n_rois,
        yeo_networks=args.yeo_networks,
        resolution_mm=args.resolution_mm,
    )
    fc = load_fc_matrix(fc_path, n_rois=args.n_rois)
    permutation = build_network_first_permutation(roi_meta, network_order)
    if args.roi_order == "network_cluster":
        permutation = build_network_cluster_permutation(
            fc=fc,
            roi_meta=roi_meta,
            network_order=network_order,
        )
    ordered_meta = [roi_meta[idx] for idx in permutation]

    fc_reordered = fc[np.ix_(permutation, permutation)]
    fc_display = fc_reordered.copy()
    if args.mask_diagonal:
        np.fill_diagonal(fc_display, 0.0)
    network_summary, network_sizes = compute_network_summary(
        fc_reordered=fc_reordered,
        ordered_meta=ordered_meta,
        network_order=network_order,
    )

    vmax = args.vmax if args.vmax is not None else infer_vmax(fc_reordered)
    title = (
        f"{fc_path.name}\n"
        f"Schaefer-{args.n_rois}, Yeo {args.yeo_networks}-network ordering"
    )
    roi_panel_title = "ROI-level FC reordered by network"
    if args.roi_order == "network_cluster":
        roi_panel_title += " + within-network clustering"
    if args.mask_diagonal:
        roi_panel_title += " (diagonal hidden)"
    fig = plot_fc(
        fc_reordered=fc_reordered,
        fc_display=fc_display,
        network_summary=network_summary,
        ordered_meta=ordered_meta,
        network_order=network_order,
        network_sizes=network_sizes,
        output_path=output_path,
        title=title,
        roi_panel_title=roi_panel_title,
        vmax=vmax,
        dpi=args.dpi,
        close_figure=not args.show,
    )
    save_network_summary_csv(
        csv_path=network_csv_path,
        network_order=network_order,
        network_sizes=network_sizes,
        summary=network_summary,
    )

    print(f"Saved figure to: {output_path}")
    print(f"Saved network summary CSV to: {network_csv_path}")
    print(f"Using symmetric color scale +/- {vmax:.3f}")

    if args.show:
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
