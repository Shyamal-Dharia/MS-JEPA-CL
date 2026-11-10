"""Figure 5: network-level attribution statistics and publication panels.

Run after ``plot_diagnosis_attribution_maps.py --directions final_all_disorder``.
It reads the seed-wise parcel attribution maps and writes:

* ``<key>_network_stats.csv``: per Schaefer-7 network
  - exact two-sided sign-flip test of the network-mean signed attribution across
    model seeds (BH-FDR across the 7 networks);
  - random parcel-set enrichment test of mean |signed attribution| on the
    seed-averaged map, against random parcel sets of matched size
    (one-sided, BH-FDR across the 7 networks). This is a Monte Carlo test, so
    p-values vary slightly with the RNG seed.
* Panel (a): pial lateral/medial surface views of the seed-averaged signed map.
* Panel (b): network summary bar chart (mean |signed attribution| on the
  seed-averaged map, colored by the sign of the network mean).
* Panel (c): orthogonal slice views of the same map.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from nilearn import datasets, plotting, surface

try:
    from .plot_diagnosis_attribution_maps import (
        benjamini_hochberg,
        exact_signflip_pvalues,
        schaefer_labels,
        schaefer_scores_to_img,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.plot_diagnosis_attribution_maps import (
        benjamini_hochberg,
        exact_signflip_pvalues,
        schaefer_labels,
        schaefer_scores_to_img,
    )


NETWORKS = [
    ("Vis", "Visual", "Visual"),
    ("SomMot", "Somatomotor", "SomMot"),
    ("DorsAttn", "Dorsal attention", "DorsAttn"),
    ("SalVentAttn", "Salience/ventral attention", "Sal/VentAttn"),
    ("Limbic", "Limbic", "Limbic"),
    ("Cont", "Control", "Control"),
    ("Default", "Default mode", "Default"),
]
HC_COLOR = "#69aac2"
DISORDER_COLOR = "#d77b6f"
PASTEL_DIVERGING = LinearSegmentedColormap.from_list(
    "pastel_teal_coral",
    ["#2f8fac", "#8cc4d3", "#dcecef", "#f6f3f1", "#f2d6d0", "#e2988c", "#c8584a"],
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--attribution-dir",
        default="results/figures/jepa_fusion/diagnosis_attribution/final_all_disorder",
    )
    parser.add_argument("--key", default="final_all_disorder")
    parser.add_argument("--output-dir", default="results/figures/jepa_fusion/figure5")
    parser.add_argument("--n-permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vmax", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--fsaverage", default="fsaverage5")
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def network_of_parcels() -> np.ndarray:
    # Schaefer labels look like "7Networks_LH_Vis_1".
    return np.asarray([label.split("_")[2] for label in schaefer_labels()])


def network_stats(signed_z: np.ndarray, n_permutations: int, seed: int) -> pd.DataFrame:
    parcel_network = network_of_parcels()
    seed_average_abs = np.abs(signed_z.mean(axis=0))
    rng = np.random.default_rng(seed)
    n_parcels_total = signed_z.shape[1]

    rows = []
    for key, label, _short in NETWORKS:
        mask = parcel_network == key
        n_parcels = int(mask.sum())
        per_seed_signed = signed_z[:, mask].mean(axis=1)
        per_seed_abs = np.abs(signed_z[:, mask]).mean(axis=1)
        p_signflip, _p_max = exact_signflip_pvalues(per_seed_signed[:, None])

        observed = float(seed_average_abs[mask].mean())
        null = np.asarray(
            [
                seed_average_abs[rng.choice(n_parcels_total, size=n_parcels, replace=False)].mean()
                for _ in range(n_permutations)
            ]
        )
        p_enrichment = (np.sum(null >= observed - 1e-12) + 1.0) / (n_permutations + 1.0)
        mean_signed = float(per_seed_signed.mean())
        rows.append(
            {
                "network": key,
                "network_label": label,
                "n_parcels": n_parcels,
                "mean_signed": mean_signed,
                "sd_signed_across_seeds": float(per_seed_signed.std(ddof=1)),
                "direction": "disorder" if mean_signed > 0 else "healthy_control",
                "p_signflip_signed_across_seeds": float(p_signflip[0]),
                "mean_abs_signed": float(per_seed_abs.mean()),
                "sd_abs_signed_across_seeds": float(per_seed_abs.std(ddof=1)),
                "p_magnitude_enrichment_random_parcels": float(p_enrichment),
                "enrichment_z": float((observed - null.mean()) / null.std()),
                "null_mean": float(null.mean()),
                "null_sd": float(null.std()),
                "display_mean_abs_signed_seed_average_map": observed,
                "display_mean_signed_seed_average_map": float(signed_z.mean(axis=0)[mask].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    frame.insert(
        frame.columns.get_loc("p_signflip_signed_across_seeds") + 1,
        "q_signflip_signed_across_seeds",
        benjamini_hochberg(frame["p_signflip_signed_across_seeds"].to_numpy()),
    )
    frame.insert(
        frame.columns.get_loc("p_magnitude_enrichment_random_parcels") + 1,
        "q_magnitude_enrichment_random_parcels",
        benjamini_hochberg(frame["p_magnitude_enrichment_random_parcels"].to_numpy()),
    )
    return frame


def save(fig, out_prefix: Path, dpi: int):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_network_summary(stats: pd.DataFrame, out_prefix: Path, dpi: int):
    short = {key: short_name for key, _label, short_name in NETWORKS}
    frame = stats.sort_values("display_mean_abs_signed_seed_average_map", ascending=True)
    colors = [
        DISORDER_COLOR if value > 0 else HC_COLOR
        for value in frame["display_mean_signed_seed_average_map"]
    ]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.barh(
        [short[key] for key in frame["network"]],
        frame["display_mean_abs_signed_seed_average_map"],
        color=colors,
        edgecolor="#666666",
        linewidth=0.8,
    )
    ax.set_xlabel("Mean |signed attribution| (z)")
    ax.set_title("Network summary")
    ax.grid(axis="x", color="#e6e0dc", linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=10, color=HC_COLOR, label="toward HC"),
        Line2D([], [], marker="o", linestyle="", markersize=10, color=DISORDER_COLOR, label="toward disorder"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    save(fig, out_prefix, dpi)


def plot_ortho(img, out_prefix: Path, vmax: float, threshold: float, dpi: int):
    display = plotting.plot_stat_map(
        img,
        display_mode="ortho",
        cut_coords=(0, -18, 24),
        cmap=PASTEL_DIVERGING,
        symmetric_cbar=True,
        threshold=threshold,
        vmax=vmax,
        colorbar=True,
        annotate=True,
        black_bg=False,
        draw_cross=True,
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    display.savefig(str(out_prefix.with_suffix(".png")), dpi=dpi)
    display.savefig(str(out_prefix.with_suffix(".pdf")))
    display.close()


def plot_surfaces(img, out_prefix: Path, vmax: float, threshold: float, fsaverage_name: str, dpi: int):
    mesh = datasets.fetch_surf_fsaverage(fsaverage_name)
    fig, axes = plt.subplots(
        2, 2, figsize=(8.0, 6.0), subplot_kw={"projection": "3d"}, gridspec_kw={"wspace": 0.0, "hspace": 0.0}
    )
    panels = [
        ("left", "lateral", axes[0, 0]),
        ("right", "lateral", axes[0, 1]),
        ("left", "medial", axes[1, 0]),
        ("right", "medial", axes[1, 1]),
    ]
    for hemi, view, ax in panels:
        pial = mesh[f"pial_{hemi}"]
        texture = surface.vol_to_surf(img, pial, interpolation="nearest", radius=2.0)
        plotting.plot_surf_stat_map(
            pial,
            texture,
            hemi=hemi,
            view=view,
            bg_map=mesh[f"sulc_{hemi}"],
            bg_on_data=True,
            darkness=0.5,
            cmap=PASTEL_DIVERGING,
            threshold=threshold,
            vmax=vmax,
            symmetric_cbar=True,
            colorbar=False,
            axes=ax,
            figure=fig,
        )
        ax.set_title(f"{hemi[0].upper()} {view}", fontsize=10)
    mappable = plt.cm.ScalarMappable(cmap=PASTEL_DIVERGING, norm=plt.Normalize(-vmax, vmax))
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Signed attribution (z)")
    save(fig, out_prefix, dpi)


def main():
    args = parse_args()
    attribution_dir = Path(args.attribution_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(attribution_dir / f"{args.key}_seedwise_attribution.npz")
    signed_z = np.asarray(payload["signed_z"], dtype=np.float64)
    print(f"Loaded signed_z with shape {signed_z.shape} for seeds {payload['seeds'].tolist()}")

    stats = network_stats(signed_z, n_permutations=args.n_permutations, seed=args.seed)
    stats_path = out_dir / f"{args.key}_network_stats.csv"
    stats.to_csv(stats_path, index=False)
    print(stats[["network", "mean_signed", "p_signflip_signed_across_seeds", "q_signflip_signed_across_seeds",
                 "p_magnitude_enrichment_random_parcels", "q_magnitude_enrichment_random_parcels"]].to_string(index=False))

    seed_average = signed_z.mean(axis=0)
    img = schaefer_scores_to_img(seed_average)
    img.to_filename(str(out_dir / f"{args.key}_signed_seed_average.nii.gz"))

    plot_surfaces(img, out_dir / "fig5a_surface", args.vmax, args.threshold, args.fsaverage, args.dpi)
    plot_network_summary(stats, out_dir / "fig5b_network_summary", args.dpi)
    plot_ortho(img, out_dir / "fig5c_ortho", args.vmax, args.threshold, args.dpi)
    print(f"Wrote {stats_path} and Figure 5 panels to {out_dir}")


if __name__ == "__main__":
    main()
