from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    from .dataset_utils import matched_records_for_dataset
    from .evaluate_jepa import extract_scan_embeddings, load_checkpoint
    from .evaluate_baselines import (
        ExplicitTransductiveCombat,
        _apply_batch_location_scale,
        _fit_batch_location_scale,
    )
    from .protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        center_site_label,
        filter_records_for_task,
        get_dataset_preset,
    )
    from ..jepa_clean.data import WindowedFMRIDataset
    from ..jepa_clean.plot_tsne_fc_vs_raw import build_fc_features, get_roi_permutation
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.dataset_utils import matched_records_for_dataset
    from src.jepa_fusion.evaluate_jepa import extract_scan_embeddings, load_checkpoint
    from src.jepa_fusion.evaluate_baselines import (
        ExplicitTransductiveCombat,
        _apply_batch_location_scale,
        _fit_batch_location_scale,
    )
    from src.jepa_fusion.protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        center_site_label,
        filter_records_for_task,
        get_dataset_preset,
    )
    from src.jepa_clean.data import WindowedFMRIDataset
    from src.jepa_clean.plot_tsne_fc_vs_raw import build_fc_features, get_roi_permutation


DEFAULT_CHECKPOINT = (
    "checkpoints/jepa_fusion/final_run_v1/"
    "final_run_v1_rotary_curr_replay_nce_seed0/phase2_final.pt"
)
DEFAULT_OUTPUT = "results/figures/four_site_tsne_static_fc_vs_jepa"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make side-by-side Static-FC and JEPA t-SNE plots for four selected sites."
    )
    parser.add_argument("--dataset", default="abide", choices=["abide", "abide1", "abide2", "srpbs"])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--sites",
        default=None,
        help="Comma-separated normalized site labels. Defaults to the four largest mixed-task sites.",
    )
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--circle-quantile", type=float, default=0.85)
    parser.add_argument("--triangle", choices=["upper", "lower"], default="upper")
    parser.add_argument("--reorder", choices=["none", "hemi", "hemi_network"], default="hemi_network")
    parser.add_argument("--max-subjects-per-site", type=int, default=None)
    parser.add_argument("--legend-font-size", type=float, default=13.0)
    parser.add_argument("--separate", action="store_true", help="Also save one clean single-panel figure per method.")
    parser.add_argument(
        "--only-separate",
        action="store_true",
        help="Save only separate single-panel figures and skip the side-by-side summary.",
    )
    parser.add_argument("--save-pdf", action="store_true", help="Also save PDF and SVG copies. Defaults to PNG plus CSV only.")
    parser.add_argument(
        "--from-coordinates",
        default=None,
        help="Redraw from a saved coordinate CSV (written by an earlier run) instead of recomputing t-SNE.",
    )
    parser.add_argument("--legend-columns", type=int, default=2, help="Legend columns for single-panel figures.")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def task_key_for_dataset(dataset_key: str):
    return "hc_disorder" if dataset_key == "srpbs" else "hc_asd"


def diagnosis_to_label(diagnosis: str):
    if diagnosis == HEALTHY_CONTROL_LABEL:
        return "HC"
    if diagnosis == AUTISM_LABEL:
        return "ASD"
    return "Disorder"


def aggregate_subject_features(features: np.ndarray, records: list):
    subject_map = {}
    for idx, record in enumerate(records):
        entry = subject_map.setdefault(
            record.subject_id,
            {
                "features": [],
                "site": str(record.site),
                "diagnosis": str(record.diagnosis),
            },
        )
        entry["features"].append(features[idx])

    subjects = sorted(subject_map)
    return {
        subject: {
            "feature": np.mean(subject_map[subject]["features"], axis=0),
            "site": subject_map[subject]["site"],
            "diagnosis": subject_map[subject]["diagnosis"],
        }
        for subject in subjects
    }


def load_static_fc_subjects(dataset_key: str, selected_sites: list[str] | None, args):
    records = matched_records_for_dataset(dataset_key)
    task_result = filter_records_for_task(
        records,
        task_key_for_dataset(dataset_key),
        mixed_sites_only=True,
        healthy_control_label=HEALTHY_CONTROL_LABEL,
        autism_label=AUTISM_LABEL,
    )
    task_records = list(task_result.records)

    if selected_sites is None:
        site_counts = {}
        for record in task_records:
            site_counts[str(record.site)] = site_counts.get(str(record.site), 0) + 1
        selected_sites = [
            site for site, _count in sorted(site_counts.items(), key=lambda item: (-item[1], item[0]))[:4]
        ]

    selected_set = set(selected_sites)
    selected_records = [record for record in task_records if str(record.site) in selected_set]
    if not selected_records:
        raise ValueError(f"No records matched selected sites: {selected_sites}")

    if args.max_subjects_per_site is not None:
        rng = np.random.default_rng(args.seed)
        by_site_subject = {}
        for record in selected_records:
            by_site_subject.setdefault(str(record.site), set()).add(record.subject_id)
        keep_subjects = set()
        for site, subjects in by_site_subject.items():
            ordered = np.asarray(sorted(subjects), dtype=object)
            if len(ordered) > args.max_subjects_per_site:
                ordered = rng.choice(ordered, size=args.max_subjects_per_site, replace=False)
            keep_subjects.update(str(subject) for subject in ordered.tolist())
        selected_records = [record for record in selected_records if record.subject_id in keep_subjects]

    permutation, _labels = get_roi_permutation(args.reorder)
    features = build_fc_features(selected_records, permutation=permutation, triangle=args.triangle)
    return aggregate_subject_features(features, selected_records), selected_sites


def load_jepa_subjects(dataset_key: str, checkpoint_path: str, device: str):
    preset = get_dataset_preset(dataset_key)
    model, _checkpoint, cfg = load_checkpoint(checkpoint_path)
    model = model.to(device)

    dataset = WindowedFMRIDataset(
        preset.raw_dir,
        window_seconds=float(cfg.get("window_seconds", 90.0)),
        stride_seconds=float(cfg.get("window_stride_seconds", 30.0)),
        target_timepoints=int(cfg.get("target_timepoints", 45)),
        sampling="center",
        exclude_unknown_diagnoses=False,
    )
    embeddings, metadata = extract_scan_embeddings(
        model,
        dataset,
        device=device,
        batch_size=128,
        window_feature="mean",
        covariance_dim=0,
        covariance_seed=0,
    )

    subject_map = {}
    for idx, meta in enumerate(metadata):
        subject = str(meta["subject_id"])
        entry = subject_map.setdefault(
            subject,
            {
                "features": [],
                "site": center_site_label(str(meta.get("site_raw", meta["site"]))),
                "diagnosis": str(meta["diag_name"]),
            },
        )
        entry["features"].append(embeddings[idx])

    return {
        subject: {
            "feature": np.mean(entry["features"], axis=0),
            "site": entry["site"],
            "diagnosis": entry["diagnosis"],
        }
        for subject, entry in subject_map.items()
    }


def reduce_to_tsne(features: np.ndarray, *, seed: int, perplexity: float, pca_components: int):
    scaled = StandardScaler().fit_transform(features)
    n_components = min(pca_components, scaled.shape[1], max(1, scaled.shape[0] - 1))
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(scaled)
    effective_perplexity = min(perplexity, max(5.0, (len(reduced) - 1) / 3.0))
    kwargs = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "random_state": seed,
        "init": "pca",
        "learning_rate": "auto",
    }
    try:
        return TSNE(max_iter=1000, **kwargs).fit_transform(reduced)
    except TypeError:
        return TSNE(n_iter=1000, **kwargs).fit_transform(reduced)


def prepare_fc_representations(fc_features: np.ndarray, sites: list[str], args):
    scaled = StandardScaler().fit_transform(fc_features)
    n_components = min(args.pca_components, scaled.shape[1], max(1, scaled.shape[0] - 1))
    static_fc = PCA(n_components=n_components, random_state=args.seed).fit_transform(scaled)

    site_arr = np.asarray(sites, dtype=object)
    covariates = np.zeros((len(site_arr), 0), dtype=np.float64)
    combat_model = ExplicitTransductiveCombat().fit(static_fc, site_arr, covariates)
    combat_fc = combat_model.transform(static_fc, site_arr, covariates)

    score_model = _fit_batch_location_scale(combat_fc, site_arr)
    covbat_fc = _apply_batch_location_scale(combat_fc, site_arr, score_model)

    return {
        "static_fc": np.asarray(static_fc, dtype=np.float32),
        "combat": np.asarray(combat_fc, dtype=np.float32),
        "covbat": np.asarray(covbat_fc, dtype=np.float32),
    }


def reduce_precomputed_to_tsne(features: np.ndarray, *, seed: int, perplexity: float):
    scaled = StandardScaler().fit_transform(features)
    effective_perplexity = min(perplexity, max(5.0, (len(scaled) - 1) / 3.0))
    kwargs = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "random_state": seed,
        "init": "pca",
        "learning_rate": "auto",
    }
    try:
        return TSNE(max_iter=1000, **kwargs).fit_transform(scaled)
    except TypeError:
        return TSNE(n_iter=1000, **kwargs).fit_transform(scaled)


def add_site_circle(ax, points: np.ndarray, color, quantile: float):
    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center[None, :], axis=1)
    radius = float(np.quantile(distances, quantile)) if len(distances) > 1 else 1.0
    radius = max(radius, 1e-3)
    ax.add_patch(
        Circle(
            center,
            radius,
            fill=False,
            edgecolor=color,
            linewidth=1.8,
            linestyle=(0, (2, 2)),
            alpha=0.95,
        )
    )


def plot_panel(
    ax,
    coords: np.ndarray,
    sites: list[str],
    title: str | None,
    colors: dict[str, tuple],
    quantile: float,
):
    site_arr = np.asarray(sites, dtype=object)
    for site in colors:
        mask = site_arr == site
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=14,
            color=colors[site],
            alpha=0.76,
            linewidths=0,
            label=site,
            rasterized=True,
        )
        add_site_circle(ax, coords[mask], colors[site], quantile)
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def write_coordinates(path: Path, subjects, sites, diagnoses, fc_coords, jepa_coords):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "subject_id",
                "site",
                "diagnosis",
                "diagnosis_short",
                "static_fc_tsne_1",
                "static_fc_tsne_2",
                "jepa_tsne_1",
                "jepa_tsne_2",
            ]
        )
        for idx, subject in enumerate(subjects):
            writer.writerow(
                [
                    subject,
                    sites[idx],
                    diagnoses[idx],
                    diagnosis_to_label(diagnoses[idx]),
                    f"{fc_coords[idx, 0]:.6g}",
                    f"{fc_coords[idx, 1]:.6g}",
                    f"{jepa_coords[idx, 0]:.6g}",
                    f"{jepa_coords[idx, 1]:.6g}",
                ]
            )


def write_all_coordinates(path: Path, subjects, sites, diagnoses, coords_by_method: dict[str, np.ndarray]):
    method_keys = list(coords_by_method)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["subject_id", "site", "diagnosis", "diagnosis_short"]
        for key in method_keys:
            header.extend([f"{key}_tsne_1", f"{key}_tsne_2"])
        writer.writerow(header)
        for idx, subject in enumerate(subjects):
            row = [subject, sites[idx], diagnoses[idx], diagnosis_to_label(diagnoses[idx])]
            for key in method_keys:
                coords = coords_by_method[key]
                row.extend([f"{coords[idx, 0]:.6g}", f"{coords[idx, 1]:.6g}"])
            writer.writerow(row)


def save_single_method_figure(
    output_prefix: Path,
    method_key: str,
    coords: np.ndarray,
    sites: list[str],
    colors: dict[str, tuple],
    quantile: float,
    save_pdf: bool,
    legend_font_size: float,
    legend_columns: int = 2,
    dpi: int = 600,
):
    # Drawn near its printed size (four panels across a page) so text stays legible.
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.7))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.2)
    plot_panel(ax, coords, sites, None, colors, quantile)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(legend_columns, len(labels)),
        frameon=False,
        fontsize=legend_font_size,
        markerscale=1.8,
        handletextpad=0.3,
        columnspacing=1.0,
        labelspacing=0.35,
        bbox_to_anchor=(0.5, 0.0),
    )
    png_path = output_prefix.with_name(f"{output_prefix.name}_{method_key}").with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        for ext in (".pdf", ".svg"):
            path = output_prefix.with_name(f"{output_prefix.name}_{method_key}").with_suffix(ext)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"Saved {path}", flush=True)
    plt.close(fig)
    print(f"Saved {png_path}", flush=True)


def load_coordinates(path: str):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    sites = [row["site"] for row in rows]
    method_keys = [name[: -len("_tsne_1")] for name in rows[0] if name.endswith("_tsne_1")]
    coords_by_method = {
        key: np.asarray([[float(row[f"{key}_tsne_1"]), float(row[f"{key}_tsne_2"])] for row in rows])
        for key in method_keys
    }
    site_order = list(dict.fromkeys(sites))
    return sites, site_order, coords_by_method


def render(coords_by_method: dict, sites: list[str], selected_sites: list[str], args):
    palette = plt.get_cmap("tab10")
    colors = {site: palette(idx) for idx, site in enumerate(selected_sites)}
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if not args.only_separate:
        fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.9), constrained_layout=True)
        panel_titles = ["Static FC", "FC + ComBat", "FC + CovBat", "MS-JEPA"]
        for ax, key, title in zip(axes, coords_by_method, panel_titles):
            plot_panel(ax, coords_by_method[key], sites, title, colors, args.circle_quantile)

        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            frameon=False,
            fontsize=args.legend_font_size + 1,
            markerscale=1.8,
            handletextpad=0.35,
            columnspacing=1.6,
            bbox_to_anchor=(0.5, -0.08),
        )
        png_path = output_prefix.with_suffix(".png")
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        if args.save_pdf:
            for ext in (".pdf", ".svg"):
                fig.savefig(output_prefix.with_suffix(ext), dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {png_path}", flush=True)

    if args.separate or args.only_separate:
        for key, coords in coords_by_method.items():
            save_single_method_figure(
                output_prefix,
                key,
                coords,
                sites,
                colors,
                args.circle_quantile,
                args.save_pdf,
                args.legend_font_size,
                args.legend_columns,
                args.dpi,
            )


def main():
    args = parse_args()
    if args.from_coordinates:
        sites, site_order, coords_by_method = load_coordinates(args.from_coordinates)
        selected = [s.strip() for s in args.sites.split(",") if s.strip()] if args.sites else site_order
        print(f"Redrawing {len(sites)} subjects from {args.from_coordinates} (sites: {', '.join(selected)})", flush=True)
        render(coords_by_method, sites, selected, args)
        return

    selected_sites = [site.strip() for site in args.sites.split(",") if site.strip()] if args.sites else None

    print(f"Loading Static-FC subjects for {args.dataset}...", flush=True)
    fc_subjects, selected_sites = load_static_fc_subjects(args.dataset, selected_sites, args)
    print(f"Selected sites: {', '.join(selected_sites)}", flush=True)

    print("Loading JEPA subjects...", flush=True)
    jepa_subjects = load_jepa_subjects(args.dataset, args.checkpoint, args.device)

    shared_subjects = [
        subject
        for subject in sorted(fc_subjects)
        if subject in jepa_subjects and fc_subjects[subject]["site"] in set(selected_sites)
    ]
    if len(shared_subjects) < 10:
        raise RuntimeError(f"Only {len(shared_subjects)} shared subjects found for selected sites.")

    fc_features = np.stack([fc_subjects[subject]["feature"] for subject in shared_subjects], axis=0)
    jepa_features = np.stack([jepa_subjects[subject]["feature"] for subject in shared_subjects], axis=0)
    sites = [fc_subjects[subject]["site"] for subject in shared_subjects]
    diagnoses = [fc_subjects[subject]["diagnosis"] for subject in shared_subjects]

    site_counts = {site: int(sum(np.asarray(sites, dtype=object) == site)) for site in selected_sites}
    print(f"Shared subjects: {len(shared_subjects)}", flush=True)
    print("Site counts: " + ", ".join(f"{site}={site_counts[site]}" for site in selected_sites), flush=True)

    print("Preparing Static-FC, ComBat, and CovBat representations...", flush=True)
    fc_representations = prepare_fc_representations(fc_features, sites, args)

    print("Running t-SNE for Static FC...", flush=True)
    fc_coords = reduce_precomputed_to_tsne(
        fc_representations["static_fc"],
        seed=args.seed,
        perplexity=args.perplexity,
    )
    print("Running t-SNE for FC + ComBat...", flush=True)
    combat_coords = reduce_precomputed_to_tsne(
        fc_representations["combat"],
        seed=args.seed,
        perplexity=args.perplexity,
    )
    print("Running t-SNE for FC + CovBat...", flush=True)
    covbat_coords = reduce_precomputed_to_tsne(
        fc_representations["covbat"],
        seed=args.seed,
        perplexity=args.perplexity,
    )
    print("Running t-SNE for JEPA...", flush=True)
    jepa_coords = reduce_to_tsne(
        jepa_features,
        seed=args.seed,
        perplexity=args.perplexity,
        pca_components=args.pca_components,
    )

    coords_by_method = {
        "static_fc": fc_coords,
        "combat": combat_coords,
        "covbat": covbat_coords,
        "jepa": jepa_coords,
    }
    render(coords_by_method, sites, selected_sites, args)

    output_prefix = Path(args.output_prefix)
    csv_path = output_prefix.with_suffix(".csv")
    write_all_coordinates(csv_path, shared_subjects, sites, diagnoses, coords_by_method)
    print(f"Saved {csv_path}", flush=True)


if __name__ == "__main__":
    main()
