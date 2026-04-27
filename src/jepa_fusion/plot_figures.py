"""
Generate paper figures:
  Figure 1: t-SNE of Static FC vs FC+ComBat vs JEPA embeddings
  Figure 2: Low-K degradation curves
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.jepa_fusion.protocol import (
    get_dataset_preset,
    filter_records_for_task,
    batch_site_label,
    HEALTHY_CONTROL_LABEL,
    AUTISM_LABEL,
)


def load_fc_features(preset, task_key, triangle="upper"):
    from src.jepa_clean.plot_tsne_fc_vs_raw import (
        load_matched_records,
        get_roi_permutation,
        build_fc_features,
    )
    fc_dirs = preset.fc_dir if isinstance(preset.fc_dir, tuple) else (preset.fc_dir,)
    raw_dirs = preset.raw_dir if isinstance(preset.raw_dir, tuple) else (preset.raw_dir,)
    all_records = []
    for fc_dir, raw_dir in zip(fc_dirs, raw_dirs):
        all_records.extend(load_matched_records(Path(fc_dir), Path(raw_dir)))
    task_records = filter_records_for_task(
        all_records, task_key, mixed_sites_only=True,
        healthy_control_label=HEALTHY_CONTROL_LABEL,
        autism_label=AUTISM_LABEL,
    )
    permutation, _ = get_roi_permutation("hemi_network")
    task_records = task_records.records
    features = build_fc_features(task_records, permutation=permutation, triangle=triangle)
    sites = [batch_site_label(r.site) for r in task_records]
    diagnoses = [r.diagnosis for r in task_records]
    subjects = [r.subject for r in task_records]
    return features, sites, diagnoses, subjects


def load_combat_features(fc_features, sites):
    from neuroCombat import neuroCombat
    import pandas as pd
    sites_arr = np.array(sites)
    # ComBat needs >=2 samples per site; filter singletons
    unique, counts = np.unique(sites_arr, return_counts=True)
    valid_sites = set(unique[counts >= 2])
    mask = np.array([s in valid_sites for s in sites_arr])
    if mask.sum() < len(mask):
        print(f"  ComBat: dropping {len(mask) - mask.sum()} singleton-site samples")
    covars = pd.DataFrame({"batch": sites_arr[mask]})
    result = neuroCombat(dat=fc_features[mask].T, covars=covars, batch_col="batch")
    # Re-insert original rows for dropped samples (use raw features)
    out = fc_features.copy()
    out[mask] = result["data"].T
    # Replace any remaining NaN with 0
    out = np.nan_to_num(out, nan=0.0)
    return out


def load_jepa_embeddings(checkpoint_path, preset, task_key, device="cuda"):
    import torch
    from src.jepa_fusion.evaluate_jepa import load_checkpoint, extract_scan_embeddings
    from src.jepa_clean.data import WindowedFMRIDataset

    model, _ckpt, checkpoint_cfg = load_checkpoint(checkpoint_path)
    model = model.to(device).eval()

    dataset = WindowedFMRIDataset(
        preset.raw_dir,
        window_seconds=float(checkpoint_cfg.get('window_seconds', 90.0)),
        stride_seconds=float(checkpoint_cfg.get('window_stride_seconds', 30.0)),
        target_timepoints=int(checkpoint_cfg.get('target_timepoints', 45)),
        sampling='center',
        exclude_unknown_diagnoses=False,
    )

    embeddings, metadata = extract_scan_embeddings(
        model, dataset, device=device,
        batch_size=128,
        window_feature='mean',
        covariance_dim=0,
        covariance_seed=0,
    )

    subject_map = {}
    for i, meta in enumerate(metadata):
        subj = str(meta['subject_id'])
        site = batch_site_label(
            str(meta.get('site_raw', meta['site'])),
            source_dataset=str(meta.get('dataset', '')),
        )
        dx = str(meta['diag_name'])
        if subj not in subject_map:
            subject_map[subj] = {'embeds': [], 'site': site, 'dx': dx}
        subject_map[subj]['embeds'].append(embeddings[i])

    subjects = sorted(subject_map.keys())
    agg_embeds = np.array([np.mean(subject_map[s]['embeds'], axis=0) for s in subjects])
    agg_sites = [subject_map[s]['site'] for s in subjects]
    agg_dx = [subject_map[s]['dx'] for s in subjects]

    keep = []
    for i, dx in enumerate(agg_dx):
        if task_key == 'hc_asd' and dx in (HEALTHY_CONTROL_LABEL, AUTISM_LABEL):
            keep.append(i)
        elif task_key == 'hc_disorder':
            keep.append(i)
        else:
            keep.append(i)
    keep = np.array(keep)
    return agg_embeds[keep], [agg_sites[i] for i in keep], [agg_dx[i] for i in keep]

def aggregate_fc_to_subject(features, sites, diagnoses, subjects):
    subject_map = {}
    for i, subj in enumerate(subjects):
        if subj not in subject_map:
            subject_map[subj] = {"feats": [], "site": sites[i], "dx": diagnoses[i]}
        subject_map[subj]["feats"].append(features[i])
    ordered = sorted(subject_map.keys())
    agg_feats = np.array([np.mean(subject_map[s]["feats"], axis=0) for s in ordered])
    agg_sites = [subject_map[s]["site"] for s in ordered]
    agg_dx = [subject_map[s]["dx"] for s in ordered]
    return agg_feats, agg_sites, agg_dx


def plot_tsne_panel(ax, coords, labels, title, legend=False, marker_size=5, alpha=0.6):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    if n_classes <= 2:
        colors = ["#1f77b4", "#d62728"][:n_classes]
    elif n_classes <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    else:
        colors = plt.cm.gist_ncar(np.linspace(0, 0.95, n_classes))
    for i in range(n_classes):
        mask = y == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[i]], s=marker_size, alpha=alpha,
                   label=le.classes_[i] if legend else None, rasterized=True)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    if legend and n_classes <= 15:
        ax.legend(fontsize=5, loc="best", markerscale=2, ncol=max(1, n_classes // 8))


def make_tsne_figure(dataset_key, checkpoint_path, output_path, device="cuda", perplexity=30, seed=42):
    preset = get_dataset_preset(dataset_key)
    task_key = preset.default_task

    print(f"Loading FC features for {dataset_key}...")
    fc_feats, sites, diagnoses, subjects = load_fc_features(preset, task_key)
    fc_subj, sites_subj, dx_subj = aggregate_fc_to_subject(fc_feats, sites, diagnoses, subjects)

    print(f"Loading JEPA embeddings...")
    jepa_subj, jepa_sites, jepa_dx = load_jepa_embeddings(checkpoint_path, preset, task_key, device)

    # PCA first (fast), then ComBat on reduced dims
    fc_std = StandardScaler().fit_transform(fc_subj)
    jepa_std = StandardScaler().fit_transform(jepa_subj)

    pca_dim = 50
    fc_pca = PCA(n_components=min(pca_dim, fc_std.shape[1])).fit_transform(fc_std)
    jepa_pca = PCA(n_components=min(pca_dim, jepa_std.shape[1])).fit_transform(jepa_std)

    print("Applying ComBat on PCA-reduced features...")
    combat_pca = load_combat_features(fc_pca, sites_subj)

    print("Running t-SNE...")
    tsne_fc = TSNE(n_components=2, perplexity=perplexity, random_state=seed, n_iter=1000).fit_transform(fc_pca)
    tsne_combat = TSNE(n_components=2, perplexity=perplexity, random_state=seed, n_iter=1000).fit_transform(combat_pca)
    tsne_jepa = TSNE(n_components=2, perplexity=perplexity, random_state=seed, n_iter=1000).fit_transform(jepa_pca)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    plot_tsne_panel(axes[0, 0], tsne_fc, sites_subj, "Static FC", legend=True)
    plot_tsne_panel(axes[0, 1], tsne_combat, sites_subj, "FC + ComBat")
    plot_tsne_panel(axes[0, 2], tsne_jepa, jepa_sites, "MS-JEPA (ours)")
    axes[0, 0].set_ylabel("Colored by Site", fontsize=12, fontweight="bold")

    plot_tsne_panel(axes[1, 0], tsne_fc, dx_subj, "Static FC", legend=True)
    plot_tsne_panel(axes[1, 1], tsne_combat, dx_subj, "FC + ComBat")
    plot_tsne_panel(axes[1, 2], tsne_jepa, jepa_dx, "MS-JEPA (ours)")
    axes[1, 0].set_ylabel("Colored by Diagnosis", fontsize=12, fontweight="bold")

    fig.suptitle(f"t-SNE Embeddings \u2014 {preset.label}", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def make_lowk_figure(output_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex="col")

    for row, dataset in enumerate(["srpbs", "abide"]):
        json_path = f"results/reports/tables/jepa_fusion/{dataset}_lowk_stress_test.json"
        with open(json_path) as f:
            data = json.load(f)

        results = data["results"]
        k_values = sorted(int(k) for k in results.keys())

        for col, (metric_group, metric_name, ylabel) in enumerate([
            ("site", "balanced_accuracy", "Site bAcc (%)"),
            ("diagnosis", "balanced_accuracy", "Dx bAcc (%)"),
        ]):
            ax = axes[row, col]
            for method, marker, color, label in [
                ("fc", None, "#888888", "Static FC"),
                ("fc_combat", "o", "#d62728", "FC + ComBat"),
                ("fc_covbat", "s", "#1f77b4", "FC + CovBat"),
            ]:
                means, stds = [], []
                for k in k_values:
                    m = results[str(k)][method][metric_group][metric_name]
                    means.append(m["mean"] * 100)
                    stds.append(m["std"] * 100)
                means, stds = np.array(means), np.array(stds)

                if method == "fc":
                    ax.axhline(means[0], color=color, linestyle="--", linewidth=1.5, label=label, alpha=0.7)
                    ax.fill_between(k_values, means[0]-stds[0], means[0]+stds[0], color=color, alpha=0.1)
                else:
                    ax.errorbar(k_values, means, yerr=stds, fmt="-"+marker, color=color,
                                label=label, capsize=3, markersize=5, linewidth=1.5)

            if col == 0:
                n_sites = 11 if dataset == "srpbs" else 30
                chance = 100.0 / n_sites
                ax.axhline(chance, color="green", linestyle=":", linewidth=1, alpha=0.7, label=f"Chance ({chance:.1f}%)")

            ax.set_ylabel(ylabel, fontsize=10)
            if row == 1:
                ax.set_xlabel("$K$ (subjects per site)", fontsize=10)
            ax.set_xticks(k_values)

            ds_label = "SRPBS-1600" if dataset == "srpbs" else "ABIDE-I + II"
            metric_label = "Site Probe" if col == 0 else "Diagnosis"
            ax.set_title(f"{ds_label} \u2014 {metric_label}", fontsize=11, fontweight="bold")

            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")



def load_covbat_features(fc_features, sites, n_pc=10, max_components=100, seed=42):
    """Whole-dataset CovBat: ComBat + PCA score batch-correction."""
    from neuroCombat import neuroCombat
    import pandas as pd
    # Step 1: ComBat on full features
    combat_feats = load_combat_features(fc_features, sites)
    # Step 2: PCA on ComBat-corrected features
    scaler = StandardScaler()
    z = scaler.fit_transform(combat_feats)
    n_comp = min(max_components, z.shape[0] - 1, z.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    scores = pca.fit_transform(z)
    # Step 3: ComBat on leading PC scores
    sites_arr = np.array(sites)
    unique, counts = np.unique(sites_arr, return_counts=True)
    valid_sites = set(unique[counts >= 2])
    mask = np.array([s in valid_sites for s in sites_arr])
    eff_npc = min(n_pc, scores.shape[1])
    covars = pd.DataFrame({"batch": sites_arr[mask]})
    adjusted = neuroCombat(dat=scores[mask, :eff_npc].T, covars=covars, batch_col="batch")
    scores_adj = scores.copy()
    scores_adj[mask, :eff_npc] = adjusted["data"].T
    # Step 4: Inverse transform
    result = scaler.inverse_transform(pca.inverse_transform(scores_adj))
    return np.nan_to_num(result, nan=0.0)


def make_site_distribution_figure(dataset_key, checkpoint_path, output_path, device="cuda"):
    """KDE of PC1 per site for all 4 methods, with silhouette score annotation."""
    from scipy.stats import gaussian_kde
    from sklearn.metrics import silhouette_score

    preset = get_dataset_preset(dataset_key)
    task_key = preset.default_task

    print(f"Loading FC features for {dataset_key}...")
    fc_feats, sites, diagnoses, subjects = load_fc_features(preset, task_key)
    fc_subj, sites_subj, dx_subj = aggregate_fc_to_subject(fc_feats, sites, diagnoses, subjects)

    print("Loading JEPA embeddings...")
    jepa_subj, jepa_sites, jepa_dx = load_jepa_embeddings(checkpoint_path, preset, task_key, device)

    # ComBat and CovBat on FULL features (correct way)
    print("Applying ComBat on full features (this may take a while)...")
    combat_subj = load_combat_features(fc_subj, sites_subj)
    print("Applying CovBat on full features...")
    covbat_subj = load_covbat_features(fc_subj, sites_subj)

    # Now PCA for visualization and silhouette
    pca_dim = 50
    fc_pca = PCA(n_components=pca_dim).fit_transform(StandardScaler().fit_transform(fc_subj))
    combat_pca = PCA(n_components=pca_dim).fit_transform(StandardScaler().fit_transform(combat_subj))
    covbat_pca = PCA(n_components=pca_dim).fit_transform(StandardScaler().fit_transform(covbat_subj))
    jepa_pca = PCA(n_components=min(pca_dim, jepa_subj.shape[1])).fit_transform(
        StandardScaler().fit_transform(jepa_subj))

    # Silhouette scores on PCA space
    le = LabelEncoder()
    fc_site_y = le.fit_transform(sites_subj)
    jepa_site_y = le.fit_transform(jepa_sites)

    sil_fc = silhouette_score(fc_pca, fc_site_y)
    sil_combat = silhouette_score(combat_pca, fc_site_y)
    sil_covbat = silhouette_score(covbat_pca, fc_site_y)
    sil_jepa = silhouette_score(jepa_pca, jepa_site_y)

    print(f"Silhouette scores — FC: {sil_fc:.3f}, ComBat: {sil_combat:.3f}, "
          f"CovBat: {sil_covbat:.3f}, JEPA: {sil_jepa:.3f}")

    # PC1 for KDE
    methods = [
        ("Static FC", fc_pca[:, 0], sites_subj, sil_fc),
        ("FC + ComBat", combat_pca[:, 0], sites_subj, sil_combat),
        ("FC + CovBat", covbat_pca[:, 0], sites_subj, sil_covbat),
        ("MS-JEPA (ours)", jepa_pca[:, 0], jepa_sites, sil_jepa),
    ]

    unique_sites = sorted(set(sites_subj))
    n_sites = len(unique_sites)
    cmap = plt.cm.tab20 if n_sites <= 20 else plt.cm.gist_ncar
    site_colors = {s: cmap(i / max(n_sites - 1, 1)) for i, s in enumerate(unique_sites)}

    # Site probe bAcc from evaluation tables (mean across 5 seeds)
    SITE_BACC = {
        "srpbs": {
            "Static FC": 58.8, "FC + ComBat": 3.6,
            "FC + CovBat": 0.9, "MS-JEPA (ours)": 19.5,
        },
        "abide": {
            "Static FC": 34.2, "FC + ComBat": 0.6,
            "FC + CovBat": 0.0, "MS-JEPA (ours)": 6.2,
        },
    }
    site_bacc = SITE_BACC.get(dataset_key, {m: 0 for m, _, _, _ in methods})

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for ax, (title, pc1, site_labels, sil) in zip(axes, methods):
        x_min, x_max = pc1.min() - 1, pc1.max() + 1
        xs = np.linspace(x_min, x_max, 300)

        for site in unique_sites:
            mask = np.array([s == site for s in site_labels])
            if mask.sum() < 3:
                continue
            kde = gaussian_kde(pc1[mask], bw_method=0.3)
            ax.plot(xs, kde(xs), color=site_colors[site], alpha=0.7, linewidth=1.5, label=site)
            ax.fill_between(xs, kde(xs), alpha=0.08, color=site_colors[site])

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("PC 1", fontsize=10)
        ax.text(0.97, 0.95, f"Silhouette = {sil:.3f}\nSite bAcc = {site_bacc[title]:.1f}%",
                transform=ax.transAxes, fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    axes[0].set_ylabel("Density", fontsize=10)
    # Shared horizontal legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(n_sites, 11),
               fontsize=7, frameon=True, bbox_to_anchor=(0.5, -0.12))

    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="figure")

    p1 = sub.add_parser("tsne")
    p1.add_argument("--dataset", choices=["abide", "srpbs"], default="srpbs")
    p1.add_argument("--checkpoint", default="checkpoints/jepa_fusion/five_seed_rotary_curr_both_seed0/phase2_final.pt")
    p1.add_argument("--output", default=None)
    p1.add_argument("--device", default="cuda")
    p1.add_argument("--perplexity", type=int, default=30)

    p2 = sub.add_parser("lowk")
    p2.add_argument("--output", default="results/figures/lowk_degradation.png")

    p3 = sub.add_parser("sitedist")
    p3.add_argument("--dataset", choices=["abide", "srpbs"], default="srpbs")
    p3.add_argument("--checkpoint", default="checkpoints/jepa_fusion/five_seed_rotary_curr_both_seed0/phase2_final.pt")
    p3.add_argument("--output", default=None)
    p3.add_argument("--device", default="cuda")

    args = parser.parse_args()
    if args.figure == "tsne":
        output = args.output or f"results/figures/tsne_{args.dataset}.png"
        make_tsne_figure(args.dataset, args.checkpoint, output, device=args.device, perplexity=args.perplexity)
    elif args.figure == "sitedist":
        output = args.output or f"results/figures/site_distribution_{args.dataset}.png"
        make_site_distribution_figure(args.dataset, args.checkpoint, output, device=args.device)
    elif args.figure == "lowk":
        make_lowk_figure(args.output)
    else:
        parser.print_help()
