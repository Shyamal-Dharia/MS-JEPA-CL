"""
Visualization utilities for MS-JEPA evaluation results.
Generates: t-SNE/UMAP embedding grids, training curves, per-site heatmaps.
Usage:  PYTHONPATH=src python -m jepa.visualize --all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE

# ── Constants ──
RESULTS_DIR = Path("results")
EMBED_DIR = RESULTS_DIR / "embeddings"
FIG_DIR = RESULTS_DIR / "figures"

# Methods to visualise (subset for readability)
KEY_METHODS = [
    "Raw_Transformer_from_scratch",
    "ComBat_Transformer_from_scratch",
    "SimCLR_Baseline",
    "Transformer-VAE",
    "JEPA_Vanilla_With_Curriculum",
    "JEPA_plus_Contrastive_With_Curriculum",
    "JEPA_plus_Adversarial_With_Curriculum",
    "JEPA_plus_VICReg_With_Curriculum",
]
KEY_METHOD_LABELS = {
    "Raw_Transformer_from_scratch": "Raw",
    "ComBat_Transformer_from_scratch": "ComBat",
    "SimCLR_Baseline": "SimCLR",
    "Transformer-VAE": "VAE",
    "JEPA_Vanilla_With_Curriculum": "JEPA-Vanilla",
    "JEPA_plus_Contrastive_With_Curriculum": "JEPA+CL",
    "JEPA_plus_Adversarial_With_Curriculum": "JEPA+Adv",
    "JEPA_plus_VICReg_With_Curriculum": "JEPA+VICReg",
}

HISTORY_FILES = {
    "JEPA-Vanilla (Curr)": "checkpoints/phase2/history_phase2.json",
    "JEPA-Vanilla (No Curr)": "checkpoints/phase2_vanilla_nocurr/history_phase2.json",
    "JEPA+CL (Curr)": "checkpoints/phase2_contrastive/history_phase2cl.json",
    "JEPA+CL (No Curr)": "checkpoints/phase2_contrastive_nocurr/history_phase2cl.json",
    "JEPA+Adv (Curr)": "checkpoints/phase2_adversarial/history_phase2advv2.json",
    "JEPA+Adv (No Curr)": "checkpoints/phase2_adversarial_nocurr/history_phase2advv2.json",
    "JEPA+VICReg (Curr)": "checkpoints/phase2_vicreg/history_phase2vicreg.json",
    "JEPA+VICReg (No Curr)": "checkpoints/phase2_vicreg_nocurr/history_phase2vicreg.json",
}

SITE_NAMES = [
    "ATT", "Osaka", "CamCAN-1", "CamCAN-2", "HUH-1", "HUH-2",
    "KUT", "KPM", "NaraNISM", "SWA", "UTO", "HRC", "UCalg"
]


# ── t-SNE / UMAP Grid ──
def plot_tsne_grid(perplexity=30, use_umap=False):
    """2-row grid: top row coloured by site, bottom by diagnosis."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sites = np.load(EMBED_DIR / "sites.npy")
    diags = np.load(EMBED_DIR / "diags.npy")

    methods = [m for m in KEY_METHODS if (EMBED_DIR / f"{m}.npy").exists()]
    if not methods:
        print("No embedding files found. Run evaluation first.")
        return
    n = len(methods)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, meth in enumerate(methods):
        emb = np.load(EMBED_DIR / f"{meth}.npy")
        label = KEY_METHOD_LABELS.get(meth, meth)

        if use_umap:
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42)
            except ImportError:
                print("umap-learn not installed, falling back to t-SNE")
                reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
        else:
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")

        Z = reducer.fit_transform(emb)

        # Top: coloured by site
        ax_s = axes[0, i]
        for sid in np.unique(sites):
            mask = sites == sid
            name = str(sid)
            ax_s.scatter(Z[mask, 0], Z[mask, 1], s=4, alpha=0.5, label=name)
        ax_s.set_title(label, fontsize=11)
        ax_s.set_xticks([])
        ax_s.set_yticks([])
        if i == 0:
            ax_s.set_ylabel("Coloured by Site", fontsize=10)

        # Bottom: coloured by diagnosis
        ax_d = axes[1, i]
        for did in np.unique(diags):
            mask = diags == did
            ax_d.scatter(Z[mask, 0], Z[mask, 1], s=4, alpha=0.5, label=f"Diag {did}")
        ax_d.set_xticks([])
        ax_d.set_yticks([])
        if i == 0:
            ax_d.set_ylabel("Coloured by Diagnosis", fontsize=10)

    # Legends
    handles_s, labels_s = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles_s, labels_s, loc="upper right", fontsize=7, title="Site",
               bbox_to_anchor=(1.0, 1.0), ncol=1)
    handles_d, labels_d = axes[1, -1].get_legend_handles_labels()
    fig.legend(handles_d, labels_d, loc="lower right", fontsize=7, title="Diagnosis",
               bbox_to_anchor=(1.0, 0.48), ncol=1)

    algo = "UMAP" if use_umap else "t-SNE"
    fig.suptitle(f"Embedding Visualisation ({algo})", fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / f"embedding_{algo.lower()}_grid.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"Saved {out} and .png")
    plt.close(fig)


# ── Training Curves ──
def plot_training_curves():
    """Plot training loss and site-probe accuracy for all JEPA variants."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    histories = {}
    for label, path in HISTORY_FILES.items():
        p = Path(path)
        if p.exists():
            with open(p) as f:
                histories[label] = json.load(f)
        else:
            print(f"Warning: {path} not found, skipping {label}")

    if not histories:
        print("No history files found.")
        return

    # Separate curr / no-curr
    curr = {k: v for k, v in histories.items() if "No Curr" not in k}
    nocurr = {k: v for k, v in histories.items() if "No Curr" in k}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Row 0: Curriculum variants
    ax_loss, ax_acc = axes[0]
    for label, h in curr.items():
        ep = h.get("epoch", list(range(1, len(h["train_loss"]) + 1)))
        ax_loss.plot(ep, h["train_loss"], label=label, linewidth=1.5)
        if "site_probe_accuracy" in h:
            ax_acc.plot(ep, h["site_probe_accuracy"], label=label, linewidth=1.5)
    ax_loss.set_title("Training Loss (Curriculum)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)
    ax_acc.set_title("Site Probe Accuracy (Curriculum)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    # Row 1: No-curriculum variants
    ax_loss2, ax_acc2 = axes[1]
    for label, h in nocurr.items():
        ep = h.get("epoch", list(range(1, len(h["train_loss"]) + 1)))
        ax_loss2.plot(ep, h["train_loss"], label=label, linewidth=1.5)
        if "site_probe_accuracy" in h:
            ax_acc2.plot(ep, h["site_probe_accuracy"], label=label, linewidth=1.5)
    ax_loss2.set_title("Training Loss (No Curriculum)")
    ax_loss2.set_xlabel("Epoch")
    ax_loss2.set_ylabel("Loss")
    ax_loss2.legend(fontsize=8)
    ax_loss2.grid(True, alpha=0.3)
    ax_acc2.set_title("Site Probe Accuracy (No Curriculum)")
    ax_acc2.set_xlabel("Epoch")
    ax_acc2.set_ylabel("Accuracy")
    ax_acc2.legend(fontsize=8)
    ax_acc2.grid(True, alpha=0.3)

    fig.suptitle("Phase 2 Training Curves", fontsize=14)
    plt.tight_layout()
    out = FIG_DIR / "training_curves.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"Saved {out} and .png")
    plt.close(fig)

    # Auxiliary losses subplot
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    aux_keys = {
        "contrastive_loss": "Contrastive Loss",
        "adv_loss": "Adversarial Loss",
        "vicreg_loss": "VICReg Loss",
        "replay_loss": "Replay Loss",
    }
    for idx, (key, title) in enumerate(aux_keys.items()):
        ax = axes2[idx // 2, idx % 2]
        found = False
        for label, h in histories.items():
            if key in h:
                ep = h.get("epoch", list(range(1, len(h[key]) + 1)))
                ax.plot(ep, h[key], label=label, linewidth=1.5)
                found = True
        if found:
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)

    fig2.suptitle("Auxiliary Loss Components", fontsize=14)
    plt.tight_layout()
    out2 = FIG_DIR / "auxiliary_losses.pdf"
    fig2.savefig(out2, bbox_inches="tight", dpi=200)
    fig2.savefig(out2.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"Saved {out2} and .png")
    plt.close(fig2)


# ── Per-site Heatmap ──
def plot_persite_heatmap():
    """Heatmap of per-site balanced accuracy from detailed analysis."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    detail_path = RESULTS_DIR / "detailed" / "detailed_analysis.json"
    if not detail_path.exists():
        print(f"{detail_path} not found. Run evaluation first.")
        return

    with open(detail_path) as f:
        detail = json.load(f)

    # Collect bAcc for frozen_linear, HC_vs_ALL
    task_key = "HC_vs_ALL"
    eval_type = "frozen_linear"
    method_order = []
    site_order = None
    matrix = []

    for entry in detail:
        if entry.get("eval_type") != eval_type or entry.get("task") != task_key:
            continue
        ps = entry.get("per_site", {})
        if not ps:
            continue
        method_order.append(entry["method"])
        if site_order is None:
            site_order = sorted(ps.keys(), key=lambda x: int(x))
        row = [ps.get(s, {}).get("balanced_accuracy", float("nan")) for s in site_order]
        matrix.append(row)

    if not matrix:
        print("No per-site data found for frozen_linear / HC_vs_ALL")
        return

    matrix = np.array(matrix)
    site_labels = [SITE_NAMES[int(s)] if int(s) < len(SITE_NAMES) else f"Site {s}"
                   for s in site_order]

    fig, ax = plt.subplots(figsize=(max(10, len(site_order) * 0.8), len(method_order) * 0.6 + 2))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(site_labels)))
    ax.set_xticklabels(site_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(method_order)))
    ax.set_yticklabels(method_order, fontsize=9)

    for i in range(len(method_order)):
        for j in range(len(site_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if 0.45 < val < 0.85 else "white")

    plt.colorbar(im, ax=ax, label="Balanced Accuracy")
    ax.set_title(f"Per-Site Balanced Accuracy — {eval_type} / {task_key}", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "persite_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"Saved {out} and .png")
    plt.close(fig)


# ── CLI ──
def main():
    parser = argparse.ArgumentParser(description="MS-JEPA Visualization")
    parser.add_argument("--tsne", action="store_true", help="t-SNE embedding grid")
    parser.add_argument("--umap", action="store_true", help="UMAP embedding grid")
    parser.add_argument("--curves", action="store_true", help="Training curves")
    parser.add_argument("--heatmap", action="store_true", help="Per-site heatmap")
    parser.add_argument("--all", action="store_true", help="Generate all plots")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    args = parser.parse_args()

    if args.all or args.tsne:
        print("Generating t-SNE grid...")
        plot_tsne_grid(perplexity=args.perplexity, use_umap=False)
    if args.all or args.umap:
        print("Generating UMAP grid...")
        plot_tsne_grid(perplexity=args.perplexity, use_umap=True)
    if args.all or args.curves:
        print("Generating training curves...")
        plot_training_curves()
    if args.all or args.heatmap:
        print("Generating per-site heatmap...")
        plot_persite_heatmap()

    if not any([args.all, args.tsne, args.umap, args.curves, args.heatmap]):
        parser.print_help()


if __name__ == "__main__":
    main()
