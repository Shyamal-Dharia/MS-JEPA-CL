from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from nilearn.datasets import fetch_atlas_schaefer_2018
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

try:
    from .evaluate_jepa_subject_vote import (
        EvalWindowRecord,
        collate_fn,
        extract_window_embeddings,
        load_checkpoint,
    )
    from .protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        center_site_label,
        filter_records_for_task,
        get_dataset_preset,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.evaluate_jepa_subject_vote import (
        EvalWindowRecord,
        collate_fn,
        extract_window_embeddings,
        load_checkpoint,
    )
    from src.jepa_fusion.protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        center_site_label,
        filter_records_for_task,
        get_dataset_preset,
    )
    from src.jepa_clean.data import WindowedFMRIDataset
    from src.jepa_clean.pooling import pool_sequence_features
    from src.jepa_clean.repro import seed_everything
else:
    from ..jepa_clean.data import WindowedFMRIDataset
    from ..jepa_clean.pooling import pool_sequence_features
    from ..jepa_clean.repro import seed_everything


@dataclass(frozen=True)
class Direction:
    key: str
    train_label: str
    eval_dataset: str
    task_key: str
    task_label: str
    mixed_sites_only: bool
    checkpoint_template: str


DIRECTIONS = {
    "final_abide": Direction(
        key="final_abide",
        train_label="SRPBS+ABIDE-I+ABIDE-II",
        eval_dataset="abide",
        task_key="hc_asd",
        task_label="HC vs ASD",
        mixed_sites_only=True,
        checkpoint_template="final_run_v1_rotary_curr_replay_nce_seed{seed}",
    ),
    "final_all_disorder": Direction(
        key="final_all_disorder",
        train_label="SRPBS+ABIDE-I+ABIDE-II",
        eval_dataset="all_disorder",
        task_key="hc_disorder",
        task_label="HC vs any disorder",
        mixed_sites_only=True,
        checkpoint_template="final_run_v1_rotary_curr_replay_nce_seed{seed}",
    ),
    "abide1_to_abide2": Direction(
        key="abide1_to_abide2",
        train_label="SRPBS+ABIDE-I",
        eval_dataset="abide2",
        task_key="hc_asd",
        task_label="HC vs ASD",
        mixed_sites_only=True,
        checkpoint_template=(
            "final_run_v1_transfer_srpbs_abide1_to_abide2_rotary_curr_replay_nce_seed{seed}"
        ),
    ),
    "abide2_to_abide1": Direction(
        key="abide2_to_abide1",
        train_label="SRPBS+ABIDE-II",
        eval_dataset="abide1",
        task_key="hc_asd",
        task_label="HC vs ASD",
        mixed_sites_only=True,
        checkpoint_template=(
            "final_run_v1_transfer_srpbs_abide2_to_abide1_rotary_curr_replay_nce_seed{seed}"
        ),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute diagnosis-probe gradient-times-input attribution maps for final "
            "MS-JEPA transfer checkpoints and project parcel scores to Schaefer-400."
        )
    )
    parser.add_argument(
        "--directions",
        default="final_abide",
        help=f"Comma-separated directions. Choices: {','.join(DIRECTIONS)}",
    )
    parser.add_argument("--model-seeds", default="0,8,16,32,128")
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--checkpoint-root", default="checkpoints/jepa_fusion/final_run_v1")
    parser.add_argument(
        "--checkpoint-template",
        default=None,
        help="Override the run-directory template, e.g. 'final_run_v1_rotary_curr_replay_seed{seed}'.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures/jepa_fusion/diagnosis_attribution",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--attr-batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=30.0)
    parser.add_argument("--target-timepoints", type=int, default=None)
    parser.add_argument(
        "--max-attribution-windows-per-site",
        type=int,
        default=0,
        help="Optional cap for attribution windows per held-out site. 0 uses all windows.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=80.0,
        help="Absolute-value percentile used only for visual thresholding.",
    )
    return parser.parse_args()


def parse_csv(text: str, cast=str):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def build_probe(seed: int, y_train: np.ndarray):
    n_comp = min(50, len(y_train))
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_comp, random_state=seed),
        LogisticRegression(
            C=0.1,
            max_iter=2000,
            random_state=seed,
            class_weight="balanced",
        ),
    )


def linear_probe_weight_in_embedding_space(clf) -> np.ndarray:
    scaler: StandardScaler = clf.named_steps["standardscaler"]
    pca: PCA = clf.named_steps["pca"]
    logreg: LogisticRegression = clf.named_steps["logisticregression"]
    weight_scaled = logreg.coef_[0] @ pca.components_
    return (weight_scaled / scaler.scale_).astype(np.float32)


def prepare_dataset_and_task_records(
    *,
    dataset_key: str,
    checkpoint_cfg: dict,
    window_seconds: float | None,
    window_stride_seconds: float,
    target_timepoints: int | None,
):
    if dataset_key == "all_disorder":
        abide_preset = get_dataset_preset("abide")
        srpbs_preset = get_dataset_preset("srpbs")
        raw_dir = tuple(abide_preset.raw_dir) + (srpbs_preset.raw_dir,)
        preset = None
    else:
        preset = get_dataset_preset(dataset_key)
        raw_dir = preset.raw_dir
    dataset = WindowedFMRIDataset(
        raw_dir,
        window_seconds=float(window_seconds or checkpoint_cfg.get("window_seconds", 90.0)),
        stride_seconds=float(window_stride_seconds),
        target_timepoints=int(target_timepoints or checkpoint_cfg.get("target_timepoints", 45)),
        sampling="sliding",
        exclude_unknown_diagnoses=False,
    )
    return dataset, preset


def filter_embeddings_for_task(
    embeddings,
    metadata,
    *,
    task_key: str,
    mixed_sites_only: bool,
    task_label: str,
):
    eval_records = [
        EvalWindowRecord(
            index=idx,
            subject=str(meta["subject_id"]),
            site=center_site_label(str(meta.get("site_raw", meta["site"]))),
            diagnosis=str(meta["diag_name"]),
        )
        for idx, meta in enumerate(metadata)
    ]
    task = filter_records_for_task(
        eval_records,
        task_key,
        mixed_sites_only=mixed_sites_only,
        healthy_control_label=HEALTHY_CONTROL_LABEL,
        autism_label=AUTISM_LABEL,
    )
    indices = np.asarray([record.index for record in task.records], dtype=int)
    filtered_metadata = []
    for record, y in zip(task.records, task.y.tolist()):
        meta = dict(metadata[record.index])
        meta["site"] = record.site
        meta["y"] = int(y)
        filtered_metadata.append(meta)
    return embeddings[indices], filtered_metadata, task.y.astype(np.int64), task


def select_attribution_indices(test_indices: np.ndarray, sites: np.ndarray, held_out: str, cap: int, seed: int):
    selected = test_indices
    if cap <= 0 or len(selected) <= cap:
        return selected
    rng = np.random.default_rng(seed)
    site_indices = np.flatnonzero(sites == held_out)
    local = np.intersect1d(selected, site_indices, assume_unique=False)
    if len(local) <= cap:
        return local
    return np.sort(rng.choice(local, size=cap, replace=False))


def compute_window_attribution(
    *,
    model,
    dataset,
    task_metadata: list[dict],
    task_indices: np.ndarray,
    selected_task_indices: np.ndarray,
    embedding_weight: np.ndarray,
    device: str,
    batch_size: int,
):
    model.eval()
    weight = torch.as_tensor(embedding_weight, dtype=torch.float32, device=device)
    selected_dataset_indices = task_indices[selected_task_indices].tolist()
    selected_subset = torch.utils.data.Subset(dataset, selected_dataset_indices)
    loader = DataLoader(
        selected_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    signed_sum = np.zeros(400, dtype=np.float64)
    abs_sum = np.zeros(400, dtype=np.float64)
    asd_sum = np.zeros(400, dtype=np.float64)
    hc_sum = np.zeros(400, dtype=np.float64)
    n_windows = 0
    n_asd = 0
    n_hc = 0
    cursor = 0

    for x, _labels in loader:
        batch_n = x.shape[0]
        x = x.to(device)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        sequence = model.encode_sequence(x)
        pooled = pool_sequence_features(sequence, mode="mean", covariance_dim=32, covariance_seed=0)
        logits = pooled @ weight
        objective = logits.sum()
        grads = torch.autograd.grad(objective, x, retain_graph=False, create_graph=False)[0]
        attr = (grads * x).detach().abs().mean(dim=1).cpu().numpy()
        signed_attr = (grads * x).detach().mean(dim=1).cpu().numpy()

        signed_sum += signed_attr.sum(axis=0)
        abs_sum += attr.sum(axis=0)
        batch_task_indices = selected_task_indices[cursor : cursor + batch_n]
        cursor += batch_n
        labels = np.asarray([task_metadata[idx]["y"] for idx in batch_task_indices], dtype=int)
        if np.any(labels == 1):
            asd_sum += attr[labels == 1].sum(axis=0)
            n_asd += int(np.sum(labels == 1))
        if np.any(labels == 0):
            hc_sum += attr[labels == 0].sum(axis=0)
            n_hc += int(np.sum(labels == 0))
        n_windows += batch_n

    return {
        "signed": signed_sum / max(n_windows, 1),
        "absolute": abs_sum / max(n_windows, 1),
        "asd_absolute": asd_sum / max(n_asd, 1),
        "hc_absolute": hc_sum / max(n_hc, 1),
        "n_windows": n_windows,
        "n_asd_windows": n_asd,
        "n_hc_windows": n_hc,
    }


def zscore_scores(scores: np.ndarray):
    scores = np.asarray(scores, dtype=np.float64)
    sd = float(np.std(scores))
    if sd <= 1e-12:
        return scores * 0.0
    return (scores - float(np.mean(scores))) / sd


def schaefer_scores_to_img(scores: np.ndarray):
    atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    atlas_img = nib.load(atlas["maps"])
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
    out = np.zeros(atlas_data.shape, dtype=np.float32)
    for roi_idx, value in enumerate(scores.tolist(), start=1):
        out[atlas_data == roi_idx] = float(value)
    return nib.Nifti1Image(out, atlas_img.affine, atlas_img.header)


def schaefer_labels():
    atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    labels = [
        label.decode("utf-8") if isinstance(label, bytes) else str(label)
        for label in atlas["labels"]
    ]
    if len(labels) == 401 and labels[0].lower().startswith("background"):
        labels = labels[1:]
    return labels[:400]


def save_brain_map(scores: np.ndarray, out_prefix: Path, title: str, threshold_percentile: float):
    img = schaefer_scores_to_img(scores)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    nifti_path = out_prefix.with_suffix(".nii.gz")
    png_path = out_prefix.with_suffix(".png")
    nib.save(img, str(nifti_path))

    abs_scores = np.abs(scores)
    nonzero = abs_scores[abs_scores > 0]
    threshold = None
    if len(nonzero):
        threshold = float(np.percentile(nonzero, threshold_percentile))
    vmax = float(np.max(abs_scores)) if len(abs_scores) else 1.0
    if vmax <= 1e-12:
        vmax = 1.0

    display = plotting.plot_stat_map(
        img,
        display_mode="ortho",
        cut_coords=(0, -18, 24),
        cmap="cold_hot",
        symmetric_cbar=True,
        threshold=threshold,
        vmax=vmax,
        title=None,
        annotate=True,
        black_bg=False,
    )
    display.savefig(str(png_path), dpi=300)
    display.close()
    return nifti_path, png_path


def save_parcel_table(scores_by_name: dict[str, np.ndarray], output_csv: Path):
    labels = schaefer_labels()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["roi", "label", *scores_by_name.keys()]
    rows = [",".join(header)]
    for idx, label in enumerate(labels):
        values = [str(idx + 1), label]
        for scores in scores_by_name.values():
            values.append(f"{float(scores[idx]):.8g}")
        rows.append(",".join(values))
    output_csv.write_text("\n".join(rows) + "\n")


def exact_signflip_pvalues(seed_scores: np.ndarray):
    """Two-sided exact paired sign-flip p-values and max-stat p-values across seeds."""
    seed_scores = np.asarray(seed_scores, dtype=np.float64)
    if seed_scores.ndim != 2:
        raise ValueError(f"Expected seed_scores with shape (n_seeds, n_rois), got {seed_scores.shape}")
    n_seeds, n_rois = seed_scores.shape
    observed = np.abs(seed_scores.mean(axis=0))
    flips = np.asarray(
        [
            [1.0 if (mask >> bit) & 1 else -1.0 for bit in range(n_seeds)]
            for mask in range(2**n_seeds)
        ],
        dtype=np.float64,
    )
    null_means = np.abs((flips @ seed_scores) / float(n_seeds))
    p_uncorrected = np.mean(null_means >= observed[None, :] - 1e-12, axis=0)
    null_max = null_means.max(axis=1)
    p_maxstat = np.mean(null_max[:, None] >= observed[None, :] - 1e-12, axis=0)
    return p_uncorrected, p_maxstat


def benjamini_hochberg(p_values: np.ndarray):
    p_values = np.asarray(p_values, dtype=np.float64)
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    q_sorted = np.empty(n, dtype=np.float64)
    running = 1.0
    for rank in range(n, 0, -1):
        idx = rank - 1
        running = min(running, sorted_p[idx] * n / rank)
        q_sorted[idx] = running
    q_values = np.empty(n, dtype=np.float64)
    q_values[order] = np.clip(q_sorted, 0.0, 1.0)
    return q_values


def save_seedwise_stats(seed_maps: list[dict], output_csv: Path):
    labels = schaefer_labels()
    metric_names = ["signed_z", "absolute_z", "asd_minus_hc_abs_z"]
    arrays = {
        name: np.stack([np.asarray(seed_map[name], dtype=np.float64) for seed_map in seed_maps], axis=0)
        for name in metric_names
    }

    stats = {}
    for name, values in arrays.items():
        p_signflip, p_maxstat = exact_signflip_pvalues(values)
        stats[name] = {
            "mean": values.mean(axis=0),
            "sd": values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros(values.shape[1]),
            "p_signflip": p_signflip,
            "q_fdr": benjamini_hochberg(p_signflip),
            "p_maxstat": p_maxstat,
        }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["roi", "label"]
    for name in metric_names:
        header.extend(
            [
                f"{name}_mean",
                f"{name}_sd",
                f"{name}_p_signflip",
                f"{name}_q_fdr",
                f"{name}_p_maxstat",
            ]
        )
    rows = [",".join(header)]
    for idx, label in enumerate(labels):
        row = [str(idx + 1), label]
        for name in metric_names:
            row.extend(
                [
                    f"{stats[name]['mean'][idx]:.8g}",
                    f"{stats[name]['sd'][idx]:.8g}",
                    f"{stats[name]['p_signflip'][idx]:.8g}",
                    f"{stats[name]['q_fdr'][idx]:.8g}",
                    f"{stats[name]['p_maxstat'][idx]:.8g}",
                ]
            )
        rows.append(",".join(row))
    output_csv.write_text("\n".join(rows) + "\n")
    return stats


def run_direction(direction: Direction, seeds: list[int], args, device: str):
    checkpoint_root = Path(args.checkpoint_root)
    out_dir = Path(args.output_dir) / direction.key
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_maps = []
    seed_rows = []

    for model_seed in seeds:
        seed_everything(int(model_seed))
        template = args.checkpoint_template or direction.checkpoint_template
        ckpt = checkpoint_root / template.format(seed=model_seed) / "phase2_final.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        print(f"[{direction.key}] seed={model_seed} loading {ckpt}", flush=True)
        model, _checkpoint, cfg = load_checkpoint(str(ckpt))
        model.to(device)

        dataset, _preset = prepare_dataset_and_task_records(
            dataset_key=direction.eval_dataset,
            checkpoint_cfg=cfg,
            window_seconds=args.window_seconds,
            window_stride_seconds=args.window_stride_seconds,
            target_timepoints=args.target_timepoints,
        )
        embeddings, metadata = extract_window_embeddings(
            model,
            dataset,
            device=device,
            batch_size=args.batch_size,
            window_feature="mean",
            covariance_dim=32,
            covariance_seed=0,
        )
        task_embeddings, task_metadata, y, task = filter_embeddings_for_task(
            embeddings,
            metadata,
            task_key=direction.task_key,
            mixed_sites_only=direction.mixed_sites_only,
            task_label=direction.task_label,
        )
        task_indices = np.asarray([record.index for record in task.records], dtype=int)
        sites = np.asarray([str(meta["site"]) for meta in task_metadata], dtype=object)
        unique_sites = np.unique(sites)

        signed_total = np.zeros(400, dtype=np.float64)
        abs_total = np.zeros(400, dtype=np.float64)
        asd_total = np.zeros(400, dtype=np.float64)
        hc_total = np.zeros(400, dtype=np.float64)
        n_total = n_asd = n_hc = 0

        for held_out in unique_sites:
            train_mask = sites != held_out
            test_mask = sites == held_out
            y_train = y[train_mask]
            y_test = y[test_mask]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            clf = build_probe(args.probe_seed, y_train)
            clf.fit(task_embeddings[train_mask], y_train)
            weight = linear_probe_weight_in_embedding_space(clf)
            selected = select_attribution_indices(
                np.flatnonzero(test_mask),
                sites,
                str(held_out),
                int(args.max_attribution_windows_per_site),
                seed=int(model_seed) + args.probe_seed,
            )
            attr = compute_window_attribution(
                model=model,
                dataset=dataset,
                task_metadata=task_metadata,
                task_indices=task_indices,
                selected_task_indices=selected,
                embedding_weight=weight,
                device=device,
                batch_size=args.attr_batch_size,
            )
            signed_total += attr["signed"] * attr["n_windows"]
            abs_total += attr["absolute"] * attr["n_windows"]
            asd_total += attr["asd_absolute"] * attr["n_asd_windows"]
            hc_total += attr["hc_absolute"] * attr["n_hc_windows"]
            n_total += attr["n_windows"]
            n_asd += attr["n_asd_windows"]
            n_hc += attr["n_hc_windows"]

        seed_map = {
            "signed_z": zscore_scores(signed_total / max(n_total, 1)),
            "absolute_z": zscore_scores(abs_total / max(n_total, 1)),
            "asd_minus_hc_abs_z": zscore_scores(
                (asd_total / max(n_asd, 1)) - (hc_total / max(n_hc, 1))
            ),
            "n_windows": n_total,
            "n_asd_windows": n_asd,
            "n_hc_windows": n_hc,
        }
        seed_maps.append(seed_map)
        seed_rows.append(
            f"| {model_seed} | {n_total} | {n_asd} | {n_hc} |"
        )
        print(
            f"[{direction.key}] seed={model_seed} attribution windows={n_total} "
            f"ASD={n_asd} HC={n_hc}",
            flush=True,
        )

    aggregate = {
        "signed_z": np.mean([m["signed_z"] for m in seed_maps], axis=0),
        "absolute_z": np.mean([m["absolute_z"] for m in seed_maps], axis=0),
        "asd_minus_hc_abs_z": np.mean([m["asd_minus_hc_abs_z"] for m in seed_maps], axis=0),
    }
    stability = np.mean(
        [np.sign(m["asd_minus_hc_abs_z"]) == np.sign(aggregate["asd_minus_hc_abs_z"]) for m in seed_maps],
        axis=0,
    )
    aggregate["asd_minus_hc_stability"] = stability

    np.savez_compressed(
        out_dir / f"{direction.key}_seedwise_attribution.npz",
        seeds=np.asarray(seeds, dtype=np.int64),
        signed_z=np.stack([m["signed_z"] for m in seed_maps], axis=0),
        absolute_z=np.stack([m["absolute_z"] for m in seed_maps], axis=0),
        asd_minus_hc_abs_z=np.stack([m["asd_minus_hc_abs_z"] for m in seed_maps], axis=0),
    )
    stats = save_seedwise_stats(seed_maps, out_dir / f"{direction.key}_parcel_seed_stats.csv")
    save_parcel_table(aggregate, out_dir / f"{direction.key}_parcel_attribution.csv")
    for name, scores in aggregate.items():
        if name.endswith("stability"):
            plot_scores = scores * np.sign(aggregate["asd_minus_hc_abs_z"])
        else:
            plot_scores = scores
        save_brain_map(
            plot_scores,
            out_dir / f"{direction.key}_{name}",
            title=f"{direction.key}: {name}",
            threshold_percentile=float(args.threshold_percentile),
        )

    summary = [
        f"# Diagnosis Attribution Map: {direction.key}",
        "",
        f"Training source: {direction.train_label}",
        f"Evaluation dataset: {direction.eval_dataset}",
        f"Evaluation task: {direction.task_label}",
        f"Model seeds: {', '.join(str(s) for s in seeds)}",
        "",
        "Attribution is gradient-times-input for the positive-class logit of the LOSO diagnosis probe "
        f"({direction.task_label}), "
        "computed through the frozen MS-JEPA encoder and averaged over held-out windows/sites/seeds. "
        "Maps are Schaefer-400 parcel-level scores projected back into MNI atlas space; they are "
        "not voxelwise structural MRI saliency maps.",
        "",
        "| Model seed | Windows | ASD windows | HC windows |",
        "|---:|---:|---:|---:|",
        *seed_rows,
        "",
        "Outputs:",
        f"- Parcel table: `{direction.key}_parcel_attribution.csv`",
        f"- Seed-wise attribution arrays: `{direction.key}_seedwise_attribution.npz`",
        f"- Seed-wise exact sign-flip statistics: `{direction.key}_parcel_seed_stats.csv`",
        f"- Orthogonal PNG/NIfTI maps: `{direction.key}_*_z.png`, `{direction.key}_*_z.nii.gz`",
        "",
        "Seed-wise inference:",
    ]
    for metric_name, metric_stats in stats.items():
        n_fdr = int(np.sum(metric_stats["q_fdr"] < 0.05))
        n_maxstat = int(np.sum(metric_stats["p_maxstat"] < 0.05))
        min_p = float(np.min(metric_stats["p_signflip"]))
        min_q = float(np.min(metric_stats["q_fdr"]))
        min_max = float(np.min(metric_stats["p_maxstat"]))
        summary.extend(
            [
                f"- `{metric_name}`: min exact sign-flip p={min_p:.4f}, "
                f"min FDR q={min_q:.4f}, min max-stat p={min_max:.4f}, "
                f"FDR-significant parcels={n_fdr}, max-stat significant parcels={n_maxstat}.",
            ]
        )
    (out_dir / "README.md").write_text("\n".join(summary) + "\n")
    return out_dir


def main():
    args = parse_args()
    directions = parse_csv(args.directions, str)
    seeds = parse_csv(args.model_seeds, int)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}", flush=True)
    output_dirs = []
    for key in directions:
        if key not in DIRECTIONS:
            raise KeyError(f"Unknown direction {key!r}. Choices: {sorted(DIRECTIONS)}")
        output_dirs.append(run_direction(DIRECTIONS[key], seeds, args, device))
    print("Wrote attribution outputs:", flush=True)
    for out_dir in output_dirs:
        print(f"  {out_dir}", flush=True)


if __name__ == "__main__":
    main()
