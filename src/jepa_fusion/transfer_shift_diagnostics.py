from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .evaluate_jepa import extract_scan_embeddings, load_checkpoint
    from .protocol import center_site_label, get_dataset_preset
    from ..jepa_clean.data import WindowedFMRIDataset
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.evaluate_jepa import extract_scan_embeddings, load_checkpoint
    from src.jepa_fusion.protocol import center_site_label, get_dataset_preset
    from src.jepa_clean.data import WindowedFMRIDataset


OVERLAP_ABIDE2_TARGET = {"KKI", "NYU", "OHSU", "SDSU", "Trinity", "UCLA_1", "USM"}
OVERLAP_ABIDE1_TARGET = OVERLAP_ABIDE2_TARGET


@dataclass(frozen=True)
class Cohort:
    embeddings: np.ndarray
    y: np.ndarray
    sites: np.ndarray
    subjects: np.ndarray


def parse_seed_list(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def diagnosis_to_binary(name: str) -> int | None:
    text = str(name).strip().lower()
    if not text or text == "unknown":
        return None
    if "healthy" in text or "control" in text:
        return 0
    return 1


def build_dataset(dataset_keys: list[str], checkpoint_cfg: dict, max_scans: int | None):
    raw_dirs = []
    for key in dataset_keys:
        preset = get_dataset_preset(key)
        if isinstance(preset.raw_dir, tuple):
            raw_dirs.extend(preset.raw_dir)
        else:
            raw_dirs.append(preset.raw_dir)
    return WindowedFMRIDataset(
        raw_dirs,
        window_seconds=float(checkpoint_cfg.get("window_seconds", 90.0)),
        stride_seconds=float(checkpoint_cfg.get("window_stride_seconds", 30.0)),
        target_timepoints=int(checkpoint_cfg.get("target_timepoints", 45)),
        sampling="center",
        max_scans=max_scans,
        exclude_unknown_diagnoses=False,
    )


def cache_key(checkpoint_path: Path, dataset_keys: list[str], max_scans: int | None) -> str:
    ds = "_".join(dataset_keys)
    cap = "all" if max_scans is None else str(max_scans)
    return f"{checkpoint_path.parent.name}_{checkpoint_path.stem}_{ds}_max{cap}.npz"


def load_or_extract_embeddings(
    checkpoint_path: Path,
    dataset_keys: list[str],
    *,
    cache_dir: Path,
    device: str,
    batch_size: int,
    max_scans: int | None,
    no_cache: bool,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / cache_key(checkpoint_path, dataset_keys, max_scans)
    if path.exists() and not no_cache:
        cached = np.load(path, allow_pickle=True)
        return cached["embeddings"], cached["diag_names"], cached["sites"], cached["subjects"]

    model, _checkpoint, cfg = load_checkpoint(str(checkpoint_path))
    model.to(device)
    dataset = build_dataset(dataset_keys, cfg, max_scans=max_scans)
    embeddings, metadata = extract_scan_embeddings(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        window_feature="mean",
        covariance_dim=32,
        covariance_seed=0,
    )
    diag_names = np.asarray([str(meta["diag_name"]) for meta in metadata], dtype=object)
    sites = np.asarray(
        [center_site_label(str(meta.get("site_raw", meta["site"]))) for meta in metadata],
        dtype=object,
    )
    subjects = np.asarray([str(meta["subject_id"]) for meta in metadata], dtype=object)
    if not no_cache:
        np.savez_compressed(
            path,
            embeddings=embeddings,
            diag_names=diag_names,
            sites=sites,
            subjects=subjects,
        )
    return embeddings, diag_names, sites, subjects


def make_cohort(
    embeddings: np.ndarray,
    diag_names: np.ndarray,
    sites: np.ndarray,
    subjects: np.ndarray,
    *,
    exclude_sites: set[str] | None = None,
) -> Cohort:
    labels = np.asarray([diagnosis_to_binary(name) for name in diag_names], dtype=object)
    keep = np.asarray([label is not None for label in labels], dtype=bool)
    if exclude_sites:
        keep &= ~np.isin(sites.astype(str), sorted(exclude_sites))
    y = np.asarray([int(label) for label in labels[keep].tolist()], dtype=np.int64)
    return Cohort(
        embeddings=np.asarray(embeddings[keep], dtype=np.float32),
        y=y,
        sites=np.asarray(sites[keep], dtype=object),
        subjects=np.asarray(subjects[keep], dtype=object),
    )


def make_probe(feature_dim: int, train_count: int, seed: int):
    n_components = min(50, feature_dim, max(1, train_count - 1))
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=seed),
        LogisticRegression(max_iter=10000, random_state=seed, class_weight="balanced"),
    )


def domain_classifier_metrics(source: Cohort, target: Cohort, seed: int, folds: int):
    x = np.concatenate([source.embeddings, target.embeddings], axis=0)
    y = np.concatenate(
        [
            np.zeros(len(source.embeddings), dtype=np.int64),
            np.ones(len(target.embeddings), dtype=np.int64),
        ]
    )
    n_splits = min(folds, int(np.bincount(y).min()))
    if n_splits < 2:
        return {"accuracy": float("nan"), "balanced_accuracy": float("nan"), "roc_auc": float("nan")}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros_like(y)
    prob = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in cv.split(x, y):
        clf = make_probe(x.shape[1], len(train_idx), seed=seed)
        clf.fit(x[train_idx], y[train_idx])
        pred[test_idx] = clf.predict(x[test_idx])
        prob[test_idx] = clf.predict_proba(x[test_idx])[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "folds": int(n_splits),
    }


def squared_distances(x: np.ndarray, y: np.ndarray):
    x_norm = np.sum(x * x, axis=1)[:, None]
    y_norm = np.sum(y * y, axis=1)[None, :]
    return np.maximum(x_norm + y_norm - 2.0 * (x @ y.T), 0.0)


def mmd_rbf(source: Cohort, target: Cohort, seed: int, max_samples: int):
    rng = np.random.default_rng(seed)
    x = source.embeddings
    y = target.embeddings
    if len(x) > max_samples:
        x = x[rng.choice(len(x), size=max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), size=max_samples, replace=False)]

    scaler = StandardScaler().fit(source.embeddings)
    x = scaler.transform(x).astype(np.float32)
    y = scaler.transform(y).astype(np.float32)
    combined = np.concatenate([x, y], axis=0)
    sample = combined
    if len(sample) > min(1000, len(sample)):
        sample = sample[rng.choice(len(sample), size=min(1000, len(sample)), replace=False)]
    d_sample = squared_distances(sample, sample)
    median = float(np.median(d_sample[d_sample > 0])) if np.any(d_sample > 0) else 1.0
    gamma = 1.0 / max(2.0 * median, 1e-8)

    k_xx = np.exp(-gamma * squared_distances(x, x))
    k_yy = np.exp(-gamma * squared_distances(y, y))
    k_xy = np.exp(-gamma * squared_distances(x, y))
    return {
        "mmd2": float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()),
        "rbf_gamma": float(gamma),
        "source_n": int(len(x)),
        "target_n": int(len(y)),
    }


def label_site_mismatch(source: Cohort, target: Cohort):
    source_rate = float(source.y.mean())
    target_rate = float(target.y.mean())
    rows = []
    for site in sorted(set(target.sites.tolist())):
        mask = target.sites == site
        rate = float(target.y[mask].mean())
        rows.append(
            {
                "site": str(site),
                "n": int(mask.sum()),
                "positive_rate": rate,
                "abs_delta_from_source": float(abs(rate - source_rate)),
            }
        )
    weighted = sum(row["n"] * row["abs_delta_from_source"] for row in rows) / max(1, len(target.y))
    return {
        "source_positive_rate": source_rate,
        "target_positive_rate": target_rate,
        "mean_abs_site_delta": float(np.mean([row["abs_delta_from_source"] for row in rows])) if rows else float("nan"),
        "weighted_abs_site_delta": float(weighted),
        "target_sites": rows,
    }


def summarize(values: list[float]):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def direction_specs(run_tag: str, checkpoint_root: Path):
    return {
        "srpbs_abide1_to_abide2": {
            "label": "SRPBS+ABIDE-I -> ABIDE-II",
            "source": ["srpbs", "abide1"],
            "target": ["abide2"],
            "exclude_target_sites": OVERLAP_ABIDE2_TARGET,
            "checkpoint": lambda seed: checkpoint_root
            / run_tag
            / f"{run_tag}_transfer_srpbs_abide1_to_abide2_rotary_curr_replay_nce_seed{seed}"
            / "phase2_final.pt",
        },
        "srpbs_abide2_to_abide1": {
            "label": "SRPBS+ABIDE-II -> ABIDE-I",
            "source": ["srpbs", "abide2"],
            "target": ["abide1"],
            "exclude_target_sites": OVERLAP_ABIDE1_TARGET,
            "checkpoint": lambda seed: checkpoint_root
            / run_tag
            / f"{run_tag}_transfer_srpbs_abide2_to_abide1_rotary_curr_replay_nce_seed{seed}"
            / "phase2_final.pt",
        },
    }


def run_direction(name: str, spec: dict, args):
    per_seed = []
    for seed in args.seeds:
        checkpoint = spec["checkpoint"](seed)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for seed {seed}: {checkpoint}")
        source_raw = load_or_extract_embeddings(
            checkpoint,
            spec["source"],
            cache_dir=args.cache_dir,
            device=args.device,
            batch_size=args.batch_size,
            max_scans=args.max_scans,
            no_cache=args.no_cache,
        )
        target_raw = load_or_extract_embeddings(
            checkpoint,
            spec["target"],
            cache_dir=args.cache_dir,
            device=args.device,
            batch_size=args.batch_size,
            max_scans=args.max_scans,
            no_cache=args.no_cache,
        )
        source = make_cohort(*source_raw)
        target = make_cohort(*target_raw, exclude_sites=spec["exclude_target_sites"])
        domain = domain_classifier_metrics(source, target, seed=seed, folds=args.folds)
        mmd = mmd_rbf(source, target, seed=seed, max_samples=args.max_mmd_samples)
        mismatch = label_site_mismatch(source, target)
        source_sites = set(source.sites.astype(str).tolist())
        target_sites = set(target.sites.astype(str).tolist())
        per_seed.append(
            {
                "seed": int(seed),
                "checkpoint": str(checkpoint),
                "source_n": int(len(source.y)),
                "target_n": int(len(target.y)),
                "source_sites": int(len(source_sites)),
                "target_sites": int(len(target_sites)),
                "target_sites_seen_in_source": int(len(source_sites & target_sites)),
                "domain_classifier": domain,
                "mmd": mmd,
                "label_site_mismatch": mismatch,
            }
        )

    return {
        "label": spec["label"],
        "source_datasets": list(spec["source"]),
        "target_datasets": list(spec["target"]),
        "excluded_target_sites": sorted(spec["exclude_target_sites"]),
        "per_seed": per_seed,
        "summary": {
            "domain_balanced_accuracy": summarize(
                [row["domain_classifier"]["balanced_accuracy"] for row in per_seed]
            ),
            "domain_roc_auc": summarize([row["domain_classifier"]["roc_auc"] for row in per_seed]),
            "mmd2": summarize([row["mmd"]["mmd2"] for row in per_seed]),
            "label_site_mismatch": summarize(
                [row["label_site_mismatch"]["weighted_abs_site_delta"] for row in per_seed]
            ),
            "target_sites_seen_in_source": summarize([row["target_sites_seen_in_source"] for row in per_seed]),
        },
    }


def build_markdown(payload: dict):
    lines = [
        "# Transfer Shift Diagnostics",
        "",
        "Domain classifier metrics distinguish source embeddings from target embeddings; higher values indicate stronger source-target separability.",
        "MMD is computed with an RBF kernel after source-fitted standardization. Label-site mismatch is the target-site-size-weighted deviation between target site positive rates and the source positive rate.",
        "",
        "| Direction | Domain bAcc (%) | Domain AUC (%) | MMD$^2$ | Label-site mismatch | Target sites seen in source |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["directions"].values():
        summary = row["summary"]
        lines.append(
            f"| {row['label']} | "
            f"{100.0 * summary['domain_balanced_accuracy']['mean']:.1f} ± {100.0 * summary['domain_balanced_accuracy']['std']:.1f} | "
            f"{100.0 * summary['domain_roc_auc']['mean']:.1f} ± {100.0 * summary['domain_roc_auc']['std']:.1f} | "
            f"{summary['mmd2']['mean']:.4f} ± {summary['mmd2']['std']:.4f} | "
            f"{summary['label_site_mismatch']['mean']:.3f} ± {summary['label_site_mismatch']['std']:.3f} | "
            f"{summary['target_sites_seen_in_source']['mean']:.1f} ± {summary['target_sites_seen_in_source']['std']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Compute lightweight transfer domain-shift diagnostics.")
    parser.add_argument("--run-tag", default="final_run_v1")
    parser.add_argument("--checkpoint-root", default="checkpoints/jepa_fusion")
    parser.add_argument("--seeds", default="0", help="Comma-separated pretraining seeds. Use 0 first for a quick run.")
    parser.add_argument("--directions", default="both", choices=["both", "a_to_b", "b_to_a"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--max-mmd-samples", type=int, default=1000)
    parser.add_argument("--cache-dir", default="results/cache/transfer_shift_diagnostics")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output-json", default="results/reports/tables/jepa_fusion/transfer_shift_diagnostics.json")
    parser.add_argument("--output-md", default="results/reports/tables/jepa_fusion/transfer_shift_diagnostics.md")
    args = parser.parse_args()
    args.seeds = parse_seed_list(args.seeds)
    args.checkpoint_root = Path(args.checkpoint_root)
    args.cache_dir = Path(args.cache_dir)
    return args


def main():
    args = parse_args()
    specs = direction_specs(args.run_tag, args.checkpoint_root)
    if args.directions == "a_to_b":
        selected = {"srpbs_abide1_to_abide2": specs["srpbs_abide1_to_abide2"]}
    elif args.directions == "b_to_a":
        selected = {"srpbs_abide2_to_abide1": specs["srpbs_abide2_to_abide1"]}
    else:
        selected = specs

    payload = {
        "config": {
            "run_tag": args.run_tag,
            "seeds": args.seeds,
            "device": args.device,
            "max_scans": args.max_scans,
            "max_mmd_samples": args.max_mmd_samples,
        },
        "directions": {},
    }
    for name, spec in selected.items():
        print(f"[transfer-shift] {spec['label']} seeds={args.seeds}", flush=True)
        payload["directions"][name] = run_direction(name, spec, args)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    output_md.write_text(build_markdown(payload))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
