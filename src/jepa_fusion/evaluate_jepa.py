"""Grouped-CV JEPA embedding evaluation with site and diagnosis probes."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..jepa_clean.data import WindowedFMRIDataset
from ..jepa_clean.evaluate_subject_vote import build_model_from_checkpoint
from ..jepa_clean.pooling import pool_sequence_features
from ..jepa_clean.repro import seed_everything
from .protocol import (
    DATASETS,
    evaluate_probe_cv,
    filter_mixed_sites,
    filter_task_records,
    format_metric,
    make_subject_id,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


def collate_fn(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), list(labels)


@torch.no_grad()
def extract_jepa_embeddings(model, dataset, device, batch_size=64):
    """Extract mean-pooled embeddings for all windows in dataset."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=0, collate_fn=collate_fn,
    )
    embeddings = []
    metadata = []
    for x, labels in loader:
        x = x.to(device)
        raw = model.module if isinstance(model, torch.nn.DataParallel) else model
        seq = raw.encode_sequence(x)
        pooled = pool_sequence_features(seq, mode="mean")
        embeddings.append(pooled.cpu().numpy())
        metadata.extend(labels)
    return np.concatenate(embeddings, axis=0), metadata


class _EmbeddingRecord:
    """Adapter so that JEPA embeddings can use protocol.evaluate_probe_cv."""
    def __init__(self, site, dataset, subject):
        self.site = site
        self.dataset = dataset
        self.subject = subject


def run_evaluation(checkpoint_path: str, seeds=(0, 1, 2, 3, 4), device=None):
    logger = logging.getLogger("JEPA-Eval")

    model, ckpt_cfg = build_model_from_checkpoint(checkpoint_path)
    if device is None:
        device = ckpt_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    window_seconds = ckpt_cfg.get("window_seconds", 90.0)
    target_timepoints = ckpt_cfg.get("target_timepoints", 45)

    results = {}
    for ds_name, spec in DATASETS.items():
        logger.info("Evaluating on %s", spec.display_name)

        dataset = WindowedFMRIDataset(
            spec.raw_dir,
            window_seconds=window_seconds,
            stride_seconds=30.0,
            target_timepoints=target_timepoints,
            sampling="center",
            exclude_unknown_diagnoses=True,
        )

        embeddings, metadata = extract_jepa_embeddings(model, dataset, device)
        logger.info("Extracted %d embeddings from %s", len(embeddings), spec.display_name)

        # Build adapter records
        records = [_EmbeddingRecord(
            site=meta["site"],
            dataset=meta["dataset"],
            subject=meta["subject"],
        ) for meta in metadata]

        # Task filter
        diag_array = np.array([int(meta["diag"]) for meta in metadata], dtype=int)
        if spec.disorder_diag == "ALL":
            mask = diag_array >= 0
            labels = (diag_array != spec.hc_diag).astype(int)
        else:
            mask = np.isin(diag_array, [spec.hc_diag, spec.disorder_diag])
            labels = (diag_array == spec.disorder_diag).astype(int)

        records = [r for r, m in zip(records, mask) if m]
        embeddings = embeddings[mask]
        labels = labels[mask]

        # Mixed-site filter
        records, labels = filter_mixed_sites(records, labels)
        keep_subjects = {make_subject_id(rec) for rec in records}
        keep_mask = np.array([make_subject_id(r) in keep_subjects for r in records])

        result = evaluate_probe_cv(
            embeddings[:len(records)], labels, records,
            n_folds=5, seeds=seeds,
        )

        results[ds_name] = {
            "display_name": spec.display_name,
            "n_samples": len(records),
            "n_sites": len(set(r.site for r in records)),
            "dx_bacc": list(result["dx_bacc"]),
            "dx_auc": list(result["dx_auc"]),
            "site_bacc": list(result["site_bacc"]),
        }

        logger.info(
            "%s | Site bAcc=%s | Dx bAcc=%s | Dx AUC=%s",
            spec.display_name,
            format_metric(result["site_bacc"]),
            format_metric(result["dx_bacc"]),
            format_metric(result["dx_auc"]),
        )

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Grouped-CV JEPA evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--save-json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    seed_everything(42, deterministic=True)
    results = run_evaluation(args.checkpoint, seeds=tuple(args.seeds), device=args.device)

    if args.save_json:
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logging.getLogger("JEPA-Eval").info("Saved: %s", path)


if __name__ == "__main__":
    main()
