"""Concatenated feature-level fusion evaluation.

Projects FC and JEPA features separately, concatenates, and runs a single
logistic regression on fused features.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from ..jepa_clean.data import WindowedFMRIDataset
from ..jepa_clean.evaluate_subject_vote import (
    build_model_from_checkpoint,
    extract_window_embeddings,
    filter_task,
)
from ..jepa_clean.repro import seed_everything
from .protocol import (
    DATASETS,
    load_fc_features,
    load_fc_records,
    filter_task_records,
    make_subject_id,
)
from .evaluate_jepa_subject_vote import DATASET_TASKS

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


def run_concat_fusion(checkpoint_path: str, device=None, seeds=(0, 1, 2, 3, 4)):
    logger = logging.getLogger("ConcatFusion")

    model, ckpt_cfg = build_model_from_checkpoint(checkpoint_path)
    if device is None:
        device = ckpt_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    window_seconds = ckpt_cfg.get("window_seconds", 90.0)
    target_timepoints = ckpt_cfg.get("target_timepoints", 45)

    all_results = {}
    for ds_name, spec in DATASETS.items():
        task_key = DATASET_TASKS[ds_name]
        logger.info("Concat fusion on %s", spec.display_name)

        # JEPA embeddings
        dataset = WindowedFMRIDataset(
            spec.raw_dir,
            window_seconds=window_seconds,
            stride_seconds=30.0,
            target_timepoints=target_timepoints,
            sampling="center",
            exclude_unknown_diagnoses=True,
        )
        jepa_emb, jepa_meta = extract_window_embeddings(
            model, dataset, device=device, batch_size=128, window_feature="mean",
        )

        # FC features
        fc_records = load_fc_records(spec.fc_dir)
        fc_records_task, fc_labels = filter_task_records(fc_records, spec.hc_diag, spec.disorder_diag)
        fc_features = load_fc_features(fc_records_task)

        # Match subjects
        jepa_subject_to_emb = {}
        jepa_subject_to_label = {}
        for emb, meta in zip(jepa_emb, jepa_meta):
            sid = meta.get("subject_id", f"{meta['dataset']}:{meta['subject']}")
            d = int(meta["diag"])
            if d < 0:
                continue
            if spec.disorder_diag == "ALL":
                lab = int(d != spec.hc_diag)
            elif d in {spec.hc_diag, spec.disorder_diag}:
                lab = int(d == spec.disorder_diag)
            else:
                continue
            jepa_subject_to_emb[sid] = emb
            jepa_subject_to_label[sid] = lab

        fc_subject_to_feat = {}
        fc_subject_to_label = {}
        fc_subject_to_site = {}
        for rec, feat, lab in zip(fc_records_task, fc_features, fc_labels):
            sid = make_subject_id(rec)
            fc_subject_to_feat[sid] = feat
            fc_subject_to_label[sid] = int(lab)
            fc_subject_to_site[sid] = rec.site

        matched = sorted(set(jepa_subject_to_emb) & set(fc_subject_to_feat))
        if not matched:
            logger.warning("No matched subjects for %s", spec.display_name)
            continue

        # Build arrays
        jepa_arr = np.stack([jepa_subject_to_emb[s] for s in matched])
        fc_arr = np.stack([fc_subject_to_feat[s] for s in matched])
        labels = np.array([fc_subject_to_label[s] for s in matched])
        sites = np.array([fc_subject_to_site[s] for s in matched])

        # LOSO with concat features
        unique_sites = np.unique(sites)
        seed_baccs = []
        seed_aucs = []

        for seed in seeds:
            all_preds = []
            all_probs = []
            all_yt = []

            for held_out in unique_sites:
                train_mask = sites != held_out
                test_mask = sites == held_out
                if len(np.unique(labels[train_mask])) < 2 or len(np.unique(labels[test_mask])) < 2:
                    continue

                # Project each branch separately, then concatenate
                n_jepa = min(50, jepa_arr.shape[1], train_mask.sum() - 1)
                n_fc = min(50, fc_arr.shape[1], train_mask.sum() - 1)

                pca_jepa = PCA(n_components=max(1, n_jepa), random_state=seed)
                pca_fc = PCA(n_components=max(1, n_fc), random_state=seed)

                j_train = pca_jepa.fit_transform(jepa_arr[train_mask])
                j_test = pca_jepa.transform(jepa_arr[test_mask])
                f_train = pca_fc.fit_transform(fc_arr[train_mask])
                f_test = pca_fc.transform(fc_arr[test_mask])

                X_train = np.concatenate([
                    StandardScaler().fit_transform(j_train),
                    StandardScaler().fit_transform(f_train),
                ], axis=1)
                ss_j = StandardScaler().fit(j_train)
                ss_f = StandardScaler().fit(f_train)
                X_test = np.concatenate([ss_j.transform(j_test), ss_f.transform(f_test)], axis=1)

                clf = LogisticRegression(max_iter=10000, class_weight="balanced", random_state=seed)
                clf.fit(X_train, labels[train_mask])
                probs = clf.predict_proba(X_test)[:, 1]
                preds = (probs >= 0.5).astype(int)

                all_preds.extend(preds.tolist())
                all_probs.extend(probs.tolist())
                all_yt.extend(labels[test_mask].tolist())

            if all_preds:
                seed_baccs.append(float(balanced_accuracy_score(all_yt, all_preds)))
                try:
                    seed_aucs.append(float(roc_auc_score(all_yt, all_probs)))
                except ValueError:
                    seed_aucs.append(float("nan"))

        all_results[ds_name] = {
            "display_name": spec.display_name,
            "bacc_mean": float(np.nanmean(seed_baccs)),
            "bacc_std": float(np.nanstd(seed_baccs)),
            "auc_mean": float(np.nanmean(seed_aucs)),
            "auc_std": float(np.nanstd(seed_aucs)),
        }
        logger.info(
            "%s | Concat bAcc=%.1f ± %.1f | AUC=%.3f ± %.3f",
            spec.display_name,
            100 * np.nanmean(seed_baccs), 100 * np.nanstd(seed_baccs),
            np.nanmean(seed_aucs), np.nanstd(seed_aucs),
        )

    return all_results


def parse_args():
    p = argparse.ArgumentParser(description="Concat-projected fusion evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--save-json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    seed_everything(42, deterministic=True)
    results = run_concat_fusion(args.checkpoint, device=args.device, seeds=tuple(args.seeds))
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
