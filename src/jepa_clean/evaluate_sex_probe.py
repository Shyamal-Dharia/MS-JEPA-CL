import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .data import WindowedFMRIDataset
from .evaluate_subject_vote import (
    build_model_from_checkpoint,
    compute_binary_metrics,
    extract_window_embeddings,
)
from .repro import seed_everything


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


def aggregate_subject_embeddings(embeddings, metadata):
    grouped = {}
    for emb, meta in zip(embeddings, metadata):
        sex = str(meta.get("sex", "") or "").strip().upper()
        if sex not in {"M", "F"}:
            continue

        subject_id = meta["subject_id"]
        entry = grouped.setdefault(
            subject_id,
            {
                "embeddings": [],
                "site": meta["site"],
                "dataset": meta["dataset"],
                "sex": sex,
                "n_windows": 0,
            },
        )
        if entry["sex"] != sex:
            raise ValueError(f"Subject {subject_id} has inconsistent sex labels")
        if entry["site"] != meta["site"]:
            raise ValueError(f"Subject {subject_id} appears at multiple sites")

        entry["embeddings"].append(emb)
        entry["n_windows"] += 1

    rows = []
    for subject_id, entry in grouped.items():
        mean_embedding = np.mean(np.stack(entry["embeddings"], axis=0), axis=0)
        rows.append(
            {
                "subject_id": subject_id,
                "site": entry["site"],
                "dataset": entry["dataset"],
                "sex": entry["sex"],
                "y": 1 if entry["sex"] == "F" else 0,
                "embedding": mean_embedding,
                "n_windows": entry["n_windows"],
            }
        )

    rows.sort(key=lambda row: (row["site"], row["subject_id"]))
    return rows


def build_probe_classifier(classifier: str, seed: int):
    if classifier == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                max_iter=2000,
                random_state=seed,
                class_weight="balanced",
            ),
        )
    if classifier == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=0.1,
                max_iter=5000,
                random_state=seed,
                class_weight="balanced",
            ),
        )
    raise ValueError(f"Unsupported classifier: {classifier}")


def get_probe_scores_and_preds(clf, x_test, classifier: str):
    if classifier == "logreg":
        scores = clf.predict_proba(x_test)[:, 1]
        preds = (scores >= 0.5).astype(int)
        return scores, preds

    if classifier == "linear_svm":
        scores = clf.decision_function(x_test)
        preds = clf.predict(x_test).astype(int)
        return scores, preds

    raise ValueError(f"Unsupported classifier: {classifier}")


def run_subject_loso(rows, seed: int, classifier: str = "logreg"):
    if not rows:
        raise ValueError("No labelled subjects with sex information were found")

    sites = sorted({row["site"] for row in rows})
    aggregated = []
    valid_sites = []

    for held_out in sites:
        train_rows = [row for row in rows if row["site"] != held_out]
        test_rows = [row for row in rows if row["site"] == held_out]
        if not test_rows:
            continue

        y_train = np.array([row["y"] for row in train_rows], dtype=int)
        y_test = np.array([row["y"] for row in test_rows], dtype=int)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        x_train = np.stack([row["embedding"] for row in train_rows], axis=0)
        x_test = np.stack([row["embedding"] for row in test_rows], axis=0)

        clf = build_probe_classifier(classifier, seed)
        clf.fit(x_train, y_train)
        probs, preds = get_probe_scores_and_preds(clf, x_test, classifier)

        for row, prob, pred in zip(test_rows, probs, preds):
            aggregated.append(
                {
                    "subject_id": row["subject_id"],
                    "site": row["site"],
                    "dataset": row["dataset"],
                    "sex": row["sex"],
                    "y": row["y"],
                    "pred": int(pred),
                    "prob": float(prob),
                    "n_windows": row["n_windows"],
                }
            )
        valid_sites.append(held_out)

    if not aggregated:
        return {
            "bacc": float("nan"),
            "auc": float("nan"),
            "f1_macro": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "n_valid_sites": 0,
            "n_subjects": 0,
            "mean_windows_per_subject": float("nan"),
        }, []

    y_true = np.array([row["y"] for row in aggregated], dtype=int)
    y_pred = np.array([row["pred"] for row in aggregated], dtype=int)
    y_prob = np.array([row["prob"] for row in aggregated], dtype=float)
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    metrics["task"] = "Sex (F vs M)"
    metrics["task_key"] = "sex_f_vs_m"
    metrics["n_valid_sites"] = len(valid_sites)
    metrics["n_subjects"] = len(aggregated)
    metrics["mean_windows_per_subject"] = float(np.mean([row["n_windows"] for row in aggregated]))
    return metrics, aggregated


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen subject-level linear probe for sex classification")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=[
            "preprocessed_data_fir/full/roi_timeseries_1600_full",
            "preprocessed_data_fir/full/roi_timeseries_abide_full",
        ],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=30.0)
    parser.add_argument("--target-timepoints", type=int, default=None)
    parser.add_argument("--sampling", choices=["center", "sliding"], default="center")
    parser.add_argument("--classifier", choices=["logreg", "linear_svm"], default="logreg")
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("JEPA-Clean-SexProbe")
    seed_everything(args.seed, deterministic=True)

    model, ckpt_cfg = build_model_from_checkpoint(args.checkpoint, override_target_timepoints=args.target_timepoints)
    device = args.device or ckpt_cfg.get("device") or "cpu"
    if device == "cuda":
        logger.warning("CUDA requested but not available in this environment; using cpu")
        device = "cpu"
    model = model.to(device)

    window_seconds = args.window_seconds or ckpt_cfg.get("window_seconds", 90.0)
    target_timepoints = args.target_timepoints or ckpt_cfg.get("target_timepoints", 45)

    dataset = WindowedFMRIDataset(
        args.data_dirs,
        window_seconds=window_seconds,
        stride_seconds=args.window_stride_seconds,
        target_timepoints=target_timepoints,
        sampling=args.sampling,
    )
    logger.info(
        "Sex probe dataset: %d windows from %d scans | sampling=%s window=%.1fs target_tp=%d",
        len(dataset),
        len(dataset.records),
        args.sampling,
        window_seconds,
        target_timepoints,
    )

    embeddings, metadata = extract_window_embeddings(model, dataset, device=device, batch_size=args.batch_size)
    subject_rows = aggregate_subject_embeddings(embeddings, metadata)
    sex_counts = Counter(row["sex"] for row in subject_rows)
    logger.info(
        "Aggregated subject embeddings: %d subjects | sex_counts=%s",
        len(subject_rows),
        dict(sorted(sex_counts.items())),
    )

    metrics, predictions = run_subject_loso(subject_rows, seed=args.seed, classifier=args.classifier)
    metrics["classifier"] = args.classifier
    logger.info(
        "%s | bAcc=%.1f%% AUC=%.3f F1=%.3f precision=%.3f recall=%.3f | valid_sites=%d subjects=%d mean_windows=%.2f",
        metrics["task"],
        100.0 * metrics["bacc"],
        metrics["auc"],
        metrics["f1_macro"],
        metrics["precision"],
        metrics["recall"],
        metrics["n_valid_sites"],
        metrics["n_subjects"],
        metrics["mean_windows_per_subject"],
    )

    if args.save_json is not None:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"summary": metrics, "subjects": predictions}, f, indent=2)
        logger.info("Saved results: %s", save_path)


if __name__ == "__main__":
    main()
