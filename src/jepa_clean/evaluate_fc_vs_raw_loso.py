import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .plot_tsne_fc_vs_raw import (
    DEFAULT_FC_DIR,
    DEFAULT_RAW_DIR,
    build_fc_features,
    build_raw_features,
    get_roi_permutation,
    load_matched_records,
)


DEFAULT_OUTPUT_JSON = "results/reports/tables/abide_fc_vs_raw_loso_diagnosis_seed0.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run matched ABIDE leave-one-site-out diagnosis probes for raw ROI time series and static FC."
    )
    parser.add_argument("--fc-dir", default=DEFAULT_FC_DIR)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--triangle", choices=["upper", "lower"], default="upper")
    parser.add_argument(
        "--reorder",
        choices=["none", "hemi", "hemi_network"],
        default="hemi_network",
    )
    parser.add_argument("--raw-target-timepoints", type=int, default=45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def run_loso_probe(features: np.ndarray, labels: list[str], sites: list[str], seed: int):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    groups = np.asarray(sites)

    y_true_all = []
    y_pred_all = []
    fold_rows = []
    skipped_sites = []

    for held_out_site in sorted(np.unique(groups)):
        test_idx = np.flatnonzero(groups == held_out_site)
        train_idx = np.flatnonzero(groups != held_out_site)
        y_train = y[train_idx]
        y_test = y[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            skipped_sites.append(
                {
                    "site": str(held_out_site),
                    "train_classes": [str(v) for v in encoder.inverse_transform(np.unique(y_train)).tolist()],
                    "test_classes": [str(v) for v in encoder.inverse_transform(np.unique(y_test)).tolist()],
                    "test_size": int(len(test_idx)),
                    "reason": "train_or_test_missing_a_class",
                }
            )
            continue

        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(50, features.shape[1], len(train_idx) - 1), random_state=seed),
            LogisticRegression(max_iter=10000),
        )
        clf.fit(features[train_idx], y_train)
        y_pred = clf.predict(features[test_idx])

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        fold_rows.append(
            {
                "site": str(held_out_site),
                "n_test": int(len(test_idx)),
                "class_counts": {
                    str(label): int(np.sum(y_test == cls_idx))
                    for cls_idx, label in enumerate(encoder.classes_)
                    if np.sum(y_test == cls_idx) > 0
                },
                "accuracy": float(np.mean(y_pred == y_test)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            }
        )

    if not y_true_all:
        raise RuntimeError("No valid LOSO folds remained after filtering.")

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(encoder.classes_)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums > 0)

    return {
        "seed": int(seed),
        "n_classes": int(len(encoder.classes_)),
        "classes": [str(value) for value in encoder.classes_.tolist()],
        "n_samples_total": int(len(y)),
        "n_samples_valid": int(len(y_true)),
        "n_valid_sites": int(len(fold_rows)),
        "n_skipped_sites": int(len(skipped_sites)),
        "valid_sites": [row["site"] for row in fold_rows],
        "skipped_sites": skipped_sites,
        "pooled_accuracy": float(np.mean(y_pred == y_true)),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mean_fold_accuracy": float(np.mean([row["accuracy"] for row in fold_rows])),
        "mean_fold_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in fold_rows])),
        "folds": fold_rows,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "cv": "leave_one_site_out",
    }


def main():
    args = parse_args()

    records = load_matched_records(Path(args.fc_dir), Path(args.raw_dir))
    if not records:
        raise FileNotFoundError("No matched FC/raw ABIDE records found.")

    permutation, _ = get_roi_permutation(args.reorder)
    fc_features = build_fc_features(records, permutation=permutation, triangle=args.triangle)
    raw_features = build_raw_features(
        records,
        permutation=permutation,
        raw_target_timepoints=args.raw_target_timepoints,
    )
    diagnoses = [record.diagnosis for record in records]
    sites = [record.site for record in records]

    raw_result = run_loso_probe(raw_features, diagnoses, sites, seed=args.seed)
    fc_result = run_loso_probe(fc_features, diagnoses, sites, seed=args.seed)

    payload = {
        "config": {
            "fc_dir": args.fc_dir,
            "raw_dir": args.raw_dir,
            "triangle": args.triangle,
            "reorder": args.reorder,
            "raw_target_timepoints": args.raw_target_timepoints,
            "seed": args.seed,
            "cv": "leave_one_site_out",
            "task": "diagnosis",
        },
        "raw": raw_result,
        "fc": fc_result,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))

    print(f"Saved JSON: {output_json}")
    print(
        "LOSO pooled bAcc | RAW vs FC | "
        f"{100.0 * raw_result['pooled_balanced_accuracy']:.1f}% vs "
        f"{100.0 * fc_result['pooled_balanced_accuracy']:.1f}%"
    )
    print(
        "LOSO mean-fold bAcc | RAW vs FC | "
        f"{100.0 * raw_result['mean_fold_balanced_accuracy']:.1f}% vs "
        f"{100.0 * fc_result['mean_fold_balanced_accuracy']:.1f}%"
    )
    print(
        "Valid LOSO sites | RAW vs FC | "
        f"{raw_result['n_valid_sites']} vs {fc_result['n_valid_sites']}"
    )


if __name__ == "__main__":
    main()
