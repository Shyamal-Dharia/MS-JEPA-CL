import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
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


DEFAULT_OUTPUT = "results/figures/abide_fc_vs_raw_site_confusion_seed0.png"
DEFAULT_JSON = "results/figures/abide_fc_vs_raw_site_confusion_seed0.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot matched ABIDE FC-vs-raw confusion matrices using the linear-probe pipeline."
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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", choices=["site", "diagnosis"], default="site")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--output-json", default=DEFAULT_JSON)
    return parser.parse_args()


def run_probe_with_predictions(features: np.ndarray, labels: list[str], seed: int, folds: int):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    min_class_count = min(np.bincount(y))
    effective_folds = min(folds, int(min_class_count))
    if effective_folds < 2:
        raise ValueError("Need at least 2 folds for the requested labels.")

    cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    preds = np.empty_like(y)
    for train_idx, test_idx in cv.split(features, y):
        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(50, features.shape[1], len(train_idx) - 1), random_state=seed),
            LogisticRegression(max_iter=10000),
        )
        clf.fit(features[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(features[test_idx])

    cm = confusion_matrix(y, preds, labels=np.arange(len(encoder.classes_)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums > 0)

    return {
        "accuracy": float(np.mean(preds == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "classes": [str(item) for item in encoder.classes_.tolist()],
        "n_classes": int(len(encoder.classes_)),
        "n_samples": int(len(y)),
        "folds": int(effective_folds),
        "seed": int(seed),
    }


def plot_confusion(ax, cm_norm: np.ndarray, class_names: list[str], title: str):
    image = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=max(0.25, float(cm_norm.max())))
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    fontsize = 6 if len(class_names) > 10 else 9
    ax.set_xticklabels(class_names, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(class_names, fontsize=fontsize)

    if len(class_names) <= 6:
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{100.0 * cm_norm[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if cm_norm[i, j] < 0.5 else "white",
                )
    return image


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
    labels = [record.site if args.task == "site" else record.diagnosis for record in records]

    fc_result = run_probe_with_predictions(fc_features, labels, seed=args.seed, folds=args.folds)
    raw_result = run_probe_with_predictions(raw_features, labels, seed=args.seed, folds=args.folds)

    class_names = fc_result["classes"]
    fc_cm_norm = np.asarray(fc_result["confusion_matrix_normalized"], dtype=np.float64)
    raw_cm_norm = np.asarray(raw_result["confusion_matrix_normalized"], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    im0 = plot_confusion(
        axes[0],
        raw_cm_norm,
        class_names,
        title=f"ROI Time Series {args.task.title()} Confusion Matrix\n(seed={args.seed}, row-normalized)",
    )
    plot_confusion(
        axes[1],
        fc_cm_norm,
        class_names,
        title=f"Static FC {args.task.title()} Confusion Matrix\n(seed={args.seed}, row-normalized)",
    )
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Recall within true class")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "config": {
            "fc_dir": args.fc_dir,
            "raw_dir": args.raw_dir,
            "triangle": args.triangle,
            "reorder": args.reorder,
            "raw_target_timepoints": args.raw_target_timepoints,
            "folds": args.folds,
            "seed": args.seed,
            "task": args.task,
        },
        "raw": raw_result,
        "fc": fc_result,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))

    print(f"Saved figure: {output}")
    print(f"Saved JSON: {output_json}")
    print(
        f"RAW {args.task} bAcc={100.0 * raw_result['balanced_accuracy']:.1f}% | "
        f"FC {args.task} bAcc={100.0 * fc_result['balanced_accuracy']:.1f}%"
    )


if __name__ == "__main__":
    main()
