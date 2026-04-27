from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid


METHOD_COLUMNS = {
    "Static FC": ("static_fc_tsne_1", "static_fc_tsne_2"),
    "FC + ComBat": ("combat_tsne_1", "combat_tsne_2"),
    "FC + CovBat": ("covbat_tsne_1", "covbat_tsne_2"),
    "MS-JEPA": ("jepa_tsne_1", "jepa_tsne_2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Report descriptive site-clustering metrics from four-site t-SNE CSV.")
    parser.add_argument("--input-csv", default="results/figures/four_site_tsne.csv")
    parser.add_argument("--output-csv", default="results/figures/four_site_tsne_metrics.csv")
    parser.add_argument("--output-md", default="results/figures/four_site_tsne_metrics.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def load_rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def site_prediction_scores(x: np.ndarray, sites: np.ndarray, *, folds: int, seed: int):
    n_splits = min(folds, min(np.unique(sites, return_counts=True)[1]))
    if n_splits < 2:
        return float("nan"), float("nan")

    true = []
    pred_knn = []
    pred_centroid = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(x, sites):
        knn = KNeighborsClassifier(n_neighbors=5).fit(x[train_idx], sites[train_idx])
        centroid = NearestCentroid().fit(x[train_idx], sites[train_idx])
        true.extend(sites[test_idx].tolist())
        pred_knn.extend(knn.predict(x[test_idx]).tolist())
        pred_centroid.extend(centroid.predict(x[test_idx]).tolist())

    true = np.asarray(true, dtype=object)
    return (
        float(np.mean(np.asarray(pred_knn, dtype=object) == true)),
        float(np.mean(np.asarray(pred_centroid, dtype=object) == true)),
    )


def method_metrics(rows: list[dict], method: str, columns: tuple[str, str], *, folds: int, seed: int):
    sites = np.asarray([row["site"] for row in rows], dtype=object)
    x = np.asarray([[float(row[columns[0]]), float(row[columns[1]])] for row in rows], dtype=np.float64)
    site_names = sorted(set(sites.tolist()))

    centroids = []
    radii = []
    for site in site_names:
        points = x[sites == site]
        centroid = points.mean(axis=0)
        centroids.append(centroid)
        radii.extend(np.linalg.norm(points - centroid[None, :], axis=1).tolist())
    centroids = np.asarray(centroids)
    pairwise_centroid_distances = [
        float(np.linalg.norm(centroids[i] - centroids[j]))
        for i, j in combinations(range(len(site_names)), 2)
    ]
    mean_centroid_distance = float(np.mean(pairwise_centroid_distances))
    within_site_radius = float(np.mean(radii))
    separation_ratio = mean_centroid_distance / max(within_site_radius, 1e-12)
    silhouette = float(silhouette_score(x, sites, metric="euclidean"))
    knn_acc, centroid_acc = site_prediction_scores(x, sites, folds=folds, seed=seed)

    return {
        "method": method,
        "n_subjects": len(rows),
        "n_sites": len(site_names),
        "mean_centroid_distance": mean_centroid_distance,
        "within_site_radius": within_site_radius,
        "centroid_separation_ratio": float(separation_ratio),
        "silhouette": silhouette,
        "knn_site_accuracy": knn_acc,
        "nearest_centroid_site_accuracy": centroid_acc,
        "chance_site_accuracy": 1.0 / len(site_names),
    }


def write_csv(path: Path, metrics: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def write_markdown(path: Path, metrics: list[dict]):
    lines = [
        "# Four-Site t-SNE Site-Clustering Metrics",
        "",
        "These are descriptive metrics computed in the t-SNE plane. Lower centroid separation ratio, lower silhouette, and lower site-prediction accuracy indicate weaker visible site clustering. Because t-SNE warps distances, use these as figure-side support rather than primary statistical evidence.",
        "",
        "| Method | Centroid/Radius | Silhouette | 5-NN Site Acc. (%) | Nearest-Centroid Site Acc. (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['method']} | "
            f"{row['centroid_separation_ratio']:.2f} | "
            f"{row['silhouette']:.3f} | "
            f"{100.0 * row['knn_site_accuracy']:.1f} | "
            f"{100.0 * row['nearest_centroid_site_accuracy']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    rows = load_rows(Path(args.input_csv))
    metrics = [
        method_metrics(rows, method, columns, folds=args.folds, seed=args.seed)
        for method, columns in METHOD_COLUMNS.items()
    ]
    write_csv(Path(args.output_csv), metrics)
    write_markdown(Path(args.output_md), metrics)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
