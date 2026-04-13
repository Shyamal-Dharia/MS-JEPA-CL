"""Site probe on fused probability vectors (grouped-CV).

Instead of probing the fused embeddings, this script probes the single fused
diagnosis probability value to test whether a scalar probability can decode
the site.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..jepa_clean.repro import seed_everything
from .protocol import DATASETS


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


def probe_fused_probabilities(fusion_json_path: str, seeds=(0, 1, 2, 3, 4)):
    """Grouped-CV site classification on subject-level fused probabilities."""
    logger = logging.getLogger("FusedProbSiteProbe")

    with open(fusion_json_path) as f:
        data = json.load(f)

    results = {}
    for ds_name in ["abide1", "abide2", "srpbs"]:
        if ds_name not in data:
            continue
        ds = data[ds_name]
        methods = ds.get("methods", {})
        ds_results = {}

        for method_name in ["Static FC + JEPA", "Static FC + ComBat + JEPA", "Static FC + CovBat + JEPA"]:
            if method_name not in methods:
                continue
            method_data = methods[method_name]
            seed_details = method_data.get("seed_details", [])

            if not seed_details:
                logger.info("No detailed subject rows for %s on %s, using aggregated metrics", method_name, ds_name)
                ds_results[method_name] = {
                    "site_bacc_mean": method_data.get("site_mean_bacc_mean", float("nan")),
                    "site_bacc_std": method_data.get("site_mean_bacc_std", float("nan")),
                }
                continue

            # If we have per-subject details, probe them
            all_probs = []
            all_sites = []
            all_subjects = []
            for row in seed_details:
                all_probs.append(row["prob"])
                all_sites.append(row["site"])
                all_subjects.append(row["subject_id"])

            if len(set(all_sites)) < 2:
                continue

            X = np.array(all_probs).reshape(-1, 1)
            site_enc = LabelEncoder()
            y = site_enc.fit_transform(all_sites)
            subjects = np.array(all_subjects)

            seed_baccs = []
            for seed in seeds:
                n_groups = len(np.unique(subjects))
                n_folds = min(5, n_groups)
                if n_folds < 2:
                    continue
                cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                fold_baccs = []
                for train_idx, test_idx in cv.split(X, y, groups=subjects):
                    if len(np.unique(y[train_idx])) < 2:
                        continue
                    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
                    clf.fit(X[train_idx], y[train_idx])
                    pred = clf.predict(X[test_idx])
                    fold_baccs.append(balanced_accuracy_score(y[test_idx], pred))
                if fold_baccs:
                    seed_baccs.append(float(np.mean(fold_baccs)))

            ds_results[method_name] = {
                "site_bacc_mean": float(np.mean(seed_baccs)) if seed_baccs else float("nan"),
                "site_bacc_std": float(np.std(seed_baccs)) if seed_baccs else float("nan"),
            }

        results[ds_name] = ds_results

    return results


def build_markdown(results: dict) -> str:
    lines = ["# Fused Probability Site Probe", ""]
    lines.append("| Dataset | FC + JEPA | FC + ComBat + JEPA | FC + CovBat + JEPA |")
    lines.append("| --- | ---: | ---: | ---: |")
    for ds_name in ["abide1", "abide2", "srpbs"]:
        if ds_name not in results:
            continue
        r = results[ds_name]
        display = DATASETS[ds_name].display_name if ds_name in DATASETS else ds_name
        cells = [f"{display} site bAcc"]
        for method in ["Static FC + JEPA", "Static FC + ComBat + JEPA", "Static FC + CovBat + JEPA"]:
            if method in r:
                m = r[method]
                cells.append(f"{100*m['site_bacc_mean']:.1f} ± {100*m['site_bacc_std']:.1f}")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(description="Site probe on fused probabilities")
    p.add_argument("--fusion-json", required=True)
    p.add_argument("--save-json", default=None)
    p.add_argument("--save-md", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    seed_everything(42, deterministic=True)
    results = probe_fused_probabilities(args.fusion_json)
    logger = logging.getLogger("FusedProbSiteProbe")

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Saved JSON: %s", args.save_json)

    if args.save_md:
        Path(args.save_md).parent.mkdir(parents=True, exist_ok=True)
        md = build_markdown(results)
        with open(args.save_md, "w") as f:
            f.write(md)
        logger.info("Saved Markdown: %s", args.save_md)


if __name__ == "__main__":
    main()
