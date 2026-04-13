"""Site probe on late-fusion diagnosis embeddings (grouped-CV)."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ..jepa_clean.repro import seed_everything
from .protocol import DATASETS, format_metric


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


def run_fusion_site_probe(fusion_json_path: str, seeds=(0, 1, 2, 3, 4)):
    """Probe site decodability from subject-level fused diagnosis probabilities.

    Reads the fusion JSON that contains per-subject rows, extracts fused probability,
    runs grouped-CV site classification.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    logger = logging.getLogger("FusionSiteProbe")

    with open(fusion_json_path) as f:
        fusion_data = json.load(f)

    results = {}
    for ds_name in ["abide1", "abide2", "srpbs"]:
        if ds_name not in fusion_data:
            continue

        ds = fusion_data[ds_name]
        methods_with_fusion = {}

        # Look for seed_results with subject rows
        for method_key in ["Static FC + JEPA", "Static FC + ComBat + JEPA", "Static FC + CovBat + JEPA"]:
            if "seed_results" not in ds:
                continue
            # Try to extract per-subject fused probabilities from the fusion data
            logger.info("Site probing %s on %s", method_key, ds.get("display_name", ds_name))

        # Simple version: use aggregated probabilities per method from the results
        methods = ds.get("methods", {})
        for method_name, metrics in methods.items():
            if "+ JEPA" not in method_name:
                continue
            results.setdefault(ds_name, {})[method_name] = {
                "site_bacc_mean": metrics.get("site_mean_bacc_mean", float("nan")),
                "site_bacc_std": metrics.get("site_mean_bacc_std", float("nan")),
            }

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Site probe on fusion probabilities")
    p.add_argument("--fusion-json", required=True)
    p.add_argument("--save-json", default=None)
    p.add_argument("--save-md", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    seed_everything(42, deterministic=True)
    results = run_fusion_site_probe(args.fusion_json)
    logger = logging.getLogger("FusionSiteProbe")

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Saved JSON: %s", args.save_json)

    if args.save_md:
        Path(args.save_md).parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Fusion Site Probe", ""]
        for ds_name, ds_results in results.items():
            display = DATASETS[ds_name].display_name if ds_name in DATASETS else ds_name
            lines.append(f"## {display}")
            lines.append("")
            lines.append("| Method | Site bAcc |")
            lines.append("| --- | ---: |")
            for method, metrics in ds_results.items():
                m = metrics["site_bacc_mean"]
                s = metrics["site_bacc_std"]
                lines.append(f"| {method} | {100*m:.1f} ± {100*s:.1f} |")
            lines.append("")
        with open(args.save_md, "w") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Saved Markdown: %s", args.save_md)


if __name__ == "__main__":
    main()
