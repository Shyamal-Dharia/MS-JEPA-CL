"""Select pilot candidates by applying the site-ceiling rule.

Ranks pilot JEPA models by their expected complementarity to harmonized FC,
filtering out models whose site probe exceeds a ceiling.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ..jepa_clean.repro import seed_everything


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


def select_pilots(
    eval_json_paths: list[str],
    site_ceiling: float = 0.15,
):
    """Rank pilot candidates.

    For each evaluation JSON, extract per-dataset site bAcc and dx bAcc.
    Filter out models whose average site bAcc exceeds the ceiling.
    Rank remaining by mean dx bAcc.
    """
    logger = logging.getLogger("SelectPilot")
    candidates = []

    for path in eval_json_paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            continue

        name = Path(path).stem
        site_baccs = []
        dx_baccs = []

        for ds_name, ds_data in data.items():
            if isinstance(ds_data, dict):
                if "site_bacc" in ds_data:
                    site_baccs.append(ds_data["site_bacc"][0] if isinstance(ds_data["site_bacc"], list) else ds_data["site_bacc"])
                if "dx_bacc" in ds_data:
                    dx_baccs.append(ds_data["dx_bacc"][0] if isinstance(ds_data["dx_bacc"], list) else ds_data["dx_bacc"])

        if not site_baccs or not dx_baccs:
            logger.info("Skipping %s: incomplete metrics", name)
            continue

        mean_site = float(np.mean(site_baccs))
        mean_dx = float(np.mean(dx_baccs))

        candidates.append({
            "name": name,
            "path": path,
            "mean_site_bacc": mean_site,
            "mean_dx_bacc": mean_dx,
            "site_safe": mean_site <= site_ceiling,
        })

    # Sort: site-safe first, then by descending dx_bacc
    candidates.sort(key=lambda c: (not c["site_safe"], -c["mean_dx_bacc"]))

    for rank, c in enumerate(candidates, 1):
        marker = "✓" if c["site_safe"] else "✗"
        logger.info(
            "Rank %d | %s | site_bacc=%.4f dx_bacc=%.4f | %s",
            rank, c["name"], c["mean_site_bacc"], c["mean_dx_bacc"], marker,
        )

    return candidates


def parse_args():
    p = argparse.ArgumentParser(description="Select pilot candidates from evaluation JSONs")
    p.add_argument("--eval-jsons", nargs="+", required=True)
    p.add_argument("--site-ceiling", type=float, default=0.15)
    p.add_argument("--save-json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    seed_everything(42, deterministic=True)
    candidates = select_pilots(args.eval_jsons, site_ceiling=args.site_ceiling)

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(candidates, f, indent=2)
        logging.getLogger("SelectPilot").info("Saved: %s", args.save_json)


if __name__ == "__main__":
    main()
