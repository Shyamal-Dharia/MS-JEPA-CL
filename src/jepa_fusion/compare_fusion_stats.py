"""Paired Wilcoxon signed-rank tests on fusion results.

Compares pairwise fusion methods using per-site metrics aggregated
over downstream probe seeds, with Holm-Bonferroni correction.
"""

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


def holm_correction(p_values):
    """Holm-Bonferroni step-down correction."""
    n = len(p_values)
    if n == 0:
        return []

    indexed = list(enumerate(p_values))
    indexed.sort(key=lambda x: x[1])

    corrected = [None] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected_p = min(1.0, p * (n - rank))
        corrected[orig_idx] = corrected_p

    # Enforce monotonicity
    indexed.sort(key=lambda x: x[1])
    prev = 0.0
    for rank, (orig_idx, _) in enumerate(indexed):
        corrected[orig_idx] = max(corrected[orig_idx], prev)
        prev = corrected[orig_idx]

    return corrected


def load_fusion_results(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def extract_per_site_metrics(fusion_json: dict, ds_name: str, method_name: str) -> dict:
    """Extract per-site aggregated metrics for a given method.

    Returns dict of {site: {metric_name: value}}.
    Falls back to subject-level aggregation if per-site data is available.
    """
    ds = fusion_json.get(ds_name, {})
    methods = ds.get("methods", {})

    if method_name not in methods:
        return {}

    method_data = methods[method_name]
    site_data = method_data.get("per_site", {})

    if site_data:
        return site_data

    # If no per-site data available, return aggregate metrics as a single entry
    return {"aggregate": {
        "bacc": method_data.get("bacc_mean", float("nan")),
        "auc": method_data.get("auc_mean", float("nan")),
    }}


def compare_methods(
    fusion_jsons: list[str],
    methods_to_compare: list[str] = None,
    metric: str = "bacc",
    alpha: float = 0.05,
):
    """Run pairwise Wilcoxon tests across fusion JSONs for each dataset."""
    logger = logging.getLogger("FusionStats")

    if methods_to_compare is None:
        methods_to_compare = [
            "Static FC + JEPA",
            "Static FC + ComBat + JEPA",
            "Static FC + CovBat + JEPA",
        ]

    # Load all results
    all_results = []
    for path in fusion_jsons:
        all_results.append(load_fusion_results(path))

    # Merge by dataset × method → list of per-site values across seeds
    comparison_results = {}

    for ds_name in ["abide1", "abide2", "srpbs"]:
        method_values = {m: [] for m in methods_to_compare}

        for fusion_json in all_results:
            for method in methods_to_compare:
                ds = fusion_json.get(ds_name, {})
                methods = ds.get("methods", {})
                if method in methods:
                    val = methods[method].get(f"{metric}_mean", float("nan"))
                    method_values[method].append(val)

        # Pairwise tests
        pairs = list(itertools.combinations(methods_to_compare, 2))
        raw_pvals = []
        test_pairs = []

        for m1, m2 in pairs:
            v1 = np.array(method_values[m1], dtype=float)
            v2 = np.array(method_values[m2], dtype=float)

            # Both must have at least 5 matched values
            valid = ~(np.isnan(v1) | np.isnan(v2))
            v1_valid = v1[valid]
            v2_valid = v2[valid]

            if len(v1_valid) < 5:
                logger.info(
                    "%s: %s vs %s — insufficient data (%d values), skipping",
                    ds_name, m1, m2, len(v1_valid),
                )
                continue

            try:
                stat, pval = stats.wilcoxon(v1_valid, v2_valid, alternative="two-sided")
            except ValueError:
                pval = 1.0

            raw_pvals.append(float(pval))
            test_pairs.append((m1, m2))
            logger.info(
                "%s: %s vs %s | %s diff=%.4f p=%.4f",
                ds_name, m1, m2, metric,
                float(np.mean(v1_valid) - np.mean(v2_valid)),
                pval,
            )

        # Holm correction
        corrected = holm_correction(raw_pvals)
        ds_comparison = []
        for i, (m1, m2) in enumerate(test_pairs):
            ds_comparison.append({
                "method_1": m1,
                "method_2": m2,
                "raw_p": raw_pvals[i],
                "corrected_p": corrected[i],
                "significant": corrected[i] < alpha,
                "mean_diff": float(np.mean(method_values[m1]) - np.mean(method_values[m2])),
            })

        comparison_results[ds_name] = {
            "metric": metric,
            "alpha": alpha,
            "comparisons": ds_comparison,
            "method_means": {m: float(np.nanmean(v)) for m, v in method_values.items()},
        }

        for comp in ds_comparison:
            sig = "***" if comp["significant"] else ""
            logger.info(
                "%s: %s vs %s | corrected p=%.4f %s",
                ds_name, comp["method_1"], comp["method_2"],
                comp["corrected_p"], sig,
            )

    return comparison_results


def parse_args():
    p = argparse.ArgumentParser(description="Paired Wilcoxon tests on fusion results")
    p.add_argument("--fusion-jsons", nargs="+", required=True)
    p.add_argument("--metric", choices=["bacc", "auc"], default="bacc")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--save-json", default=None)
    p.add_argument("--save-md", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("FusionStats")

    results = compare_methods(
        args.fusion_jsons,
        metric=args.metric,
        alpha=args.alpha,
    )

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Saved JSON: %s", args.save_json)

    if args.save_md:
        Path(args.save_md).parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Fusion Statistical Comparison", ""]
        lines.append(f"Metric: `{args.metric}` | α = {args.alpha}")
        lines.append("")
        for ds_name, ds_result in results.items():
            from .protocol import DATASETS
            display = DATASETS[ds_name].display_name if ds_name in DATASETS else ds_name
            lines.append(f"## {display}")
            lines.append("")
            lines.append("| Method 1 | Method 2 | Mean Diff | Raw p | Corrected p | Sig |")
            lines.append("| --- | --- | ---: | ---: | ---: | :---: |")
            for comp in ds_result["comparisons"]:
                sig = "✓" if comp["significant"] else ""
                lines.append(
                    f"| {comp['method_1']} | {comp['method_2']} "
                    f"| {comp['mean_diff']:.4f} | {comp['raw_p']:.4f} "
                    f"| {comp['corrected_p']:.4f} | {sig} |"
                )
            lines.append("")

        with open(args.save_md, "w") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Saved Markdown: %s", args.save_md)


if __name__ == "__main__":
    main()
