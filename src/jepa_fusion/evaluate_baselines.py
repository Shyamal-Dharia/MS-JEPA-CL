"""Classical baseline evaluation: ROI, Static FC, FC+ComBat, FC+CovBat.

Uses grouped subject CV with fold-wise harmonization.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .protocol import (
    DATASETS,
    evaluate_probe_cv,
    filter_mixed_sites,
    filter_task_records,
    format_metric,
    load_fc_features,
    load_fc_records,
    load_raw_features,
    load_raw_records,
    make_subject_id,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


def match_fc_raw(fc_records, raw_records):
    """Match FC and raw records by subject."""
    raw_by_subject = {}
    for rec in raw_records:
        raw_by_subject[make_subject_id(rec)] = rec

    matched_fc = []
    matched_raw = []
    for fc_rec in fc_records:
        subj = make_subject_id(fc_rec)
        if subj in raw_by_subject:
            matched_fc.append(fc_rec)
            matched_raw.append(raw_by_subject[subj])
    return matched_fc, matched_raw


def run_baselines(dataset_name: str, seeds=(0, 1, 2, 3, 4), n_folds=5):
    logger = logging.getLogger("Baselines")
    spec = DATASETS[dataset_name]

    # Load records
    fc_records = load_fc_records(spec.fc_dir)
    raw_records = load_raw_records(spec.raw_dir)
    logger.info("Loaded %d FC records, %d raw records", len(fc_records), len(raw_records))

    # Match by subject
    fc_records, raw_records = match_fc_raw(fc_records, raw_records)
    logger.info("Matched %d FC/raw record pairs", len(fc_records))

    # Task filtering
    fc_records, fc_labels = filter_task_records(fc_records, spec.hc_diag, spec.disorder_diag)
    raw_records, raw_labels = filter_task_records(raw_records, spec.hc_diag, spec.disorder_diag)
    logger.info("After task filter: FC=%d raw=%d", len(fc_records), len(raw_records))

    # Mixed-site filtering
    fc_records, fc_labels = filter_mixed_sites(fc_records, fc_labels)
    raw_records, raw_labels = filter_mixed_sites(raw_records, raw_labels)
    logger.info("After mixed-site filter: FC=%d raw=%d", len(fc_records), len(raw_records))

    # Load features
    fc_features = load_fc_features(fc_records)
    raw_features = load_raw_features(raw_records)
    logger.info("FC features: %s | Raw features: %s", fc_features.shape, raw_features.shape)

    n_sites = len(set(rec.site for rec in fc_records))
    chance_site = 100.0 / n_sites

    # Evaluate methods
    rows = {}
    methods = [
        ("ROI time series", raw_features, raw_records, raw_labels, None),
        ("Static FC", fc_features, fc_records, fc_labels, None),
        ("Static FC + ComBat", fc_features, fc_records, fc_labels, "combat"),
        ("Static FC + CovBat", fc_features, fc_records, fc_labels, "covbat"),
    ]
    for method_name, features, records, labels, harmonize in methods:
        logger.info("Evaluating: %s", method_name)
        result = evaluate_probe_cv(
            features, labels, records,
            n_folds=n_folds, seeds=seeds,
            harmonize=harmonize,
        )
        rows[method_name] = result
        logger.info(
            "%s | Site bAcc=%s | Dx bAcc=%s | Dx AUC=%s",
            method_name,
            format_metric(result["site_bacc"]),
            format_metric(result["dx_bacc"]),
            format_metric(result["dx_auc"]),
        )

    # Build table
    lines = [
        "| Method | Site bAcc | Dx bAcc | Dx ROC-AUC |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, result in rows.items():
        lines.append(
            f"| {name} | {format_metric(result['site_bacc'])} "
            f"| {format_metric(result['dx_bacc'])} "
            f"| {format_metric(result['dx_auc'])} |"
        )
    lines.append(f"| Chance | {chance_site:.1f} | 50.0 | 50.0 |")
    table = "\n".join(lines)

    # Task display name
    if spec.disorder_diag == "ALL":
        task_display = "HC vs ALL Disorders"
    elif spec.disorder_diag == 1:
        task_display = "HC vs ASD"
    else:
        task_display = f"HC vs diag-{spec.disorder_diag}"

    results = {
        "dataset": dataset_name,
        "display_name": spec.display_name,
        "task_display": task_display,
        "n_sites": n_sites,
        "n_records": len(fc_records),
        "methods": {name: {k: list(v) for k, v in res.items()} for name, res in rows.items()},
        "table": table,
    }

    # Output paths
    if spec.disorder_diag == "ALL":
        tag = f"{dataset_name}_hc_disorder_mixedsites_baseline_table"
    else:
        tag = f"{dataset_name}_hc_asd_mixedsites_baseline_table"

    out_dir = Path("results/reports/tables/jepa_fusion")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{tag}.json"
    md_path = out_dir / f"{tag}.md"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved JSON: %s", json_path)

    with open(md_path, "w") as f:
        f.write(f"# {spec.display_name} Baselines\n\n")
        f.write(f"Task: `{task_display}`\n\n")
        f.write(table)
        f.write("\n")
    logger.info("Saved Markdown: %s", md_path)

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Classical baselines for jepa_fusion")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-folds", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    run_baselines(args.dataset, seeds=tuple(args.seeds), n_folds=args.n_folds)


if __name__ == "__main__":
    main()
