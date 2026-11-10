"""Table 7: aggregate per-seed demographic probe outputs into mean +/- s.d.

Reads ``<prefix>_seed<S>_demographics.json`` files written by
``evaluate_demographics.py`` (one run per MS-JEPA pretraining seed, probe seed = model seed)
and writes a seed-level mean +/- sample s.d. table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHOD_LABELS = {
    "fc": "Static FC",
    "fc_combat": "Static FC + ComBat",
    "fc_covbat": "Static FC + CovBat",
    "jepa": "MS-JEPA",
}
METRICS = [
    ("sex", "balanced_accuracy", "Sex bAcc", 100.0, 2),
    ("sex", "roc_auc", "Sex ROC-AUC", 100.0, 2),
    ("age", "mae", "Age MAE", 1.0, 2),
    ("age", "rmse", "Age RMSE", 1.0, 2),
    ("age", "r2", "Age R2", 1.0, 3),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="results/reports/tables/jepa_fusion/demographics")
    parser.add_argument("--prefix", default="rotary_curr_replay")
    parser.add_argument("--seeds", default="0,8,16,32,128")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    payloads = [
        json.loads((results_dir / f"{args.prefix}_seed{seed}_demographics.json").read_text())
        for seed in seeds
    ]

    lines = [
        "# Demographic Prediction: Seed Mean and Std",
        "",
        f"Aggregated across seeds: {', '.join(map(str, seeds))}. Values are mean ± sample std across seeds.",
        "",
        "| Dataset | Method | " + " | ".join(label for *_x, label, _s, _d in METRICS) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in METRICS) + " |",
    ]
    for dataset_key in payloads[0]["order"]:
        dataset_label = payloads[0]["datasets"][dataset_key]["config"]["dataset_label"]
        for method_key in payloads[0]["methods"]:
            cells = []
            for task, metric, _label, scale, digits in METRICS:
                values = np.asarray(
                    [
                        p["datasets"][dataset_key]["methods"][method_key]["summary"][task][metric]["mean"]
                        for p in payloads
                    ],
                    dtype=np.float64,
                ) * scale
                cells.append(f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}")
            lines.append(
                f"| {dataset_label} | {METHOD_LABELS.get(method_key, method_key)} | " + " | ".join(cells) + " |"
            )

    out_md = results_dir / f"{args.prefix}_demographics_seed_mean_std.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out_md}")


if __name__ == "__main__":
    main()
