"""Table 9: zero-shot transfer summary and paired sign-flip tests.

Reads the no-exclusion zero-shot outputs written by ``run_zero_shot_no_exclusion.sh``
(overlapping ABIDE-I/ABIDE-II centers retained in the target), summarizes each
method as mean +/- s.d. over seeds, and tests MS-JEPA against Masked Recon. SSL
with exact paired sign-flip tests over matched pretraining seeds, Holm-corrected
across the six primary tests (2 directions x 3 metrics).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_RESULTS_ROOT = "results/reports/tables/jepa_fusion/zero_shot_no_exclusion"
DEFAULT_SEEDS = "0,4,8,16,32,64,128,256,512,1024"
TRANSFER_SPECS = [
    {
        "key": "srpbs_abide1_to_abide2",
        "label": "SRPBS + ABIDE-I -> ABIDE-II",
        "dataset": "abide2",
        "stem": "srpbs_abide1_to_abide2",
    },
    {
        "key": "srpbs_abide2_to_abide1",
        "label": "SRPBS + ABIDE-II -> ABIDE-I",
        "dataset": "abide1",
        "stem": "srpbs_abide2_to_abide1",
    },
]
METRICS = [
    ("site_bacc", "Site bAcc", "lower"),
    ("dx_bacc", "Diag. bAcc", "higher"),
    ("dx_auc", "Diag. ROC-AUC", "higher"),
]


def mean_sd(values: list[float]):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


def load_learned_rows(base: Path, prefix: str, dataset: str, seeds: list[int]):
    """Site bAcc from the grouped-CV site probe; diagnosis from LOSO subject vote."""
    rows = []
    for seed in seeds:
        grouped = base / f"{prefix}_seed{seed}_no_exclusion_groupedcv.json"
        subject = base / f"{prefix}_seed{seed}_no_exclusion_subject_vote.json"
        gd = json.loads(grouped.read_text())["datasets"][dataset]
        sd = json.loads(subject.read_text())["datasets"][dataset]
        rows.append(
            {
                "seed": int(seed),
                "site_bacc": 100.0 * float(gd["summary"]["site"]["balanced_accuracy"]["mean"]),
                "dx_bacc": 100.0 * float(sd["summary"]["bacc"]["mean"]),
                "dx_auc": 100.0 * float(sd["summary"]["auc"]["mean"]),
            }
        )
    return rows


def load_static_rows(path: str):
    payload = json.loads(Path(path).read_text())
    rows = []
    if isinstance(payload["runs"], list):
        for run in payload["runs"]:
            rows.append(
                {
                    "seed": int(run["seed"]),
                    "site_bacc": 100.0 * float(run["site"]["balanced_accuracy"]),
                    "dx_bacc": 100.0 * float(run["subject"]["balanced_accuracy"]),
                    "dx_auc": 100.0 * float(run["subject"]["roc_auc"]),
                }
            )
    else:
        # Newer multi-method schema, if regenerated later.
        for run in payload["runs"]["fc"]:
            rows.append(
                {
                    "seed": int(run["seed"]),
                    "site_bacc": 100.0 * float(run["site"]["balanced_accuracy"]),
                    "dx_bacc": 100.0 * float(run["subject"]["balanced_accuracy"]),
                    "dx_auc": 100.0 * float(run["subject"]["roc_auc"]),
                }
            )
    return sorted(rows, key=lambda row: row["seed"])


def paired_effect(primary_rows: list[dict], baseline_rows: list[dict], metric: str, direction: str):
    primary_by_seed = {row["seed"]: row for row in primary_rows}
    baseline_by_seed = {row["seed"]: row for row in baseline_rows}
    seeds = sorted(set(primary_by_seed) & set(baseline_by_seed))
    if direction == "lower":
        diffs = np.asarray([baseline_by_seed[seed][metric] - primary_by_seed[seed][metric] for seed in seeds])
    else:
        diffs = np.asarray([primary_by_seed[seed][metric] - baseline_by_seed[seed][metric] for seed in seeds])
    return seeds, diffs


def sign_flip_test(diffs: np.ndarray):
    observed = float(diffs.mean())
    n = int(len(diffs))
    means = []
    for signs in itertools.product([-1.0, 1.0], repeat=n):
        means.append(float((diffs * np.asarray(signs)).mean()))
    means_arr = np.asarray(means)
    p_two = float(np.mean(np.abs(means_arr) >= abs(observed) - 1e-12))
    p_one = float(np.mean(means_arr >= observed - 1e-12))
    return {
        "mean_effect": observed,
        "p_two_sided": p_two,
        "p_one_sided_primary_better": p_one,
        "n_pairs": n,
    }


def unpaired_permutation_test(primary: list[float], baseline: list[float], direction: str):
    primary = np.asarray(primary, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    observed = (
        float(baseline.mean() - primary.mean())
        if direction == "lower"
        else float(primary.mean() - baseline.mean())
    )
    combined = np.concatenate([primary, baseline])
    n_primary = len(primary)
    effects = []
    for primary_idx in itertools.combinations(range(len(combined)), n_primary):
        primary_mask = np.zeros(len(combined), dtype=bool)
        primary_mask[list(primary_idx)] = True
        perm_primary = combined[primary_mask]
        perm_baseline = combined[~primary_mask]
        effect = (
            float(perm_baseline.mean() - perm_primary.mean())
            if direction == "lower"
            else float(perm_primary.mean() - perm_baseline.mean())
        )
        effects.append(effect)
    effects = np.asarray(effects)
    return {
        "mean_effect": observed,
        "p_two_sided": float(np.mean(np.abs(effects) >= abs(observed) - 1e-12)),
        "p_one_sided_primary_better": float(np.mean(effects >= observed - 1e-12)),
        "n_primary": int(len(primary)),
        "n_baseline": int(len(baseline)),
        "note": "Exploratory only: Static FC probe seeds are not pretraining seeds.",
    }


def holm_adjust(p_values: list[float]):
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (idx, p_value) in enumerate(indexed, start=1):
        running = max(running, min(1.0, (m - rank + 1) * p_value))
        adjusted[idx] = running
    return adjusted


def fmt_pm(stats: dict):
    return f"{stats['mean']:.1f} ± {stats['sd']:.1f}"


def fmt_p(value: float):
    if value < 1e-4:
        return f"{value:.1e}"
    return f"{value:.4f}"


def build_report(payload: dict):
    seeds_text = ",".join(str(seed) for seed in payload["config"]["seeds"])
    lines = [
        "# Zero-Shot Transfer Statistical Tests (Table 9)",
        "",
        "Target evaluation keeps overlapping ABIDE-I/ABIDE-II centers (no target-site exclusion).",
        "",
        f"Primary inference compares `MS-JEPA` against `Masked Recon. SSL` using exact paired sign-flip tests over matched pretraining seeds (`{seeds_text}`), Holm-corrected over all six primary tests.",
        "Positive effect means `MS-JEPA` is better: lower site bAcc, higher diagnosis bAcc, or higher diagnosis ROC-AUC.",
        "",
        "Static FC comparisons are listed separately as exploratory unpaired permutation tests because Static-FC variation comes from probe seeds, not pretraining seeds.",
        "",
    ]
    for spec in payload["directions"]:
        lines.extend(
            [
                f"## {spec['label']}",
                "",
                "| Method | Site bAcc | Diag. bAcc | Diag. ROC-AUC |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for method in ["Static FC", "Masked Recon. SSL", "MS-JEPA"]:
            summary = spec["summary"][method]
            lines.append(
                f"| {method} | {fmt_pm(summary['site_bacc'])} | "
                f"{fmt_pm(summary['dx_bacc'])} | {fmt_pm(summary['dx_auc'])} |"
            )
        lines.extend(
            [
                "",
                "### Primary Paired Tests: MS-JEPA vs Masked Recon. SSL",
                "",
                "| Metric | Effect | p(one-sided) | p(two-sided) | Holm q(one-sided) |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in spec["primary_tests"]:
            lines.append(
                f"| {row['metric_label']} | {row['mean_effect']:.2f} | "
                f"{fmt_p(row['p_one_sided_primary_better'])} | {fmt_p(row['p_two_sided'])} | "
                f"{fmt_p(row['q_one_sided_holm'])} |"
            )
        lines.extend(
            [
                "",
                "### Exploratory Unpaired Tests: MS-JEPA vs Static FC",
                "",
                "| Metric | Effect | p(one-sided) | p(two-sided) |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in spec["static_tests"]:
            lines.append(
                f"| {row['metric_label']} | {row['mean_effect']:.2f} | "
                f"{fmt_p(row['p_one_sided_primary_better'])} | {fmt_p(row['p_two_sided'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="Matched pretraining seeds.")
    parser.add_argument("--run-tag", default="final_run_v1")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Defaults to <results-root>/zero_shot_transfer_stats (.json and .md).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    directions = []
    all_primary_p = []
    primary_refs = []
    for spec in TRANSFER_SPECS:
        stem = spec["stem"]
        jepa = load_learned_rows(
            root / "msjepa",
            f"{args.run_tag}_transfer_{stem}_rotary_curr_replay_nce",
            spec["dataset"],
            seeds,
        )
        masked = load_learned_rows(
            root / "masked_recon", f"masked_recon_ssl_{stem}", spec["dataset"], seeds
        )
        static = load_static_rows(
            str(root / "static_fc" / f"static_fc_zero_shot_{stem}_hc_disorder_no_exclusion.json")
        )
        summary = {}
        for name, rows in [("MS-JEPA", jepa), ("Masked Recon. SSL", masked), ("Static FC", static)]:
            summary[name] = {
                metric: mean_sd([row[metric] for row in rows])
                for metric, _label, _direction in METRICS
            }

        primary_tests = []
        static_tests = []
        for metric, metric_label, direction in METRICS:
            seeds, diffs = paired_effect(jepa, masked, metric, direction)
            test = sign_flip_test(diffs)
            test.update({"metric": metric, "metric_label": metric_label, "paired_seeds": seeds})
            primary_refs.append((len(directions), len(primary_tests)))
            all_primary_p.append(test["p_one_sided_primary_better"])
            primary_tests.append(test)

            static_test = unpaired_permutation_test(
                [row[metric] for row in jepa],
                [row[metric] for row in static],
                direction,
            )
            static_test.update({"metric": metric, "metric_label": metric_label})
            static_tests.append(static_test)

        directions.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "summary": summary,
                "primary_tests": primary_tests,
                "static_tests": static_tests,
            }
        )

    q_values = holm_adjust(all_primary_p)
    for (direction_idx, test_idx), q_value in zip(primary_refs, q_values):
        directions[direction_idx]["primary_tests"][test_idx]["q_one_sided_holm"] = q_value

    payload = {
        "config": {
            "results_root": str(root),
            "seeds": seeds,
            "primary_test": "exact paired sign-flip",
            "exploratory_static_test": "exact unpaired permutation",
            "primary_p_adjustment": "Holm over all primary one-sided tests",
            "metrics": [metric for metric, _label, _direction in METRICS],
        },
        "directions": directions,
    }

    prefix = Path(args.output_prefix) if args.output_prefix else root / "zero_shot_transfer_stats"
    out_json = prefix.with_suffix(".json")
    out_md = prefix.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    out_md.write_text(build_report(payload))
    print(build_report(payload))
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()
