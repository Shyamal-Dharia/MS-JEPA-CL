from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare


METHOD_ORDER = [
    "Static FC",
    "FC + ComBat",
    "FC + CovBat",
    "JEPA Rotary (Pooled)",
    "JEPA Rotary (Curriculum + Replay)",
]
BASELINE_METHODS = [
    "Static FC",
    "FC + ComBat",
    "FC + CovBat",
]
JEPA_METHODS = [
    "JEPA Rotary (Pooled)",
    "JEPA Rotary (Curriculum + Replay)",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run omnibus Friedman tests and planned paired permutation posthoc tests "
            "on site-level summaries derived from site-Wasserstein outputs."
        )
    )
    parser.add_argument(
        "--report-json",
        default="results/reports/tables/jepa_fusion/site_wasserstein_pca50/site_wasserstein_pca50_all.json",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output-prefix",
        default="site_wasserstein_stats_pca50",
    )
    return parser.parse_args()


def maybe_json(value):
    if isinstance(value, dict):
        return {str(k): maybe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [maybe_json(v) for v in value]
    if isinstance(value, tuple):
        return [maybe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [maybe_json(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    return value


def bonferroni_correction(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    return np.clip(np.asarray(p_values, dtype=np.float64) * n, 0.0, 1.0).tolist()


def overall_pair_map(method_row: dict) -> dict[tuple[str, str], float]:
    acc: dict[tuple[str, str], list[float]] = {}
    for run in method_row["runs"]:
        for pair in run["overall"]["pairwise"]:
            key = (str(pair["site_a"]), str(pair["site_b"]))
            acc.setdefault(key, []).append(float(pair["swd"]))
    return {key: float(np.mean(vals)) for key, vals in acc.items()}


def conditional_pair_map(method_row: dict) -> dict[tuple[str, str, str], float]:
    acc: dict[tuple[str, str, str], list[float]] = {}
    for run in method_row["runs"]:
        for diagnosis, payload in run["conditional"]["per_diagnosis"].items():
            for pair in payload["pairwise"]:
                key = (str(diagnosis), str(pair["site_a"]), str(pair["site_b"]))
                acc.setdefault(key, []).append(float(pair["swd"]))
    return {key: float(np.mean(vals)) for key, vals in acc.items()}


def pair_map_to_site_units(pair_map: dict) -> dict[tuple, float]:
    acc: dict[tuple, list[float]] = {}
    for key, value in pair_map.items():
        if len(key) == 2:
            site_a, site_b = key
            acc.setdefault((site_a,), []).append(float(value))
            acc.setdefault((site_b,), []).append(float(value))
        elif len(key) == 3:
            diagnosis, site_a, site_b = key
            acc.setdefault((diagnosis, site_a), []).append(float(value))
            acc.setdefault((diagnosis, site_b), []).append(float(value))
        else:
            raise ValueError(f"Unexpected key length: {key}")
    return {unit: float(np.mean(vals)) for unit, vals in acc.items()}


def site_level_matrix(label_to_row: dict[str, dict], metric: str) -> tuple[list[tuple], dict[str, np.ndarray]]:
    mapper = overall_pair_map if metric == "overall" else conditional_pair_map
    site_maps = {
        label: pair_map_to_site_units(mapper(label_to_row[label]))
        for label in METHOD_ORDER
        if label in label_to_row
    }
    common_units = sorted(set.intersection(*[set(site_map) for site_map in site_maps.values()]))
    matrix = {
        label: np.asarray([site_maps[label][unit] for unit in common_units], dtype=np.float64)
        for label in METHOD_ORDER
        if label in site_maps
    }
    return common_units, matrix


def omnibus_friedman(matrix: dict[str, np.ndarray]) -> dict:
    arrays = [matrix[label] for label in METHOD_ORDER if label in matrix]
    if len(arrays) < 3:
        return {"statistic": float("nan"), "p_value": float("nan")}
    stat, p_value = friedmanchisquare(*arrays)
    return {"statistic": float(stat), "p_value": float(p_value)}


def paired_signflip_permutation(diff: np.ndarray, n_permutations: int, rng: np.random.Generator) -> dict:
    diff = np.asarray(diff, dtype=np.float64)
    observed = float(np.mean(diff))
    if len(diff) == 0:
        return {"observed_mean_difference": float("nan"), "p_value_two_sided": float("nan"), "p_value_greater": float("nan")}
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_permutations), len(diff)))
    permuted = np.mean(signs * diff[None, :], axis=1)
    p_two = float((np.sum(np.abs(permuted) >= abs(observed)) + 1.0) / (len(permuted) + 1.0))
    p_greater = float((np.sum(permuted >= observed) + 1.0) / (len(permuted) + 1.0))
    return {
        "observed_mean_difference": observed,
        "p_value_two_sided": p_two,
        "p_value_greater": p_greater,
    }


def planned_posthoc(matrix: dict[str, np.ndarray], n_permutations: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    pvals = []
    for baseline in BASELINE_METHODS:
        for jepa in JEPA_METHODS:
            diff = matrix[baseline] - matrix[jepa]
            result = paired_signflip_permutation(diff, n_permutations=n_permutations, rng=rng)
            row = {
                "baseline": baseline,
                "jepa": jepa,
                "n_units": int(len(diff)),
                "baseline_mean": float(np.mean(matrix[baseline])),
                "jepa_mean": float(np.mean(matrix[jepa])),
                "mean_difference": float(np.mean(diff)),
                "median_difference": float(np.median(diff)),
                **result,
            }
            rows.append(row)
            pvals.append(row["p_value_two_sided"])
    qvals = bonferroni_correction(pvals)
    for row, q in zip(rows, qvals):
        row["q_value_two_sided"] = float(q)
    return rows


def build_markdown(payload: dict) -> str:
    lines = [
        "# Site Wasserstein Statistics",
        "",
        "Omnibus test: Friedman test on matched site-level summary SWD units across all methods.",
        "Posthoc tests: paired sign-flip permutation tests on planned baseline-vs-JEPA contrasts.",
        "Site-level summaries average each site's pairwise SWD to all other available sites.",
        "For conditional SWD, units are diagnosis-site combinations.",
        f"Two-sided posthoc p-values are Bonferroni corrected within each dataset/metric family over {len(BASELINE_METHODS) * len(JEPA_METHODS)} planned contrasts.",
        "",
    ]
    for dataset_key, dataset_payload in payload["datasets"].items():
        lines.extend(
            [
                f"## {dataset_payload['dataset_label']}",
                "",
            ]
        )
        for metric_name, metric_payload in dataset_payload["metrics"].items():
            omni = metric_payload["omnibus"]
            lines.extend(
                [
                    f"### {metric_name.title()}",
                    "",
                    f"Friedman chi-square = `{omni['statistic']:.4f}`, p = `{omni['p_value']:.3e}`, matched site-level units = `{metric_payload['n_units']}`",
                    "",
                    "| Baseline | JEPA | Mean Diff | Median Diff | p(two-sided) | q(Bonf.) | p(one-sided baseline>JEPA) |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in metric_payload["posthoc"]:
                lines.append(
                    f"| {row['baseline']} | {row['jepa']} | {row['mean_difference']:.4f} | {row['median_difference']:.4f} | "
                    f"{row['p_value_two_sided']:.4e} | {row['q_value_two_sided']:.4e} | {row['p_value_greater']:.4e} |"
                )
            lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    report = json.loads(Path(args.report_json).read_text())
    payload = {
        "source_report": str(args.report_json),
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
        "datasets": {},
    }

    for dataset_key, dataset_payload in report["datasets"].items():
        label_to_row = {row["label"]: row for row in dataset_payload["methods"]}
        metrics = {}
        for metric_idx, metric in enumerate(("overall", "conditional")):
            common_units, matrix = site_level_matrix(label_to_row, metric)
            metrics[metric] = {
                "n_units": int(len(common_units)),
                "omnibus": omnibus_friedman(matrix),
                "posthoc": planned_posthoc(
                    matrix,
                    n_permutations=args.n_permutations,
                    seed=args.seed + 1000 * metric_idx,
                ),
            }
        payload["datasets"][dataset_key] = {
            "dataset_label": dataset_payload["dataset_label"],
            "metrics": metrics,
        }

    outdir = Path("results/reports/tables/jepa_fusion") / args.output_prefix
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / f"{args.output_prefix}.json"
    out_md = outdir / f"{args.output_prefix}.md"
    out_json.write_text(json.dumps(maybe_json(payload), indent=2))
    out_md.write_text(build_markdown(payload))
    print(json.dumps({"json": str(out_json), "md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
