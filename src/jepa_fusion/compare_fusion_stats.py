from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run paired site-level statistical comparisons for fusion subject-vote results. "
            "Metrics are averaged over seeds within each held-out site, then compared with a "
            "paired Wilcoxon signed-rank test across sites."
        )
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="JSON file produced by evaluate_fusion_subject_vote.py",
    )
    parser.add_argument(
        "--target",
        default="fusion::fc_covbat",
        help="Method key to test as the target/best method.",
    )
    parser.add_argument(
        "--comparators",
        nargs="*",
        default=None,
        help="Optional list of comparator method keys. Defaults to all other methods in the JSON.",
    )
    parser.add_argument(
        "--alternative",
        choices=["greater", "less", "two-sided"],
        default="greater",
        help="Alternative hypothesis for the Wilcoxon test.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path. Defaults next to the input file.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional output Markdown path. Defaults next to the input file.",
    )
    return parser.parse_args()


def default_output_path(input_json: Path, suffix: str) -> Path:
    return input_json.with_name(f"{input_json.stem}_stats{suffix}")


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def format_p(value: float | None) -> str:
    if value is None:
        return "--"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def holm_adjust(pvalues: list[float]) -> list[float]:
    indexed = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [0.0] * len(pvalues)
    running_max = 0.0
    total = len(pvalues)
    for rank, (idx, pval) in enumerate(indexed):
        candidate = min(1.0, (total - rank) * pval)
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted


def summarize_sites(dataset_payload: dict, method: str, metric: str) -> dict[str, float]:
    per_site_values: dict[str, list[float]] = defaultdict(list)
    for run in dataset_payload["runs"][method]:
        for row in run["summary"]["site_metrics"]:
            per_site_values[row["site"]].append(float(row[metric]))
    return {
        site: float(sum(values) / len(values))
        for site, values in sorted(per_site_values.items())
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def run_paired_wilcoxon(x: list[float], y: list[float], alternative: str) -> float | None:
    diffs = [xi - yi for xi, yi in zip(x, y)]
    nonzero = [diff for diff in diffs if abs(diff) > 1e-12]
    if not nonzero:
        return 1.0
    result = wilcoxon(x, y, alternative=alternative, zero_method="wilcox", method="auto")
    return float(result.pvalue)


def method_label(dataset_payload: dict, method: str) -> str:
    return dataset_payload.get("method_labels", {}).get(method, method)


def compare_dataset(dataset_key: str, dataset_payload: dict, target: str, comparators: list[str], alternative: str):
    target_bacc = summarize_sites(dataset_payload, target, "bacc")
    target_auc = summarize_sites(dataset_payload, target, "auc")
    rows = []
    raw_bacc_p = []
    raw_auc_p = []
    valid_comparators = []

    for comparator in comparators:
        comp_bacc = summarize_sites(dataset_payload, comparator, "bacc")
        comp_auc = summarize_sites(dataset_payload, comparator, "auc")
        sites = sorted(set(target_bacc) & set(comp_bacc) & set(target_auc) & set(comp_auc))
        target_bacc_values = [target_bacc[site] for site in sites]
        comp_bacc_values = [comp_bacc[site] for site in sites]
        target_auc_values = [target_auc[site] for site in sites]
        comp_auc_values = [comp_auc[site] for site in sites]
        bacc_p = run_paired_wilcoxon(target_bacc_values, comp_bacc_values, alternative=alternative)
        auc_p = run_paired_wilcoxon(target_auc_values, comp_auc_values, alternative=alternative)
        row = {
            "dataset": dataset_key,
            "target": target,
            "comparator": comparator,
            "comparator_label": method_label(dataset_payload, comparator),
            "n_sites": len(sites),
            "target_mean_bacc": mean(target_bacc_values),
            "comparator_mean_bacc": mean(comp_bacc_values),
            "delta_bacc": mean(target_bacc_values) - mean(comp_bacc_values),
            "p_bacc": bacc_p,
            "target_mean_auc": mean(target_auc_values),
            "comparator_mean_auc": mean(comp_auc_values),
            "delta_auc": mean(target_auc_values) - mean(comp_auc_values),
            "p_auc": auc_p,
        }
        rows.append(row)
        raw_bacc_p.append(1.0 if bacc_p is None else bacc_p)
        raw_auc_p.append(1.0 if auc_p is None else auc_p)
        valid_comparators.append(comparator)

    adjusted_bacc = holm_adjust(raw_bacc_p)
    adjusted_auc = holm_adjust(raw_auc_p)
    for row, bacc_adj, auc_adj in zip(rows, adjusted_bacc, adjusted_auc):
        row["p_bacc_holm"] = float(bacc_adj)
        row["p_auc_holm"] = float(auc_adj)
    return rows


def build_markdown(payload: dict) -> str:
    lines = [
        "# Fusion Statistical Comparisons",
        "",
        f"Input: `{payload['input_json']}`",
        "",
        f"Target method: `{payload['target']}`",
        "",
        (
            "Test: paired Wilcoxon signed-rank across held-out sites after averaging each site's "
            f"metric over seeds. Alternative hypothesis: `{payload['alternative']}`."
        ),
        "",
    ]
    for dataset_key in payload["order"]:
        comparisons = payload["datasets"][dataset_key]["comparisons"]
        target_label = payload["datasets"][dataset_key]["target_label"]
        lines.append(f"## {dataset_key}")
        lines.append("")
        lines.append(f"Target label: `{target_label}`")
        lines.append("")
        lines.append(
            "| Comparator | n sites | Target bAcc | Comparator bAcc | Delta bAcc | p | Holm p | "
            "Target AUC | Comparator AUC | Delta AUC | p | Holm p |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in comparisons:
            lines.append(
                f"| {row['comparator_label']} | {row['n_sites']} | "
                f"{format_pct(row['target_mean_bacc'])} | {format_pct(row['comparator_mean_bacc'])} | "
                f"{format_pct(row['delta_bacc'])} | {format_p(row['p_bacc'])} | {format_p(row['p_bacc_holm'])} | "
                f"{format_pct(row['target_mean_auc'])} | {format_pct(row['comparator_mean_auc'])} | "
                f"{format_pct(row['delta_auc'])} | {format_p(row['p_auc'])} | {format_p(row['p_auc_holm'])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    input_json = Path(args.input_json)
    payload = json.loads(input_json.read_text())
    output_json = Path(args.output_json) if args.output_json else default_output_path(input_json, ".stats.json")
    output_md = Path(args.output_md) if args.output_md else default_output_path(input_json, ".stats.md")

    datasets_output = {}
    dataset_order = list(payload["datasets"].keys())
    for dataset_key in dataset_order:
        dataset_payload = payload["datasets"][dataset_key]
        methods = list(dataset_payload["runs"].keys())
        if args.target not in methods:
            raise ValueError(f"Target method {args.target!r} not found in dataset {dataset_key!r}")
        comparators = args.comparators or [method for method in methods if method != args.target]
        missing = [method for method in comparators if method not in methods]
        if missing:
            raise ValueError(f"Comparators missing in {dataset_key!r}: {missing}")
        comparisons = compare_dataset(
            dataset_key=dataset_key,
            dataset_payload=dataset_payload,
            target=args.target,
            comparators=comparators,
            alternative=args.alternative,
        )
        datasets_output[dataset_key] = {
            "target": args.target,
            "target_label": method_label(dataset_payload, args.target),
            "comparisons": comparisons,
        }

    result_payload = {
        "input_json": str(input_json),
        "target": args.target,
        "alternative": args.alternative,
        "order": dataset_order,
        "datasets": datasets_output,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result_payload, indent=2))
    output_md.write_text(build_markdown(result_payload))
    print(f"Saved JSON: {output_json}")
    print(f"Saved Markdown: {output_md}")


if __name__ == "__main__":
    main()
