from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

CEILINGS = {
    "srpbs": 0.157,
    "abide": 0.102,
    "abide1": 0.102,
    "abide2": 0.126,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rank jepa_fusion pilot evaluations using grouped-CV site probes plus LOSO diagnosis "
            "when matching subject-vote JSON files are available."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Evaluation JSON files from evaluate_jepa.py")
    parser.add_argument("--output-json", default="results/reports/tables/jepa_fusion/pilot_selection.json")
    parser.add_argument("--output-md", default="results/reports/tables/jepa_fusion/pilot_selection.md")
    return parser.parse_args()


def candidate_label(payload: dict):
    cfg = payload["checkpoint"]["config"]
    family = cfg.get("model_family", "temporal")
    if family == "temporal":
        return f"temporal:{cfg.get('temporal_variant', 'unknown')}"
    return "st_roi_patch"


def subject_vote_path_for(grouped_eval_path: str) -> Path:
    path = Path(grouped_eval_path)
    if path.name.endswith("_jepa_eval.json"):
        return path.with_name(path.name.replace("_jepa_eval.json", "_subject_vote.json"))
    return path.with_name(f"{path.stem}_subject_vote.json")


def diagnosis_summary_for(grouped_payload: dict, grouped_eval_path: str):
    subject_vote_path = subject_vote_path_for(grouped_eval_path)
    if subject_vote_path.exists():
        with subject_vote_path.open() as handle:
            subject_vote_payload = json.load(handle)
        return subject_vote_payload, {
            dataset_key: subject_vote_payload["datasets"][dataset_key]["summary"]
            for dataset_key in subject_vote_payload["order"]
        }
    return None, {
        dataset_key: grouped_payload["datasets"][dataset_key]["summary"]["diagnosis"]
        for dataset_key in grouped_payload["order"]
    }


def mean_std(values: list[float]):
    if not values:
        raise ValueError("Cannot aggregate an empty list of values.")
    mean = float(sum(values) / len(values))
    if len(values) == 1:
        return {"mean": mean, "std": 0.0, "n": 1}
    variance = sum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
    return {"mean": mean, "std": variance ** 0.5, "n": len(values)}


def train_seed_for(payload: dict):
    cfg = payload["checkpoint"].get("config", {})
    seed = cfg.get("seed")
    if seed is None:
        return None
    return int(seed)


def score_payload(payload: dict, grouped_eval_path: str):
    subject_vote_payload, diagnosis_map = diagnosis_summary_for(payload, grouped_eval_path)
    rows = []
    pass_count = 0
    dx_bacc = []
    dx_auc = []
    site_bacc = []
    site_auc = []
    for dataset_key in payload["order"]:
        dataset = payload["datasets"][dataset_key]
        site_mean = dataset["summary"]["site"]["balanced_accuracy"]["mean"]
        site_auc_mean = dataset["summary"]["site"]["roc_auc"]["mean"]
        diagnosis_summary = diagnosis_map[dataset_key]
        dx_bacc_key = "bacc" if "bacc" in diagnosis_summary else "balanced_accuracy"
        dx_auc_key = "auc" if "auc" in diagnosis_summary else "roc_auc"
        dx_bacc_mean = diagnosis_summary[dx_bacc_key]["mean"]
        dx_auc_mean = diagnosis_summary[dx_auc_key]["mean"]
        passes = site_mean <= CEILINGS[dataset_key]
        pass_count += int(passes)
        dx_bacc.append(dx_bacc_mean)
        dx_auc.append(dx_auc_mean)
        site_bacc.append(site_mean)
        site_auc.append(site_auc_mean)
        rows.append(
            {
                "dataset": dataset_key,
                "site_bacc": site_mean,
                "site_auc": site_auc_mean,
                "dx_bacc": dx_bacc_mean,
                "dx_auc": dx_auc_mean,
                "ceiling": CEILINGS[dataset_key],
                "passes": bool(passes),
            }
        )
    return {
        "candidate": candidate_label(payload),
        "checkpoint": payload["checkpoint"]["path"],
        "train_seed": train_seed_for(payload),
        "datasets": rows,
        "pass_count": int(pass_count),
        "mean_dx_bacc": float(sum(dx_bacc) / len(dx_bacc)),
        "mean_dx_auc": float(sum(dx_auc) / len(dx_auc)),
        "mean_site_bacc": float(sum(site_bacc) / len(site_bacc)),
        "mean_site_auc": float(sum(site_auc) / len(site_auc)),
        "diagnosis_protocol": "loso_subject_vote" if subject_vote_payload is not None else "grouped_cv_probe",
        "subject_vote_json": None if subject_vote_payload is None else str(subject_vote_path_for(grouped_eval_path)),
    }


def aggregate_candidate_rows(candidate: str, seed_rows: list[dict]):
    dataset_order = [dataset_row["dataset"] for dataset_row in seed_rows[0]["datasets"]]
    dataset_aggregates = []
    for dataset_key in dataset_order:
        matching_rows = []
        for seed_row in seed_rows:
            for dataset_row in seed_row["datasets"]:
                if dataset_row["dataset"] == dataset_key:
                    matching_rows.append(dataset_row)
                    break
        dataset_aggregates.append(
            {
                "dataset": dataset_key,
                "site_bacc": mean_std([row["site_bacc"] for row in matching_rows]),
                "site_auc": mean_std([row["site_auc"] for row in matching_rows]),
                "dx_bacc": mean_std([row["dx_bacc"] for row in matching_rows]),
                "dx_auc": mean_std([row["dx_auc"] for row in matching_rows]),
                "ceiling": matching_rows[0]["ceiling"],
                "pass_count": int(sum(1 for row in matching_rows if row["passes"])),
                "n_pretrain_seeds": int(len(matching_rows)),
            }
        )

    protocols = sorted({row["diagnosis_protocol"] for row in seed_rows})
    pretrain_seeds = [row["train_seed"] for row in seed_rows if row["train_seed"] is not None]
    return {
        "candidate": candidate,
        "n_pretrain_seeds": int(len(seed_rows)),
        "pretrain_seeds": sorted(pretrain_seeds),
        "checkpoints": [row["checkpoint"] for row in seed_rows],
        "grouped_eval_jsons": [row["grouped_eval_json"] for row in seed_rows],
        "subject_vote_jsons": [row["subject_vote_json"] for row in seed_rows if row["subject_vote_json"] is not None],
        "diagnosis_protocols": protocols,
        "pass_count": mean_std([float(row["pass_count"]) for row in seed_rows]),
        "mean_dx_bacc": mean_std([row["mean_dx_bacc"] for row in seed_rows]),
        "mean_dx_auc": mean_std([row["mean_dx_auc"] for row in seed_rows]),
        "mean_site_bacc": mean_std([row["mean_site_bacc"] for row in seed_rows]),
        "mean_site_auc": mean_std([row["mean_site_auc"] for row in seed_rows]),
        "datasets": dataset_aggregates,
        "seed_rows": seed_rows,
    }


def ranking_key(row: dict):
    return (
        float(row["pass_count"]["mean"]),
        float(row["mean_dx_bacc"]["mean"]),
        float(row["mean_dx_auc"]["mean"]),
        -float(row["mean_site_bacc"]["mean"]),
    )


def format_pct(value: float):
    return f"{100.0 * value:.1f}"


def format_pct_stat(value: dict):
    return f"{format_pct(value['mean'])} +- {format_pct(value['std'])}"


def format_count_stat(value: dict):
    return f"{value['mean']:.2f} +- {value['std']:.2f}"


def build_markdown(ranked_rows: list[dict]):
    lines = [
        "# JEPA Pilot Selection",
        "",
        "Aggregated across JEPA pretraining seeds for each candidate.",
        "",
        "Site columns come from grouped-CV site probes. Diagnosis columns come from LOSO subject-vote when available.",
        "",
        "| Candidate | Pretrain Seeds | Mean Pass Count | Mean Site bAcc | Mean Site ROC-AUC | Mean Dx bAcc | Mean Dx ROC-AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked_rows:
        pretrain_seeds = ",".join(str(seed) for seed in row["pretrain_seeds"]) if row["pretrain_seeds"] else str(row["n_pretrain_seeds"])
        lines.append(
            f"| {row['candidate']} | {pretrain_seeds} | {format_count_stat(row['pass_count'])}/{len(row['datasets'])} | "
            f"{format_pct_stat(row['mean_site_bacc'])} | {format_pct_stat(row['mean_site_auc'])} | "
            f"{format_pct_stat(row['mean_dx_bacc'])} | {format_pct_stat(row['mean_dx_auc'])} |"
        )
    lines.extend(["", "## Per-Dataset Checks", ""])
    for row in ranked_rows:
        lines.append(f"### {row['candidate']}")
        lines.append("")
        lines.append("| Dataset | Site bAcc | Site ROC-AUC | Ceiling | Pass Rate | Dx bAcc | Dx ROC-AUC |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for dataset_row in row["datasets"]:
            lines.append(
                f"| {dataset_row['dataset']} | {format_pct_stat(dataset_row['site_bacc'])} | "
                f"{format_pct_stat(dataset_row['site_auc'])} | {format_pct(dataset_row['ceiling'])} | "
                f"{dataset_row['pass_count']}/{dataset_row['n_pretrain_seeds']} | "
                f"{format_pct_stat(dataset_row['dx_bacc'])} | {format_pct_stat(dataset_row['dx_auc'])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    scored = []
    for input_path in args.inputs:
        with open(input_path) as handle:
            payload = json.load(handle)
        seed_row = score_payload(payload, input_path)
        seed_row["grouped_eval_json"] = str(input_path)
        scored.append(seed_row)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        grouped[row["candidate"]].append(row)

    aggregated = [aggregate_candidate_rows(candidate, seed_rows) for candidate, seed_rows in grouped.items()]
    ranked = sorted(aggregated, key=ranking_key, reverse=True)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump({"ranked": ranked, "ceilings": CEILINGS}, handle, indent=2)
    output_md.write_text(build_markdown(ranked))
    print(f"Saved JSON: {output_json}")
    print(f"Saved Markdown: {output_md}")


if __name__ == "__main__":
    main()
