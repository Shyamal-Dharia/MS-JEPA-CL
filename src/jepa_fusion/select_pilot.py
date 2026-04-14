from __future__ import annotations

import argparse
import json
from pathlib import Path

CEILINGS = {
    "srpbs": 0.157,
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
        "datasets": rows,
        "pass_count": int(pass_count),
        "mean_dx_bacc": float(sum(dx_bacc) / len(dx_bacc)),
        "mean_dx_auc": float(sum(dx_auc) / len(dx_auc)),
        "mean_site_bacc": float(sum(site_bacc) / len(site_bacc)),
        "mean_site_auc": float(sum(site_auc) / len(site_auc)),
        "diagnosis_protocol": "loso_subject_vote" if subject_vote_payload is not None else "grouped_cv_probe",
        "subject_vote_json": None if subject_vote_payload is None else str(subject_vote_path_for(grouped_eval_path)),
    }


def ranking_key(row: dict):
    return (
        int(row["pass_count"]),
        float(row["mean_dx_bacc"]),
        float(row["mean_dx_auc"]),
        -float(row["mean_site_bacc"]),
    )


def format_pct(value: float):
    return f"{100.0 * value:.1f}"


def build_markdown(ranked_rows: list[dict]):
    lines = [
        "# JEPA Pilot Selection",
        "",
        "Site columns come from grouped-CV site probes. Diagnosis columns come from LOSO subject-vote when available.",
        "",
        "| Candidate | Pass Count | Mean Site bAcc | Mean Site ROC-AUC | Mean Dx bAcc | Mean Dx ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked_rows:
        lines.append(
            f"| {row['candidate']} | {row['pass_count']}/3 | "
            f"{format_pct(row['mean_site_bacc'])} | {format_pct(row['mean_site_auc'])} | "
            f"{format_pct(row['mean_dx_bacc'])} | {format_pct(row['mean_dx_auc'])} |"
        )
    lines.extend(["", "## Per-Dataset Checks", ""])
    for row in ranked_rows:
        lines.append(f"### {row['candidate']}")
        lines.append("")
        lines.append("| Dataset | Site bAcc | Site ROC-AUC | Ceiling | Pass | Dx bAcc | Dx ROC-AUC |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for dataset_row in row["datasets"]:
            lines.append(
                f"| {dataset_row['dataset']} | {format_pct(dataset_row['site_bacc'])} | "
                f"{format_pct(dataset_row['site_auc'])} | {format_pct(dataset_row['ceiling'])} | "
                f"{'yes' if dataset_row['passes'] else 'no'} | {format_pct(dataset_row['dx_bacc'])} | "
                f"{format_pct(dataset_row['dx_auc'])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    scored = []
    for input_path in args.inputs:
        with open(input_path) as handle:
            payload = json.load(handle)
        scored.append(score_payload(payload, input_path))

    ranked = sorted(scored, key=ranking_key, reverse=True)
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
