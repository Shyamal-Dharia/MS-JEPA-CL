from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.preprocessing import LabelEncoder

try:
    from .evaluate_jepa import format_pct, run_grouped_cv, summarize_runs
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.evaluate_jepa import format_pct, run_grouped_cv, summarize_runs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate grouped-CV site probes on subject-level fusion probabilities "
            "from evaluate_fusion_subject_vote outputs."
        )
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--representation", choices=["fused_prob", "prob_pair"], default="fused_prob")
    parser.add_argument("--methods", default="fusion::fc,fusion::fc_combat,fusion::fc_covbat")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def parse_method_list(method_text: str):
    methods = []
    for item in method_text.split(","):
        item = item.strip()
        if item:
            methods.append(item)
    if not methods:
        raise ValueError("At least one method is required.")
    return methods


def default_output_paths(input_json: str, representation: str):
    src = Path(input_json)
    stem = src.stem + f"_{representation}_site_probe"
    return src.with_name(stem + ".json"), src.with_name(stem + ".md")


def build_method_labels(method_keys: list[str]):
    labels = {}
    for method_key in method_keys:
        if method_key == "fusion::fc":
            labels[method_key] = "Static FC + JEPA (late-prob site probe)"
        elif method_key == "fusion::fc_combat":
            labels[method_key] = "Static FC + ComBat + JEPA (late-prob site probe)"
        elif method_key == "fusion::fc_covbat":
            labels[method_key] = "Static FC + CovBat + JEPA (late-prob site probe)"
        else:
            labels[method_key] = method_key
    return labels


def build_features(subject_rows: list[dict], representation: str):
    if representation == "fused_prob":
        return np.asarray([[float(row["prob"])] for row in subject_rows], dtype=np.float32)
    if representation == "prob_pair":
        return np.asarray(
            [[float(row["fc_prob"]), float(row["jepa_prob"])] for row in subject_rows],
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported representation: {representation}")


def evaluate_dataset(dataset_payload: dict, method_keys: list[str], representation: str, folds: int):
    method_labels = build_method_labels(method_keys)
    runs_by_method = {method_key: [] for method_key in method_keys}

    for method_key in method_keys:
        for run in dataset_payload["runs"][method_key]:
            subject_rows = run["subject_rows"]
            features = build_features(subject_rows, representation)
            sites = np.asarray([str(row["site"]) for row in subject_rows], dtype=object)
            groups = np.asarray([str(row["subject_id"]) for row in subject_rows], dtype=object)
            site_encoder = LabelEncoder()
            y_site = site_encoder.fit_transform(sites)
            site_class_names = [str(name) for name in site_encoder.classes_.tolist()]
            runs_by_method[method_key].append(
                run_grouped_cv(
                    features,
                    y_site,
                    site_class_names,
                    groups,
                    seed=int(run["seed"]),
                    folds=folds,
                )
            )

    return {
        "config": {
            "dataset": dataset_payload["config"]["dataset"],
            "dataset_label": dataset_payload["config"]["dataset_label"],
            "representation": representation,
            "folds": int(folds),
        },
        "method_order": list(method_keys),
        "method_labels": method_labels,
        "summary": {method_key: summarize_runs(runs_by_method[method_key]) for method_key in method_keys},
    }


def build_markdown(payload: dict):
    rep_desc = "fused diagnosis probability" if payload["config"]["representation"] == "fused_prob" else "2D `(p_fc, p_jepa)` pair"
    lines = [
        f"# {payload['checkpoint']['label']} Late-Probability Fusion Site Probe",
        "",
        f"Source fusion results: `{payload['source_json']}`",
        "",
        f"Site probe uses grouped subject-level cross-validation on the subject-level {rep_desc}.",
        "",
        "| Dataset | Method | Site bAcc | Site ROC-AUC | Subjects |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for dataset_key in payload["order"]:
        dataset_payload = payload["datasets"][dataset_key]
        for method_key in dataset_payload["method_order"]:
            label = dataset_payload["method_labels"][method_key]
            summary = dataset_payload["summary"][method_key]
            lines.append(
                f"| {dataset_payload['config']['dataset_label']} | {label} | "
                f"{format_pct(summary['balanced_accuracy'])} | {format_pct(summary['roc_auc'])} | "
                f"{summary['n_samples']} |"
            )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    method_keys = parse_method_list(args.methods)
    source_json = Path(args.input_json)
    payload_in = json.loads(source_json.read_text())

    payload = {
        "checkpoint": payload_in["checkpoint"],
        "config": {
            "representation": args.representation,
            "folds": int(args.folds),
            "methods": method_keys,
        },
        "source_json": str(source_json),
        "order": list(payload_in["order"]),
        "datasets": {},
    }

    for dataset_key in payload_in["order"]:
        payload["datasets"][dataset_key] = evaluate_dataset(
            payload_in["datasets"][dataset_key],
            method_keys,
            args.representation,
            args.folds,
        )

    output_json = Path(args.output_json) if args.output_json else default_output_paths(args.input_json, args.representation)[0]
    output_md = Path(args.output_md) if args.output_md else default_output_paths(args.input_json, args.representation)[1]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(payload, f, indent=2)
    output_md.write_text(build_markdown(payload))


if __name__ == "__main__":
    main()
