from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from .analyze_embedding_invariance import (
        align_bundles_to_common_subjects,
        compute_common_subjects,
        method_specs,
        parse_seed_list,
        prepare_fc_bundles,
        prepare_jepa_bundles,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.analyze_embedding_invariance import (
        align_bundles_to_common_subjects,
        compute_common_subjects,
        method_specs,
        parse_seed_list,
        prepare_fc_bundles,
        prepare_jepa_bundles,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Report how much variance the downstream probe PCA keeps for FC and JEPA "
            "representations without rerunning the evaluation pipelines."
        )
    )
    parser.add_argument("--dataset", choices=["abide", "srpbs", "all"], default="all")
    parser.add_argument("--run-tag", default="final_run_v1")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--seeds", default="0,8,16,32,128")
    parser.add_argument("--include-curriculum-nce", action="store_true")
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--triangle", choices=["upper", "lower"], default="upper")
    parser.add_argument("--reorder", choices=["none", "hemi", "hemi_network"], default="hemi_network")
    parser.add_argument("--harmonization-covariates", default="age,sex")
    parser.add_argument("--harmonization-fit-controls-only", action="store_true", default=True)
    parser.add_argument("--harmonization-age-min", type=float, default=0.0)
    parser.add_argument("--harmonization-age-max", type=float, default=100.0)
    parser.add_argument("--harmonization-seed", type=int, default=0)
    parser.add_argument("--window-feature", choices=["mean", "mean_std", "mean_cov"], default="mean")
    parser.add_argument("--covariance-dim", type=int, default=32)
    parser.add_argument("--covariance-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", default="results/cache/jepa_embedding_invariance")
    parser.add_argument("--output-prefix", default="pca_retained_variance")
    return parser.parse_args()


def mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        return {"mean": float(arr[0]), "std": 0.0}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1))}


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


def retained_variance(features: np.ndarray, n_components: int, standardize: bool) -> dict:
    x = np.asarray(features, dtype=np.float64)
    if standardize:
        x = StandardScaler().fit_transform(x)
    max_components = min(int(n_components), x.shape[0] - 1, x.shape[1])
    if max_components < 1:
        raise RuntimeError("Not enough samples/features for PCA.")
    pca = PCA(n_components=max_components, random_state=0)
    pca.fit(x)
    return {
        "variance_kept": float(np.sum(pca.explained_variance_ratio_)),
        "n_components": int(max_components),
        "standardized_before_pca": bool(standardize),
    }


def method_payload(method_label: str, by_seed: dict[int, dict]) -> dict:
    runs = [{"seed": int(seed), **row} for seed, row in sorted(by_seed.items())]
    return {
        "label": method_label,
        "summary": {
            "variance_kept": mean_std([row["variance_kept"] for row in by_seed.values()]),
            "n_components": int(next(iter(by_seed.values()))["n_components"]),
            "standardized_before_pca": bool(next(iter(by_seed.values()))["standardized_before_pca"]),
            "n_runs": int(len(by_seed)),
        },
        "runs": runs,
    }


def build_markdown(payload: dict) -> str:
    lines = [
        "# PCA Retained Variance",
        "",
        f"Downstream PCA components: `{payload['n_components']}`",
        f"Standardize before PCA: `{payload['standardized_before_pca']}`",
        "FC + ComBat/CovBat use controls-only harmonization with age and sex covariates.",
        "",
    ]
    for dataset_key, dataset_payload in payload["datasets"].items():
        lines.extend(
            [
                f"## {dataset_payload['dataset_label']}",
                "",
                f"Common subjects across methods: `{dataset_payload['n_common_subjects']}`",
                "",
                "| Method | Variance Kept (%) | SD (%) | Runs |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in dataset_payload["methods"]:
            summary = row["summary"]["variance_kept"]
            lines.append(
                f"| {row['label']} | {100.0 * summary['mean']:.1f} | {100.0 * summary['std']:.1f} | {row['summary']['n_runs']} |"
            )
        lines.append("")
    return "\n".join(lines)


def output_paths(prefix: str, dataset_choice: str) -> dict[str, Path]:
    base = Path("results/reports/tables/jepa_fusion") / prefix
    stem = f"{prefix}_{dataset_choice}"
    return {
        "json": base / f"{stem}.json",
        "md": base / f"{stem}.md",
    }


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    dataset_keys = ["abide", "srpbs"] if args.dataset == "all" else [args.dataset]
    results_dir = Path(args.results_dir) if args.results_dir else Path("results/reports/tables/jepa_fusion") / args.run_tag
    specs = method_specs(args.run_tag, include_curriculum_nce=args.include_curriculum_nce)
    selected_specs = [spec for spec in specs if spec.key in {"fc", "fc_combat", "fc_covbat"} or spec.kind == "jepa"]
    standardized = not bool(args.no_standardize)

    payload = {
        "run_tag": args.run_tag,
        "dataset_choice": args.dataset,
        "n_components": int(args.n_components),
        "standardized_before_pca": bool(standardized),
        "datasets": {},
    }

    for dataset_key in dataset_keys:
        fc_bundles, meta = prepare_fc_bundles(dataset_key, args)
        method_seed_bundles = {
            "fc": {0: fc_bundles["fc"]},
            "fc_combat": {0: fc_bundles["fc_combat"]},
            "fc_covbat": {0: fc_bundles["fc_covbat"]},
        }
        for spec in selected_specs:
            if spec.kind != "jepa":
                continue
            method_seed_bundles[spec.key] = prepare_jepa_bundles(dataset_key, spec, seeds, results_dir, args)

        common_subjects = compute_common_subjects(method_seed_bundles)
        if not common_subjects:
            raise RuntimeError(f"No common subjects found for dataset {dataset_key}.")
        align_bundles_to_common_subjects(method_seed_bundles, common_subjects)

        method_rows = []
        for spec in selected_specs:
            by_seed = {
                seed: retained_variance(bundle.features, args.n_components, standardized)
                for seed, bundle in method_seed_bundles[spec.key].items()
            }
            method_rows.append(method_payload(spec.label, by_seed))

        payload["datasets"][dataset_key] = {
            "dataset_label": meta["dataset_label"],
            "task_label": meta["task_label"],
            "n_common_subjects": int(len(common_subjects)),
            "methods": method_rows,
        }

    outputs = output_paths(args.output_prefix, args.dataset)
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(maybe_json(payload), indent=2))
    outputs["md"].write_text(build_markdown(payload))
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
