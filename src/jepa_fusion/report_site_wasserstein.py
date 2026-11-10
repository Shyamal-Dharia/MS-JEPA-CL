from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wasserstein_distance
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


METHOD_KEYS = (
    "fc",
    "fc_combat",
    "fc_covbat",
    "jepa_rotary_pooled",
    "jepa_rotary_curriculum",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute site-distribution sliced Wasserstein distances for FC and JEPA "
            "representations in a shared standardized PCA space."
        )
    )
    parser.add_argument("--dataset", choices=["abide", "srpbs", "all"], default="all")
    parser.add_argument("--run-tag", default="final_run_v1")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--seeds", default="0,8,16,32,128")
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
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--n-projections", type=int, default=256)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument(
        "--site-label-level",
        choices=["auto", "batch", "center"],
        default="center",
        help="Which site labels to use for Wasserstein comparison. 'auto' collapses ABIDE to center labels and keeps batch labels elsewhere.",
    )
    parser.add_argument("--output-prefix", default="site_wasserstein_pca50_centermerged")
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


def reduce_features(features: np.ndarray, pca_dim: int) -> np.ndarray:
    x = StandardScaler().fit_transform(np.asarray(features, dtype=np.float64))
    n_components = min(int(pca_dim), x.shape[0] - 1, x.shape[1])
    n_components = max(2, n_components)
    return PCA(n_components=n_components, random_state=0).fit_transform(x)


def random_directions(dim: int, n_projections: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(int(n_projections), int(dim)))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return dirs / norms


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, directions: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vals = []
    for direction in directions:
        x_proj = np.asarray(x @ direction, dtype=np.float64)
        y_proj = np.asarray(y @ direction, dtype=np.float64)
        vals.append(float(wasserstein_distance(x_proj, y_proj)))
    return float(np.mean(vals)) if vals else float("nan")


def pairwise_site_swd(features: np.ndarray, site_labels: np.ndarray, directions: np.ndarray) -> dict:
    unique_sites = sorted(str(site) for site in np.unique(site_labels))
    values = []
    pairs = []
    for i, site_a in enumerate(unique_sites):
        x = features[np.asarray(site_labels, dtype=object) == site_a]
        for site_b in unique_sites[i + 1 :]:
            y = features[np.asarray(site_labels, dtype=object) == site_b]
            dist = sliced_wasserstein(x, y, directions)
            if np.isfinite(dist):
                values.append(dist)
                pairs.append({"site_a": site_a, "site_b": site_b, "swd": float(dist)})
    return {
        "mean_swd": float(np.mean(values)) if values else float("nan"),
        "n_site_pairs": int(len(values)),
        "pairwise": pairs,
    }


def conditional_site_swd(features: np.ndarray, site_labels: np.ndarray, diagnoses: np.ndarray, directions: np.ndarray) -> dict:
    out = {}
    for diagnosis in sorted(str(x) for x in np.unique(diagnoses)):
        mask = np.asarray(diagnoses, dtype=object) == diagnosis
        result = pairwise_site_swd(features[mask], site_labels[mask], directions)
        out[diagnosis] = result
    means = [row["mean_swd"] for row in out.values() if np.isfinite(row["mean_swd"])]
    return {
        "mean_swd": float(np.mean(means)) if means else float("nan"),
        "per_diagnosis": out,
    }


def comparison_site_labels(dataset_key: str, bundle, site_label_level: str) -> np.ndarray:
    if site_label_level == "center":
        return np.asarray(bundle.site_centers, dtype=object)
    if site_label_level == "batch":
        return np.asarray(bundle.site_batches, dtype=object)
    if dataset_key == "abide":
        return np.asarray(bundle.site_centers, dtype=object)
    return np.asarray(bundle.site_batches, dtype=object)


def build_markdown(payload: dict) -> str:
    lines = [
        "# Site Wasserstein Audit",
        "",
        f"Shared PCA dimension: `{payload['pca_dim']}`",
        f"Sliced Wasserstein projections: `{payload['n_projections']}`",
        "Representations are standardized within method, reduced with unsupervised PCA, then compared across site distributions.",
        "Conditional SWD is averaged across diagnosis-specific site comparisons.",
        "",
    ]
    for dataset_key, dataset_payload in payload["datasets"].items():
        lines.extend(
            [
                f"## {dataset_payload['dataset_label']}",
                "",
                f"Common subjects across methods: `{dataset_payload['n_common_subjects']}`",
                "",
                "| Method | Overall SWD | SD | Conditional SWD | SD | Runs |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in dataset_payload["methods"]:
            overall = row["summary"]["overall_swd"]
            conditional = row["summary"]["conditional_swd"]
            lines.append(
                f"| {row['label']} | {overall['mean']:.4f} | {overall['std']:.4f} | "
                f"{conditional['mean']:.4f} | {conditional['std']:.4f} | {row['summary']['n_runs']} |"
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


def summarize_runs(runs: list[dict]) -> dict:
    return {
        "overall_swd": mean_std([run["overall"]["mean_swd"] for run in runs if np.isfinite(run["overall"]["mean_swd"])]),
        "conditional_swd": mean_std(
            [run["conditional"]["mean_swd"] for run in runs if np.isfinite(run["conditional"]["mean_swd"])]
        ),
        "n_runs": int(len(runs)),
    }


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    dataset_keys = ["abide", "srpbs"] if args.dataset == "all" else [args.dataset]
    results_dir = Path(args.results_dir) if args.results_dir else Path("results/reports/tables/jepa_fusion") / args.run_tag
    specs = {spec.key: spec for spec in method_specs(args.run_tag, include_curriculum_nce=False)}

    payload = {
        "run_tag": args.run_tag,
        "dataset_choice": args.dataset,
        "pca_dim": int(args.pca_dim),
        "n_projections": int(args.n_projections),
        "projection_seed": int(args.projection_seed),
        "site_label_level": str(args.site_label_level),
        "datasets": {},
    }

    for dataset_key in dataset_keys:
        fc_bundles, meta = prepare_fc_bundles(dataset_key, args)
        method_seed_bundles = {
            "fc": {0: fc_bundles["fc"]},
            "fc_combat": {0: fc_bundles["fc_combat"]},
            "fc_covbat": {0: fc_bundles["fc_covbat"]},
            "jepa_rotary_pooled": prepare_jepa_bundles(dataset_key, specs["jepa_rotary_pooled"], seeds, results_dir, args),
            "jepa_rotary_curriculum": prepare_jepa_bundles(
                dataset_key, specs["jepa_rotary_curriculum"], seeds, results_dir, args
            ),
        }
        common_subjects = compute_common_subjects(method_seed_bundles)
        if not common_subjects:
            raise RuntimeError(f"No common subjects found for dataset {dataset_key}.")
        align_bundles_to_common_subjects(method_seed_bundles, common_subjects)

        method_rows = []
        for method_key in METHOD_KEYS:
            method = specs[method_key]
            runs = []
            for seed, bundle in sorted(method_seed_bundles[method_key].items()):
                reduced = reduce_features(bundle.features, args.pca_dim)
                directions = random_directions(reduced.shape[1], args.n_projections, args.projection_seed + int(seed))
                site_labels = comparison_site_labels(dataset_key, bundle, args.site_label_level)
                overall = pairwise_site_swd(reduced, site_labels, directions)
                conditional = conditional_site_swd(reduced, site_labels, bundle.diagnoses, directions)
                runs.append(
                    {
                        "seed": int(seed),
                        "overall": overall,
                        "conditional": conditional,
                    }
                )
            method_rows.append(
                {
                    "key": method_key,
                    "label": method.label,
                    "summary": summarize_runs(runs),
                    "runs": runs,
                }
            )

        payload["datasets"][dataset_key] = {
            "dataset_label": meta["dataset_label"],
            "task_label": meta["task_label"],
            "n_common_subjects": int(len(common_subjects)),
            "n_comparison_sites": int(len(np.unique(comparison_site_labels(dataset_key, next(iter(method_seed_bundles["fc"].values())), args.site_label_level)))),
            "methods": method_rows,
        }

    outputs = output_paths(args.output_prefix, args.dataset)
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(json.dumps(maybe_json(payload), indent=2))
    outputs["md"].write_text(build_markdown(payload))
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
