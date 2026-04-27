"""Low-K stress test for ComBat / CovBat baselines.

For each K in {10, 20, 30, 40, 50}, subsample K subjects per site for
harmonization calibration, transform the full dataset, then evaluate
both site probe (grouped subject-level CV) and diagnosis (LOSO).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.preprocessing import LabelEncoder

try:
    from .evaluate_baselines import (
        CombatTransformer,
        CovBatTransformer,
        mean_std,
        parse_seed_list,
        run_grouped_cv,
        run_loso_diagnosis,
    )
    from .dataset_utils import matched_records_for_preset
    from .protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        filter_records_for_task,
        get_dataset_preset,
    )
    from ..jepa_clean.plot_tsne_fc_vs_raw import (
        build_fc_features,
        get_roi_permutation,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.evaluate_baselines import (
        CombatTransformer,
        CovBatTransformer,
        mean_std,
        parse_seed_list,
        run_grouped_cv,
        run_loso_diagnosis,
    )
    from src.jepa_fusion.dataset_utils import matched_records_for_preset
    from src.jepa_fusion.protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        filter_records_for_task,
        get_dataset_preset,
    )
    from src.jepa_clean.plot_tsne_fc_vs_raw import (
        build_fc_features,
        get_roi_permutation,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Low-K stress test for ComBat/CovBat baselines.")
    p.add_argument("--dataset", choices=["abide", "abide1", "abide2", "srpbs"], required=True)
    p.add_argument("--task", choices=["default", "hc_asd", "hc_disorder"], default="default")
    p.add_argument("--all-sites", action="store_true")
    p.add_argument("--k-values", default="10,20,30,40,50", help="Comma-separated K values")
    p.add_argument("--triangle", choices=["upper", "lower"], default="upper")
    p.add_argument("--reorder", choices=["none", "hemi", "hemi_network"], default="hemi_network")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--covbat-n-pc", type=int, default=50)
    p.add_argument("--covbat-pct-var", type=float, default=0.95)
    p.add_argument("--covbat-max-components", type=int, default=256)
    p.add_argument("--healthy-control-label", default=HEALTHY_CONTROL_LABEL)
    p.add_argument("--autism-label", default=AUTISM_LABEL)
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-md", default=None)
    return p.parse_args()


def subsample_calibration_indices(
    subjects: np.ndarray,
    sites: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """Select up to K unique subjects per site, return record indices."""
    rng = np.random.default_rng(seed)
    selected = []
    for site in np.unique(sites):
        site_mask = sites == site
        site_indices = np.flatnonzero(site_mask)
        site_subjects = subjects[site_indices]
        unique_subjects = np.unique(site_subjects)
        n_pick = min(k, len(unique_subjects))
        chosen_subjects = set(rng.choice(unique_subjects, size=n_pick, replace=False))
        for idx in site_indices:
            if subjects[idx] in chosen_subjects:
                selected.append(idx)
    return np.array(sorted(selected), dtype=np.int64)


def lowk_transform(
    method: str,
    fc_features: np.ndarray,
    sites: np.ndarray,
    calib_idx: np.ndarray,
    *,
    seed: int,
    covbat_n_pc: int,
    covbat_pct_var: float,
    covbat_max_components: int,
) -> np.ndarray:
    """Fit ComBat/CovBat on calibration subset, transform ALL data."""
    x_calib = fc_features[calib_idx]
    sites_calib = sites[calib_idx]

    if method == "fc_combat":
        transformer = CombatTransformer()
    elif method == "fc_covbat":
        transformer = CovBatTransformer(
            n_pc=covbat_n_pc,
            pct_var=covbat_pct_var,
            max_components=covbat_max_components,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    _, x_all_transformed = transformer.fit_transform(
        x_calib, sites_calib,
        fc_features, sites,
    )
    return x_all_transformed


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    preset = get_dataset_preset(args.dataset)
    task_key = preset.default_task if args.task == "default" else args.task
    mixed_sites_only = False if args.all_sites else preset.default_mixed_sites_only

    # Load and filter records
    records = matched_records_for_preset(preset)
    task_result = filter_records_for_task(
        records, task_key,
        mixed_sites_only=mixed_sites_only,
        healthy_control_label=args.healthy_control_label,
        autism_label=args.autism_label,
    )
    task_records = task_result.records
    task_y = task_result.y

    # Build FC features
    permutation, _ = get_roi_permutation(args.reorder)
    fc_features = build_fc_features(task_records, permutation=permutation, triangle=args.triangle)

    subjects = np.array([str(r.subject_id) for r in task_records], dtype=object)
    holdout_sites = np.array([str(r.site) for r in task_records], dtype=object)
    batch_sites = np.array([str(r.site_batch) for r in task_records], dtype=object)
    site_encoder = LabelEncoder()
    y_site = site_encoder.fit_transform(batch_sites)
    site_class_names = [str(n) for n in site_encoder.classes_.tolist()]

    methods = ["fc", "fc_combat", "fc_covbat"]

    results = {}
    for k in k_values:
        results[k] = {}
        for method in methods:
            site_runs = []
            dx_runs = []
            for seed in seeds:
                if method == "fc":
                    features = fc_features
                else:
                    calib_idx = subsample_calibration_indices(subjects, batch_sites, k, seed)
                    try:
                        features = lowk_transform(
                            method, fc_features, batch_sites, calib_idx,
                            seed=seed,
                            covbat_n_pc=args.covbat_n_pc,
                            covbat_pct_var=args.covbat_pct_var,
                            covbat_max_components=args.covbat_max_components,
                        )
                    except Exception as e:
                        print(f"  K={k} {method} seed={seed}: FAILED - {e}")
                        continue

                # Site probe (grouped subject-level CV)
                site_run = run_grouped_cv(
                    features, y_site,
                    class_names=site_class_names,
                    groups=subjects,
                    seed=seed,
                    folds=args.folds,
                )
                site_runs.append(site_run)

                # Diagnosis (LOSO)
                dx_run = run_loso_diagnosis(
                    features, task_y,
                    class_names=task_result.class_names,
                    sites=holdout_sites,
                    seed=seed,
                )
                dx_runs.append(dx_run)
                print(f"  K={k:>3} {method:<12} seed={seed}: "
                      f"site_bAcc={100*site_run['balanced_accuracy']:.1f}  "
                      f"site_AUC={100*site_run['roc_auc']:.1f}  "
                      f"dx_bAcc={100*dx_run['balanced_accuracy']:.1f}  "
                      f"dx_AUC={100*dx_run['roc_auc']:.1f}")

            if site_runs and dx_runs:
                results[k][method] = {
                    "site": {
                        "balanced_accuracy": mean_std([r["balanced_accuracy"] for r in site_runs]),
                        "roc_auc": mean_std([r["roc_auc"] for r in site_runs]),
                    },
                    "diagnosis": {
                        "balanced_accuracy": mean_std([r["balanced_accuracy"] for r in dx_runs]),
                        "roc_auc": mean_std([r["roc_auc"] for r in dx_runs]),
                    },
                    "n_runs": len(dx_runs),
                }

    # Summary table
    print(f"\n{'='*80}")
    print(f"Low-K Stress Test: {preset.label}")
    print(f"{'='*80}")
    print(f"{'K':>5} {'Method':<15} {'site_bAcc':>12} {'site_AUC':>12} {'dx_bAcc':>12} {'dx_AUC':>12}")
    print("-" * 75)
    for k in k_values:
        for method in methods:
            if method in results[k]:
                r = results[k][method]
                sb = r["site"]["balanced_accuracy"]
                sa = r["site"]["roc_auc"]
                db = r["diagnosis"]["balanced_accuracy"]
                da = r["diagnosis"]["roc_auc"]
                print(f"{k:>5} {method:<15} "
                      f"{100*sb['mean']:>5.1f}±{100*sb['std']:.1f}  "
                      f"{100*sa['mean']:>5.1f}±{100*sa['std']:.1f}  "
                      f"{100*db['mean']:>5.1f}±{100*db['std']:.1f}  "
                      f"{100*da['mean']:>5.1f}±{100*da['std']:.1f}")
        print()

    # Save
    payload = {
        "config": {
            "dataset": preset.key,
            "dataset_label": preset.label,
            "task_key": task_key,
            "k_values": k_values,
            "seeds": seeds,
            "methods": methods,
            "n_records": len(task_records),
            "n_subjects": len(set(subjects.tolist())),
            "n_sites": len(set(batch_sites.tolist())),
        },
        "results": {
            str(k): {
                method: results[k][method]
                for method in methods
                if method in results[k]
            }
            for k in k_values
        },
    }

    base = Path("results/reports/tables/jepa_fusion")
    out_json = Path(args.output_json) if args.output_json else base / f"{preset.key}_lowk_stress_test.json"
    out_md = Path(args.output_md) if args.output_md else base / f"{preset.key}_lowk_stress_test.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    # Markdown
    lines = [f"# Low-K Stress Test: {preset.label}", "",
             f"K = calibration subjects per site for ComBat/CovBat. FC = no harmonization (reference).", "",
             f"| K | Method | Site bAcc (%) | Site AUC (%) | Dx bAcc (%) | Dx AUC (%) |",
             f"| ---: | --- | ---: | ---: | ---: | ---: |"]
    for k in k_values:
        for method in methods:
            if method in results[k]:
                r = results[k][method]
                sb = r["site"]["balanced_accuracy"]
                sa = r["site"]["roc_auc"]
                db = r["diagnosis"]["balanced_accuracy"]
                da = r["diagnosis"]["roc_auc"]
                lines.append(f"| {k} | {method} | {100*sb['mean']:.1f} ± {100*sb['std']:.1f} | "
                             f"{100*sa['mean']:.1f} ± {100*sa['std']:.1f} | "
                             f"{100*db['mean']:.1f} ± {100*db['std']:.1f} | "
                             f"{100*da['mean']:.1f} ± {100*da['std']:.1f} |")
    out_md.write_text("\n".join(lines) + "\n")

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
