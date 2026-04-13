"""FC + JEPA late fusion with subject-vote LOSO evaluation.

Runs matched-scan LOSO for JEPA-only, FC-only, and FC+JEPA fusion rows with
optional ComBat/CovBat harmonization on the FC branch.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..jepa_clean.data import WindowedFMRIDataset
from ..jepa_clean.evaluate_subject_vote import (
    build_model_from_checkpoint,
    extract_window_embeddings,
    filter_task,
    run_subject_vote_loso,
    TASKS,
)
from ..jepa_clean.repro import seed_everything
from .protocol import (
    DATASETS,
    apply_combat_foldwise,
    apply_covbat_foldwise,
    load_fc_features,
    load_fc_records,
    make_subject_id,
    normalize_site,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


DATASET_TASKS = {
    "abide1": "hc_asd",
    "abide2": "hc_asd",
    "srpbs": "hc_all",
}


def _build_fc_subject_embeddings(fc_records, fc_features, fc_labels):
    """Build subject-level FC feature dict for LOSO."""
    subject_data = {}
    for rec, feat, lab in zip(fc_records, fc_features, fc_labels):
        subj_id = make_subject_id(rec)
        subject_data[subj_id] = {
            "feature": feat,
            "site": rec.site,
            "label": int(lab),
        }
    return subject_data


def run_fc_loso(fc_features, fc_records, fc_labels, seed, harmonize=None):
    """Run FC-only subject-level LOSO with optional harmonization."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sites = np.array([rec.site for rec in fc_records])
    subjects = np.array([make_subject_id(rec) for rec in fc_records])
    unique_sites = np.unique(sites)

    all_subject_rows = []

    for held_out in unique_sites:
        train_mask = sites != held_out
        test_mask = sites == held_out

        X_train = fc_features[train_mask].copy()
        X_test = fc_features[test_mask].copy()
        y_train = fc_labels[train_mask]
        y_test = fc_labels[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        if harmonize == "combat":
            X_train, X_test = apply_combat_foldwise(
                X_train, X_test, sites[train_mask], sites[test_mask],
            )
        elif harmonize == "covbat":
            X_train, X_test = apply_covbat_foldwise(
                X_train, X_test, sites[train_mask], sites[test_mask],
            )

        n_comp = min(50, X_train.shape[0] - 1, X_train.shape[1])
        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=max(1, n_comp), random_state=seed),
            LogisticRegression(
                max_iter=10000,
                class_weight="balanced",
                random_state=seed,
            ),
        )
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]

        # Subject-level aggregation (one scan per subject in FC)
        test_subjects = subjects[test_mask]
        for subj, prob, y in zip(test_subjects, probs, y_test):
            all_subject_rows.append({
                "subject_id": subj,
                "site": held_out,
                "y": int(y),
                "prob": float(prob),
                "pred": int(prob >= 0.5),
            })

    if not all_subject_rows:
        return None, None

    y_true = np.array([r["y"] for r in all_subject_rows])
    y_pred = np.array([r["pred"] for r in all_subject_rows])
    y_prob = np.array([r["prob"] for r in all_subject_rows])

    bacc = float(balanced_accuracy_score(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    # Per-site metrics
    site_baccs = []
    for site in np.unique([r["site"] for r in all_subject_rows]):
        site_rows = [r for r in all_subject_rows if r["site"] == site]
        yt = np.array([r["y"] for r in site_rows])
        yp = np.array([r["pred"] for r in site_rows])
        if len(np.unique(yt)) >= 2:
            site_baccs.append(float(balanced_accuracy_score(yt, yp)))

    return {
        "bacc": bacc,
        "auc": auc,
        "site_mean_bacc": float(np.mean(site_baccs)) if site_baccs else float("nan"),
        "n_valid_sites": len(set(r["site"] for r in all_subject_rows)),
    }, all_subject_rows


def run_fusion_evaluation(
    checkpoint_path: str,
    fc_weight: float = 0.5,
    device=None,
    probe_seeds=(0, 1, 2, 3, 4),
):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    logger = logging.getLogger("Fusion-SubjectVote")

    model, ckpt_cfg = build_model_from_checkpoint(checkpoint_path)
    if device is None:
        device = ckpt_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    window_seconds = ckpt_cfg.get("window_seconds", 90.0)
    target_timepoints = ckpt_cfg.get("target_timepoints", 45)
    jepa_weight = 1.0 - fc_weight

    all_results = {}
    for ds_name, spec in DATASETS.items():
        task_key = DATASET_TASKS[ds_name]
        logger.info("Evaluating fusion on %s", spec.display_name)

        # Load JEPA embeddings
        dataset = WindowedFMRIDataset(
            spec.raw_dir,
            window_seconds=window_seconds,
            stride_seconds=30.0,
            target_timepoints=target_timepoints,
            sampling="sliding",
            exclude_unknown_diagnoses=True,
        )
        jepa_emb, jepa_meta = extract_window_embeddings(
            model, dataset, device=device, batch_size=128, window_feature="mean",
        )

        # Filter to task
        task_name, task_emb, task_meta, y = filter_task(jepa_emb, jepa_meta, task_key)

        # Load FC records
        fc_records = load_fc_records(spec.fc_dir)
        fc_features = load_fc_features(fc_records)

        # Match to task
        from .protocol import filter_task_records
        fc_records, fc_labels = filter_task_records(fc_records, spec.hc_diag, spec.disorder_diag)
        fc_features_task = load_fc_features(fc_records)

        # Matched subjects
        jepa_subjects = {meta["subject_id"] for meta in task_meta}
        fc_subjects = {make_subject_id(rec) for rec in fc_records}
        matched_subjects = jepa_subjects & fc_subjects

        # Filter FC to matched
        fc_mask = np.array([make_subject_id(r) in matched_subjects for r in fc_records])
        fc_records_m = [r for r, m in zip(fc_records, fc_mask) if m]
        fc_features_m = fc_features_task[fc_mask]
        fc_labels_m = fc_labels[fc_mask]

        # Run per-seed evaluations
        seed_summaries = defaultdict(list)
        methods = [
            ("JEPA", None), ("Static FC", None), ("Static FC + ComBat", "combat"),
            ("Static FC + CovBat", "covbat"),
            ("Static FC + JEPA", None), ("Static FC + ComBat + JEPA", "combat"),
            ("Static FC + CovBat + JEPA", "covbat"),
        ]

        for seed in probe_seeds:
            # JEPA-only LOSO
            jepa_metrics, jepa_rows = run_subject_vote_loso(
                task_emb, task_meta, y, seed=seed, classifier="logreg",
            )
            seed_summaries["JEPA"].append(jepa_metrics)

            # FC-only LOSO (each variant)
            for harmonize_name, harmonize in [
                ("Static FC", None),
                ("Static FC + ComBat", "combat"),
                ("Static FC + CovBat", "covbat"),
            ]:
                fc_m, fc_r = run_fc_loso(
                    fc_features_m, fc_records_m, fc_labels_m,
                    seed=seed, harmonize=harmonize,
                )
                seed_summaries[harmonize_name].append(fc_m or {"bacc": float("nan"), "auc": float("nan")})

            # Fusion: combine JEPA and FC predictions per subject
            jepa_subject_probs = {}
            if jepa_rows:
                for row in jepa_rows:
                    jepa_subject_probs[row["subject_id"]] = row["prob"]

            for harmonize_name, harmonize in [
                ("Static FC + JEPA", None),
                ("Static FC + ComBat + JEPA", "combat"),
                ("Static FC + CovBat + JEPA", "covbat"),
            ]:
                fc_m_result, fc_rows = run_fc_loso(
                    fc_features_m, fc_records_m, fc_labels_m,
                    seed=seed, harmonize=harmonize.replace(" + JEPA", "") if harmonize else None,
                )
                if fc_rows:
                    fused_rows = []
                    for row in fc_rows:
                        jepa_p = jepa_subject_probs.get(row["subject_id"])
                        if jepa_p is not None:
                            fused_prob = fc_weight * row["prob"] + jepa_weight * jepa_p
                            fused_rows.append({
                                "subject_id": row["subject_id"],
                                "site": row["site"],
                                "y": row["y"],
                                "prob": float(fused_prob),
                                "pred": int(fused_prob >= 0.5),
                            })

                    if fused_rows:
                        yt = np.array([r["y"] for r in fused_rows])
                        yp = np.array([r["pred"] for r in fused_rows])
                        ypr = np.array([r["prob"] for r in fused_rows])
                        bacc = float(balanced_accuracy_score(yt, yp))
                        try:
                            auc = float(roc_auc_score(yt, ypr))
                        except ValueError:
                            auc = float("nan")

                        # Per-site
                        site_baccs = []
                        for site in np.unique([r["site"] for r in fused_rows]):
                            sr = [r for r in fused_rows if r["site"] == site]
                            st = np.array([r["y"] for r in sr])
                            sp = np.array([r["pred"] for r in sr])
                            if len(np.unique(st)) >= 2:
                                site_baccs.append(float(balanced_accuracy_score(st, sp)))

                        seed_summaries[harmonize_name].append({
                            "bacc": bacc,
                            "auc": auc,
                            "site_mean_bacc": float(np.mean(site_baccs)) if site_baccs else float("nan"),
                            "n_valid_sites": len(set(r["site"] for r in fused_rows)),
                        })
                    else:
                        seed_summaries[harmonize_name].append({"bacc": float("nan"), "auc": float("nan")})
                else:
                    seed_summaries[harmonize_name].append({"bacc": float("nan"), "auc": float("nan")})

        # Aggregate
        ds_results = {}
        for method_name in [m[0] for m in methods]:
            summaries = seed_summaries[method_name]
            baccs = [s["bacc"] for s in summaries if s is not None]
            aucs = [s.get("auc", float("nan")) for s in summaries if s is not None]
            site_baccs = [s.get("site_mean_bacc", float("nan")) for s in summaries if s is not None]
            ds_results[method_name] = {
                "bacc_mean": float(np.nanmean(baccs)),
                "bacc_std": float(np.nanstd(baccs)),
                "auc_mean": float(np.nanmean(aucs)),
                "auc_std": float(np.nanstd(aucs)),
                "site_mean_bacc_mean": float(np.nanmean(site_baccs)),
                "site_mean_bacc_std": float(np.nanstd(site_baccs)),
                "n_valid_sites": summaries[0].get("n_valid_sites", 0) if summaries else 0,
            }

        all_results[ds_name] = {
            "display_name": spec.display_name,
            "methods": ds_results,
        }

    return all_results


def build_markdown(results: dict, checkpoint_path: str, fc_weight: float = 0.5) -> str:
    ckpt = Path(checkpoint_path)
    ckpt_label = f"{ckpt.parent.name}/{ckpt.name}"
    lines = [
        f"# {ckpt_label} FC+JEPA Subject-Vote Fusion",
        "",
        f"Checkpoint: `{checkpoint_path}`",
        "",
        "Diagnosis uses leave-one-site-out subject-level evaluation on matched FC/raw scans.",
        "FC-only and FC+JEPA rows use the same subject-level LOSO protocol.",
        "",
        f"Late fusion uses fixed probabilities: `p = {fc_weight:.2f} * p_fc + {1 - fc_weight:.2f} * p_jepa`.",
        "",
        "| Method | Dx LOSO bAcc | Dx LOSO ROC-AUC | Mean Per-Site Dx bAcc | Valid Sites |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for ds_name in ["abide1", "abide2", "srpbs"]:
        if ds_name not in results:
            continue
        r = results[ds_name]
        lines.append(f"\n## {r['display_name']}\n")
        for method_name, m in r["methods"].items():
            method_label = method_name
            if method_name == "JEPA":
                method_label = f"JEPA ({ckpt_label})"
            elif "+ JEPA" in method_name:
                method_label = f"{method_name} (late fusion, fc_weight={fc_weight:.2f})"
                if "ComBat" in method_name or "CovBat" in method_name:
                    method_label += "; transductive LOSO)"
                else:
                    method_label += ")"
            elif "ComBat" in method_name or "CovBat" in method_name:
                method_label = f"{method_name} (transductive LOSO)"

            lines.append(
                f"| {method_label} "
                f"| {100 * m['bacc_mean']:.1f} ± {100 * m['bacc_std']:.1f} "
                f"| {100 * m['auc_mean']:.1f} ± {100 * m['auc_std']:.1f} "
                f"| {100 * m['site_mean_bacc_mean']:.1f} ± {100 * m['site_mean_bacc_std']:.1f} "
                f"| {m['n_valid_sites']} |"
            )
        lines.append(f"| Chance | 50.0 | 50.0 | 50.0 | {r['methods'].get('JEPA', {}).get('n_valid_sites', 0)} |")

    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(description="FC+JEPA late fusion subject-vote evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--fc-weight", type=float, default=0.5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--save-json", default=None)
    p.add_argument("--save-md", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    seed_everything(42, deterministic=True)

    results = run_fusion_evaluation(
        args.checkpoint,
        fc_weight=args.fc_weight,
        device=args.device,
        probe_seeds=tuple(args.seeds),
    )

    logger = logging.getLogger("Fusion-SubjectVote")

    ckpt = Path(args.checkpoint)
    ckpt_tag = f"{ckpt.parent.name}_{ckpt.stem}"
    out_dir = Path("results/reports/tables/jepa_fusion")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.save_json or str(out_dir / f"{ckpt_tag}_fusion_subject_vote.json")
    md_path = args.save_md or str(out_dir / f"{ckpt_tag}_fusion_subject_vote.md")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved JSON: %s", json_path)

    md_content = build_markdown(results, args.checkpoint, fc_weight=args.fc_weight)
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info("Saved Markdown: %s", md_path)


if __name__ == "__main__":
    main()
