import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from .dataset_utils import default_eval_dataset_order, matched_records_for_preset
    from .protocol import filter_records_for_task, get_dataset_preset
    from ..jepa_clean.evaluate_subject_vote import compute_binary_metrics
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.dataset_utils import default_eval_dataset_order, matched_records_for_preset
    from src.jepa_fusion.protocol import filter_records_for_task, get_dataset_preset
    from src.jepa_clean.evaluate_subject_vote import compute_binary_metrics


DATASET_CONFIG = {
    "abide": {
        "dataset_label": "ABIDE-I + ABIDE-II",
        "task": "hc_asd",
    },
    "abide1": {
        "dataset_label": "ABIDE-I",
        "task": "hc_asd",
    },
    "abide2": {
        "dataset_label": "ABIDE-II",
        "task": "hc_asd",
    },
    "srpbs": {
        "dataset_label": "SRPBS-1600",
        "task": "hc_disorder",
    },
}

FC_METHOD_LABELS = {
    "fc": "Static FC + JEPA (concat fusion)",
    "fc_combat": "Static FC + ComBat + JEPA (concat fusion)",
    "fc_covbat": "Static FC + CovBat + JEPA (concat fusion)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FC+JEPA concatenated fusion.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="all", choices=["all", "abide", "abide1", "abide2", "srpbs"])
    parser.add_argument("--fc-methods", default="fc,fc_combat,fc_covbat")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def parse_csv_list(text: str):
    return [part.strip() for part in text.split(",") if part.strip()]


def format_pct(mean: float, std: float):
    return f"{100.0 * mean:.1f} ± {100.0 * std:.1f}"


def default_outputs(checkpoint: str):
    ckpt = Path(checkpoint)
    stem = f"{ckpt.parent.name}_{ckpt.stem}_concat_fusion"
    out_dir = Path("results/reports/tables/jepa_fusion")
    return out_dir / f"{stem}.json", out_dir / f"{stem}.md"


def load_cache_payloads(cache_glob: str, predicate):
    matches = []
    for json_path in sorted(Path("results/cache/jepa_fusion").glob(cache_glob)):
        payload = json.loads(json_path.read_text()).get("cache_payload", {})
        if predicate(payload):
            matches.append((json_path, payload))
    if not matches:
        raise FileNotFoundError(f"No cache metadata matched {cache_glob}")
    return matches


def select_fc_cache(dataset: str, task: str, subsample_seed: int):
    matches = load_cache_payloads(
        "fc_subject_vote_*.json",
        lambda p: p.get("dataset") == dataset
        and p.get("task") == task
        and bool(p.get("mixed_sites_only")) is True
        and int(p.get("subsample_seed", -1)) == subsample_seed,
    )
    json_path, _ = matches[0]
    return json_path.with_suffix(".npz")


def select_jepa_cache(checkpoint: str, dataset: str, task: str, subsample_seed: int):
    matches = load_cache_payloads(
        "jepa_subject_vote_*.json",
        lambda p: p.get("checkpoint") == checkpoint
        and p.get("dataset") == dataset
        and p.get("task") == task
        and bool(p.get("mixed_sites_only")) is True
        and int(p.get("subsample_seed", -1)) == subsample_seed,
    )
    json_path, _ = matches[0]
    return json_path.with_suffix(".npz")


def diagnosis_to_binary(dataset: str, diagnosis: str):
    if dataset in {"abide1", "abide2"}:
        if diagnosis == "Healthy Control":
            return 0
        if diagnosis == "Autism Spectrum Disorder":
            return 1
        return None
    if diagnosis == "unknown":
        return None
    return 0 if diagnosis == "Healthy Control" else 1


def load_filtered_records(dataset: str):
    preset = get_dataset_preset(dataset)
    task_result = filter_records_for_task(
        matched_records_for_preset(preset),
        preset.default_task,
        mixed_sites_only=preset.default_mixed_sites_only,
    )
    rows = []
    for record, y in zip(task_result.records, task_result.y.tolist()):
        rows.append(
            {
                "scan_id": record.sample_id,
                "subject_id": record.subject_id,
                "site": record.site,
                "site_batch": record.site_batch,
                "y": int(y),
            }
        )
    return rows


def aggregate_scan_embeddings(jepa_npz: Path):
    data = np.load(jepa_npz, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32, copy=False)
    scan_ids = data["scan_id"]
    acc = defaultdict(list)
    for idx, scan_id in enumerate(scan_ids):
        acc[str(scan_id)].append(embeddings[idx])
    return {scan_id: np.mean(np.stack(v, axis=0), axis=0) for scan_id, v in acc.items()}


def build_concat_representation(dataset: str, checkpoint: str, fc_methods, subsample_seed: int):
    cfg = DATASET_CONFIG[dataset]
    rows = load_filtered_records(dataset)
    fc_npz = np.load(select_fc_cache(dataset, cfg["task"], subsample_seed), allow_pickle=True)
    jepa_scan_map = aggregate_scan_embeddings(select_jepa_cache(checkpoint, dataset, cfg["task"], subsample_seed))

    if fc_npz[fc_methods[0]].shape[0] != len(rows):
        raise RuntimeError(
            f"Record/cache mismatch for {dataset}: rows={len(rows)} fc={fc_npz[fc_methods[0]].shape[0]}"
        )

    base = {
        "subject_id": np.asarray([row["subject_id"] for row in rows]),
        "site": np.asarray([row["site"] for row in rows]),
        "site_batch": np.asarray([row["site_batch"] for row in rows]),
        "y": np.asarray([row["y"] for row in rows], dtype=np.int64),
        "scan_id": np.asarray([row["scan_id"] for row in rows]),
    }

    jepa_features = []
    missing = []
    for scan_id in base["scan_id"]:
        emb = jepa_scan_map.get(str(scan_id))
        if emb is None:
            missing.append(str(scan_id))
        else:
            jepa_features.append(emb)
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Missing JEPA scan embeddings for {len(missing)} scans in {dataset}: {preview}")
    jepa_features = np.stack(jepa_features, axis=0).astype(np.float32, copy=False)

    features = {}
    for method in fc_methods:
        features[method] = np.concatenate([fc_npz[method].astype(np.float32, copy=False), jepa_features], axis=1)
    return base, features


def make_probe(seed: int, x_train: np.ndarray):
    n_components = min(50, x_train.shape[1], max(1, x_train.shape[0] - 1))
    steps = [StandardScaler()]
    if n_components >= 1:
        steps.append(PCA(n_components=n_components, random_state=seed))
    steps.append(
        LogisticRegression(
            C=0.1,
            max_iter=5000,
            random_state=seed,
            class_weight="balanced",
        )
    )
    return make_pipeline(*steps)


def run_grouped_cv(features, y, groups, seed, n_splits, return_auc):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = np.zeros(len(y), dtype=np.int64)
    y_prob = np.full(len(y), np.nan, dtype=np.float64)

    for train_idx, test_idx in splitter.split(features, y, groups):
        clf = make_probe(seed, features[train_idx])
        clf.fit(features[train_idx], y[train_idx])
        y_pred[test_idx] = clf.predict(features[test_idx])
        if return_auc:
            if hasattr(clf, "predict_proba"):
                y_prob[test_idx] = clf.predict_proba(features[test_idx])[:, 1]
            else:
                scores = clf.decision_function(features[test_idx])
                y_prob[test_idx] = 1.0 / (1.0 + np.exp(-scores))

    metrics = {"bacc": float(balanced_accuracy_score(y, y_pred))}
    if return_auc:
        try:
            metrics["auc"] = float(roc_auc_score(y, y_prob))
        except ValueError:
            metrics["auc"] = float("nan")
    return metrics


def run_subject_loso(features, y, sites, subjects, seed):
    subject_rows = []
    for heldout_site in sorted(np.unique(sites)):
        test_mask = sites == heldout_site
        train_mask = ~test_mask
        clf = make_probe(seed, features[train_mask])
        clf.fit(features[train_mask], y[train_mask])
        if hasattr(clf, "predict_proba"):
            scan_prob = clf.predict_proba(features[test_mask])[:, 1]
        else:
            scores = clf.decision_function(features[test_mask])
            scan_prob = 1.0 / (1.0 + np.exp(-scores))

        by_subject = defaultdict(list)
        label_by_subject = {}
        for subject_id, site, target, prob in zip(
            subjects[test_mask], sites[test_mask], y[test_mask], scan_prob
        ):
            by_subject[str(subject_id)].append(float(prob))
            label_by_subject[str(subject_id)] = (int(target), str(site))

        for subject_id, probs in by_subject.items():
            target, site = label_by_subject[subject_id]
            prob = float(np.mean(probs))
            pred = int(prob >= 0.5)
            subject_rows.append(
                {
                    "subject_id": subject_id,
                    "site": site,
                    "y": target,
                    "prob": prob,
                    "pred": pred,
                    "n_scans": len(probs),
                }
            )

    y_true = np.asarray([row["y"] for row in subject_rows], dtype=np.int64)
    y_pred = np.asarray([row["pred"] for row in subject_rows], dtype=np.int64)
    y_prob = np.asarray([row["prob"] for row in subject_rows], dtype=np.float64)
    return compute_binary_metrics(y_true, y_pred, y_prob), subject_rows


def summarize_runs(runs, keys):
    out = {}
    for key in keys:
        vals = np.asarray([run[key] for run in runs], dtype=np.float64)
        out[key] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
        }
    return out


def evaluate_dataset(dataset, checkpoint, fc_methods, seeds, folds, subsample_seed):
    base, features_by_method = build_concat_representation(dataset, checkpoint, fc_methods, subsample_seed)
    site_encoder = LabelEncoder()
    y_site = site_encoder.fit_transform(base["site_batch"])

    result = {
        "config": DATASET_CONFIG[dataset],
        "meta": {
            "n_scans": int(len(base["y"])),
            "n_subjects": int(len(np.unique(base["subject_id"]))),
            "n_sites": int(len(site_encoder.classes_)),
            "site_labels": [str(x) for x in site_encoder.classes_.tolist()],
        },
        "runs": {},
        "summary": {},
    }

    for method in fc_methods:
        method_runs = []
        x = features_by_method[method]
        for seed in seeds:
            site_metrics = run_grouped_cv(x, y_site, base["subject_id"], seed, folds, return_auc=False)
            dx_cv_metrics = run_grouped_cv(x, base["y"], base["subject_id"], seed, folds, return_auc=True)
            dx_loso_metrics, subject_rows = run_subject_loso(
                x, base["y"], base["site"], base["subject_id"], seed
            )
            method_runs.append(
                {
                    "seed": int(seed),
                    "grouped_cv_site": site_metrics,
                    "grouped_cv_diagnosis": dx_cv_metrics,
                    "loso_diagnosis": dx_loso_metrics,
                    "subject_rows": subject_rows,
                }
            )
        result["runs"][method] = method_runs
        result["summary"][method] = {
            "grouped_cv_site": summarize_runs(
                [run["grouped_cv_site"] for run in method_runs], ["bacc"]
            ),
            "grouped_cv_diagnosis": summarize_runs(
                [run["grouped_cv_diagnosis"] for run in method_runs], ["bacc", "auc"]
            ),
            "loso_diagnosis": summarize_runs(
                [run["loso_diagnosis"] for run in method_runs], ["bacc", "auc"]
            ),
        }
    return result


def build_markdown(payload):
    lines = ["# Concat Fusion Summary", ""]
    lines.append(
        "Feature fusion uses scan-level concatenation of the FC representation and the mean JEPA scan embedding, "
        "followed by the same `StandardScaler -> PCA -> LogisticRegression` downstream probe family."
    )
    lines.append("")
    for dataset in payload["order"]:
        block = payload["datasets"][dataset]
        lines.append(f"## {block['config']['dataset_label']}")
        lines.append("")
        lines.append("| Method | Grouped-CV Site bAcc | Grouped-CV Dx bAcc | Grouped-CV Dx AUC | LOSO Dx bAcc | LOSO Dx AUC |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for method in payload["fc_methods"]:
            summary = block["summary"][method]
            lines.append(
                "| "
                + " | ".join(
                    [
                        FC_METHOD_LABELS[method],
                        format_pct(summary["grouped_cv_site"]["bacc"]["mean"], summary["grouped_cv_site"]["bacc"]["std"]),
                        format_pct(summary["grouped_cv_diagnosis"]["bacc"]["mean"], summary["grouped_cv_diagnosis"]["bacc"]["std"]),
                        format_pct(summary["grouped_cv_diagnosis"]["auc"]["mean"], summary["grouped_cv_diagnosis"]["auc"]["std"]),
                        format_pct(summary["loso_diagnosis"]["bacc"]["mean"], summary["loso_diagnosis"]["bacc"]["std"]),
                        format_pct(summary["loso_diagnosis"]["auc"]["mean"], summary["loso_diagnosis"]["auc"]["std"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    fc_methods = parse_csv_list(args.fc_methods)
    seeds = [int(seed) for seed in parse_csv_list(args.seeds)]
    if args.dataset == "all":
        dataset_order = default_eval_dataset_order(args.dataset)
    else:
        dataset_order = [args.dataset]

    output_json, output_md = default_outputs(args.checkpoint)
    if args.output_json:
        output_json = Path(args.output_json)
    if args.output_md:
        output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint": args.checkpoint,
        "fc_methods": fc_methods,
        "seeds": seeds,
        "order": dataset_order,
        "datasets": {},
    }
    for dataset in dataset_order:
        payload["datasets"][dataset] = evaluate_dataset(
            dataset=dataset,
            checkpoint=args.checkpoint,
            fc_methods=fc_methods,
            seeds=seeds,
            folds=args.folds,
            subsample_seed=args.subsample_seed,
        )

    output_json.write_text(json.dumps(payload, indent=2))
    output_md.write_text(build_markdown(payload))
    print(f"[concat-fusion] wrote {output_json}")
    print(f"[concat-fusion] wrote {output_md}")


if __name__ == "__main__":
    main()
