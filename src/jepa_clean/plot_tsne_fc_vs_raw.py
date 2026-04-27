import argparse
import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .data import resample_time_window


DEFAULT_FC_DIR = "preprocessed_data_fir/FC_abide_full"
DEFAULT_RAW_DIR = "preprocessed_data_fir/full/roi_timeseries_abide_full"
DEFAULT_OUTPUT = "results/figures/abide_fc_vs_raw_tsne_by_site.png"
DEFAULT_EMBEDDING_JSON = "results/figures/abide_fc_vs_raw_tsne_embeddings.json"


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    subject: str
    run: str
    site: str
    diagnosis: str
    age: float | None
    sex: str
    fc_path: Path
    raw_path: Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare site structure in ABIDE FC and raw ROI time-series with matched 2D embeddings."
    )
    parser.add_argument("--fc-dir", default=DEFAULT_FC_DIR)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--triangle", choices=["upper", "lower"], default="upper")
    parser.add_argument(
        "--reorder",
        choices=["none", "hemi", "hemi_network"],
        default="hemi_network",
        help="ROI reordering used before FC vectorization or raw flattening.",
    )
    parser.add_argument(
        "--raw-target-timepoints",
        type=int,
        default=45,
        help="Resample each raw ROI time series to this many timepoints before flattening.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument(
        "--embedding-method",
        choices=["tsne", "pca"],
        default="tsne",
        help="2D embedding used for visualization. Site-probe numbers are still computed on PCA features.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--label-by", choices=["site", "diagnosis"], default="site")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-json", default=DEFAULT_EMBEDDING_JSON)
    parser.add_argument(
        "--site-probe-folds",
        type=int,
        default=5,
        help="Folds used for site and diagnosis probes. Set to 0 to skip.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def get_meta_value(meta: dict, *keys, default="unknown"):
    for key in keys:
        value = meta.get(key)
        if value is not None and value != "":
            return value
    return default


def load_matched_records(fc_dir: Path, raw_dir: Path):
    fc_paths = sorted(fc_dir.glob("*_fc.npy"))
    raw_map = {path.name.replace("_roi_ts.npy", ""): path for path in raw_dir.glob("*_roi_ts.npy")}

    records = []
    for fc_path in fc_paths:
        sample_id = fc_path.name.replace("_fc.npy", "")
        raw_path = raw_map.get(sample_id)
        if raw_path is None:
            continue

        fc_meta = load_json(fc_path.with_name(fc_path.name.replace("_fc.npy", "_fc_meta.json")))
        raw_meta = load_json(raw_path.with_name(raw_path.name.replace("_roi_ts.npy", "_meta.json")))
        meta = dict(raw_meta)
        meta.update({k: v for k, v in fc_meta.items() if v is not None})

        diagnosis = str(get_meta_value(meta, "diag_name", "dx_name", default="unknown"))
        records.append(
            SampleRecord(
                sample_id=sample_id,
                subject=str(get_meta_value(meta, "subject", default="unknown")),
                run=str(get_meta_value(meta, "run", default="")),
                site=str(get_meta_value(meta, "site", default="unknown")),
                diagnosis=diagnosis,
                age=(
                    None
                    if get_meta_value(meta, "age", default=None) in (None, "", "unknown")
                    else float(get_meta_value(meta, "age", default=None))
                ),
                sex=str(get_meta_value(meta, "sex", default="unknown")),
                fc_path=fc_path,
                raw_path=raw_path,
            )
        )

    return records


def maybe_subsample_records(records: list[SampleRecord], max_samples: int | None, seed: int):
    if max_samples is None or len(records) <= max_samples:
        return records

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(records))[:max_samples]
    return [records[idx] for idx in indices]


def _parse_schaefer_label(label: str):
    parts = label.split("_")
    hemi = parts[1] if len(parts) > 1 else "UNK"
    network = parts[2] if len(parts) > 2 else "UNK"
    match = re.search(r"_(\d+)$", label)
    parcel_index = int(match.group(1)) if match else 0
    return hemi, network, parcel_index


def get_roi_permutation(mode: str):
    atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    labels = [lbl.decode("utf-8") if isinstance(lbl, bytes) else str(lbl) for lbl in atlas["labels"]]
    indices = list(range(len(labels)))

    if mode == "none":
        return np.asarray(indices, dtype=np.int64), labels

    if mode == "hemi":
        order = sorted(indices, key=lambda idx: (0 if _parse_schaefer_label(labels[idx])[0] == "LH" else 1, idx))
        return np.asarray(order, dtype=np.int64), labels

    if mode == "hemi_network":
        order = sorted(
            indices,
            key=lambda idx: (
                0 if _parse_schaefer_label(labels[idx])[0] == "LH" else 1,
                _parse_schaefer_label(labels[idx])[1],
                _parse_schaefer_label(labels[idx])[2],
            ),
        )
        return np.asarray(order, dtype=np.int64), labels

    raise ValueError(f"Unsupported reorder mode: {mode}")


def vectorize_fc(fc: np.ndarray, triangle: str):
    if fc.shape[0] != fc.shape[1]:
        raise ValueError(f"Expected square FC matrix, got {fc.shape}")
    tri_fn = np.triu_indices if triangle == "upper" else np.tril_indices
    tri = tri_fn(fc.shape[0], k=1 if triangle == "upper" else -1)
    return fc[tri].astype(np.float32, copy=False)


def build_fc_features(records, permutation: np.ndarray, triangle: str):
    features = []
    for record in records:
        fc = np.load(record.fc_path).astype(np.float32, copy=False)
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        if permutation is not None:
            fc = fc[np.ix_(permutation, permutation)]
        features.append(vectorize_fc(fc, triangle=triangle))
    return np.stack(features, axis=0)


def build_raw_features(records, permutation: np.ndarray, raw_target_timepoints: int):
    features = []
    for record in records:
        ts = np.load(record.raw_path).astype(np.float32, copy=False)
        ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
        if permutation is not None:
            ts = ts[:, permutation]
        ts = resample_time_window(ts, raw_target_timepoints)
        features.append(ts.reshape(-1))
    return np.stack(features, axis=0)


def compute_volume_windows(n_timepoints: int, window_size: int, stride: int):
    if window_size < 1:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}")
    if n_timepoints < window_size:
        return [(0, n_timepoints)]

    starts = list(range(0, n_timepoints - window_size + 1, stride))
    if not starts:
        starts = [0]
    last_start = n_timepoints - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + window_size) for start in starts]


def compute_second_windows(n_timepoints: int, tr: float, window_seconds: float, stride_seconds: float):
    native_window = max(4, int(round(window_seconds / max(tr, 1e-6))))
    native_window = min(native_window, n_timepoints)
    if n_timepoints <= native_window:
        return [(0, n_timepoints)]

    stride = max(1, int(round(stride_seconds / max(tr, 1e-6))))
    starts = list(range(0, n_timepoints - native_window + 1, stride))
    last_start = n_timepoints - native_window
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + native_window) for start in starts]


def build_raw_window_bags(
    records,
    permutation: np.ndarray,
    target_timepoints: int,
    window_size: int | None = None,
    stride: int | None = None,
    window_seconds: float | None = None,
    stride_seconds: float | None = None,
):
    use_volume_windows = window_size is not None or stride is not None
    use_second_windows = window_seconds is not None or stride_seconds is not None
    if use_volume_windows == use_second_windows:
        raise ValueError(
            "Specify exactly one raw windowing scheme: either volume-based or second-based."
        )

    if use_volume_windows:
        if window_size is None or stride is None:
            raise ValueError("Both window_size and stride are required for volume-based windows.")
    else:
        if window_seconds is None or stride_seconds is None:
            raise ValueError(
                "Both window_seconds and stride_seconds are required for second-based windows."
            )

    kept_records = []
    window_bags = []

    for record in records:
        ts = np.load(record.raw_path).astype(np.float32, copy=False)
        ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
        if permutation is not None:
            ts = ts[:, permutation]

        if use_volume_windows:
            spans = compute_volume_windows(ts.shape[0], window_size=window_size, stride=stride)
        else:
            raw_meta = load_json(record.raw_path.with_name(record.raw_path.name.replace("_roi_ts.npy", "_meta.json")))
            tr = float(raw_meta.get("tr", 2.0) or 2.0)
            spans = compute_second_windows(
                ts.shape[0],
                tr=tr,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
            )

        windows = [resample_time_window(ts[start:end], target_timepoints).reshape(-1) for start, end in spans]
        window_bags.append(np.stack(windows, axis=0).astype(np.float32, copy=False))
        kept_records.append(record)

    return kept_records, window_bags


def apply_combat_harmonization(
    features: np.ndarray,
    sites: list[str],
    categorical_covariates: dict[str, list[str]] | None = None,
    *,
    eb: bool = True,
    parametric: bool = True,
    mean_only: bool = False,
    verbose: bool = False,
):
    """Apply neuroCombat to sample-level features using site as the batch variable."""
    if len(features) != len(sites):
        raise ValueError("features and sites must have matching sample counts.")

    try:
        import pandas as pd
        from neuroCombat import neuroCombat
    except ImportError as exc:
        raise ImportError(
            "neuroCombat is required for --include-fc-combat. Install it with `pip install neuroCombat`."
        ) from exc

    covars_dict = {"site": [str(site) for site in sites]}
    categorical_cols = []
    for name, values in (categorical_covariates or {}).items():
        if len(values) != len(sites):
            raise ValueError(f"ComBat covariate {name!r} length does not match sites.")
        covars_dict[name] = [str(value) for value in values]
        categorical_cols.append(name)

    covars = pd.DataFrame(covars_dict)
    data = np.asarray(features, dtype=np.float32).T.copy()

    stream = None if verbose else io.StringIO()
    redirect_context = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(stream)
    with redirect_context:
        result = neuroCombat(
            dat=data,
            covars=covars,
            batch_col="site",
            categorical_cols=categorical_cols,
            eb=eb,
            parametric=parametric,
            mean_only=mean_only,
        )

    harmonized = np.asarray(result["data"], dtype=np.float32).T
    return np.nan_to_num(harmonized, nan=0.0, posinf=0.0, neginf=0.0)


def apply_covbat_harmonization(
    features: np.ndarray,
    sites: list[str],
    *,
    n_pc: int = 50,
    pct_var: float = 0.95,
    verbose: bool = False,
):
    """Apply the local CovBat implementation to sample-level features."""
    if len(features) != len(sites):
        raise ValueError("features and sites must have matching sample counts.")

    try:
        import pandas as pd
        from covbat import covbat
    except ImportError as exc:
        raise ImportError(
            "The local covbat.py module and its dependencies are required for --include-fc-covbat."
        ) from exc

    sample_columns = list(range(len(sites)))
    data = pd.DataFrame(np.asarray(features, dtype=np.float32).T, columns=sample_columns)
    batch = pd.Series([str(site) for site in sites], index=sample_columns)
    effective_n_pc = int(n_pc)
    if effective_n_pc > 0:
        effective_n_pc = min(effective_n_pc, features.shape[0], features.shape[1])

    stream = None if verbose else io.StringIO()
    redirect_context = (
        contextlib.nullcontext()
        if verbose
        else contextlib.ExitStack()
    )
    if verbose:
        with redirect_context:
            harmonized = covbat(data, batch, pct_var=pct_var, n_pc=effective_n_pc)
    else:
        with redirect_context as stack:
            stack.enter_context(contextlib.redirect_stdout(stream))
            stack.enter_context(contextlib.redirect_stderr(stream))
            harmonized = covbat(data, batch, pct_var=pct_var, n_pc=effective_n_pc)

    harmonized_arr = np.asarray(harmonized, dtype=np.float32).T
    return np.nan_to_num(harmonized_arr, nan=0.0, posinf=0.0, neginf=0.0)


def _make_linear_probe(features: np.ndarray, train_count: int, seed: int):
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=min(50, features.shape[1], train_count - 1), random_state=seed),
        LogisticRegression(max_iter=10000),
    )


def _effective_group_folds(y: np.ndarray, groups: np.ndarray, folds: int):
    group_counts = []
    for cls in np.unique(y):
        cls_groups = np.unique(groups[y == cls])
        group_counts.append(len(cls_groups))
    return min(folds, min(group_counts))


def _aggregate_window_predictions(clf, window_features: np.ndarray, aggregation: str):
    if aggregation not in {"mean_proba", "vote"}:
        raise ValueError(f"Unsupported aggregation mode: {aggregation}")

    if aggregation == "mean_proba":
        mean_proba = clf.predict_proba(window_features).mean(axis=0)
        return int(np.argmax(mean_proba))

    window_preds = clf.predict(window_features)
    vote_counts = np.bincount(window_preds.astype(np.int64), minlength=len(clf.classes_))
    winners = np.flatnonzero(vote_counts == vote_counts.max())
    if len(winners) == 1:
        return int(winners[0])

    mean_proba = clf.predict_proba(window_features).mean(axis=0)
    tied_scores = mean_proba[winners]
    return int(winners[np.argmax(tied_scores)])


def compute_embedding(
    features: np.ndarray,
    method: str,
    pca_components: int,
    perplexity: float,
    seed: int,
):
    if len(features) < 3:
        raise ValueError("Need at least 3 samples for visualization.")
    n_samples = features.shape[0]
    pca_dim = min(pca_components, features.shape[1], n_samples - 1)

    pca = PCA(n_components=pca_dim, random_state=seed)
    reduced = pca.fit_transform(StandardScaler().fit_transform(features))

    if method == "pca":
        embedding = reduced[:, :2]
        return embedding, reduced

    perplexity = min(perplexity, max(2.0, (n_samples - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    )
    embedding = tsne.fit_transform(reduced)
    return embedding, reduced


def run_probe(features: np.ndarray, labels: list[str], seed: int, folds: int, task_name: str):
    if folds <= 1:
        return None

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    min_class_count = min(np.bincount(y))
    effective_folds = min(folds, int(min_class_count))
    if effective_folds < 2:
        return None

    cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    preds = np.empty_like(y)
    for train_idx, test_idx in cv.split(features, y):
        clf = _make_linear_probe(features, train_count=len(train_idx), seed=seed)
        clf.fit(features[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(features[test_idx])

    return {
        "task": task_name,
        "accuracy": float(np.mean(preds == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "n_classes": int(len(np.unique(y))),
        "n_samples": int(len(y)),
        "folds": int(effective_folds),
        "classes": [str(value) for value in encoder.classes_.tolist()],
    }


def run_grouped_probe(
    features: np.ndarray,
    labels: list[str],
    groups: list[str],
    seed: int,
    folds: int,
    task_name: str,
):
    if folds <= 1:
        return None
    if len(features) != len(labels) or len(labels) != len(groups):
        raise ValueError("features, labels, and groups must have matching lengths.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    groups_arr = np.asarray(groups)
    effective_folds = _effective_group_folds(y, groups_arr, folds)
    if effective_folds < 2:
        return None

    cv = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    preds = np.empty_like(y)
    for train_idx, test_idx in cv.split(features, y, groups=groups_arr):
        clf = _make_linear_probe(features, train_count=len(train_idx), seed=seed)
        clf.fit(features[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(features[test_idx])

    return {
        "task": task_name,
        "accuracy": float(np.mean(preds == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "n_classes": int(len(np.unique(y))),
        "n_samples": int(len(y)),
        "folds": int(effective_folds),
        "classes": [str(value) for value in encoder.classes_.tolist()],
        "cv": "stratified_group_kfold",
        "n_groups": int(len(np.unique(groups_arr))),
    }


def run_window_bag_probe(
    window_bags: list[np.ndarray],
    labels: list[str],
    seed: int,
    folds: int,
    task_name: str,
    aggregation: str = "mean_proba",
):
    if folds <= 1:
        return None
    if len(window_bags) != len(labels):
        raise ValueError(
            f"Window bag count ({len(window_bags)}) did not match label count ({len(labels)})"
        )

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    min_class_count = min(np.bincount(y))
    effective_folds = min(folds, int(min_class_count))
    if effective_folds < 2:
        return None

    cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    preds = np.empty_like(y)
    sample_indices = np.arange(len(y), dtype=np.int64)

    for train_idx, test_idx in cv.split(sample_indices, y):
        train_windows = np.concatenate([window_bags[idx] for idx in train_idx], axis=0)
        train_labels = np.concatenate(
            [
                np.full(window_bags[idx].shape[0], y[idx], dtype=np.int64)
                for idx in train_idx
            ],
            axis=0,
        )

        clf = _make_linear_probe(train_windows, train_count=len(train_windows), seed=seed)
        clf.fit(train_windows, train_labels)

        for sample_idx in test_idx:
            preds[sample_idx] = _aggregate_window_predictions(
                clf,
                window_bags[sample_idx],
                aggregation=aggregation,
            )

    window_counts = np.asarray([bag.shape[0] for bag in window_bags], dtype=np.int64)
    return {
        "task": task_name,
        "accuracy": float(np.mean(preds == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "n_classes": int(len(np.unique(y))),
        "n_samples": int(len(y)),
        "folds": int(effective_folds),
        "classes": [str(value) for value in encoder.classes_.tolist()],
        "aggregation": aggregation,
        "n_windows_total": int(window_counts.sum()),
        "mean_windows_per_sample": float(window_counts.mean()),
        "median_windows_per_sample": float(np.median(window_counts)),
    }


def run_window_bag_grouped_probe(
    window_bags: list[np.ndarray],
    labels: list[str],
    groups: list[str],
    seed: int,
    folds: int,
    task_name: str,
    aggregation: str = "mean_proba",
):
    if folds <= 1:
        return None
    if len(window_bags) != len(labels) or len(labels) != len(groups):
        raise ValueError("window_bags, labels, and groups must have matching lengths.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    groups_arr = np.asarray(groups)
    effective_folds = _effective_group_folds(y, groups_arr, folds)
    if effective_folds < 2:
        return None

    cv = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    preds = np.empty_like(y)
    sample_indices = np.arange(len(y), dtype=np.int64)

    for train_idx, test_idx in cv.split(sample_indices, y, groups=groups_arr):
        train_windows = np.concatenate([window_bags[idx] for idx in train_idx], axis=0)
        train_labels = np.concatenate(
            [
                np.full(window_bags[idx].shape[0], y[idx], dtype=np.int64)
                for idx in train_idx
            ],
            axis=0,
        )

        clf = _make_linear_probe(train_windows, train_count=len(train_windows), seed=seed)
        clf.fit(train_windows, train_labels)

        for sample_idx in test_idx:
            preds[sample_idx] = _aggregate_window_predictions(
                clf,
                window_bags[sample_idx],
                aggregation=aggregation,
            )

    window_counts = np.asarray([bag.shape[0] for bag in window_bags], dtype=np.int64)
    return {
        "task": task_name,
        "accuracy": float(np.mean(preds == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "n_classes": int(len(np.unique(y))),
        "n_samples": int(len(y)),
        "folds": int(effective_folds),
        "classes": [str(value) for value in encoder.classes_.tolist()],
        "aggregation": aggregation,
        "n_windows_total": int(window_counts.sum()),
        "mean_windows_per_sample": float(window_counts.mean()),
        "median_windows_per_sample": float(np.median(window_counts)),
        "cv": "stratified_group_kfold",
        "n_groups": int(len(np.unique(groups_arr))),
    }


def run_window_bag_loso_probe(
    window_bags: list[np.ndarray],
    labels: list[str],
    sites: list[str],
    seed: int,
    aggregation: str = "mean_proba",
):
    if len(window_bags) != len(labels) or len(labels) != len(sites):
        raise ValueError("window_bags, labels, and sites must have matching lengths.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    site_arr = np.asarray(sites)

    y_true_all = []
    y_pred_all = []
    fold_rows = []
    skipped_sites = []

    for held_out_site in sorted(np.unique(site_arr)):
        test_idx = np.flatnonzero(site_arr == held_out_site)
        train_idx = np.flatnonzero(site_arr != held_out_site)
        y_train = y[train_idx]
        y_test = y[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            skipped_sites.append(
                {
                    "site": str(held_out_site),
                    "train_classes": [str(v) for v in encoder.inverse_transform(np.unique(y_train)).tolist()],
                    "test_classes": [str(v) for v in encoder.inverse_transform(np.unique(y_test)).tolist()],
                    "test_size": int(len(test_idx)),
                    "reason": "train_or_test_missing_a_class",
                }
            )
            continue

        train_windows = np.concatenate([window_bags[idx] for idx in train_idx], axis=0)
        train_labels = np.concatenate(
            [
                np.full(window_bags[idx].shape[0], y[idx], dtype=np.int64)
                for idx in train_idx
            ],
            axis=0,
        )
        clf = _make_linear_probe(train_windows, train_count=len(train_windows), seed=seed)
        clf.fit(train_windows, train_labels)

        y_pred = np.asarray(
            [
                _aggregate_window_predictions(clf, window_bags[idx], aggregation=aggregation)
                for idx in test_idx
            ],
            dtype=np.int64,
        )

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        fold_rows.append(
            {
                "site": str(held_out_site),
                "n_test": int(len(test_idx)),
                "class_counts": {
                    str(label): int(np.sum(y_test == cls_idx))
                    for cls_idx, label in enumerate(encoder.classes_)
                    if np.sum(y_test == cls_idx) > 0
                },
                "accuracy": float(np.mean(y_pred == y_test)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            }
        )

    if not y_true_all:
        raise RuntimeError("No valid LOSO folds remained after filtering.")

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    window_counts = np.asarray([bag.shape[0] for bag in window_bags], dtype=np.int64)

    return {
        "seed": int(seed),
        "n_classes": int(len(encoder.classes_)),
        "classes": [str(value) for value in encoder.classes_.tolist()],
        "n_samples_total": int(len(y)),
        "n_samples_valid": int(len(y_true)),
        "n_valid_sites": int(len(fold_rows)),
        "n_skipped_sites": int(len(skipped_sites)),
        "valid_sites": [row["site"] for row in fold_rows],
        "skipped_sites": skipped_sites,
        "pooled_accuracy": float(np.mean(y_pred == y_true)),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mean_fold_accuracy": float(np.mean([row["accuracy"] for row in fold_rows])),
        "mean_fold_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in fold_rows])),
        "folds": fold_rows,
        "aggregation": aggregation,
        "n_windows_total": int(window_counts.sum()),
        "mean_windows_per_sample": float(window_counts.mean()),
        "median_windows_per_sample": float(np.median(window_counts)),
        "cv": "leave_one_site_out",
    }


def make_palette(labels: list[str]):
    unique = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab20", len(unique))
    return {label: cmap(i) for i, label in enumerate(unique)}


def plot_embeddings(
    fc_embedding: np.ndarray,
    raw_embedding: np.ndarray,
    labels: list[str],
    title_label: str,
    output_path: Path,
    embedding_method: str,
):
    palette = make_palette(labels)
    unique_labels = list(palette)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    method_label = "t-SNE" if embedding_method == "tsne" else "PCA"
    for ax, embedding, title in [
        (axes[0], fc_embedding, "FC"),
        (axes[1], raw_embedding, "Raw ROI Time Series"),
    ]:
        for label in unique_labels:
            mask = np.array(labels) == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=26,
                alpha=0.8,
                c=[palette[label]],
                label=label,
                edgecolors="none",
            )
        ax.set_title(f"{title} {method_label} colored by {title_label}")
        ax.set_xlabel(f"{method_label} 1")
        ax.set_ylabel(f"{method_label} 2")
        ax.grid(alpha=0.15, linewidth=0.5)

    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        ncol=1,
        title=title_label.title(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_embedding_json(
    path: Path,
    records: list[SampleRecord],
    labels: list[str],
    fc_embedding: np.ndarray,
    raw_embedding: np.ndarray,
    fc_site_probe: dict | None,
    raw_site_probe: dict | None,
    fc_diagnosis_probe: dict | None,
    raw_diagnosis_probe: dict | None,
    args,
):
    payload = {
        "config": {
            "fc_dir": args.fc_dir,
            "raw_dir": args.raw_dir,
            "triangle": args.triangle,
            "reorder": args.reorder,
            "raw_target_timepoints": args.raw_target_timepoints,
            "perplexity": args.perplexity,
            "pca_components": args.pca_components,
            "embedding_method": args.embedding_method,
            "label_by": args.label_by,
            "seed": args.seed,
            "max_samples": args.max_samples,
        },
        "site_probe": {
            "fc": fc_site_probe,
            "raw": raw_site_probe,
        },
        "diagnosis_probe": {
            "fc": fc_diagnosis_probe,
            "raw": raw_diagnosis_probe,
        },
        "samples": [],
    }

    for record, label, fc_xy, raw_xy in zip(records, labels, fc_embedding, raw_embedding):
        payload["samples"].append(
            {
                "sample_id": record.sample_id,
                "subject": record.subject,
                "run": record.run,
                "site": record.site,
                "diagnosis": record.diagnosis,
                "label": label,
                "fc_embedding": [float(fc_xy[0]), float(fc_xy[1])],
                "raw_embedding": [float(raw_xy[0]), float(raw_xy[1])],
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    fc_dir = Path(args.fc_dir)
    raw_dir = Path(args.raw_dir)
    if not fc_dir.exists():
        raise FileNotFoundError(f"FC directory not found: {fc_dir}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw ROI directory not found: {raw_dir}")

    records = load_matched_records(fc_dir=fc_dir, raw_dir=raw_dir)
    if not records:
        raise FileNotFoundError("No matched FC/raw samples found.")

    records = maybe_subsample_records(records, max_samples=args.max_samples, seed=args.seed)

    permutation, _labels = get_roi_permutation(args.reorder)
    fc_features = build_fc_features(records, permutation=permutation, triangle=args.triangle)
    raw_features = build_raw_features(
        records,
        permutation=permutation,
        raw_target_timepoints=args.raw_target_timepoints,
    )

    label_values = [getattr(record, args.label_by) for record in records]
    fc_embedding, _ = compute_embedding(
        fc_features,
        method=args.embedding_method,
        pca_components=args.pca_components,
        perplexity=args.perplexity,
        seed=args.seed,
    )
    raw_embedding, _ = compute_embedding(
        raw_features,
        method=args.embedding_method,
        pca_components=args.pca_components,
        perplexity=args.perplexity,
        seed=args.seed,
    )

    fc_site_probe = run_probe(
        fc_features,
        [record.site for record in records],
        seed=args.seed,
        folds=args.site_probe_folds,
        task_name="site",
    )
    raw_site_probe = run_probe(
        raw_features,
        [record.site for record in records],
        seed=args.seed,
        folds=args.site_probe_folds,
        task_name="site",
    )
    fc_diagnosis_probe = run_probe(
        fc_features,
        [record.diagnosis for record in records],
        seed=args.seed,
        folds=args.site_probe_folds,
        task_name="diagnosis",
    )
    raw_diagnosis_probe = run_probe(
        raw_features,
        [record.diagnosis for record in records],
        seed=args.seed,
        folds=args.site_probe_folds,
        task_name="diagnosis",
    )

    print(f"Matched samples: {len(records)}")
    print(f"FC features: {fc_features.shape} | raw features: {raw_features.shape}")
    if fc_site_probe is not None:
        print(
            "Site probe | FC    | "
            f"acc={fc_site_probe['accuracy']:.3f} "
            f"bAcc={fc_site_probe['balanced_accuracy']:.3f} "
            f"classes={fc_site_probe['n_classes']}"
        )
    if raw_site_probe is not None:
        print(
            "Site probe | RAW   | "
            f"acc={raw_site_probe['accuracy']:.3f} "
            f"bAcc={raw_site_probe['balanced_accuracy']:.3f} "
            f"classes={raw_site_probe['n_classes']}"
        )
    if fc_diagnosis_probe is not None:
        print(
            "Dx probe   | FC    | "
            f"acc={fc_diagnosis_probe['accuracy']:.3f} "
            f"bAcc={fc_diagnosis_probe['balanced_accuracy']:.3f} "
            f"classes={fc_diagnosis_probe['n_classes']}"
        )
    if raw_diagnosis_probe is not None:
        print(
            "Dx probe   | RAW   | "
            f"acc={raw_diagnosis_probe['accuracy']:.3f} "
            f"bAcc={raw_diagnosis_probe['balanced_accuracy']:.3f} "
            f"classes={raw_diagnosis_probe['n_classes']}"
        )
    print(
        f"Note: the 2D {args.embedding_method.upper()} plot is qualitative; use the site probe numbers as the cleaner comparison of site information."
    )

    output_path = Path(args.output)
    plot_embeddings(
        fc_embedding=fc_embedding,
        raw_embedding=raw_embedding,
        labels=label_values,
        title_label=args.label_by,
        output_path=output_path,
        embedding_method=args.embedding_method,
    )
    print(f"Saved figure: {output_path}")

    embedding_json = Path(args.embedding_json)
    save_embedding_json(
        path=embedding_json,
        records=records,
        labels=label_values,
        fc_embedding=fc_embedding,
        raw_embedding=raw_embedding,
        fc_site_probe=fc_site_probe,
        raw_site_probe=raw_site_probe,
        fc_diagnosis_probe=fc_diagnosis_probe,
        raw_diagnosis_probe=raw_diagnosis_probe,
        args=args,
    )
    print(f"Saved embedding JSON: {embedding_json}")


if __name__ == "__main__":
    main()
