from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

try:
    from ..jepa_clean.data import normalize_site_name
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_clean.data import normalize_site_name


HEALTHY_CONTROL_LABEL = "Healthy Control"
AUTISM_LABEL = "Autism Spectrum Disorder"
UNKNOWN_LABELS = {"", "Unknown", "unknown", "UNK", "unk"}


@dataclass(frozen=True)
class DatasetPreset:
    key: str
    label: str
    fc_dir: str | tuple[str, ...]
    raw_dir: str | tuple[str, ...]
    default_task: str
    default_mixed_sites_only: bool


DATASET_PRESETS = {
    "abide": DatasetPreset(
        key="abide",
        label="ABIDE-I + ABIDE-II",
        fc_dir=(
            "preprocessed_data_fir/FC_abide_full",
            "preprocessed_data_fir/FC_abide2_full",
        ),
        raw_dir=(
            "preprocessed_data_fir/full/roi_timeseries_abide_full",
            "preprocessed_data_fir/full/roi_timeseries_abide2_full",
        ),
        default_task="hc_asd",
        default_mixed_sites_only=True,
    ),
    "abide1": DatasetPreset(
        key="abide1",
        label="ABIDE-I",
        fc_dir="preprocessed_data_fir/FC_abide_full",
        raw_dir="preprocessed_data_fir/full/roi_timeseries_abide_full",
        default_task="hc_asd",
        default_mixed_sites_only=True,
    ),
    "abide2": DatasetPreset(
        key="abide2",
        label="ABIDE-II",
        fc_dir="preprocessed_data_fir/FC_abide2_full",
        raw_dir="preprocessed_data_fir/full/roi_timeseries_abide2_full",
        default_task="hc_asd",
        default_mixed_sites_only=True,
    ),
    "srpbs": DatasetPreset(
        key="srpbs",
        label="SRPBS-1600",
        fc_dir="preprocessed_data_fir/FC_1600",
        raw_dir="preprocessed_data_fir/full/roi_timeseries_1600_full",
        default_task="hc_disorder",
        default_mixed_sites_only=True,
    ),
}


@dataclass(frozen=True)
class TaskResult:
    records: list
    y: np.ndarray
    class_names: list[str]
    task_label: str
    task_key: str
    removed_unknown: int
    removed_task_mismatch: int
    removed_single_class_sites: int
    mixed_sites_kept: list[str]


def get_dataset_preset(key: str) -> DatasetPreset:
    if key not in DATASET_PRESETS:
        raise KeyError(f"Unknown dataset preset: {key}")
    return DATASET_PRESETS[key]


def dataset_order_for_choice(dataset_choice: str):
    if dataset_choice == "all":
        return ["abide", "srpbs"]
    return [dataset_choice]


def source_dataset_keys(dataset_key: str):
    if dataset_key == "abide":
        return ("abide1", "abide2")
    return (dataset_key,)


def source_dataset_prefix(dataset_key: str):
    mapping = {
        "abide": "abide",
        "abide1": "abide_i",
        "abide2": "abide_ii",
        "srpbs": "srpbs_1600",
    }
    if dataset_key not in mapping:
        raise KeyError(f"Unknown dataset key: {dataset_key}")
    return mapping[dataset_key]


def center_site_label(site: str):
    return normalize_site_name(str(site))


def batch_site_label(site: str, *, source_dataset: str | None = None):
    raw_site = str(site)
    if source_dataset == "abide_i":
        return raw_site if raw_site.startswith("ABIDEI-") else f"ABIDEI-{raw_site}"
    if source_dataset == "abide_ii":
        return raw_site if raw_site.startswith("ABIDEII-") else f"ABIDEII-{raw_site}"
    return raw_site


def _known_diagnosis(record) -> bool:
    return str(record.diagnosis) not in UNKNOWN_LABELS


def _site_class_sets(records, y: np.ndarray):
    site_to_classes = defaultdict(set)
    for record, label in zip(records, y.tolist()):
        site_to_classes[str(record.site)].add(int(label))
    return site_to_classes


def filter_records_for_task(
    records: list,
    task_key: str,
    *,
    mixed_sites_only: bool,
    healthy_control_label: str = HEALTHY_CONTROL_LABEL,
    autism_label: str = AUTISM_LABEL,
) -> TaskResult:
    known_records = [record for record in records if _known_diagnosis(record)]
    removed_unknown = len(records) - len(known_records)

    filtered_records = []
    labels = []

    if task_key == "hc_asd":
        task_label = "HC vs ASD"
        class_names = [healthy_control_label, autism_label]
        for record in known_records:
            diagnosis = str(record.diagnosis)
            if diagnosis == healthy_control_label:
                filtered_records.append(record)
                labels.append(0)
            elif diagnosis == autism_label:
                filtered_records.append(record)
                labels.append(1)
    elif task_key == "hc_disorder":
        task_label = "HC vs ALL Disorders"
        class_names = [healthy_control_label, "Disorder"]
        for record in known_records:
            diagnosis = str(record.diagnosis)
            filtered_records.append(record)
            labels.append(0 if diagnosis == healthy_control_label else 1)
    else:
        raise ValueError(f"Unsupported task_key: {task_key}")

    removed_task_mismatch = len(known_records) - len(filtered_records)
    if not filtered_records:
        raise ValueError(f"Task {task_key} removed every record.")

    y = np.asarray(labels, dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise ValueError(f"Task {task_key} has fewer than two classes after filtering.")

    removed_single_class_sites = 0
    mixed_sites_kept: list[str] = []
    if mixed_sites_only:
        site_to_classes = _site_class_sets(filtered_records, y)
        mixed_sites_kept = sorted(site for site, labels_set in site_to_classes.items() if labels_set == {0, 1})
        kept_records = []
        kept_labels = []
        for record, label in zip(filtered_records, y.tolist()):
            if str(record.site) in mixed_sites_kept:
                kept_records.append(record)
                kept_labels.append(label)
        removed_single_class_sites = len(filtered_records) - len(kept_records)
        filtered_records = kept_records
        y = np.asarray(kept_labels, dtype=np.int64)
        if not filtered_records:
            raise ValueError(f"Mixed-site filtering removed every record for task {task_key}.")
        if len(np.unique(y)) < 2:
            raise ValueError(f"Mixed-site filtering left fewer than two classes for task {task_key}.")

    return TaskResult(
        records=filtered_records,
        y=y,
        class_names=class_names,
        task_label=task_label,
        task_key=task_key,
        removed_unknown=removed_unknown,
        removed_task_mismatch=removed_task_mismatch,
        removed_single_class_sites=removed_single_class_sites,
        mixed_sites_kept=mixed_sites_kept,
    )


def class_counts(y: np.ndarray, class_names: list[str]) -> dict[str, int]:
    counts = Counter(int(v) for v in y.tolist())
    return {
        class_name: int(counts.get(class_idx, 0))
        for class_idx, class_name in enumerate(class_names)
    }
