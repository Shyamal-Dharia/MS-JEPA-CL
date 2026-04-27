from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

try:
    from .protocol import (
        batch_site_label,
        center_site_label,
        dataset_order_for_choice,
        get_dataset_preset,
        source_dataset_prefix,
        source_dataset_keys,
    )
    from ..jepa_clean.plot_tsne_fc_vs_raw import load_matched_records
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.protocol import (
        batch_site_label,
        center_site_label,
        dataset_order_for_choice,
        get_dataset_preset,
        source_dataset_prefix,
        source_dataset_keys,
    )
    from src.jepa_clean.plot_tsne_fc_vs_raw import load_matched_records


@dataclass(frozen=True)
class MatchedEvaluationRecord:
    sample_id: str
    subject: str
    subject_id: str
    run: str
    site: str
    site_batch: str
    source_dataset: str
    diagnosis: str
    age: float | None
    sex: str
    fc_path: Path
    raw_path: Path


def default_eval_dataset_order(dataset_choice: str):
    return dataset_order_for_choice(dataset_choice)


def matched_records_for_dataset(dataset_key: str):
    records = []
    for source_key in source_dataset_keys(dataset_key):
        preset = get_dataset_preset(source_key)
        source_dataset = source_dataset_prefix(source_key)
        source_records = load_matched_records(Path(preset.fc_dir), Path(preset.raw_dir))
        for record in source_records:
            raw_site = str(record.site)
            records.append(
                MatchedEvaluationRecord(
                    sample_id=str(record.sample_id),
                    subject=str(record.subject),
                    subject_id=f"{source_dataset}:{record.subject}",
                    run=str(record.run),
                    site=center_site_label(raw_site),
                    site_batch=batch_site_label(raw_site, source_dataset=source_dataset),
                    source_dataset=source_dataset,
                    diagnosis=str(record.diagnosis),
                    age=None if record.age is None else float(record.age),
                    sex=str(record.sex),
                    fc_path=record.fc_path,
                    raw_path=record.raw_path,
                )
            )
    return records


def matched_records_for_preset(preset):
    return matched_records_for_dataset(preset.key)
