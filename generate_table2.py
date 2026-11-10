"""Table 2: pretraining cohort summary (SRPBS incl. traveling subjects, ABIDE-I, ABIDE-II).

Counts come from the Schaefer-400 ROI time-series sidecars actually used for JEPA
pretraining. N = unique subjects, Scans = all sessions/runs, Sites = site labels
after normalizing the ABIDE-I/ABIDE-II shared centers. Age, sex and diagnosis are
summarized at the subject level (age over subjects with a recorded age).

Usage:
    python generate_table2.py [--data-root preprocessed_data_fir/full]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from src.jepa_clean.data import load_scan_records, make_subject_id


DATASETS = {
    "SRPBS": ["roi_timeseries_1600_full", "roi_timeseries_traveling_full"],
    "ABIDE-I": ["roi_timeseries_abide_full"],
    "ABIDE-II": ["roi_timeseries_abide2_full"],
}
DIAGNOSIS_COLUMNS = [
    ("HC", "Healthy Control"),
    ("ASD", "Autism Spectrum Disorder"),
    ("MDD", "Major Depressive Disorder"),
    ("SCZ", "Schizophrenia"),
    ("BD", "Bipolar Disorder"),
    ("Pain", "Pain"),
    ("Stroke", "Stroke"),
    ("Dys.", "Dysthymia"),
    ("Oth.", "Others"),
]


def load_scans(data_root: Path) -> pd.DataFrame:
    frames = []
    for name, dirs in DATASETS.items():
        records = load_scan_records([str(data_root / d) for d in dirs])
        frames.append(
            pd.DataFrame(
                {
                    "dataset": name,
                    "subject_id": make_subject_id(r),
                    "site": r.site,
                    "sex": r.sex.upper(),
                    "age": r.age,
                    "diagnosis": r.diag_name,
                }
                for r in records
            )
        )
    return pd.concat(frames, ignore_index=True)


def summarize(scans: pd.DataFrame, label: str) -> dict:
    subjects = scans.groupby("subject_id").first()
    ages = subjects["age"][subjects["age"] > 0]
    known = {name for _short, name in DIAGNOSIS_COLUMNS}
    row = {
        "Dataset": label,
        "N": len(subjects),
        "Scans": len(scans),
        "Sites": scans["site"].nunique(),
        "Age": f"{ages.mean():.2f} ± {ages.std():.2f}",
        "M/F/Unk.": (
            f"{(subjects['sex'] == 'M').sum()} / {(subjects['sex'] == 'F').sum()} / "
            f"{(~subjects['sex'].isin(['M', 'F'])).sum()}"
        ),
    }
    for short, name in DIAGNOSIS_COLUMNS:
        row[short] = int((subjects["diagnosis"] == name).sum())
    row["Unk."] = int((~subjects["diagnosis"].isin(known)).sum())
    return row


def to_latex(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{l" + "c" * (len(table.columns) - 1) + "}",
        r"\toprule",
        " & ".join(table.columns).replace("±", r"$\pm$") + r" \\",
        r"\midrule",
    ]
    for idx, row in table.iterrows():
        if row["Dataset"] == "Total":
            lines.append(r"\midrule")
        lines.append(" & ".join(str(v) for v in row.values).replace("±", r"$\pm$") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="preprocessed_data_fir/full")
    args = parser.parse_args()

    scans = load_scans(Path(args.data_root))
    rows = [summarize(scans[scans["dataset"] == name], name) for name in DATASETS]
    rows.append(summarize(scans, "Total"))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False), file=sys.stderr)
    print(to_latex(table))


if __name__ == "__main__":
    main()
