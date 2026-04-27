#!/usr/bin/env python3
"""
Extract Schaefer 400 ROI time series from fMRIPrep outputs for ABIDE-I.

Dataset: ABIDE-I (~1,112 subjects across 24 sites)
  - Diagnosis: ASD (DX_GROUP=1) vs. Controls (DX_GROUP=2)
  - Multi-site resting-state fMRI

Pipeline per subject:
  1. Load fMRIPrep BOLD (MNI152NLin2009cAsym, 2mm)
  2. Resolve TR (NIfTI header -> fMRIPrep JSON -> BIDS sidecar)
  3. Confound regression: 6 motion + 5 aCompCor
  4. Bandpass filter: 0.01-0.1 Hz
  5. Trim 7 start + 7 end volumes
  6. Keep the full post-trim signal
  7. Extract mean signal per Schaefer 400 ROI (7-network)
  8. Z-score each ROI
  9. Save .npy + metadata JSON

Outputs:
  derivatives/roi_timeseries_abide/
  derivatives/abide_metadata.csv
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker

# ======================== Config ========================
TRIM_START = 7
TRIM_END = 7
N_ROIS = 400
NETWORKS = 7
DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1]
CONFOUNDS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
    "a_comp_cor_00", "a_comp_cor_01",
    "a_comp_cor_02", "a_comp_cor_03",
    "a_comp_cor_04",
]
HIGH_PASS = 0.01
LOW_PASS = 0.1

# ABIDE-I raw DX_GROUP: 1 = ASD, 2 = Control
# Remap to: 0 = Control, 1 = ASD
DX_REMAP = {1: 1, 2: 0}
DX_NAME = {0: "Healthy Control", 1: "Autism Spectrum Disorder"}

# ABIDE raw SEX: 1 = Male, 2 = Female
SEX_MAP = {1: "M", 2: "F", "1": "M", "2": "F", "M": "M", "F": "F"}


def setup_logging(log_file=None):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def load_site_map(site_map_path):
    """Load site_map.tsv -> {sub_id: site}."""
    site_map = {}
    if not site_map_path.exists():
        logging.warning(f"site_map.tsv not found: {site_map_path}")
        return site_map
    with open(site_map_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            site_map[row["subject"].strip()] = row["site"].strip()
    return site_map


def load_abide_participants(raw_dir):
    """Load all per-site participants.tsv files -> {sub_id: {dx, age, sex, hand}}."""
    participants = {}
    if not raw_dir.exists():
        logging.warning(f"ABIDE raw dir not found: {raw_dir}")
        return participants

    for site_dir in sorted(raw_dir.iterdir()):
        if not site_dir.is_dir() or site_dir.name == "sidecards":
            continue
        tsv_path = site_dir / "participants.tsv"
        if not tsv_path.exists():
            continue
        with open(tsv_path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                raw_id = row["participant_id"].strip()
                if raw_id.startswith("sub-"):
                    sub_id = raw_id
                else:
                    sub_id = f"sub-{int(raw_id):07d}"

                try:
                    raw_dx = int(row["DX_GROUP"].strip())
                    dx = DX_REMAP.get(raw_dx, -1)
                except (KeyError, ValueError):
                    dx = -1
                try:
                    age = float(row["AGE_AT_SCAN"].strip())
                except (KeyError, ValueError):
                    age = -1
                sex_raw = row.get("SEX", "").strip()
                hand_raw = row.get("HANDEDNESS_CATEGORY", "").strip()

                participants[sub_id] = {
                    "dx": dx,
                    "dx_name": DX_NAME.get(dx, "Unknown"),
                    "age": age,
                    "sex": SEX_MAP.get(sex_raw, sex_raw),
                    "hand": hand_raw if hand_raw not in ("n/a", "-9999") else "",
                }
    return participants


def resolve_tr(bold_path, bids_dir, sub):
    """Resolve TR: NIfTI header -> fMRIPrep JSON -> BIDS sidecar."""
    img = nib.load(bold_path)
    zooms = img.header.get_zooms()
    if len(zooms) > 3:
        tr = float(zooms[3])
        if tr > 0:
            return tr, "header"

    json_path = bold_path.with_suffix("").with_suffix(".json")
    if json_path.exists():
        with open(json_path) as f:
            tr = json.load(f).get("RepetitionTime", 0.0)
        if tr > 0:
            return tr, "fmriprep_json"

    bids_func = bids_dir / sub / "func"
    if bids_func.exists():
        for jf in sorted(bids_func.glob("*_bold.json")):
            with open(jf) as f:
                tr = json.load(f).get("RepetitionTime", 0.0)
            if tr > 0:
                return tr, "bids_json"

    return 0.0, "unknown"


def extract_roi_ts(bold_path, confounds_tsv, tr, atlas_maps, target_vols=None):
    """
    Core extraction: confound regression, bandpass, parcellation, z-score, trim.

    Parameters
    ----------
    target_vols : int | None
        If set, truncate to this many timepoints after trimming.
        If None, keep the full post-trim signal.
    """
    img = nib.load(bold_path)
    n_vols = img.shape[3]

    min_keep = 1 if target_vols is None else target_vols
    min_needed = TRIM_START + TRIM_END + min_keep
    if n_vols < min_needed:
        return None, n_vols, 0, f"too_short_{n_vols}"

    if not confounds_tsv.exists():
        return None, n_vols, 0, "no_confounds"

    df = pd.read_csv(confounds_tsv, sep="\t")
    available = [c for c in CONFOUNDS if c in df.columns]
    conf = df[available].fillna(0.0)
    assert len(conf) == n_vols

    end_idx = n_vols - TRIM_END
    bold_trimmed = img.slicer[..., TRIM_START:end_idx]
    conf_trimmed = conf.iloc[TRIM_START:end_idx].reset_index(drop=True)
    n_post_trim = end_idx - TRIM_START

    if n_post_trim <= 0:
        return None, n_vols, 0, f"too_short_after_trim_{n_post_trim}"

    if target_vols is not None:
        bold_proc = bold_trimmed.slicer[..., :target_vols]
        conf_proc = conf_trimmed.iloc[:target_vols].reset_index(drop=True)
        expected_timepoints = target_vols
    else:
        bold_proc = bold_trimmed
        conf_proc = conf_trimmed
        expected_timepoints = n_post_trim

    masker = NiftiLabelsMasker(
        labels_img=atlas_maps,
        standardize="zscore_sample",
        high_pass=HIGH_PASS,
        low_pass=LOW_PASS,
        t_r=tr,
    )
    ts = masker.fit_transform(bold_proc, confounds=conf_proc)
    assert ts.shape == (expected_timepoints, N_ROIS), f"Unexpected shape {ts.shape}"
    return ts, n_vols, n_post_trim, "ok"


def process_abide_subject(bold_path, site_map, participants, bids_dir, out_dir, atlas_maps):
    """Process one ABIDE BOLD file (one run of one subject)."""
    sub = bold_path.parts[-3]
    run_parts = [p for p in bold_path.name.split("_") if p.startswith("run-")]
    run_label = run_parts[0] if run_parts else None
    out_name = f"{sub}_{run_label}" if run_label else sub

    try:
        tr, tr_source = resolve_tr(bold_path, bids_dir, sub)
        if tr == 0.0:
            logging.warning(f"[ABIDE] {out_name}: Could not resolve TR, skipping")
            return None

        if run_label:
            confounds_tsv = bold_path.parent / f"{sub}_task-rest_{run_label}_desc-confounds_timeseries.tsv"
        else:
            confounds_tsv = bold_path.parent / f"{sub}_task-rest_desc-confounds_timeseries.tsv"

        ts, n_vols, post_trim_vols, status = extract_roi_ts(
            bold_path, confounds_tsv, tr, atlas_maps, target_vols=None
        )
        if ts is None:
            logging.warning(f"[ABIDE] {out_name}: {status}")
            return None

        site = site_map.get(sub, "")
        info = participants.get(sub, {})

        meta = {
            "subject": sub,
            "run": run_label or "",
            "dataset": "abide_i",
            "site": site,
            "dx": info.get("dx", -1),
            "dx_name": info.get("dx_name", "Unknown"),
            "age": info.get("age", -1),
            "sex": info.get("sex", ""),
            "hand": info.get("hand", ""),
            "tr": tr,
            "tr_source": tr_source,
            "original_vols": n_vols,
            "post_trim_vols": post_trim_vols,
            "n_timepoints": int(ts.shape[0]),
            "target_vols": None,
            "timepoints_cropped": False,
            "n_rois": N_ROIS,
            "trim_start": TRIM_START,
            "trim_end": TRIM_END,
            "confounds": CONFOUNDS,
            "bandpass": [HIGH_PASS, LOW_PASS],
            "atlas": f"schaefer_{N_ROIS}_{NETWORKS}networks",
        }

        np.save(out_dir / f"{out_name}_roi_ts.npy", ts.astype(np.float32))
        with open(out_dir / f"{out_name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logging.info(f"[ABIDE] {out_name}: OK (site={site}, dx={info.get('dx','')}, TR={tr})")
        return meta

    except Exception as e:
        logging.error(f"[ABIDE] {out_name}: FAILED -- {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract Schaefer 400 ROI time series for ABIDE-I")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--subjects", type=str, nargs="*", default=None,
                        help="Process only these subjects (e.g. sub-0050002)")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess and overwrite runs already present in the output directory")
    args = parser.parse_args()

    setup_logging(args.log_file)
    base = args.base_dir

    # Paths kept the same as requested
    fmriprep_dir = base / "derivatives" / "fmriprep_abide"
    bids_dir = base / "ABIDE_I_BIDS"
    raw_dir = base / "ABIDE_I" / "raw"
    site_map_path = bids_dir / "site_map.tsv"
    out_dir = base / "derivatives" / "roi_timeseries_abide"

    logging.info(f"Fetching Schaefer {N_ROIS} atlas ({NETWORKS}-network)...")
    atlas = fetch_atlas_schaefer_2018(n_rois=N_ROIS, resolution_mm=2)
    atlas_maps = atlas["maps"]

    site_map = load_site_map(site_map_path)
    logging.info(f"[ABIDE] Loaded site map for {len(site_map)} subjects")

    participants = load_abide_participants(raw_dir)
    logging.info(f"[ABIDE] Loaded phenotypic data for {len(participants)} subjects")

    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"[ABIDE] Writing full post-trim time series to: {out_dir}")

    bold_files = sorted(
        fmriprep_dir.glob("sub-*/func/*_task-rest_run-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    )
    if args.subjects:
        bold_files = [b for b in bold_files if b.parts[-3] in args.subjects]

    if args.force:
        todo = bold_files
        existing = set()
    else:
        existing = {f.stem.replace("_roi_ts", "") for f in out_dir.glob("*_roi_ts.npy")}
        todo = []
        for b in bold_files:
            sub = b.parts[-3]
            run_parts = [p for p in b.name.split("_") if p.startswith("run-")]
            out_name = f"{sub}_{run_parts[0]}" if run_parts else sub
            if out_name not in existing:
                todo.append(b)

    logging.info(f"[ABIDE] {len(bold_files)} total runs, {len(existing)} done, {len(todo)} to process")

    all_metadata = []
    if todo:
        results = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process_abide_subject)(b, site_map, participants, bids_dir, out_dir, atlas_maps)
            for b in todo
        )
        all_metadata.extend([r for r in results if r is not None])
        ok = sum(1 for r in results if r is not None)
        logging.info(f"[ABIDE] Done: {ok}/{len(results)} succeeded")

    for meta_file in sorted(out_dir.glob("*_meta.json")):
        with open(meta_file) as f:
            m = json.load(f)
            if m not in all_metadata:
                all_metadata.append(m)

    if all_metadata:
        csv_path = base / "derivatives" / "abide_metadata.csv"
        columns = [
            "subject", "run", "dataset", "site",
            "dx", "dx_name", "age", "sex", "hand",
            "tr", "tr_source", "original_vols", "post_trim_vols",
            "n_timepoints", "target_vols", "timepoints_cropped", "n_rois",
        ]
        rows = []
        for m in all_metadata:
            row = {c: m.get(c, "") for c in columns}
            rows.append(row)
        df = pd.DataFrame(rows, columns=columns)
        df = df.sort_values(["site", "subject", "run"]).reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved metadata CSV: {csv_path} ({len(df)} rows)")

    logging.info("All done!")


if __name__ == "__main__":
    main()
