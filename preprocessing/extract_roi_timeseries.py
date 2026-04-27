#!/usr/bin/env python3
"""
Extract Schaefer 400 ROI time series from fMRIPrep outputs with rich metadata.

Handles TWO datasets:
  1. SRPBS-1600: 1,627 subjects, single-site, mixed healthy/disorder
  2. Traveling Subjects: 9 subjects x ~15 sites, all healthy

Pipeline per subject/session:
  1. Load fMRIPrep BOLD (MNI152, 2mm)
  2. Resolve TR (header -> JSON -> participants.tsv protocol)
  3. Confound regression: 6 motion + WM + CSF
  4. Bandpass filter: 0.01-0.1 Hz
  5. Trim 7 start + 7 end volumes
  6. Keep full post-trim signal for SRPBS-1600
  7. Keep full post-trim signal for traveling subjects
  8. Extract mean signal per Schaefer 400 ROI (7-network)
  9. Z-score each ROI
 10. Save .npy + metadata JSON

Outputs:
  preprocessed_data_fir/roi_timeseries_1600_full/
  preprocessed_data_fir/roi_timeseries_traveling_full/
  preprocessed_data_fir/subject_metadata.csv
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

PROTOCOL_TR = {
    1: 2.5, 2: 2.0, 3: 2.0, 4: 2.7, 5: 2.5,
    6: 2.5, 7: 2.0, 8: 2.5, 9: 2.5, 10: 2.5,
    11: 2.5, 12: 2.5, 13: 2.5, 14: 2.5,
}

DIAG_MAP = {
    0: "Healthy Control",
    1: "Autism Spectrum Disorder",
    2: "Major Depressive Disorder",
    3: "Obsessive Compulsive Disorder",
    4: "Schizophrenia",
    5: "Pain",
    6: "Stroke",
    7: "Bipolar Disorder",
    8: "Dysthymia",
    99: "Others",
}

SITE_NAMES = {
    "SWA": "Showa University",
    "HUH": "Hiroshima University Hospital",
    "HRC": "Hiroshima Rehabilitation Center",
    "HKH": "Hiroshima Kajikawa Hospital",
    "COI": "Hiroshima COI",
    "KUT": "Kyoto University TimTrio",
    "KTT": "Kyoto University Trio",
    "UTO": "University of Tokyo Hospital",
    "ATT": "ATR TimTrio",
    "ATV": "ATR Verio",
    "OSU": "Osaka University",
    "CIN": "CiNet",
    "NKN": "Nishinomiya Kyouritsu Hospital",
    "KPM": "Kobe Parkinsons",
    "KUS": "Kyushu University",
    "YC1": "Yaesu Clinic 1",
    "YC2": "Yaesu Clinic 2",
    "HKH": "Hiroshima Kajikawa Hospital",
}

SEX_MAP = {1: "M", 2: "F", "1": "M", "2": "F", "M": "M", "F": "F"}
HAND_MAP = {1: "R", 2: "L", "1": "R", "2": "L"}


def setup_logging(log_file=None):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def load_1600_participants(tsv_path):
    """Load SRPBS-1600 participants.tsv -> {sub_id: {site, protocol, diag, age, sex, hand}}."""
    participants = {}
    if not tsv_path.exists():
        logging.warning(f"participants.tsv not found: {tsv_path}")
        return participants
    with open(tsv_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sub_id = row["participant_id"].strip()
            try:
                diag = int(row["diag"].strip())
            except (KeyError, ValueError):
                diag = -1
            try:
                protocol = int(row["protocol"].strip())
            except (KeyError, ValueError):
                protocol = -1
            try:
                age = int(row["age"].strip())
            except (KeyError, ValueError):
                age = -1
            sex_raw = row.get("sex", "").strip()
            hand_raw = row.get("hand", "").strip()
            participants[sub_id] = {
                "site": row.get("site", "").strip(),
                "protocol": protocol,
                "diag": diag,
                "diag_name": DIAG_MAP.get(diag, "Unknown"),
                "age": age,
                "sex": SEX_MAP.get(sex_raw, sex_raw),
                "hand": HAND_MAP.get(hand_raw, hand_raw),
            }
    return participants


def load_ts_participants(tsv_path):
    """Load traveling subject participants.tsv -> {sub_id: {age, sex}}."""
    participants = {}
    if not tsv_path.exists():
        logging.warning(f"TS participants.tsv not found: {tsv_path}")
        return participants
    with open(tsv_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sub_id = row["participant_id"].strip()
            try:
                age = int(row["age"].strip())
            except (KeyError, ValueError):
                age = -1
            sex_raw = row.get("sex", "").strip()
            participants[sub_id] = {
                "age": age,
                "sex": SEX_MAP.get(sex_raw, sex_raw),
            }
    return participants


def resolve_tr(bold_path, sub_id, participants_1600):
    """Resolve TR with multi-level fallback."""
    img = nib.load(bold_path)
    tr = float(img.header.get_zooms()[3])
    if tr > 0:
        return tr, "header"

    json_path = bold_path.with_suffix("").with_suffix(".json")
    if json_path.exists():
        with open(json_path) as f:
            tr = json.load(f).get("RepetitionTime", 0.0)
        if tr > 0:
            return tr, "fmriprep_json"

    info = participants_1600.get(sub_id, {})
    protocol = info.get("protocol")
    if protocol and protocol in PROTOCOL_TR:
        return PROTOCOL_TR[protocol], f"protocol_{protocol}"

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

    # Trim start/end
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


# ======================= SRPBS-1600 =======================
def process_1600_subject(bold_path, participants_1600, out_dir, atlas_maps):
    sub = bold_path.parts[-3]
    try:
        tr, tr_source = resolve_tr(bold_path, sub, participants_1600)
        if tr == 0.0:
            logging.warning(f"[1600] {sub}: Could not resolve TR, skipping")
            return None

        confounds_tsv = bold_path.parent / f"{sub}_task-rest_desc-confounds_timeseries.tsv"
        ts, n_vols, post_trim_vols, status = extract_roi_ts(
            bold_path, confounds_tsv, tr, atlas_maps, target_vols=None
        )
        if ts is None:
            logging.warning(f"[1600] {sub}: {status}")
            return None

        info = participants_1600.get(sub, {})
        meta = {
            "subject": sub,
            "dataset": "srpbs_1600",
            "is_traveling_subject": False,
            "site": info.get("site", ""),
            "site_name": SITE_NAMES.get(info.get("site", ""), ""),
            "protocol": info.get("protocol", -1),
            "diag": info.get("diag", -1),
            "diag_name": info.get("diag_name", "Unknown"),
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

        np.save(out_dir / f"{sub}_roi_ts.npy", ts.astype(np.float32))
        with open(out_dir / f"{sub}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logging.info(f"[1600] {sub}: OK (site={info.get('site','')}, diag={info.get('diag','')}, TR={tr})")
        return meta

    except Exception as e:
        logging.error(f"[1600] {sub}: FAILED -- {e}")
        return None


# ==================== Traveling Subjects ====================
def process_ts_session(bold_path, ts_participants, out_dir, atlas_maps):
    """Process one session (run-01) of a traveling subject."""
    sub = bold_path.parts[-4]    # e.g. sub-01
    ses = bold_path.parts[-3]    # e.g. ses-siteSWA
    # Path: .../sub-01/ses-siteSWA/func/sub-01_ses-siteSWA_task-rest_run-01_..._bold.nii.gz
    parts = bold_path.parts
    ses = [p for p in parts if p.startswith("ses-")][0]
    site_code = ses.replace("ses-site", "")  # e.g. "SWA", "ATTd1"
    # Normalize site code: remove day suffixes like "d1", "d2" etc.
    site_base = site_code.rstrip("0123456789").rstrip("d")
    out_name = f"{sub}_{ses}"

    try:
        img = nib.load(bold_path)
        tr = float(img.header.get_zooms()[3])
        if tr == 0.0:
            # TS data is always TR=2.5
            tr = 2.5
        tr_source = "header" if float(img.header.get_zooms()[3]) > 0 else "ts_default"

        # Find confounds -- match same run
        run_part = [p for p in bold_path.name.split("_") if p.startswith("run-")][0]
        confounds_tsv = bold_path.parent / f"{sub}_{ses}_task-rest_{run_part}_desc-confounds_timeseries.tsv"
        ts, n_vols, post_trim_vols, status = extract_roi_ts(
            bold_path, confounds_tsv, tr, atlas_maps, target_vols=None
        )
        if ts is None:
            logging.warning(f"[TS] {out_name}: {status}")
            return None

        info = ts_participants.get(sub, {})
        meta = {
            "subject": sub,
            "session": ses,
            "dataset": "traveling_subject",
            "is_traveling_subject": True,
            "site": site_base,
            "site_name": SITE_NAMES.get(site_base, site_code),
            "site_session": site_code,
            "protocol": -1,
            "diag": 0,
            "diag_name": "Healthy Control",
            "age": info.get("age", -1),
            "sex": info.get("sex", ""),
            "hand": "",
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

        logging.info(f"[TS] {out_name}: OK (site={site_code}, TR={tr})")
        return meta

    except Exception as e:
        logging.error(f"[TS] {out_name}: FAILED -- {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract Schaefer 400 ROI time series")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--dataset", choices=["1600", "traveling", "both", "metadata"], default="both",
                        help="Which dataset to process")
    parser.add_argument("--subjects", type=str, nargs="*", default=None,
                        help="Process only these subjects")
    args = parser.parse_args()

    setup_logging(args.log_file)
    base = args.base_dir

    # Paths
    fmriprep_1600 = base / "preprocessed_data_fir" / "fmriprep_1600"
    fmriprep_ts = base / "preprocessed_data_fir" / "fmriprep_TS_persession"
    participants_1600_path = base / "raw_data" / "SRPBS_1600" / "SRPBS_1600" / "participants.tsv"
    participants_ts_path = base / "raw_data" / "SRPBS_Raw_TS_Data" / "SRPBS_TS" / "sourcedata" / "participants.tsv"
    out_1600 = base / "preprocessed_data_fir" / "roi_timeseries_1600_full"
    out_ts = base / "preprocessed_data_fir" / "roi_timeseries_traveling_full"

    # Atlas
    logging.info(f"Fetching Schaefer {N_ROIS} atlas ({NETWORKS}-network)...")
    atlas = fetch_atlas_schaefer_2018(n_rois=N_ROIS, resolution_mm=2)
    atlas_maps = atlas["maps"]

    # ===================== SRPBS-1600 =====================
    if args.dataset in ("1600", "both"):
        out_1600.mkdir(parents=True, exist_ok=True)
        participants_1600 = load_1600_participants(participants_1600_path)
        logging.info(f"[1600] Loaded info for {len(participants_1600)} subjects")
        logging.info(f"[1600] Writing full post-trim time series to: {out_1600}")

        bold_files = sorted(fmriprep_1600.glob("*/func/*MNI152*res-2*preproc_bold.nii.gz"))
        if args.subjects:
            bold_files = [b for b in bold_files if b.parts[-3] in args.subjects]
        existing = {f.stem.replace("_roi_ts", "") for f in out_1600.glob("*_roi_ts.npy")}
        todo = [b for b in bold_files if b.parts[-3] not in existing]
        logging.info(f"[1600] {len(bold_files)} total, {len(existing)} done, {len(todo)} to process")

        if todo:
            results = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(process_1600_subject)(b, participants_1600, out_1600, atlas_maps) for b in todo
            )
            ok = sum(1 for r in results if r is not None)
            logging.info(f"[1600] Done: {ok}/{len(results)} succeeded")

    # ================ Traveling Subjects ================
    if args.dataset in ("traveling", "both"):
        out_ts.mkdir(parents=True, exist_ok=True)
        ts_participants = load_ts_participants(participants_ts_path)
        logging.info(f"[TS] Loaded info for {len(ts_participants)} traveling subjects")
        logging.info(f"[TS] Writing full post-trim time series to: {out_ts}")

        # Find run-01 BOLD files for each session
        bold_files = sorted(fmriprep_ts.glob("*/ses-*/func/*run-01*MNI152*res-2*preproc_bold.nii.gz"))
        if args.subjects:
            bold_files = [b for b in bold_files if b.parts[-4] in args.subjects]
        existing = {f.stem.replace("_roi_ts", "") for f in out_ts.glob("*_roi_ts.npy")}
        todo_filtered = []
        for b in bold_files:
            sub = b.parts[-4]
            ses = [p for p in b.parts if p.startswith("ses-")][0]
            out_name = f"{sub}_{ses}"
            if out_name not in existing:
                todo_filtered.append(b)
        logging.info(f"[TS] {len(bold_files)} total sessions, {len(existing)} done, {len(todo_filtered)} to process")

        if todo_filtered:
            results = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(process_ts_session)(b, ts_participants, out_ts, atlas_maps) for b in todo_filtered
            )
            ok = sum(1 for r in results if r is not None)
            logging.info(f"[TS] Done: {ok}/{len(results)} succeeded")

    # ================== Save Metadata CSV ==================
    all_metadata = []
    for meta_dir in [out_1600, out_ts]:
        if not meta_dir.exists():
            continue
        for meta_file in sorted(meta_dir.glob("*_meta.json")):
            with open(meta_file) as f:
                all_metadata.append(json.load(f))

    if all_metadata:
        csv_path = base / "preprocessed_data_fir" / "subject_metadata.csv"
        columns = [
            "subject", "session", "dataset", "is_traveling_subject",
            "site", "site_name", "site_session", "protocol",
            "diag", "diag_name", "age", "sex", "hand",
            "tr", "tr_source", "original_vols", "post_trim_vols",
            "n_timepoints", "target_vols", "timepoints_cropped", "n_rois",
        ]
        rows = []
        for m in all_metadata:
            row = {c: m.get(c, "") for c in columns}
            rows.append(row)
        df = pd.DataFrame(rows, columns=columns)
        df = df.sort_values(["dataset", "subject", "session"]).reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved metadata CSV: {csv_path} ({len(df)} rows)")
    else:
        logging.warning("No metadata JSON files found; subject_metadata.csv was not rewritten")

    logging.info("All done!")


if __name__ == "__main__":
    main()
