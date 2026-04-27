#!/usr/bin/env python3
"""
convert_srpbs1600_to_bids.py
============================
Convert the SRPBS 1600-subject multi-disorder dataset to BIDS format
suitable for fMRIPrep processing.

Source layout (per subject):
    data/sub-XXXX/t1/defaced_mprage.nii
    data/sub-XXXX/rsfmri/vol_001 .. vol_NNN   (individual 3-D NIfTI volumes, no extension)
    data/sub-XXXX/fmap/data_0000.nii  data_0001.nii  phase.nii

Output BIDS layout:
    sub-XXXX/anat/sub-XXXX_T1w.nii.gz
    sub-XXXX/func/sub-XXXX_task-rest_bold.nii.gz   (merged 4-D)
    sub-XXXX/func/sub-XXXX_task-rest_bold.json
    sub-XXXX/fmap/sub-XXXX_magnitude1.nii.gz
    sub-XXXX/fmap/sub-XXXX_magnitude2.nii.gz
    sub-XXXX/fmap/sub-XXXX_phasediff.nii.gz
    sub-XXXX/fmap/sub-XXXX_phasediff.json

Usage:
    python convert_srpbs1600_to_bids.py                   # all subjects
    python convert_srpbs1600_to_bids.py --subjects sub-0001 sub-0002
    python convert_srpbs1600_to_bids.py --dry-run         # preview only
    python convert_srpbs1600_to_bids.py --nprocs 8        # parallel jobs
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import gzip
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================================
# PATHS -- override with CLI arguments if needed
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "raw_data" / "SRPBS_1600" / "SRPBS_1600"
BIDS_ROOT = REPO_ROOT / "raw_data" / "SRPBS_1600_BIDS"

# ============================================================================
# MRI PROTOCOL PARAMETERS (from MRI_protocols_rsMRI.tsv & MRI_protocols_T1w.tsv)
# ============================================================================

# rsfMRI (EPI) parameters by protocol number
# PhaseEncodingDirection: PA = j- (posterior-to-anterior), AP = j (anterior-to-posterior)
BOLD_PARAMS = {
    1:  {"RepetitionTime": 2.5, "EchoTime": 0.030, "FlipAngle": 80, "PhaseEncodingDirection": "j-",  "Manufacturer": "Siemens", "ManufacturersModelName": "Verio", "MagneticFieldStrength": 3},
    2:  {"RepetitionTime": 2.0, "EchoTime": 0.027, "FlipAngle": 90, "PhaseEncodingDirection": "j-",  "Manufacturer": "GE", "ManufacturersModelName": "Signa HDxt", "MagneticFieldStrength": 3},
    3:  {"RepetitionTime": 2.0, "EchoTime": 0.027, "FlipAngle": 90, "PhaseEncodingDirection": "j",   "Manufacturer": "GE", "ManufacturersModelName": "Signa HDxt", "MagneticFieldStrength": 3},
    4:  {"RepetitionTime": 2.7, "EchoTime": 0.031, "FlipAngle": 90, "PhaseEncodingDirection": "j",   "Manufacturer": "Siemens", "ManufacturersModelName": "Spectra", "MagneticFieldStrength": 3},
    5:  {"RepetitionTime": 2.5, "EchoTime": 0.030, "FlipAngle": 80, "PhaseEncodingDirection": "j",   "Manufacturer": "Siemens", "ManufacturersModelName": "MAGNETOM Verio", "MagneticFieldStrength": 3},
    6:  {"RepetitionTime": 2.5, "EchoTime": 0.030, "FlipAngle": 80, "PhaseEncodingDirection": "j-",  "Manufacturer": "Siemens", "ManufacturersModelName": "TimTrio", "MagneticFieldStrength": 3},
    7:  {"RepetitionTime": 2.0, "EchoTime": 0.030, "FlipAngle": 90, "PhaseEncodingDirection": "j",   "Manufacturer": "Siemens", "ManufacturersModelName": "Trio", "MagneticFieldStrength": 3},
    8:  {"RepetitionTime": 2.5, "EchoTime": 0.030, "FlipAngle": 80, "PhaseEncodingDirection": "j-",  "Manufacturer": "GE", "ManufacturersModelName": "Discovery MR750w", "MagneticFieldStrength": 3},
    9:  {"RepetitionTime": 2.5, "EchoTime": 0.030, "FlipAngle": 80, "PhaseEncodingDirection": "j-",  "Manufacturer": "Siemens", "ManufacturersModelName": "Verio", "MagneticFieldStrength": 3},
    10: {"RepetitionTime": 2.5, "EchoTime": 0.030, "FlipAngle": 80, "PhaseEncodingDirection": "j-",  "Manufacturer": "Siemens", "ManufacturersModelName": "Verio", "MagneticFieldStrength": 3},
    11: {"RepetitionTime": 2.5, "EchoTime": 0.030, "FlipAngle": 80, "PhaseEncodingDirection": "j-",  "Manufacturer": "Siemens", "ManufacturersModelName": "TimTrio", "MagneticFieldStrength": 3},
    12: {"RepetitionTime": 2.5, "EchoTime": 0.030, "FlipAngle": 80, "PhaseEncodingDirection": "j-",  "Manufacturer": "Siemens", "ManufacturersModelName": "TimTrio", "MagneticFieldStrength": 3},
    13: {"RepetitionTime": 2.5, "EchoTime": 0.030, "FlipAngle": 80, "PhaseEncodingDirection": "j-",  "Manufacturer": "Siemens", "ManufacturersModelName": "TimTrio", "MagneticFieldStrength": 3},
    14: {"RepetitionTime": 2.5, "EchoTime": 0.030, "FlipAngle": 80, "PhaseEncodingDirection": "j",   "Manufacturer": "Siemens", "ManufacturersModelName": "TimTrio", "MagneticFieldStrength": 3},
}

# Fieldmap echo times per manufacturer (verified against TS dataset DICOM headers)
# Siemens: TE1=4.92ms, TE2=7.38ms (confirmed from ATT/SWA/KUT/COI/ATV sites)
# GE:      TE1=4.90ms, TE2=7.40ms (confirmed from UTO site - Discovery MR750w)
FMAP_ECHO_TIMES = {
    "Siemens": {"EchoTime1": 0.00492, "EchoTime2": 0.00738},
    "GE":      {"EchoTime1": 0.0049,  "EchoTime2": 0.0074},
}

# Site-specific readout timing parameters (from travelling subject dataset)
# These are required for fieldmap-based susceptibility distortion correction.
# Sources:
#   HUH/HRC (GE Signa HDxt)     → TS sub-01/ses-siteHUH
#   HKH     (Siemens Spectra)    → TS sub-01/ses-siteHKH
#   KTT     (Siemens Trio)       → TS sub-01/ses-siteKUT
#   CIN/NKN/OSU (Siemens TimTrio)→ TS sub-01/ses-siteATTd1
SITE_READOUT_PARAMS = {
    "HUH": {"TotalReadoutTime": 0.045612,  "EffectiveEchoSpacing": 0.000724},
    "HRC": {"TotalReadoutTime": 0.045612,  "EffectiveEchoSpacing": 0.000724},
    "HKH": {"TotalReadoutTime": 0.0359104, "EffectiveEchoSpacing": 0.000570006},
    "KTT": {"TotalReadoutTime": 0.0352797, "EffectiveEchoSpacing": 0.000559996},
    "CIN": {"TotalReadoutTime": 0.0308698, "EffectiveEchoSpacing": 0.000489996},
    "NKN": {"TotalReadoutTime": 0.0308698, "EffectiveEchoSpacing": 0.000489996},
    "OSU": {"TotalReadoutTime": 0.0308698, "EffectiveEchoSpacing": 0.000489996},
}

TASK_NAME = "rest"


def load_participants(raw_root):
    """Load participants.tsv -> dict: sub_id -> {site, protocol, ...}."""
    tsv_path = raw_root / "participants.tsv"
    participants = {}
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sub_id = row["participant_id"].strip()
            participants[sub_id] = {
                "site": row["site"].strip(),
                "protocol": int(row["protocol"].strip()),
                "diag": row.get("diag", "").strip(),
                "age": row.get("age", "").strip(),
                "sex": row.get("sex", "").strip(),
                "hand": row.get("hand", "").strip(),
            }
    return participants


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"    wrote {path.name}")


def gzip_copy(src, dst):
    with open(src, "rb") as f_in:
        with gzip.open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def fsl_convert(src, dst_stem):
    """Convert a NIfTI file to .nii.gz via fslchfiletype."""
    try:
        subprocess.run(
            ["fslchfiletype", "NIFTI_GZ", str(src), dst_stem],
            check=True, capture_output=True, text=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


def merge_volumes_fsl(vol_files, output_path):
    """Merge individual 3-D NIfTI volumes (no extension) into a 4-D file.
    
    Creates temporary symlinks with .nii extension since FSL requires
    recognized NIfTI extensions.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="bids_merge_"))
    symlinks = []
    try:
        # Create symlinks with .nii extension
        for v in vol_files:
            link = tmpdir / f"{v.name}.nii"
            link.symlink_to(v)
            symlinks.append(str(link))

        # fslmerge -t outputs .nii.gz by default
        dst_stem = str(output_path).replace(".nii.gz", "")
        cmd = ["fslmerge", "-t", dst_stem] + symlinks
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return True
    finally:
        # Clean up symlinks and temp dir
        for s in symlinks:
            try:
                os.unlink(s)
            except OSError:
                pass
        try:
            tmpdir.rmdir()
        except OSError:
            pass


def create_dataset_description(bids_root):
    desc = {
        "Name": "SRPBS Multi-disorder MRI Dataset",
        "BIDSVersion": "1.6.0",
        "DatasetType": "raw",
        "License": "CC BY 4.0",
        "Authors": [
            "Ayumu Yamashita", "Noriaki Yahata", "Takashi Itahashi",
            "Giuseppe Lisi", "Takashi Yamada", "Naho Ichikawa",
            "Masahiro Takamura", "Yujiro Yoshihara", "Akira Kunimatsu",
            "Naohiro Okada", "Hirotaka Yamagata", "Koji Matsuo",
            "Ryuichiro Hashimoto", "Go Okada", "Yuki Sakai",
            "Jun Morimoto", "Jin Narumoto", "Yasuhiro Shimada",
            "Kiyoto Kasai", "Nobumasa Kato", "Hidehiko Takahashi",
            "Yasumasa Okamoto", "Saori C Tanaka", "Okito Yamashita",
            "Mitsuo Kawato", "Hiroshi Imamizu"
        ],
        "ReferencesAndLinks": [
            "Yamashita A, et al. (2019) PLOS Biology 17(4): e3000042. https://doi.org/10.1371/journal.pbio.3000042"
        ],
        "Funding": [
            "Japan Agency for Medical Research and Development (AMED) Grant Numbers JP18dm0307008, JP17dm0107044, JP18dm0307002, JP18dm0307004, JP18dm0307009."
        ],
        "GeneratedBy": [{
            "Name": "convert_srpbs1600_to_bids.py",
            "Description": "Custom BIDS conversion script for SRPBS-1600 dataset"
        }]
    }
    write_json(bids_root / "dataset_description.json", desc)


def create_participants_files(raw_root, bids_root, participants):
    tsv_out = bids_root / "participants.tsv"
    with open(tsv_out, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["participant_id", "site", "protocol", "diag", "age", "sex", "hand"])
        for sub_id in sorted(participants.keys()):
            p = participants[sub_id]
            writer.writerow([sub_id, p["site"], p["protocol"], p["diag"], p["age"], p["sex"], p["hand"]])
    print(f"  wrote participants.tsv ({len(participants)} subjects)")

    pjson = {
        "participant_id": {"Description": "Subject identifier"},
        "site": {
            "Description": "Imaging site code",
            "Levels": {
                "SWA": "Showa University", "HUH": "Hiroshima University Hospital",
                "HRC": "Hiroshima Rehabilitation Center", "HKH": "Hiroshima Kajikawa Hospital",
                "COI": "Hiroshima COI", "KUT": "Kyoto University (TimTrio)",
                "KTT": "Kyoto University (Trio)", "UTO": "University of Tokyo",
                "ATV": "ATR (Verio)", "ATT": "ATR (TimTrio)",
                "NKN": "CiNet (Nishinomiya Kyouritsu)", "OSU": "CiNet (Osaka University)",
                "CIN": "CiNet"
            }
        },
        "protocol": {"Description": "MRI protocol number (1-14)"},
        "diag": {"Description": "Diagnosis", "Levels": {"0": "Healthy control", "1": "Patient"}},
        "age": {"Description": "Age in years", "Units": "years"},
        "sex": {"Description": "Biological sex", "Levels": {"1": "Male", "2": "Female"}},
        "hand": {"Description": "Handedness", "Levels": {"1": "Right", "2": "Left"}}
    }
    write_json(bids_root / "participants.json", pjson)


def create_task_json(bids_root):
    write_json(bids_root / "task-rest_bold.json", {
        "TaskName": "rest",
        "TaskDescription": "Resting-state fMRI with eyes open, fixating on crosshair."
    })



def identify_fmap_files(fmap_dir):
    """
    Identify magnitude and phasediff files from various naming conventions.

    All patterns follow the same structure:
      - 2 magnitude images (or 1 dual-echo magnitude)
      - 1 phasediff image

    Returns: (mag1_path, mag2_path, phasediff_path)
             mag2_path may be None if only 2 files are present.
    """
    import re as _re

    files = sorted([f for f in fmap_dir.iterdir() if f.is_file()])
    names = [f.name for f in files]

    if len(files) == 0:
        return None, None, None

    # --- Pattern 1: data_0000.nii, data_0001.nii, phase.nii ---
    if "data_0000.nii" in names:
        return (fmap_dir / "data_0000.nii",
                fmap_dir / "data_0001.nii",
                fmap_dir / "phase.nii")

    # --- Pattern 2: fm49_01.nii, fm49_02.nii, fm74.nii (or fm52 variants) ---
    fm_mag = sorted([f for f in files if _re.match(r'fm\d+_\d+\.nii', f.name)])
    fm_phase = sorted([f for f in files if _re.match(r'fm\d+\.nii', f.name)])
    if len(fm_mag) >= 2 and len(fm_phase) >= 1:
        return fm_mag[0], fm_mag[1], fm_phase[0]

    # --- Pattern 3: TE4.9_1, TE4.9_2, TE7.4_1, etc. (extensionless) ---
    te_short = sorted([f for f in files if _re.match(r'TE4\.\d+_\d+', f.name)])
    te_long = sorted([f for f in files if _re.match(r'TE7\.\d+_\d+', f.name)])
    if len(te_short) >= 1 and len(te_long) >= 1:
        mag1 = te_short[0]
        mag2 = te_short[1] if len(te_short) > 1 else None
        phase = te_long[-1]
        return mag1, mag2, phase
    if len(te_long) >= 2:
        return te_long[0], te_long[1] if len(te_long) > 2 else None, te_long[-1]

    # --- Pattern 4: fieldmaps003a1001_1.nii, _2.nii, fieldmaps004a2001.nii ---
    mag_files = sorted([f for f in files if _re.search(r'_[12]\.nii$', f.name)])
    phase_files = sorted([f for f in files if f.name.endswith('.nii') and not _re.search(r'_[12]\.nii$', f.name)])
    if len(mag_files) >= 1 and len(phase_files) >= 1:
        mag1 = mag_files[0]
        mag2 = mag_files[1] if len(mag_files) > 1 else None
        return mag1, mag2, phase_files[0]

    # --- Pattern 5: 2 or 3 .nii files (generic) ---
    nii_files = sorted([f for f in files if f.name.endswith('.nii')])
    if len(nii_files) == 2:
        return nii_files[0], None, nii_files[1]
    if len(nii_files) == 3:
        return nii_files[0], nii_files[1], nii_files[2]

    # --- Fallback for extensionless files ---
    if len(files) == 3:
        return files[0], files[1], files[2]
    if len(files) == 2:
        return files[0], None, files[1]
    if len(files) == 6:
        return files[0], files[1], files[-1]

    return None, None, None

def convert_subject(sub_id, raw_root, bids_root, participant_info, dry_run=False):
    """Convert a single subject to BIDS format."""
    src = raw_root / "data" / sub_id
    if not src.exists():
        print(f"  [SKIP] {sub_id}: source not found")
        return False

    protocol = participant_info["protocol"]
    site = participant_info["site"]

    if protocol not in BOLD_PARAMS:
        print(f"  [WARN] {sub_id}: unknown protocol {protocol}, using 1")
        protocol = 1
    bold_params = BOLD_PARAMS[protocol]

    dst = bids_root / sub_id
    anat_dir = dst / "anat"
    func_dir = dst / "func"
    fmap_dir = dst / "fmap"

    if dry_run:
        print(f"  [DRY-RUN] {sub_id} (site={site}, protocol={protocol})")
        return True

    print(f"  Converting {sub_id} (site={site}, protocol={protocol})...")
    anat_dir.mkdir(parents=True, exist_ok=True)
    func_dir.mkdir(parents=True, exist_ok=True)
    fmap_dir.mkdir(parents=True, exist_ok=True)

    success = True

    # ---- ANAT (T1w) ----
    t1_src = src / "t1" / "defaced_mprage.nii"
    t1_dst = anat_dir / f"{sub_id}_T1w.nii.gz"
    if t1_src.exists() and not t1_dst.exists():
        dst_stem = str(t1_dst).replace(".nii.gz", "")
        if not fsl_convert(t1_src, dst_stem):
            gzip_copy(t1_src, t1_dst)
        print(f"    anat: {t1_dst.name}")
    elif t1_dst.exists():
        print(f"    anat: exists, skipping")
    else:
        print(f"    [WARN] T1w not found"); success = False

    # ---- FUNC (rsfMRI) -- merge volumes ----
    rsfmri_dir = src / "rsfmri"
    bold_dst = func_dir / f"{sub_id}_task-rest_bold.nii.gz"
    bold_json_dst = func_dir / f"{sub_id}_task-rest_bold.json"

    if rsfmri_dir.exists() and not bold_dst.exists():
        vol_files = sorted(rsfmri_dir.glob("vol_*"))
        if len(vol_files) == 0:
            print(f"    [WARN] No volumes found"); success = False
        else:
            try:
                merge_volumes_fsl(vol_files, bold_dst)
                print(f"    func: merged {len(vol_files)} volumes -> {bold_dst.name}")
            except Exception as e:
                print(f"    [ERROR] fslmerge: {e}"); success = False
    elif bold_dst.exists():
        print(f"    func: exists, skipping")
    else:
        print(f"    [WARN] rsfMRI not found"); success = False

    # BOLD JSON
    if not bold_json_dst.exists():
        bold_json = {
            "TaskName": TASK_NAME,
            "RepetitionTime": bold_params["RepetitionTime"],
            "EchoTime": bold_params["EchoTime"],
            "FlipAngle": bold_params["FlipAngle"],
            "PhaseEncodingDirection": bold_params["PhaseEncodingDirection"],
            "Manufacturer": bold_params["Manufacturer"],
            "ManufacturersModelName": bold_params["ManufacturersModelName"],
            "MagneticFieldStrength": bold_params["MagneticFieldStrength"],
        }
        # Add site-specific readout timing if available (needed for fieldmap SDC)
        if site in SITE_READOUT_PARAMS:
            bold_json["TotalReadoutTime"] = SITE_READOUT_PARAMS[site]["TotalReadoutTime"]
            bold_json["EffectiveEchoSpacing"] = SITE_READOUT_PARAMS[site]["EffectiveEchoSpacing"]
        write_json(bold_json_dst, bold_json)

    # ---- FMAP (fieldmaps) ----
    fmap_src = src / "fmap"
    if fmap_src.exists():
        mag1_src, mag2_src, phase_src = identify_fmap_files(fmap_src)

        mag1_dst = fmap_dir / f"{sub_id}_magnitude1.nii.gz"
        mag2_dst = fmap_dir / f"{sub_id}_magnitude2.nii.gz"
        phdiff_dst = fmap_dir / f"{sub_id}_phasediff.nii.gz"
        phdiff_json_dst = fmap_dir / f"{sub_id}_phasediff.json"

        if mag1_src is None or phase_src is None:
            print(f"    [WARN] Could not identify fmap files in {fmap_src}")
            print(f"           Files: {[f.name for f in sorted(fmap_src.iterdir())]}")
        else:
            fmap_pairs = [(mag1_src, mag1_dst, "magnitude1")]
            if mag2_src is not None:
                fmap_pairs.append((mag2_src, mag2_dst, "magnitude2"))
            fmap_pairs.append((phase_src, phdiff_dst, "phasediff"))

            for src_f, dst_f, label in fmap_pairs:
                if src_f is not None and not dst_f.exists():
                    try:
                        subprocess.run(
                            ["fslchfiletype", "NIFTI_GZ", str(src_f),
                             str(dst_f.with_suffix("").with_suffix(""))],
                            check=True, capture_output=True, text=True
                        )
                        print(f"    fmap: {dst_f.name}")
                    except subprocess.CalledProcessError:
                        import gzip as gz
                        # Handle extensionless files - read raw bytes
                        with open(src_f, "rb") as f_in:
                            with gz.open(dst_f, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        print(f"    fmap: {dst_f.name} (gzip fallback)")
                elif dst_f.exists():
                    print(f"    fmap: {label} already exists")

            # Phasediff JSON sidecar with IntendedFor
            if not phdiff_json_dst.exists():
                manufacturer = bold_params["Manufacturer"]
                echo_times = FMAP_ECHO_TIMES.get(manufacturer, FMAP_ECHO_TIMES["Siemens"])
                phdiff_json = {
                    "EchoTime1": echo_times["EchoTime1"],
                    "EchoTime2": echo_times["EchoTime2"],
                    "IntendedFor": f"bids::{sub_id}/func/{sub_id}_task-rest_bold.nii.gz"
                }
                write_json(phdiff_json_dst, phdiff_json)
    else:
        print(f"    [INFO] No fieldmap directory for {sub_id}")

    return success


def main():
    parser = argparse.ArgumentParser(description="Convert SRPBS-1600 to BIDS")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("--raw-root", type=str, default=str(RAW_ROOT))
    parser.add_argument("--bids-root", type=str, default=str(BIDS_ROOT))
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    bids_root = Path(args.bids_root)

    print("=" * 60)
    print("  SRPBS-1600 -> BIDS Conversion")
    print("=" * 60)
    print(f"  Raw data : {raw_root}")
    print(f"  BIDS out : {bids_root}")
    print(f"  Dry run  : {args.dry_run}")
    print(f"  Processes: {args.nprocs}")
    print("=" * 60)

    participants = load_participants(raw_root)
    print(f"\nLoaded metadata for {len(participants)} participants")

    subject_list = args.subjects if args.subjects else sorted(participants.keys())
    print(f"Subjects to convert: {len(subject_list)}")

    if not args.dry_run:
        bids_root.mkdir(parents=True, exist_ok=True)
        create_dataset_description(bids_root)
        create_participants_files(raw_root, bids_root, participants)
        create_task_json(bids_root)
        readme_src = raw_root / "README.txt"
        readme_dst = bids_root / "README"
        if readme_src.exists() and not readme_dst.exists():
            shutil.copy2(readme_src, readme_dst)
            print("  Copied README")
        bidsignore = bids_root / ".bidsignore"
        if not bidsignore.exists():
            bidsignore.write_text("# Ignore non-BIDS files\n")

    print(f"\n--- Converting {len(subject_list)} subjects ---\n")
    succeeded = failed = 0

    if args.nprocs > 1 and not args.dry_run:
        with ProcessPoolExecutor(max_workers=args.nprocs) as executor:
            futures = {}
            for sub_id in subject_list:
                if sub_id not in participants:
                    print(f"  [SKIP] {sub_id}: not in participants.tsv"); continue
                futures[executor.submit(convert_subject, sub_id, raw_root, bids_root, participants[sub_id], args.dry_run)] = sub_id
            for future in as_completed(futures):
                try:
                    if future.result(): succeeded += 1
                    else: failed += 1
                except Exception as e:
                    print(f"  [ERROR] {futures[future]}: {e}"); failed += 1
    else:
        for sub_id in subject_list:
            if sub_id not in participants:
                print(f"  [SKIP] {sub_id}: not in participants.tsv"); continue
            try:
                if convert_subject(sub_id, raw_root, bids_root, participants[sub_id], args.dry_run):
                    succeeded += 1
                else: failed += 1
            except Exception as e:
                print(f"  [ERROR] {sub_id}: {e}"); failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Done! Succeeded: {succeeded}, Failed: {failed}")
    print(f"  BIDS dir: {bids_root}")
    print(f"{'=' * 60}")
    if not args.dry_run:
        print(f"\nNext steps:")
        print(f"  1. Validate: bids-validator {bids_root}")
        print(f"  2. Run: bash run_fmriprep_1600.sh")


if __name__ == "__main__":
    main()
