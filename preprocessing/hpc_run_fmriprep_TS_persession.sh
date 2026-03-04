#!/bin/bash
#SBATCH --job-name=fmriprep_TS
#SBATCH --output=logs/fmriprep_TS_%A_%a.out
#SBATCH --error=logs/fmriprep_TS_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --tmp=50G
#
# hpc_run_fmriprep_TS_persession.sh — fMRIPrep for SRPBS Travelling Subjects
# ============================================================================
#
# FINAL production script for per-session independent processing.
#
# Each subject-session is processed INDEPENDENTLY so that each site's own T1w
# drives the anatomical pipeline. This ensures site-specific scanner bias
# propagates through preprocessing identically to how the 1600-subject dataset
# is processed — critical for multi-site generalization studies.
#
# Features:
#   - Per-session BIDS filter: each job sees only ONE session's anat + bold + fmap
#   - Auto-detection of SDC strategy (fieldmap vs SyN fallback)
#   - Skip-if-already-completed logic for safe restarts
#   - Node-local scratch for I/O performance (SLURM_TMPDIR)
#
# Dataset: 9 subjects × ~16 sessions (sites) = ~141 subject-sessions
#
# Prerequisites:
#   1. bash hpc_setup_fmriprep.sh  (pulls container)
#   2. SRPBS_TS_BIDS/ with subject_session_list.txt
#      Format: one "sub-XX ses-siteYYY" pair per line
#   3. FreeSurfer license at FS_LICENSE path
#
# Usage:
#   # Process all subject-sessions (24 concurrent):
#   sbatch --array=1-$(wc -l < /scratch/shyamal/SRPBS_TS_BIDS/subject_session_list.txt)%24 hpc_run_fmriprep_TS_persession.sh
#
#   # Process first 5 as a test:
#   sbatch --array=1-5 hpc_run_fmriprep_TS_persession.sh
#
#   # Process a single subject-session (e.g., line 17):
#   sbatch --array=17 hpc_run_fmriprep_TS_persession.sh
#
set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

BIDS_DIR="/scratch/shyamal/SRPBS_TS_BIDS"
OUTPUT_DIR="/scratch/shyamal/derivatives/fmriprep_TS_persession"
WORK_DIR="/scratch/shyamal/work/fmriprep_TS_persession"
FS_LICENSE="/scratch/shyamal/freesurfer_license.txt"

FMRIPREP_VERSION="24.1.1"
SIF_FILE="/scratch/shyamal/containers/fmriprep-${FMRIPREP_VERSION}.sif"

# Resources (must match #SBATCH directives)
NTHREADS=8
MEM_MB=22000
OMP_NTHREADS=4

# ============================================================================
# ENVIRONMENT
# ============================================================================

module load apptainer/1.3.5

TEMPLATEFLOW_DIR="/scratch/shyamal/templateflow"
mkdir -p "${TEMPLATEFLOW_DIR}"
export APPTAINERENV_TEMPLATEFLOW_HOME=/templateflow
export SINGULARITYENV_TEMPLATEFLOW_HOME=/templateflow

mkdir -p "${OUTPUT_DIR}" "${WORK_DIR}" logs

# Use SLURM_TMPDIR for fast node-local I/O (auto-cleaned by SLURM)
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR}" ]; then
    LOCAL_WORK="${SLURM_TMPDIR}/fmriprep_TS_${SLURM_ARRAY_TASK_ID}"
    mkdir -p "${LOCAL_WORK}"
else
    LOCAL_WORK="${WORK_DIR}"
    mkdir -p "${LOCAL_WORK}"
fi

# ============================================================================
# RESOLVE SUBJECT + SESSION FROM ARRAY INDEX
# ============================================================================

SESSION_LIST="${BIDS_DIR}/subject_session_list.txt"

if [ ! -f "${SESSION_LIST}" ]; then
    echo "ERROR: Session list not found: ${SESSION_LIST}"
    exit 1
fi

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SESSION_LIST}")

if [ -z "${LINE}" ]; then
    echo "ERROR: No entry for array index ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

SUB=$(echo "${LINE}" | awk '{print $1}')
SES=$(echo "${LINE}" | awk '{print $2}')
SUB_LABEL="${SUB#sub-}"
SES_LABEL="${SES#ses-}"

echo "============================================"
echo "  fMRIPrep — SRPBS Travelling Subjects"
echo "  (Per-Session Independent Processing)"
echo "============================================"
echo "Job ID          : ${SLURM_JOB_ID}"
echo "Array Task ID   : ${SLURM_ARRAY_TASK_ID}"
echo "Subject         : ${SUB}"
echo "Session         : ${SES}"
echo "Node            : $(hostname)"
echo "CPUs            : ${NTHREADS}"
echo "Memory          : ${MEM_MB} MB"
echo "BIDS directory  : ${BIDS_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Work directory  : ${LOCAL_WORK}"
echo "Start time      : $(date)"
echo "============================================"

# ============================================================================
# SKIP IF ALREADY COMPLETED
# ============================================================================

DONE_FILE="${OUTPUT_DIR}/${SUB}/${SES}/func/${SUB}_${SES}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
if [ -f "${DONE_FILE}" ]; then
    echo ">>> ${SUB} ${SES} already processed. Skipping."
    exit 0
fi

# ============================================================================
# AUTO-SELECT SDC STRATEGY
# ============================================================================
# Must determine this BEFORE writing the BIDS filter file, because we
# conditionally include the fmap entry (omitting it avoids confusing
# fMRIPrep's fieldmap estimator when no fmap data exists for this session).

SES_DIR="${BIDS_DIR}/${SUB}/${SES}"
SDC_ARGS=""
HAS_FMAPS=false

if ls "${SES_DIR}/fmap/"*.nii* 1>/dev/null 2>&1; then
    echo ">>> SDC: Fieldmap-based (fieldmaps found for ${SES})"
    HAS_FMAPS=true
else
    echo ">>> SDC: SyN (no fieldmaps for ${SES})"
    SDC_ARGS="--ignore fieldmaps --use-syn-sdc warn --force-syn"
fi

# ============================================================================
# CREATE BIDS FILTER FILE — restrict to this single session
# ============================================================================
# Key design: fMRIPrep only sees anat + bold (+ optionally fmap) from THIS
# session, so this site's T1w drives the entire anatomical pipeline.

FILTER_FILE="${LOCAL_WORK}/bids_filter_${SUB_LABEL}_${SES_LABEL}.json"
FILTER_FILE_CONTAINER="/work/bids_filter_${SUB_LABEL}_${SES_LABEL}.json"

if [ "${HAS_FMAPS}" = true ]; then
    cat > "${FILTER_FILE}" <<EOF
{
    "t1w": {
        "datatype": "anat",
        "suffix": "T1w",
        "session": "${SES_LABEL}"
    },
    "t2w": {
        "datatype": "anat",
        "suffix": "T2w",
        "session": "${SES_LABEL}"
    },
    "bold": {
        "datatype": "func",
        "suffix": "bold",
        "session": "${SES_LABEL}"
    },
    "fmap": {
        "datatype": "fmap",
        "session": "${SES_LABEL}"
    }
}
EOF
else
    cat > "${FILTER_FILE}" <<EOF
{
    "t1w": {
        "datatype": "anat",
        "suffix": "T1w",
        "session": "${SES_LABEL}"
    },
    "t2w": {
        "datatype": "anat",
        "suffix": "T2w",
        "session": "${SES_LABEL}"
    },
    "bold": {
        "datatype": "func",
        "suffix": "bold",
        "session": "${SES_LABEL}"
    }
}
EOF
fi

echo ">>> BIDS filter file:"
cat "${FILTER_FILE}"
echo ""

# ============================================================================
# RUN FMRIPREP
# ============================================================================

echo ">>> Running fMRIPrep for ${SUB} / ${SES} ..."

apptainer run --cleanenv \
    --bind "${BIDS_DIR}":/data:ro \
    --bind "${OUTPUT_DIR}":/out \
    --bind "${LOCAL_WORK}":/work \
    --bind "${FS_LICENSE}":/opt/freesurfer/license.txt:ro \
    --bind "${TEMPLATEFLOW_DIR}":/templateflow \
    --env TEMPLATEFLOW_HOME=/templateflow \
    "${SIF_FILE}" \
    /data /out participant \
    --participant-label "${SUB_LABEL}" \
    --bids-filter-file "${FILTER_FILE_CONTAINER}" \
    --output-spaces MNI152NLin2009cAsym:res-2 anat \
    --nprocs ${NTHREADS} \
    --omp-nthreads ${OMP_NTHREADS} \
    --mem ${MEM_MB} \
    --work-dir /work \
    --fs-no-reconall \
    --skip-bids-validation \
    --notrack \
    --clean-workdir \
    ${SDC_ARGS}

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Finished ${SUB} / ${SES}"
echo "  Exit code: ${EXIT_CODE}"
echo "  End time : $(date)"
echo "============================================"

exit ${EXIT_CODE}
