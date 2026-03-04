#!/bin/bash
#SBATCH --job-name=fmriprep_1600
#SBATCH --output=logs/fmriprep_1600_%A_%a.out
#SBATCH --error=logs/fmriprep_1600_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=50G
#
# hpc_run_fmriprep_1600.sh — fMRIPrep preprocessing for SRPBS 1600 dataset
# =========================================================================
#
# FINAL production script incorporating all patches and fixes:
#   - Auto-detection of SDC strategy (fieldmap vs SyN fallback)
#   - TotalReadoutTime now baked into BIDS JSONs (via updated conversion script)
#   - Handles subjects with no fieldmaps (SyN SDC)
#   - Handles subjects with fieldmaps but missing TotalReadoutTime (SyN fallback)
#   - Skip-if-already-completed logic for safe restarts
#   - Node-local scratch for I/O performance (SLURM_TMPDIR)
#
# Prerequisites:
#   1. bash hpc_setup_fmriprep.sh         (pulls container + generates subject list)
#   2. python convert_srpbs1600_to_bids.py (converts raw data to BIDS with all fixes)
#   3. FreeSurfer license at FS_LICENSE path
#
# Usage:
#   # Process all subjects (24 concurrent):
#   sbatch --array=1-$(wc -l < /scratch/shyamal/SRPBS_1600_BIDS/subject_list.txt)%24 hpc_run_fmriprep_1600.sh
#
#   # Process first 10 as a test:
#   sbatch --array=1-10 hpc_run_fmriprep_1600.sh
#
#   # Process a single subject (e.g., #42):
#   sbatch --array=42 hpc_run_fmriprep_1600.sh
#
#   # Rerun only incomplete subjects (from a rerun list):
#   sbatch --array=1-$(wc -l < /scratch/shyamal/rerun_subject_list.txt)%24 hpc_run_fmriprep_1600.sh
#
set -euo pipefail

# ============================================================================
# CONFIGURATION — Edit these paths to match your HPC layout
# ============================================================================

BIDS_DIR="/scratch/shyamal/SRPBS_1600_BIDS"
OUTPUT_DIR="/scratch/shyamal/derivatives/fmriprep_1600"
WORK_DIR="/scratch/shyamal/work/fmriprep_1600"
FS_LICENSE="/scratch/shyamal/freesurfer_license.txt"

FMRIPREP_VERSION="24.1.1"
SIF_FILE="/scratch/shyamal/containers/fmriprep-${FMRIPREP_VERSION}.sif"

# Subject list — switch to rerun_subject_list.txt for reruns
SUBJECT_LIST="${BIDS_DIR}/subject_list.txt"

# Resources (must match #SBATCH directives)
NTHREADS=8
MEM_MB=30000
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
    LOCAL_WORK="${SLURM_TMPDIR}/fmriprep_${SLURM_ARRAY_TASK_ID}"
    mkdir -p "${LOCAL_WORK}"
else
    LOCAL_WORK="${WORK_DIR}"
    mkdir -p "${LOCAL_WORK}"
fi

# ============================================================================
# RESOLVE SUBJECT FROM ARRAY INDEX
# ============================================================================

if [ ! -f "${SUBJECT_LIST}" ]; then
    echo "ERROR: Subject list not found: ${SUBJECT_LIST}"
    echo "       Run hpc_setup_fmriprep.sh first."
    exit 1
fi

SUB=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SUBJECT_LIST}")

if [ -z "${SUB}" ]; then
    echo "ERROR: No subject found for array index ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

SUB_LABEL="${SUB#sub-}"

echo "============================================"
echo "  fMRIPrep — SRPBS 1600"
echo "============================================"
echo "Job ID          : ${SLURM_JOB_ID}"
echo "Array Task ID   : ${SLURM_ARRAY_TASK_ID}"
echo "Subject         : ${SUB}"
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

DONE_FILE="${OUTPUT_DIR}/${SUB}/func/${SUB}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
if [ -f "${DONE_FILE}" ]; then
    echo ">>> Subject ${SUB} already processed. Skipping."
    exit 0
fi

# ============================================================================
# AUTO-SELECT SDC STRATEGY
# ============================================================================
# Fieldmap-based SDC requires BOTH:
#   1. Fieldmap NIfTIs in fmap/ directory
#   2. TotalReadoutTime in the BOLD JSON sidecar
#
# The updated convert_srpbs1600_to_bids.py now includes TotalReadoutTime for
# all sites where it was available from the travelling subject dataset:
#   HUH, HRC, HKH, KTT, CIN, NKN, OSU
#
# If either fieldmaps or TotalReadoutTime is missing, we fall back to SyN SDC.

FMAP_DIR="${BIDS_DIR}/${SUB}/fmap"
BOLD_JSON="${BIDS_DIR}/${SUB}/func/${SUB}_task-rest_bold.json"
SDC_ARGS=""

HAS_FMAP=false
HAS_TRT=false

if ls "${FMAP_DIR}"/*.nii* 1>/dev/null 2>&1; then
    HAS_FMAP=true
fi

if [ -f "${BOLD_JSON}" ] && grep -q '"TotalReadoutTime"' "${BOLD_JSON}"; then
    HAS_TRT=true
fi

if ${HAS_FMAP} && ${HAS_TRT}; then
    echo ">>> SDC: Fieldmap-based (fieldmaps + TotalReadoutTime present)"
elif ${HAS_FMAP} && ! ${HAS_TRT}; then
    echo ">>> SDC: SyN fallback (fieldmaps exist but TotalReadoutTime missing)"
    SDC_ARGS="--ignore fieldmaps --use-syn-sdc warn --force-syn"
else
    echo ">>> SDC: SyN (no fieldmaps available)"
    SDC_ARGS="--ignore fieldmaps --use-syn-sdc warn --force-syn"
fi

# ============================================================================
# RUN FMRIPREP
# ============================================================================

echo ">>> Running fMRIPrep for ${SUB} ..."

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
echo "  Finished ${SUB}"
echo "  Exit code: ${EXIT_CODE}"
echo "  End time : $(date)"
echo "============================================"

exit ${EXIT_CODE}
