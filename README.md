# MS-JEPA-CL

Multi-site JEPA with curriculum learning for resting-state fMRI site
harmonization and diagnosis transfer.

This repository contains the code used for the paper experiments around
SRPBS-1600, ABIDE-I, and ABIDE-II. The data, checkpoints, logs, cached arrays,
and generated paper outputs are not part of the repository because they are
large and dataset-access controlled.

## What This Code Does

The project compares fixed fMRI representations under site-aware evaluation:

| Method family | Purpose |
| --- | --- |
| Static FC | Classical functional-connectivity baseline |
| Static FC + ComBat/CovBat | Statistical harmonization baselines |
| Raw ROI time series | Non-harmonized temporal baseline |
| JEPA embeddings | Self-supervised temporal representations |
| FC + JEPA fusion | Complementary static and temporal features |

The main evaluation rule is:

- diagnosis transfer uses leave-one-site-out or subject-level voting protocols
- site leakage uses a separate grouped subject-level site probe
- all fold-local preprocessing, PCA, harmonization, and probes are fit only on
  training folds

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/jepa_fusion/` | Main paper pipeline: JEPA training, baseline evaluation, fusion, transfer, figures, and final table builders |
| `src/jepa_clean/` | Earlier clean JEPA implementation and FC-vs-raw analysis utilities reused by the fusion pipeline |
| `preprocessing/` | fMRIPrep launchers and Schaefer-400 ROI time-series extraction |
| `convert_to_bids/` | SRPBS raw-to-BIDS conversion utility |
| `run_paper_tables.sh` | End-to-end final paper table launcher |
| `run_transfer_5seed.sh` | Five-seed zero-shot transfer launcher |
| `run_subject_vote_ablation.sh` | Subject-vote auxiliary-loss ablation launcher |
| `generate_table2.py` | Dataset summary table generator |

## Environment

Use Python 3.10 or newer. The project is not packaged yet, so run commands from
the repository root with the root on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Core Python dependencies:

```text
numpy
pandas
scipy
scikit-learn
torch
matplotlib
nibabel
nilearn
joblib
xgboost
neuroCombat
```

Some scripts also call external neuroimaging tools through shell launchers,
especially fMRIPrep and SLURM (`sbatch`) for HPC preprocessing.

## Expected Data Layout

The code expects preprocessed and raw data under the repository root:

```text
preprocessed_data_fir/
  FC_1600/
  FC_abide_full/
  FC_abide2_full/
  full/
    roi_timeseries_1600_full/
    roi_timeseries_abide_full/
    roi_timeseries_abide2_full/
    subject_metadata_full.csv
    abide_metadata_full.csv
    abide2_metadata.csv
raw_data/
  SRPBS_1600/
  SRPBS_1600_BIDS/
```

These directories are intentionally ignored by Git.

## Preprocessing

Convert SRPBS raw data to BIDS:

```bash
python convert_to_bids/convert_srpbs1600_to_bids.py
```

Run fMRIPrep on the HPC system:

```bash
sbatch preprocessing/hpc_run_fmriprep_1600.sh
sbatch preprocessing/hpc_run_fmriprep_TS_persession.sh
```

Extract Schaefer-400 ROI time series:

```bash
python preprocessing/extract_roi_timeseries.py --dataset both --n-jobs 8
python preprocessing/extract_roi_timeseries_abide.py --n-jobs 8
```

## Main Paper Workflow

Run the full staged paper workflow:

```bash
bash run_paper_tables.sh all
```

Run selected stages only:

```bash
bash run_paper_tables.sh dataset_summary baselines
bash run_paper_tables.sh pilot_pooled curriculum aux
bash run_paper_tables.sh transfer lowk tables
```

Useful environment variables:

```bash
RUN_TAG=final_run_v1 DEVICE=cuda SKIP_EXISTING=1 bash run_paper_tables.sh all
```

Generated outputs go to:

```text
checkpoints/jepa_fusion/<RUN_TAG>/
results/reports/tables/jepa_fusion/<RUN_TAG>/
logs/<RUN_TAG>/
```

These outputs are ignored by Git.

## Common Direct Commands

Evaluate FC and ROI baselines:

```bash
python -m src.jepa_fusion.evaluate_baselines --dataset abide --seeds 0,8,16,32,128
python -m src.jepa_fusion.evaluate_baselines --dataset srpbs --seeds 0,8,16,32,128
```

Run JEPA pilot sweeps:

```bash
python -m src.jepa_fusion.run_pilot \
  --training-mode pooled \
  --train-seeds 0,8,16,32,128 \
  --eval-seeds 0,8,16,32,128 \
  --device cuda

python -m src.jepa_fusion.run_pilot \
  --training-mode curriculum \
  --train-seeds 0,8,16,32,128 \
  --eval-seeds 0,8,16,32,128 \
  --device cuda
```

Run multiseed JEPA confirmation:

```bash
python -m src.jepa_fusion.run_multiseed_confirmation \
  --seeds 0,8,16,32,128 \
  --probe-seeds 0,8,16,32,128 \
  --device cuda
```

Run zero-shot transfer:

```bash
bash run_transfer_5seed.sh A  # train SRPBS + ABIDE-I, evaluate ABIDE-II
bash run_transfer_5seed.sh B  # train SRPBS + ABIDE-II, evaluate ABIDE-I
```

Run low-K harmonization stress tests:

```bash
python -m src.jepa_fusion.evaluate_baselines_lowk_newbaseline \
  --dataset abide \
  --k-values 10,20,30,40,50

python -m src.jepa_fusion.evaluate_baselines_lowk_newbaseline \
  --dataset srpbs \
  --k-values 10,20,30,40,50
```

Run subject-level demographic probes for sex classification and age regression. JEPA uses sliding windows by default and averages window embeddings to one subject embedding:

```bash
python -m src.jepa_fusion.evaluate_demographics \
  --dataset all \
  --checkpoint checkpoints/jepa_fusion/final_run_v1/<run_name>/phase2_final.pt \
  --seeds 0,8,16,32,128 \
  --device cuda
```

## Key Modules

| Module | Purpose |
| --- | --- |
| `src/jepa_fusion/protocol.py` | Dataset presets, task labels, site normalization, task filtering |
| `src/jepa_fusion/dataset_utils.py` | Matched FC/raw record loading for evaluation |
| `src/jepa_fusion/model.py` | Main FusionJEPA architecture and rotary temporal encoder variants |
| `src/jepa_fusion/model_fourier_patch.py` | Fourier patch JEPA prototype architecture |
| `src/jepa_fusion/train_jepa.py` | Pooled and curriculum JEPA training |
| `src/jepa_fusion/evaluate_baselines.py` | Static FC, ComBat/CovBat, and ROI baseline evaluation |
| `src/jepa_fusion/evaluate_jepa.py` | Grouped-CV JEPA embedding evaluation |
| `src/jepa_fusion/evaluate_jepa_subject_vote.py` | LOSO subject-vote JEPA evaluation |
| `src/jepa_fusion/evaluate_demographics.py` | Sex and age prediction from FC variants and JEPA embeddings |
| `src/jepa_fusion/evaluate_fusion_subject_vote.py` | FC + JEPA subject-vote fusion evaluation |
| `src/jepa_fusion/build_final_run_tables.py` | Final JSON/Markdown-to-LaTeX table builder |
| `src/jepa_clean/data.py` | Windowed fMRI dataset helpers and ABIDE site normalization |
