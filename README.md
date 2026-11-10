# MS-JEPA: Zero-Shot Harmonization of Multi-Site rs-fMRI

Code for **"Zero-Shot Harmonization of Multi-Site Resting-State fMRI with
Joint-Embedding Predictive Architectures"** (Shyamal Y. Dharia, Stephen D. Smith,
Camilo E. Valderrama).

MS-JEPA learns site-robust rs-fMRI representations by predicting masked temporal
windows in latent space. A traveling-subject curriculum with replay alignment
encourages cross-site consistency. Representations are evaluated with frozen
linear probes for diagnosis and acquisition-site leakage, against Static FC,
ComBat/CovBat-harmonized FC, raw ROI time series, and a matched
masked-reconstruction SSL baseline. Zero-shot transfer is source-fitted: target
data is never used for pretraining, normalization, PCA, or probe fitting.

Data (SRPBS, ABIDE-I, ABIDE-II), checkpoints, logs, and generated results are not
included. The datasets are access-controlled and must be obtained from their
providers.

## Setup

Python 3.10, single GPU (experiments used an NVIDIA RTX A6000).

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Run every command from the repository root.

## Data layout

```text
preprocessed_data_fir/
  FC_1600/  FC_abide_full/  FC_abide2_full/        # static FC (Schaefer-400)
  full/
    roi_timeseries_1600_full/                     # SRPBS
    roi_timeseries_traveling_full/                # SRPBS traveling subjects
    roi_timeseries_abide_full/                    # ABIDE-I
    roi_timeseries_abide2_full/                   # ABIDE-II
    subject_metadata_full.csv  abide_metadata_full.csv  abide2_metadata.csv
```

Each scan is stored as `<scan>_roi_ts.npy` (time x 400) with a `<scan>_meta.json`
sidecar (site, subject, diagnosis, age, sex, TR).

## Preprocessing

```bash
python convert_to_bids/convert_srpbs1600_to_bids.py
SCRATCH_ROOT=/scratch/$USER sbatch preprocessing/hpc_run_fmriprep_1600.sh
SCRATCH_ROOT=/scratch/$USER sbatch preprocessing/hpc_run_fmriprep_TS_persession.sh
python preprocessing/extract_roi_timeseries.py --dataset both --n-jobs 8
python preprocessing/extract_roi_timeseries_abide.py --n-jobs 8
```

fMRIPrep 24.1.1, 6 motion + 5 aCompCor regressors, 0.01–0.1 Hz band-pass,
first/last 7 volumes trimmed, Schaefer-400 (7-network) parcellation.

## Reproducing the paper

Results go to `results/`, checkpoints to `checkpoints/jepa_fusion/final_run_v1/`.
Probe and pretraining seeds are `0,8,16,32,128` unless stated otherwise.

| Paper item | How to reproduce |
| --- | --- |
| Table 2: cohort summary | `python generate_table2.py` |
| Table 5: baselines | `bash run_paper_tables.sh baselines` |
| Figure 2: low-K calibration | `bash run_paper_tables.sh lowk` |
| Table 6: JEPA variants + curriculum | `bash run_paper_tables.sh pilot_pooled curriculum tables` |
| Table 8: auxiliary-loss ablation | `bash run_paper_tables.sh aux tables` |
| Table 7: demographic probes | [Demographics](#table-7-demographics) |
| Figure 3: Sliced Wasserstein distance | [Site alignment](#figure-3-site-alignment) |
| Figure 4: four-site t-SNE | `python -m src.jepa_fusion.plot_four_site_tsne --dataset abide` |
| Table 9: zero-shot transfer | [Zero-shot transfer](#table-9-zero-shot-transfer) |
| Figure 5: parcel attribution | [Attribution](#figure-5-attribution) |

`bash run_paper_tables.sh all` runs every staged step. Set `DEVICE=cuda`,
`RUN_TAG=final_run_v1`, and `SKIP_EXISTING=1` as needed.

### Models and checkpoints

| Checkpoint family | Objective | Used for |
| --- | --- | --- |
| `final_run_v1_pilot_temporal_*_pooled_seed*` | JEPA + VICReg, pooled | Table 6 (pooled rows), Table 8 |
| `final_run_v1_rotary_curr_replay_seed*` | JEPA + VICReg + replay alignment (λ=0.1), curriculum | Table 6 (Curriculum + Replay), Table 7, Figure 3 |
| `final_run_v1_rotary_curr_replay_{adv,nce,both}_seed*` | as above + adversarial and/or InfoNCE (0.1) | Table 8 |
| `final_run_v1_rotary_curr_replay_nce_seed*` | curriculum + replay + InfoNCE (0.1), SRPBS + ABIDE-I + ABIDE-II | Figure 5 (attribution); default for Figure 4 |
| `final_run_v1_transfer_*_rotary_curr_replay_nce_seed*` | curriculum + replay + InfoNCE (0.1), SRPBS + one ABIDE release | Table 9 (MS-JEPA) |
| `masked_recon_ssl_*_seed*` | masked ROI reconstruction, same encoder and masking | Table 9 (Masked Recon. SSL) |

### Table 7: demographics

```bash
for s in 0 8 16 32 128; do
  python -m src.jepa_fusion.evaluate_demographics --dataset all --seeds $s \
    --checkpoint checkpoints/jepa_fusion/final_run_v1/final_run_v1_rotary_curr_replay_seed$s/phase2_final.pt \
    --output-json results/reports/tables/jepa_fusion/demographics/rotary_curr_replay_seed${s}_demographics.json \
    --output-md results/reports/tables/jepa_fusion/demographics/rotary_curr_replay_seed${s}_demographics.md
done
python -m src.jepa_fusion.summarize_demographics_seeds
```

### Figure 3: site alignment

Sliced Wasserstein distance uses 50 PCA components, 256 random projections, and
center-level site labels (shared ABIDE-I/II centers merged). It is followed by
Friedman tests and Bonferroni-corrected paired permutation tests.

```bash
python -m src.jepa_fusion.report_site_wasserstein --dataset all --device cuda
python -m src.jepa_fusion.analyze_site_wasserstein_stats
python -m src.jepa_fusion.plot_site_wasserstein_distribution
```

### Table 9: zero-shot transfer

The direction A source is SRPBS + ABIDE-I and the target is ABIDE-II. Direction B
uses SRPBS + ABIDE-II as the source and ABIDE-I as the target. Ten matched
pretraining seeds are used per method (`0,4,8,16,32,64,128,256,512,1024`).
Overlapping ABIDE-I/II centers are kept in the target evaluation.

```bash
# 1. Pretrain MS-JEPA and Masked Recon. SSL (seeds 0,8,16,32,128, then 4,64,256,512,1024)
bash run_paper_tables.sh transfer
for d in A B; do
  bash run_masked_recon_transfer_5seed.sh $d
  bash run_extra_zero_shot_seeds.sh jepa $d
  bash run_extra_zero_shot_seeds.sh masked $d
done

# 2. Evaluate all checkpoints and Static FC on the full target cohort,
#    then compute Table 9 means and Holm-corrected paired sign-flip tests
bash run_zero_shot_no_exclusion.sh
```

The steps in (1) also write evaluations that exclude overlapping target centers.
Those outputs are not used for Table 9. The table and its significance markers
come from `results/reports/tables/jepa_fusion/zero_shot_no_exclusion/zero_shot_transfer_stats.md`.

### Figure 5: attribution

```bash
python -m src.jepa_fusion.plot_diagnosis_attribution_maps \
  --directions final_all_disorder --model-seeds 0,8,16,32,128 --device cuda
python -m src.jepa_fusion.plot_attribution_figure5
```

The first command computes gradient-times-input attribution for the HC-vs-disorder
leave-one-site-out probe and runs parcel-level exact sign-flip tests across seeds.
The second computes network-level tests (sign-flip across seeds, and enrichment
against random parcel sets) and draws panels (a)–(c).

## Code layout

| Path | Contents |
| --- | --- |
| `src/jepa_fusion/model.py`, `masking.py`, `losses.py`, `train_jepa.py` | MS-JEPA encoder variants, temporal masking, losses, pooled and curriculum training |
| `src/jepa_fusion/train_masked_recon_ssl.py` | Masked-reconstruction SSL baseline |
| `src/jepa_fusion/evaluate_baselines*.py`, `evaluate_static_fc_zero_shot.py` | Static FC / ComBat / CovBat / ROI baselines, low-K, zero-shot Static FC |
| `src/jepa_fusion/evaluate_jepa.py`, `evaluate_jepa_subject_vote.py` | Frozen-encoder site probe (grouped CV) and diagnosis probe (LOSO) |
| `src/jepa_fusion/protocol.py`, `dataset_utils.py` | Dataset presets, site normalization, task filtering |
| `src/jepa_clean/` | Windowed dataset, pooling, and reproducibility helpers used by the pipeline |
| `preprocessing/`, `convert_to_bids/` | fMRIPrep launchers, ROI extraction, SRPBS BIDS conversion |

Other modules in `src/jepa_fusion/` (FC + JEPA fusion, Fourier-patch JEPA,
pilot sweeps) are exploratory and not used for the reported results.

## License

MIT. See `LICENSE`.
