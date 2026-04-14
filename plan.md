# Exact Methods Plan From `plan.docx`

This file replaces the older branch-only fusion roadmap. It follows the actual paper methods narrative in `plan.docx`, while mapping each step to the current `src/jepa_fusion` codebase.

## Paper Objective

fMRI site harmonization is the bottleneck.

- `ComBat` and `CovBat` are the accepted spatial baselines on static FC.
- They work well, but they are batch-driven and need enough data from a new site.
- JEPA is the proposed inductive alternative: learn temporal, site-robust latent features without needing a batch estimate for every new site.
- The claim is not `replace ComBat`.
- The claim is `add a complementary temporal harmonization path that remains usable when new-site data are scarce or unavailable`.

## Exact Experimental Sequence

Protocol rule for this repo:

- diagnosis evaluation should use `LOSO` across all methods and phases
- site leakage should still be measured with a separate site probe
- the site probe should use grouped subject-level CV, not LOSO, because a site-ID classifier cannot predict an unseen held-out site label it was never trained on

### 1. Establish the trusted classical baseline first

Run the same benchmark on:

- `ABIDE-I`
- `ABIDE-II`
- `SRPBS`

For each dataset:

- show that raw static FC leaks site information
- apply `ComBat` and `CovBat`
- show site identification drops toward chance while diagnosis remains largely intact
- use `5` random seeds

Baseline rows to keep in every first-pass table:

- primary baseline: `Static FC`
- primary baseline: `Static FC + ComBat`
- primary baseline: `Static FC + CovBat`
- secondary sanity baseline: `ROI time series`

Primary outputs:

- baseline diagnosis table
- baseline site-decoding table
- one shared probe family for all fixed feature sets
- protocol pairing:
  - site probe: grouped subject-level CV
  - diagnosis: LOSO

Shared downstream probe:

- `StandardScaler`
- `PCA`
- `LogisticRegression`

Repo mapping:

- baseline script: `python -m src.jepa_fusion.evaluate_baselines`
- stage runner: `python -m src.jepa_fusion.run_methods_plan --stage baseline_tables`

### 2. Treat JEPA strictly as representation learning

JEPA must be evaluated with the same downstream probe family as the FC baselines. The comparison should stay about feature quality, not about classifier complexity.

This step should use `5` evaluation seeds when selecting the best JEPA architecture. Do not select the pilot winner from a single-seed result.

The architecture sweep must be built from raw ROI time series and should proceed in this order:

- temporal masking vs spatio-temporal masking
- sin-cos vs rotary positional encoding
- no conv block vs conv block
- pooled training first
- curriculum training next

Current `jepa_fusion` mapping:

- pooled architecture sweep:
  - `python -m src.jepa_fusion.run_pilot --training-mode pooled`
- curriculum architecture sweep:
  - `python -m src.jepa_fusion.run_pilot --training-mode curriculum`
- one-shot stage runner:
  - `python -m src.jepa_fusion.run_methods_plan --stage pilot_pooled pilot_curriculum`

Expected outputs:

- grouped-CV site-probe JEPA evaluation JSON/Markdown tables
- LOSO diagnosis JEPA evaluation JSON/Markdown tables
- pilot ranking for pooled training
- pilot ranking for curriculum training
- for every candidate and every dataset:
  - site probe balanced accuracy
  - site probe ROC-AUC
  - diagnosis balanced accuracy
  - diagnosis ROC-AUC

### 3. Refine the training strategy after the best JEPA architecture is chosen

The document sequence is:

- curriculum learning with travelling subjects first, then the main dataset
- replay during phase 2
- domain adversarial loss
- contrastive/noise-augmentation loss

What `jepa_fusion` supports now:

- curriculum training: supported
- replay alignment loss: supported
- domain adversarial loss: supported
- pooled vs curriculum comparisons: supported
- grouped-CV 5-seed confirmation: supported

What is still missing for an exact paper match:

- contrastive/noise-augmentation JEPA branch inside `src/jepa_fusion`

Current commands:

- 5-seed pilot sweep with grouped site probe plus LOSO diagnosis:
  - `python -m src.jepa_fusion.run_pilot --training-mode pooled`
  - `python -m src.jepa_fusion.run_pilot --training-mode curriculum`
- same stage via the higher-level runner:
  - `python -m src.jepa_fusion.run_methods_plan --stage pilot_pooled pilot_curriculum`

### 4. Show where batch-driven harmonization breaks

This is a required paper experiment from `plan.docx`.

Hold out a new site and restrict the number of available samples used to estimate `ComBat` / `CovBat`:

- `K = 30`
- `K = 20`
- `K = 10`

The paper result should show:

- statistical harmonization degrades as `K` shrinks
- the trained JEPA encoder remains usable because it is inductive

Status in `jepa_fusion`:

- not implemented yet

Needed code addition:

- a low-`K` held-out-site evaluator for FC + `ComBat` / `CovBat`
- matched comparison against a frozen JEPA checkpoint

### 5. Final cross-dataset zero-shot transfer

This is the final paper test in `plan.docx`.

Direction A:

- train the best JEPA configuration on `SRPBS + ABIDE-I`
- evaluate on unseen `ABIDE-II` sites

Direction B:

- train the best JEPA configuration on `SRPBS + ABIDE-II`
- evaluate on unseen `ABIDE-I` sites

This is the core `K = 0` argument:

- JEPA does not need any new-site calibration batch to start producing usable embeddings

Current repo support:

- implemented through train-time dataset selection in `src.jepa_fusion.train_jepa`
- runnable via `src.jepa_fusion.run_methods_plan`

Commands:

- `python -m src.jepa_fusion.run_methods_plan --stage transfer_abide2`
- `python -m src.jepa_fusion.run_methods_plan --stage transfer_abide1`

Default transfer profile:

- `rotary_curr_align`

Rationale:

- this is the current best match to the document's “optimized JEPA model” language within the existing `jepa_fusion` branch

### 6. Final analysis and reporting

The paper should finish with:

- diagnosis and site metrics reported over `5` seeds
- statistical tests for main comparisons
- embedding visualizations
- per-site breakdown tables
- a final narrative that FC harmonization is the trusted spatial baseline and JEPA is the temporal, inductive complement

Current repo pieces already relevant here:

- `src/jepa_fusion.compare_fusion_stats`
- grouped-CV tables in `results/reports/tables/jepa_fusion`

Still needed for the exact paper plan:

- publication figures tied to the JEPA-only narrative
- explicit low-`K` summary tables
- final zero-shot transfer summary table

## Runnable Stage Order In This Repo

Use this order unless we deliberately branch off for debugging:

1. `baseline_tables`
2. `pilot_pooled`
3. `pilot_curriculum`
4. `groupedcv_confirmation`
5. `transfer_abide2`
6. `transfer_abide1`

One command for the full runnable subset:

```bash
python -m src.jepa_fusion.run_methods_plan --device cuda --cuda-visible-devices 0 --skip-existing
```

Dry-run first:

```bash
python -m src.jepa_fusion.run_methods_plan --dry-run --device cuda --cuda-visible-devices 0
```

## Implementation Status

Implemented now:

- baseline benchmark tables on `ABIDE-I`, `ABIDE-II`, `SRPBS`
- pooled JEPA architecture sweep
- curriculum JEPA architecture sweep
- grouped-CV 5-seed confirmation
- cross-dataset transfer directions with train-time dataset selection
- replay alignment and domain adversarial JEPA variants

Still missing for the exact document plan:

- low-`K` held-out-site `ComBat` / `CovBat` stress test
- contrastive/noise-augmentation JEPA branch inside `src/jepa_fusion`
- final paper figures and summary tables tied to the exact document narrative

## Files That Now Carry This Plan

- `plan.md`
- `src/jepa_fusion/run_methods_plan.py`
- `src/jepa_fusion/run_pilot.py`
- `src/jepa_fusion/train_jepa.py`

Keep future work aligned to this file unless the paper plan itself changes.
