# JEPA Fusion Branch

This package is the clean workspace for the next branch:

- keep harmonized FC as the strong topological baseline
- treat JEPA embeddings as complementary temporal features
- optimize for a defensible fusion result, not a forced "JEPA beats ComBat" claim

## Main Claim

Primary paper claim for this branch:

- `ComBat/CovBat-harmonized static FC + JEPA embeddings` improves diagnosis over harmonized static FC alone

Secondary claim:

- JEPA embeddings capture complementary temporal information that static FC does not preserve

Non-claim:

- do not frame this branch as `JEPA alone beats ComBat/CovBat`

## Order Of Work

Yes: baseline scripts should come before fusion scripts.

The first job is to lock a clean benchmark table with one shared evaluation protocol. Fusion only matters if the FC baselines and ROI baselines are already stable and leak-free.

## Phase 0: Baselines

Initial baseline rows:

- Static FC
- Static FC + ComBat
- Static FC + CovBat
- ROI time series

Optional later baseline:

- ROI time series + ComBat/CovBat

That optional row is not part of the core paper story. The core comparison is harmonized FC versus JEPA-derived temporal features.

## Shared Probe

For the first benchmark, use the same downstream probe family for all fixed feature sets:

- `StandardScaler`
- `PCA`
- `LogisticRegression`

This is the right default because it keeps the comparison about feature quality, not classifier complexity.

Important constraints:

- harmonization must be fit on training folds only, then applied to held-out data
- PCA must be fit on training folds only
- any tuning must happen inside the training split only
- use the same split definitions for all methods

Reasonable starting probe:

- `StandardScaler()`
- `PCA(n_components=min(50, train_n - 1, feature_dim))`
- `LogisticRegression(max_iter=10000, class_weight="balanced")`

## Phase 1: JEPA Feature Baselines

Add JEPA-derived rows after the classical baselines are locked:

- ROI time series -> JEPA embeddings, no curriculum
- ROI time series -> JEPA embeddings, curriculum

Both should use the same frozen-feature probe family as the baselines above.

Required checks:

- diagnosis probe
- site probe
- overlap-aware evaluation notes for ABIDE-I / ABIDE-II shared sites

This stage answers:

- does curriculum improve the JEPA features?
- are JEPA embeddings complementary to FC?
- how much site information remains in each JEPA variant?

## Phase 2: Fusion

Fusion rows come after Phase 0 and Phase 1 are stable:

- Static FC + ComBat + JEPA(no curriculum)
- Static FC + CovBat + JEPA(no curriculum)
- Static FC + ComBat + JEPA(curriculum)
- Static FC + CovBat + JEPA(curriculum)

Primary recommendation: start with late fusion only.

Why:

- FC vectors and JEPA embeddings live in different spaces
- late fusion is safer and easier to interpret
- it reduces the risk that one branch dominates because of raw scale or dimension

## Fusion Modes

Start with these three:

- `soft_vote`: train separate FC and JEPA classifiers, average predicted probabilities
- `weighted_vote`: same as soft vote, but learn or tune a single branch weight on the training split
- `concat_projected`: project FC and JEPA separately, then concatenate projected features before a final logistic regression

Recommended primary method:

- `soft_vote`

Safest feature-level method:

- `concat_projected`

More complex option for later only:

- `stacking`: train a meta-classifier on out-of-fold FC and JEPA logits

Do not start with deep learned fusion. The sample size is too small and the paper does not need that complexity yet.

## Evaluation Guardrails

Keep these fixed for the branch:

- one main diagnosis metric: balanced accuracy
- one secondary metric: ROC-AUC
- mandatory site probe for JEPA features
- ideally also a site probe for fused features
- overlap-aware reporting for ABIDE-I / ABIDE-II shared sites
- no full-dataset ComBat/CovBat fitting before cross-validation

## Script Roadmap

Suggested first scripts:

- `protocol.py`
- `features.py`
- `evaluate_baselines.py`
- `train_jepa.py`
- `evaluate_jepa.py`
- `select_pilot.py`

Suggested responsibilities:

- `protocol.py`: split definitions, overlap-aware filtering, shared metrics
- `features.py`: FC feature building, ROI flattening, harmonization wrappers
- `evaluate_baselines.py`: Static FC / ComBat FC / CovBat FC / raw ROI
- `train_jepa.py`: pooled or curriculum JEPA pretraining for temporal and ST ROI-patch families
- `evaluate_jepa.py`: grouped-CV JEPA embedding evaluation with site and diagnosis bAcc/AUC
- `select_pilot.py`: apply the site-ceiling rule and rank pilot candidates

## Minimal First Deliverable

Before any new JEPA training, the first clean table should be:

- Static FC
- Static FC + ComBat
- Static FC + CovBat
- ROI time series

with one fixed protocol and one reusable reporting format.

## Current JEPA Commands

Pilot training examples:

- `python -m src.jepa_fusion.train_jepa --model-family temporal --temporal-variant basic --training-mode pooled --save-dir checkpoints/jepa_fusion/pilot_t_basic`
- `python -m src.jepa_fusion.train_jepa --model-family temporal --temporal-variant conv_net --training-mode pooled --save-dir checkpoints/jepa_fusion/pilot_t_conv_net`
- `python -m src.jepa_fusion.train_jepa --model-family temporal --temporal-variant rotary_conv_net --training-mode pooled --save-dir checkpoints/jepa_fusion/pilot_t_rotary_conv_net`
- `python -m src.jepa_fusion.train_jepa --model-family st_roi_patch --training-mode pooled --save-dir checkpoints/jepa_fusion/pilot_st_roi_patch`

Evaluation examples:

- `python -m src.jepa_fusion.evaluate_jepa --checkpoint checkpoints/jepa_fusion/pilot_t_conv_net/pooled_final.pt`
- `python -m src.jepa_fusion.evaluate_jepa --checkpoint checkpoints/jepa_fusion/pilot_st_roi_patch/pooled_final.pt`

Pilot ranking example:

- `python -m src.jepa_fusion.select_pilot --inputs results/reports/tables/jepa_fusion/pilot_t_basic_pooled_final_jepa_eval.json results/reports/tables/jepa_fusion/pilot_t_conv_net_pooled_final_jepa_eval.json results/reports/tables/jepa_fusion/pilot_t_rotary_conv_net_pooled_final_jepa_eval.json results/reports/tables/jepa_fusion/pilot_st_roi_patch_pooled_final_jepa_eval.json`

One-shot pilot launcher:

- `python -m src.jepa_fusion.run_pilot --cuda-visible-devices 0 --device cuda`
