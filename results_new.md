# Finetune Classification Results (LOSO Cross-Validation)

**Generated:** 2026-03-10  
**Evaluation:** Leave-One-Site-Out (LOSO) | Finetune protocol only  
**Data:** 1626 subjects, 93 ROIs, 400 timepoints, 13 sites, 9 diagnoses

---

## Binary Classification Tasks

### HC vs ALL (N=1626 | HC=950, Disease=676 | chance=50.0% | 11 LOSO sites)

| Method | bAcc | AUC | F1 |
|---|---|---|---|
| **JEPA Vanilla (Curr)** | **80.0%±0.9%** | **0.883±0.006** | **0.797** |
| JEPA + VICReg (Curr) | 78.4%±0.7% | 0.868±0.005 | 0.782 |
| JEPA + Contrastive (Curr) | 70.2%±0.3% | 0.783±0.005 | 0.700 |
| JEPA + Adversarial (Curr) | 59.6%±1.3% | 0.651±0.010 | 0.596 |
| JEPA Vanilla (NoCurr) | 59.4%±1.8% | 0.631±0.016 | 0.594 |
| JEPA + Adversarial (NoCurr) | 58.3%±0.8% | 0.614±0.016 | 0.582 |
| JEPA + Contrastive (NoCurr) | 57.6%±1.3% | 0.612±0.007 | 0.574 |
| JEPA + VICReg (NoCurr) | 57.0%±0.5% | 0.606±0.016 | 0.566 |
| Raw (from scratch) | 55.7%±1.3% | 0.582±0.009 | 0.555 |
| ComBat (from scratch) | 55.4%±2.1% | 0.584±0.008 | 0.552 |
| SimCLR Baseline | 55.1%±1.5% | 0.578±0.006 | 0.549 |
| Transformer-VAE | 50.9%±1.1% | 0.522±0.007 | 0.498 |

---

### HC vs MDD (N=1205 | HC=950, Disease=255 | chance=50.0% | 6 LOSO sites)

| Method | bAcc | AUC | F1 |
|---|---|---|---|
| **JEPA Vanilla (Curr)** | **74.0%±2.1%** | **0.822±0.007** | **0.715** |
| JEPA + VICReg (Curr) | 72.5%±1.4% | 0.808±0.003 | 0.707 |
| JEPA + Contrastive (Curr) | 65.8%±1.0% | 0.704±0.011 | 0.640 |
| JEPA + Adversarial (Curr) | 56.4%±2.3% | 0.592±0.012 | 0.559 |
| JEPA Vanilla (NoCurr) | 54.8%±1.1% | 0.563±0.019 | 0.546 |
| SimCLR Baseline | 54.7%±1.4% | 0.557±0.012 | 0.539 |
| JEPA + Contrastive (NoCurr) | 54.7%±0.5% | 0.549±0.007 | 0.546 |
| JEPA + Adversarial (NoCurr) | 54.2%±1.6% | 0.567±0.012 | 0.530 |
| ComBat (from scratch) | 54.1%±1.1% | 0.547±0.016 | 0.534 |
| Raw (from scratch) | 53.5%±2.5% | 0.548±0.018 | 0.532 |
| JEPA + VICReg (NoCurr) | 52.8%±1.0% | 0.533±0.021 | 0.521 |
| Transformer-VAE | 50.5%±1.8% | 0.512±0.008 | 0.483 |

---

### HC vs SCZ (N=1097 | HC=950, Disease=147 | chance=50.0% | 4 LOSO sites)

| Method | bAcc | AUC | F1 |
|---|---|---|---|
| **JEPA Vanilla (Curr)** | **73.7%±1.8%** | **0.838±0.017** | 0.678 |
| JEPA + VICReg (Curr) | 73.4%±2.3% | 0.805±0.020 | **0.690** |
| JEPA + Contrastive (Curr) | 65.5%±2.1% | 0.724±0.014 | 0.619 |
| JEPA + Adversarial (Curr) | 57.7%±2.2% | 0.627±0.019 | 0.570 |
| ComBat (from scratch) | 57.5%±1.8% | 0.611±0.023 | 0.569 |
| Raw (from scratch) | 57.2%±2.0% | 0.614±0.019 | 0.560 |
| JEPA + Adversarial (NoCurr) | 57.1%±2.1% | 0.581±0.027 | 0.564 |
| SimCLR Baseline | 56.5%±1.1% | 0.592±0.017 | 0.560 |
| JEPA Vanilla (NoCurr) | 55.9%±2.2% | 0.595±0.025 | 0.562 |
| JEPA + Contrastive (NoCurr) | 55.8%±1.8% | 0.562±0.023 | 0.557 |
| JEPA + VICReg (NoCurr) | 54.7%±1.8% | 0.558±0.040 | 0.547 |
| Transformer-VAE | 50.2%±1.8% | 0.484±0.007 | 0.473 |

---

### HC vs ASD (N=1075 | HC=950, Disease=125 | chance=50.0% | 2 LOSO sites)

| Method | bAcc | AUC | F1 |
|---|---|---|---|
| **JEPA Vanilla (Curr)** | **51.3%±4.9%** | 0.534±0.038 | **0.488** |
| ComBat (from scratch) | 49.9%±3.9% | **0.569±0.061** | 0.469 |
| JEPA + Adversarial (Curr) | 49.1%±2.0% | 0.391±0.036 | 0.435 |
| JEPA + VICReg (Curr) | 48.8%±2.0% | 0.490±0.052 | 0.459 |
| JEPA Vanilla (NoCurr) | 48.0%±3.3% | 0.482±0.039 | 0.423 |
| JEPA + Contrastive (Curr) | 47.7%±5.2% | 0.434±0.042 | 0.453 |
| Raw (from scratch) | 47.4%±2.5% | 0.529±0.051 | 0.447 |
| JEPA + Contrastive (NoCurr) | 47.0%±1.7% | 0.389±0.033 | 0.395 |
| JEPA + VICReg (NoCurr) | 46.8%±1.3% | 0.482±0.021 | 0.409 |
| Transformer-VAE | 44.5%±4.0% | 0.207±0.005 | 0.412 |
| SimCLR Baseline | 44.3%±3.0% | 0.501±0.027 | 0.408 |
| JEPA + Adversarial (NoCurr) | 44.1%±1.3% | 0.441±0.036 | 0.392 |

---

### HC vs Pain (N=1016 | HC=950, Disease=66 | chance=50.0% | 3 LOSO sites)

| Method | bAcc | AUC | F1 |
|---|---|---|---|
| **JEPA Vanilla (Curr)** | **68.2%±3.7%** | **0.807±0.079** | **0.655** |
| JEPA + VICReg (Curr) | 64.9%±4.3% | 0.730±0.050 | 0.615 |
| JEPA + Contrastive (Curr) | 62.9%±3.8% | 0.720±0.049 | 0.600 |
| JEPA + Adversarial (Curr) | 54.6%±2.4% | 0.603±0.030 | 0.464 |
| SimCLR Baseline | 54.5%±2.8% | 0.618±0.025 | 0.510 |
| Raw (from scratch) | 54.1%±2.9% | 0.564±0.026 | 0.487 |
| JEPA + VICReg (NoCurr) | 54.0%±2.4% | 0.510±0.024 | 0.486 |
| JEPA + Adversarial (NoCurr) | 53.9%±2.7% | 0.611±0.024 | 0.490 |
| ComBat (from scratch) | 53.7%±2.7% | 0.558±0.031 | 0.508 |
| JEPA Vanilla (NoCurr) | 53.3%±2.4% | 0.555±0.047 | 0.437 |
| JEPA + Contrastive (NoCurr) | 52.5%±1.7% | 0.480±0.039 | 0.422 |
| Transformer-VAE | 48.4%±2.9% | 0.447±0.044 | 0.463 |

---

### HC vs Stroke (N=960 | HC=950, Disease=10 | chance=50.0% | 2 LOSO sites)

| Method | bAcc | AUC | F1 |
|---|---|---|---|
| JEPA + VICReg (Curr) | **53.0%±2.4%** | 0.706±0.096 | **0.485** |
| JEPA Vanilla (Curr) | 52.0%±2.4% | **0.718±0.063** | 0.465 |
| Raw (from scratch) | 51.7%±2.1% | 0.501±0.126 | 0.462 |
| JEPA + Contrastive (Curr) | 51.0%±2.0% | 0.731±0.037 | 0.446 |
| JEPA + Contrastive (NoCurr) | 51.0%±2.0% | 0.563±0.113 | 0.446 |
| JEPA + Adversarial (NoCurr) | 51.0%±2.0% | 0.563±0.062 | 0.446 |
| JEPA + Adversarial (Curr) | 50.0%±0.0% | 0.486±0.048 | 0.426 |
| JEPA + VICReg (NoCurr) | 50.0%±0.0% | 0.615±0.084 | 0.426 |
| Transformer-VAE | 49.7%±0.7% | 0.452±0.028 | 0.425 |
| JEPA Vanilla (NoCurr) | 49.7%±0.7% | 0.630±0.138 | 0.425 |
| ComBat (from scratch) | 49.6%±1.2% | 0.451±0.056 | 0.436 |
| SimCLR Baseline | 49.2%±4.7% | 0.606±0.063 | 0.455 |

---

### Skipped Binary Tasks

| Task | Reason |
|---|---|
| HC vs Bipolar (N=991, Disease=41) | Disease subjects only at 1 site — LOSO impossible |
| HC vs Dysthymia (N=954, Disease=4) | Disease subjects only at 1 site — LOSO impossible |
| HC vs Others (N=978, Disease=28) | Disease subjects only at 1 site — LOSO impossible |

---

## Multiclass Classification Tasks

### Multiclass ALL (N=1626 | 9 classes | chance=11.1% | 11 LOSO sites)

Class counts: HC=950, ASD=125, MDD=255, SCZ=147, Pain=66, Stroke=10, Bipolar=41, Dysthymia=4, Others=28

| Method | bAcc | AUC | F1 |
|---|---|---|---|
| **JEPA Vanilla (Curr)** | **14.3%±0.4%** | **0.701±0.003** | **0.144** |
| JEPA + VICReg (Curr) | 13.8%±0.6% | 0.679±0.007 | 0.140 |
| JEPA + Contrastive (Curr) | 12.5%±0.9% | 0.617±0.009 | 0.128 |
| JEPA + Contrastive (NoCurr) | 11.9%±0.2% | 0.492±0.007 | 0.113 |
| ComBat (from scratch) | 11.6%±0.5% | 0.520±0.005 | 0.111 |
| Raw (from scratch) | 11.4%±0.3% | 0.514±0.006 | 0.110 |
| JEPA Vanilla (NoCurr) | 11.3%±0.6% | 0.506±0.006 | 0.106 |
| SimCLR Baseline | 11.2%±0.6% | 0.519±0.006 | 0.111 |
| JEPA + VICReg (NoCurr) | 11.2%±0.3% | 0.492±0.015 | 0.106 |
| JEPA + Adversarial (NoCurr) | 9.2%±0.3% | 0.469±0.004 | 0.087 |
| JEPA + Adversarial (Curr) | 8.5%±0.4% | 0.497±0.007 | 0.081 |
| Transformer-VAE | 6.8%±1.0% | 0.406±0.005 | 0.067 |

---

### Multiclass Multi-Site (N=1553 | 6 classes | chance=16.7% | 11 LOSO sites) — PARTIAL

Class counts: HC=950, ASD=125, MDD=255, SCZ=147, Pain=66, Stroke=10

| Method | bAcc | AUC | F1 |
|---|---|---|---|
| **JEPA Vanilla (Curr)** | **22.7%±0.5%** | **0.749±0.004** | **0.227** |
| SimCLR Baseline | 18.5%±1.3% | 0.551±0.009 | 0.183 |
| Raw (from scratch) | 18.2%±0.6% | 0.553±0.011 | 0.177 |
| ComBat (from scratch) | 18.2%±0.7% | 0.554±0.010 | 0.177 |
| *Remaining 8 methods* | *still running...* | | |

---

## Notes

- **Curr** = With Curriculum learning; **NoCurr** = No Curriculum learning
- All results use LOSO (Leave-One-Site-Out) cross-validation
- bAcc = Balanced Accuracy; AUC = Area Under ROC Curve; F1 = F1-score
- ± values are standard error across LOSO folds
- Δ values (not shown) represent difference from chance level
- Multiclass Multi-Site task is still running as of 2026-03-10



# Evaluation Metrics — How We Calculate AUC and Thresholds

## 1. Binary Classification (HC vs Disease)

### Threshold Calibration (Youden's J Statistic)

For binary tasks (e.g., HC vs MDD), the classifier outputs a probability score between 0 and 1 for each subject. Instead of using the default 0.5 cutoff, we find the **optimal threshold** that best separates HC from disease:

1. **Split the training data** into 80% train-inner and 20% validation (stratified by class).
2. **Train the classifier** on train-inner.
3. **Search for the best threshold** on the validation set by testing 200 evenly spaced values from 0.01 to 0.99.
4. For each candidate threshold, compute:
   - **Sensitivity** = correctly identified disease subjects / total disease subjects
   - **Specificity** = correctly identified HC subjects / total HC subjects
   - **Youden's J** = Sensitivity + Specificity − 1
5. **Pick the threshold** that maximizes J — this gives the best trade-off between catching disease cases and avoiding false alarms.
6. **Apply that threshold** to the held-out test site's probability scores to make final predictions.

This is done independently for each LOSO fold and each seed, so the threshold adapts to the training data without ever seeing the test site.

### AUC (Binary)

For binary tasks, AUC is the standard **Area Under the ROC Curve**. It measures how well the model's probability scores separate HC from disease, regardless of the chosen threshold. A score of 0.5 means random; 1.0 means perfect separation.

Computed using `sklearn.metrics.roc_auc_score(y_true, y_prob)` on the predicted probabilities for the positive (disease) class.

---

## 2. Multiclass Classification (9 diagnoses)

### Predictions (No Threshold — Argmax)

For multiclass tasks (9 classes: HC, ASD, MDD, SCZ, Pain, Bipolar, Stroke, Dysthymia, Others), there is **no threshold calibration**. Instead:

1. The classifier outputs a probability for each of the 9 classes.
2. The predicted class is simply the one with the **highest probability** (argmax).

This is the standard approach — threshold calibration only makes sense for binary decisions.

### AUC (Multiclass — One-vs-Rest Weighted)

For multiclass, AUC is computed using the **One-vs-Rest (OvR) weighted** strategy:

1. For each of the 9 classes, treat it as a separate binary problem: "this class vs all others."
2. Compute the binary AUC for each class using the model's predicted probability for that class.
3. Take the **weighted average** across all 9 classes, where each class's weight is proportional to the number of subjects in that class (support-weighted).

This means classes with more subjects (e.g., HC with 950) contribute more to the final AUC than rare classes (e.g., Dysthymia with 4).

Computed using `sklearn.metrics.roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")`.

---

## Summary Table

| Aspect | Binary (HC vs X) | Multiclass (9 classes) |
|---|---|---|
| **How predictions are made** | Probability → calibrated threshold → 0 or 1 | Probability per class → argmax → predicted class |
| **Threshold** | Youden's J on validation split (per fold, per seed) | None (argmax) |
| **AUC type** | Standard ROC AUC (single probability vs binary label) | One-vs-Rest weighted AUC (probability matrix vs multi-label) |
| **AUC range** | 0.5 (random) to 1.0 (perfect) | 0.5 (random) to 1.0 (perfect) |
| **Chance-level bAcc** | 50% | 11.1% (1/9 classes) |
