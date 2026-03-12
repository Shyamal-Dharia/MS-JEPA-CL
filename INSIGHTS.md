
---

## Bug #3 FIX APPLIED (March 8, 2026)

**File**: `src/jepa/train_phase2_contrastive.py` — complete rewrite

### Changes made:
1. **Restored `HCOnlyDataset`** — trains on 950 HC subjects only (was using all 1626)
2. **Restored `make_site_balanced_sampler`** — rare sites sampled equally (was deleted)
3. **Restored `ReplayBuffer`** — 15% of each batch from travelling subjects (was deleted)
4. **Removed `augment=True, num_views=2`** — no more fake augmented positive pairs
5. **Contrastive loss on replay only** — uses real `subject_ids` from travelling data metadata, not `pseudo_ids = list(range(B)) * 2`
6. **Fixed `.flatten(start_dim=1)` → `.mean(dim=1)`** in `get_pooled_embedding()` — contrastive loss now operates on 400-dim mean-pooled embeddings

### Training loop now matches vanilla pattern:
```
For each batch:
  1. Draw 54 HC samples (site-balanced) + 10 replay samples (travelling)
  2. Concatenate → 64 samples
  3. JEPA MSE loss on all 64
  4. NT-Xent contrastive loss on ONLY the 10 replay samples (real cross-site pairs)
  5. total_loss = jepa_loss + 0.5 * cl_loss
```

### Old (broken) backup at:
`src/jepa/train_phase2_contrastive.py.bak`

### Next step: Retrain
```bash
cd /data/s.dharia-ra/multi-site-problem
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 nohup python -m jepa.train_phase2_contrastive > results/train_logs/phase2_cl_curr.log 2>&1 &
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 nohup python -m jepa.train_phase2_contrastive --no_curriculum > results/train_logs/phase2_cl_nocurr.log 2>&1 &
```

## Bug #4: CPU thread thrashing in evaluation

8 parallel `evaluate_all.py` processes each consuming 770%+ CPU (8 threads each), saturating all cores.
GPU utilization was 0% during finetune LOSO because PyTorch defaults to using all CPU threads for internal operations.

**Fix for next eval run**: Set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` before launching processes.

---

## Bug #5: `@torch.no_grad()` on contrastive training's `get_pooled_embedding()` (March 8, 2026)

**File**: `src/jepa/train_phase2_contrastive.py` line 255

The `get_pooled_embedding()` function that produces embeddings for the NT-Xent contrastive loss
was decorated with `@torch.no_grad()`. This meant the contrastive loss **could not backpropagate
gradients** to the encoder — the CL loss was computed but had zero effect on training.

**Fix**: Removed the `@torch.no_grad()` decorator. The function is also called by the site probe
(via `extract_embeddings()`), which has its own `@torch.no_grad()` context manager.

---

## Bug #6: Adversarial script issues (March 8, 2026)

**File**: `src/jepa/train_p2_advv.py` — 7 fixes applied

1. `embed_dim = 64` → `embed_dim = 400` (must match Phase 1 checkpoint)
2. `.flatten(start_dim=1)` → `.mean(dim=1)` in `get_pooled_embedding()` 
3. `.flatten(start_dim=1)` → `.mean(dim=1)` in `extract_embeddings()`
4. `flattened_dim = cfg.embed_dim * 93` → `flattened_dim = cfg.embed_dim` (site classifier input)
5. Save dir: `phase2_adversarial_v2` → `phase2_adversarial`
6. Added `--no_curriculum` argparse flag
7. Added `model_final.pt` save for evaluator

**Backup**: `src/jepa/train_p2_advv.py.bak`

---

## Expanded to 12-Method Evaluation (March 8, 2026)

### Standalone VICReg script created
**New file**: `src/jepa/train_phase2_vicreg.py`

Standalone VICReg (MSE + VICReg only, NO contrastive loss) for clean ablation:
each Phase 2 variant tests exactly ONE auxiliary loss on top of JEPA MSE.

| Variant | Loss | Script |
|---------|------|--------|
| Vanilla | MSE | train_phase2.py |
| Adversarial | MSE + GRL adversarial | train_p2_advv.py |
| Contrastive | MSE + NT-Xent | train_phase2_contrastive.py |
| VICReg | MSE + VICReg(var+cov) | train_phase2_vicreg.py |

### 12 methods evaluated
```
 1. Raw (Transformer from scratch)
 2. ComBat (Transformer from scratch)
 3. SimCLR Baseline
 4. Transformer-VAE
 5. JEPA Vanilla (No Curriculum)
 6. JEPA Vanilla (With Curriculum)
 7. JEPA + Contrastive (No Curriculum)
 8. JEPA + Contrastive (With Curriculum)
 9. JEPA + Adversarial (No Curriculum)
10. JEPA + Adversarial (With Curriculum)
11. JEPA + VICReg (No Curriculum)
12. JEPA + VICReg (With Curriculum)
```

### Early results (HC vs ALL — linear & MLP probes only, pre-finetune)

| Method | Linear AUC | MLP AUC |
|--------|-----------|---------|
| Raw | 0.523 | 0.561 |
| ComBat | 0.528 | 0.516 |
| SimCLR | 0.521 | 0.539 |
| VAE | 0.513 | 0.477 |
| Vanilla (NoCurr) | 0.547 | 0.580 |
| **Vanilla (Curr)** | **0.629** | **0.764** |
| CL (NoCurr) | 0.524 | 0.545 |
| CL (Curr) | 0.598 | 0.645 |
| ADV (NoCurr) | 0.553 | 0.532 |
| ADV (Curr) | 0.620 | 0.543 |
| VICReg (NoCurr) | 0.517 | 0.508 |
| **VICReg (Curr)** | **0.633** | **0.745** |

**Key observations**:
- Curriculum makes a massive difference (Vanilla NoCurr MLP: 0.580 vs Curr: 0.764)
- Standalone VICReg (Curr) MLP: 0.745 — competitive with Vanilla!
- Contrastive CL (Curr): 0.645 — lower than previous broken run (0.742) despite gradient fix
- Adversarial destroys signal regardless of curriculum
- Finetune results (the dominant AUC) still running...
