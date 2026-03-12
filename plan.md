# Plan: MS-JEPA-CL for NeurIPS 2026

**Your project has strong results** (AUC 0.893 on HC vs ALL, curriculum learning as the clear differentiator). To make it NeurIPS main-conference caliber, the gaps are: second dataset, stronger baselines, proper ablations, interpretability, and theoretical framing.

---

## Phase A: Solidify Current Results (Week 1–2)

1. **Let the 5-seed eval finish** — this already upgrades your error bars from 3→5 seeds
2. **Add statistical significance tests** — paired bootstrap or corrected resampled t-test (Nadeau & Bengio 2003) comparing JEPA Vanilla (Curr) vs each baseline across LOSO folds. Reviewers will demand p-values.
3. **Fix ComBat baseline** — your logs show neuroCombat wasn't installed and fell back to raw data. This invalidates that baseline row entirely.
4. **Explain the adversarial paradox** — best site-invariance (8.1% MLP probe) but worst diagnostic performance. This is actually a *finding*, not a bug: adversarial over-regularizes and removes diagnostically-relevant signal. Frame it as evidence that curriculum > adversarial.

---

## Phase B: Second Dataset — ABIDE I/II (Week 2–4) *[CRITICAL]*

NeurIPS reviewers will reject a single-dataset paper. ABIDE is ideal:
- **Publicly available**, ~1,112 subjects, **17 sites**, ASD vs HC diagnosis
- Tests a completely different scanner ecosystem

5. **Download & preprocess ABIDE I** with your existing pipeline (Schaefer 400, bandpass, confound regression) — reuse `preprocessing/extract_roi_timeseries.py`
6. **Three transfer scenarios** — this is the killer experiment:
   - **Zero-shot**: SRPBS-pretrained encoder applied directly to ABIDE → LOSO eval
   - **Curriculum transfer**: Phase 1 on SRPBS travelling subjects → Phase 2 on ABIDE HCs → LOSO eval
   - **Pooled**: Phase 2 on SRPBS + ABIDE HCs → eval on both
7. **Cross-dataset transfer table** — shows the curriculum generalizes beyond the training dataset. This alone could carry the paper.

---

## Phase C: Stronger Baselines (Week 2–3, *parallel with B*)

8. **Add MAE (Masked Autoencoder)** — JEPA's closest competitor in the masked-prediction family. Reviewers will 100% ask for this.
9. **Add MIDA** (Dinsdale et al., 2021) — adversarial domain adaptation, the strongest existing deep-learning harmonization method. This is the real competitor.
10. **Add CovBat** — extends ComBat to covariance harmonization; standard in neuroimaging

---

## Phase D: Ablation Study (Week 3–4) *[CRITICAL]*

11. **Curriculum component ablations** — each tested in isolation:

| Ablation | Tests |
|----------|-------|
| No Phase 1 at all (Phase 2 from scratch) | Is Phase 1 necessary? |
| Phase 1 only (no Phase 2) | Does Phase 2 add anything? |
| Phase 2 without replay buffer | Is replay crucial? |
| Phase 2 without site-balanced sampler | Is balancing important? |
| Replay fractions: 5%, 10%, 15%, 25% | Sensitivity to replay % |
| Masking: temporal only / spatial only / both | Which masking matters? |

12. **Hyperparameter sensitivity** — EMA decay, learning rate, warmup epochs (can go in appendix)

---

## Phase E: Analysis & Interpretability (Week 3–5)

13. **t-SNE/UMAP figures**: Raw → Phase 1 → Phase 2 embeddings, colored by site and by diagnosis. Show site clusters *dissolving* while diagnostic clusters *emerging*. Already have `src/jepa/visualize.py` — extend it.
14. **Travelling subject fingerprinting**: For each of the 9 travelling subjects, compute cross-site embedding similarity. Show embeddings become subject-consistent across sites after curriculum training.
15. **CKA or MMD analysis**: Quantitative measure of site-distribution alignment beyond just probe accuracy.
16. **Per-site performance heatmap**: Methods × sites matrix — identify failure modes (small sites, unusual scanners).

---

## Phase F: Theoretical Framing (Week 4–5)

17. **Frame as progressive domain adaptation**: Phase 1 learns a cross-domain alignment on *paired* data (travelling subjects = natural domain pairs), Phase 2 uses replay to prevent catastrophic forgetting. Connect to Ben-David et al. (2010) domain adaptation theory.
18. **Information-theoretic argument**: Phase 1 minimizes I(Z; S) on paired data, Phase 2 maintains this while maximizing I(Z; X). This is the Information Bottleneck perspective.
19. **Why adversarial fails**: It forces I(Z; S) ≈ 0 globally, which also destroys diagnostically-relevant, site-correlated signal. Curriculum achieves *selective invariance*.

---

## Phase G: Paper Writing (Week 5–8)

20. **Key figures** (6–8 for main paper):
   - Method overview (3-panel: Phase 1 → Phase 2 → Eval)
   - Main results (SRPBS + ABIDE, all methods)
   - Curriculum ablation
   - t-SNE/UMAP (site vs diagnosis coloring, before/after)
   - Cross-dataset transfer
   - Per-site heatmap
   - Training dynamics (site probe dropping during training)
   - Fingerprinting analysis

21. **Paper structure** (9 pages):
   - Intro (1p): Multi-site fMRI problem, curriculum insight
   - Related Work (0.75p): Harmonization, SSL for neuro, JEPA
   - Method (2p): Phase 1 + Phase 2 + architecture
   - Experiments (3.5p): Two datasets, main results, ablations, transfer
   - Analysis (1p): Why curriculum works, adversarial paradox, limitations
   - Conclusion (0.25p)

---

## Relevant Files

- `src/jepa/evaluate_all.py` — extend with statistical tests, MAE/MIDA baselines
- `src/jepa/train_phase2.py` — reference for ablation variants (replay removal, sampler removal)
- `src/jepa/visualize.py` — extend with publication figures
- `src/jepa/dataset.py` — add ABIDE data loader
- `preprocessing/extract_roi_timeseries.py` — reuse for ABIDE preprocessing
- `results/EVALUATION_REPORT.md` — current results baseline

---

## Verification Checklist

- [ ] All diagnostic tasks: mean ± std across 5 seeds, paired statistical tests with p-values
- [ ] ABIDE results confirm SRPBS narrative (curriculum helps on new dataset)
- [ ] ComBat baseline actually runs neuroCombat (fix the import bug)
- [ ] Ablation table complete — every curriculum component tested in isolation
- [ ] Reproducibility: `requirements.txt`, seeded randomness, saved configs

---

## Key Decisions

- **Primary story**: Curriculum learning via travelling subjects, not the JEPA architecture itself
- **Second dataset**: ABIDE I (public, multi-site, psychiatric)
- **Include adversarial paradox**: Best site-invariance but worst diagnosis = important finding
- **Multiclass (bAcc 0.151)**: Report only in appendix — focus on binary tasks in main paper

---

## Further Considerations

1. **Synthetic travelling subjects** — If ABIDE has no travelling subjects, could data augmentation simulate cross-site pairs? Interesting but risky for timeline. *Recommend: mention as future work.*
2. **Model scaling** — Current is small (3 layers). A 2 vs 3 vs 6 layer comparison would help but isn't critical. *Recommend: appendix if time allows.*
3. **Multimodal** — If any ABIDE sites have structural MRI, adding T1 features could strengthen results. *Recommend: skip unless trivial.*
