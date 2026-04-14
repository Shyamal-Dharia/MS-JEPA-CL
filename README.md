# MS-JEPA-CL

## Current Workflow

The active paper workflow is baseline-first and is tracked in [plan.md](plan.md).

Protocol rule:

- diagnosis evaluation uses `LOSO`
- site leakage uses a separate grouped subject-level site probe

## First Step

The first step is the baseline stage:

- `ROI time series`
- `Static FC`
- `Static FC + ComBat`
- `Static FC + CovBat`

Outputs are written under the clean namespace:

- `checkpoints/final_version_jepa_fusion`
- `results/reports/tables/final_version_jepa_fusion`
- `logs/final_version_jepa_fusion`

Launch command:

```bash
cd /data/s.dharia-ra/multi-site-problem

mkdir -p logs/final_version_jepa_fusion
mkdir -p checkpoints/final_version_jepa_fusion
mkdir -p results/reports/tables/final_version_jepa_fusion

nohup python -m src.jepa_fusion.run_methods_plan \
  --stage baseline_tables \
  --checkpoints-dir /data/s.dharia-ra/multi-site-problem/checkpoints/final_version_jepa_fusion \
  --results-dir /data/s.dharia-ra/multi-site-problem/results/reports/tables/final_version_jepa_fusion \
  > /data/s.dharia-ra/multi-site-problem/logs/final_version_jepa_fusion/baseline_tables.log 2>&1 &

echo $!
```

What it does:

- runs the baseline stage for `ABIDE-I`, `ABIDE-II`, and `SRPBS`
- computes grouped subject-level site probe metrics
- computes LOSO diagnosis metrics
- writes Markdown and JSON tables into `results/reports/tables/final_version_jepa_fusion`
- writes the live log to `logs/final_version_jepa_fusion/baseline_tables.log`

Monitor progress:

```bash
tail -f /data/s.dharia-ra/multi-site-problem/logs/final_version_jepa_fusion/baseline_tables.log
```
