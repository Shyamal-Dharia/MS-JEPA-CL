# src/jepa/train_phase3.py
"""
Phase 3: End-to-end fine-tuning for diagnosis classification.

Instead of frozen probes on top of JEPA embeddings, this unfreezes the encoder
and trains it jointly with a classification head. Uses:
  - Lower LR for encoder (pretrained), higher LR for head (random init)
  - Optional site-adversarial regularization
  - LOSO evaluation: for each held-out site, trains from scratch on N-1 sites

Usage:
    python src/jepa/train_phase3.py
"""

import sys
import os
import json
import logging
from pathlib import Path
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.dataset import FMRIDataset
from jepa.model import JEPA


# ─────────────────────────── Config ───────────────────────────

class Phase3Config:
    # Which pretrained checkpoint to fine-tune
    checkpoint_path = "checkpoints/phase2/model_A_phase2_final.pt"
    checkpoint_name = "Phase2 van (64)"  # for logging

    # Model architecture (must match checkpoint)
    embed_dim = 64
    encoder_depth = 3
    predictor_depth = 2
    num_heads = 4
    mlp_ratio = 4.0
    dropout = 0.1
    ema_decay = 0.996

    # Data
    data_dir = "preprocessed_data_fir/roi_timeseries_1600"

    # Classification tasks to evaluate
    # Each: (task_name, hc_diag_code, disorder_diag_code)
    # Use 'ALL' as disorder_diag to group all non-HC diagnoses
    tasks = [
        # ── Primary: healthy vs any disease ──
        ('HC vs ALL',      0, 'ALL'),
        # ── Per-disorder breakdowns ──
        ('HC vs MDD',      0, 2),
        ('HC vs SCZ',      0, 4),
        ('HC vs ASD',      0, 1),
        ('HC vs Bipolar',  0, 7),
        ('HC vs Pain',     0, 5),
        ('HC vs Stroke',   0, 6),
        ('HC vs Dysthymia',0, 8),
        ('HC vs Others',   0, 99),
    ]

    # Fine-tuning hyperparameters
    ft_epochs = 30
    ft_batch_size = 32
    ft_encoder_lr = 1e-5        # low LR for pretrained encoder
    ft_head_lr = 1e-3           # high LR for random classification head
    ft_weight_decay = 0.01
    ft_warmup_epochs = 3

    # Classification head
    head_hidden = 128
    head_dropout = 0.3

    # Evaluation
    n_cv_folds = 5
    seeds = [42, 123, 777]      # average over multiple seeds

    # Output
    save_dir = "checkpoints/phase3"
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────── Classification Head ──────────────────────────

class DiagnosisHead(nn.Module):
    """2-layer MLP classification head on top of pooled encoder output."""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


# ────────────── End-to-end Fine-tuning Model ──────────────────

class FineTuneModel(nn.Module):
    """
    Wraps a pretrained JEPA encoder + classification head.
    Forward: raw fMRI → pooled embedding → logits
    """

    def __init__(self, jepa_model, head):
        super().__init__()
        self.token_embed = jepa_model.token_embed
        self.context_encoder = jepa_model.context_encoder
        self.head = head

    def forward(self, x):
        """
        x: (batch, 93, 400) raw fMRI timeseries
        Returns: (batch, 2) logits for binary classification
        """
        seq_len = x.size(1)
        idx = torch.arange(seq_len, device=x.device)
        emb = self.token_embed(x, idx)          # (B, 93, embed_dim)
        rep = self.context_encoder(emb)          # (B, 93, embed_dim)
        pooled = rep.flatten(start_dim=1)                 # (B, 93 * embed_dim)
        logits = self.head(pooled)               # (B, 2)
        return logits


# ────────────── Single Fine-Tuning Run ────────────────────────

def finetune_one_split(
    checkpoint_path, cfg, X_train, y_train, X_test, y_test,
    seed=42, device='cuda'
):
    """
    Load pretrained JEPA, attach head, fine-tune on (X_train, y_train),
    evaluate on (X_test, y_test).

    X_train, X_test: np.ndarray of shape (N, 93, 400) — raw fMRI
    y_train, y_test: np.ndarray of shape (N,) — binary labels
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load pretrained JEPA
    jepa = JEPA(
        input_dim=400, max_seq_len=93,
        embed_dim=cfg.embed_dim, encoder_depth=cfg.encoder_depth,
        predictor_depth=cfg.predictor_depth, num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout, ema_decay=cfg.ema_decay,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    jepa.load_state_dict(ckpt['model_state_dict'])

    # 2. Build fine-tune model
    head = DiagnosisHead(cfg.embed_dim, cfg.head_hidden, cfg.head_dropout)
    model = FineTuneModel(jepa, head).to(device)

    # 3. Separate param groups: low LR encoder, high LR head
    encoder_params = list(model.token_embed.parameters()) + list(model.context_encoder.parameters())
    head_params = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': cfg.ft_encoder_lr},
        {'params': head_params, 'lr': cfg.ft_head_lr},
    ], weight_decay=cfg.ft_weight_decay)

    # 4. Class-balanced loss
    n0, n1 = (y_train == 0).sum(), (y_train == 1).sum()
    w = torch.tensor([1.0 / n0, 1.0 / n1], dtype=torch.float32)
    w = w / w.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=w.to(device))

    # 5. DataLoader
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=cfg.ft_batch_size, shuffle=True)

    # 6. Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.ft_epochs)

    # 7. Training loop
    model.train()
    for epoch in range(cfg.ft_epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

    # 8. Evaluate
    model.eval()
    X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_te_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    return preds, probs


# ──────────────────── Main Evaluation ─────────────────────────

def evaluate_phase3(cfg: Phase3Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("Phase3")
    device = cfg.device

    # ── Load all data ──
    ds = FMRIDataset(cfg.data_dir, 'temporal')
    logger.info(f"Loaded {len(ds)} subjects")

    all_ts = []
    meta_list = []
    for i in range(len(ds)):
        x, labels = ds[i]
        all_ts.append(x.numpy())
        meta_list.append(labels)

    all_ts = np.array(all_ts)  # (N, 93, 400)
    diags = np.array([m['diag'] for m in meta_list])
    sites = np.array([m['site'] for m in meta_list])

    logger.info(f"Data shape: {all_ts.shape}")

    # ── Also evaluate checkpoints to compare ──
    checkpoints_to_eval = [
        (cfg.checkpoint_name, cfg.checkpoint_path, cfg.embed_dim, cfg.encoder_depth),
    ]

    all_results = []

    for ckpt_name, ckpt_path, embed_dim, enc_depth in checkpoints_to_eval:
        # Update config for this checkpoint
        this_cfg = deepcopy(cfg)
        this_cfg.embed_dim = embed_dim
        this_cfg.encoder_depth = enc_depth

        for task_name, hc_diag, disorder_diag in cfg.tasks:
            if disorder_diag == 'ALL':
                # HC vs all non-HC diagnoses
                mask = np.ones(len(diags), dtype=bool)
                y_task = (diags[mask] != hc_diag).astype(int)
            else:
                mask = np.isin(diags, [hc_diag, disorder_diag])
                y_task = (diags[mask] == disorder_diag).astype(int)
            X_task = all_ts[mask]
            sites_task = sites[mask]

            logger.info(f"\n{'='*70}")
            logger.info(f"TASK: {task_name} | Checkpoint: {ckpt_name}")
            logger.info(f"N={mask.sum()} (HC={sum(y_task==0)}, Disorder={sum(y_task==1)})")

            # ── LOSO evaluation ──
            unique_sites = np.unique(sites_task)
            loso_preds = np.full(len(y_task), np.nan)
            loso_probs = np.full(len(y_task), np.nan)
            sites_used = 0

            for held_out_site in unique_sites:
                test_mask = sites_task == held_out_site
                train_mask = ~test_mask

                if len(np.unique(y_task[test_mask])) < 2:
                    continue
                if len(np.unique(y_task[train_mask])) < 2:
                    continue

                # Average over seeds
                all_probs_seeds = []
                for seed in cfg.seeds:
                    _, probs = finetune_one_split(
                        ckpt_path, this_cfg,
                        X_task[train_mask], y_task[train_mask],
                        X_task[test_mask], y_task[test_mask],
                        seed=seed, device=device,
                    )
                    all_probs_seeds.append(probs)

                avg_probs = np.mean(all_probs_seeds, axis=0)
                loso_preds[test_mask] = (avg_probs >= 0.5).astype(int)
                loso_probs[test_mask] = avg_probs
                sites_used += 1
                logger.info(f"  Site {held_out_site}: done ({sites_used})")

            valid = ~np.isnan(loso_preds)
            loso_bacc = balanced_accuracy_score(y_task[valid], loso_preds[valid]) if valid.sum() > 0 else np.nan
            try:
                loso_auc = roc_auc_score(y_task[valid], loso_probs[valid]) if valid.sum() > 0 else np.nan
            except ValueError:
                loso_auc = np.nan

            # ── 5-fold CV evaluation ──
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=cfg.n_cv_folds, shuffle=True, random_state=42)
            cv_baccs, cv_aucs = [], []

            for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X_task, y_task)):
                all_probs_cv = []
                for seed in cfg.seeds:
                    _, probs = finetune_one_split(
                        ckpt_path, this_cfg,
                        X_task[tr_idx], y_task[tr_idx],
                        X_task[te_idx], y_task[te_idx],
                        seed=seed, device=device,
                    )
                    all_probs_cv.append(probs)

                avg_probs_cv = np.mean(all_probs_cv, axis=0)
                avg_preds_cv = (avg_probs_cv >= 0.5).astype(int)
                cv_baccs.append(balanced_accuracy_score(y_task[te_idx], avg_preds_cv))
                try:
                    cv_aucs.append(roc_auc_score(y_task[te_idx], avg_probs_cv))
                except ValueError:
                    pass
                logger.info(f"  CV fold {fold_idx+1}/{cfg.n_cv_folds}: done")

            cv_bacc = np.mean(cv_baccs)
            cv_auc = np.mean(cv_aucs) if cv_aucs else np.nan

            result = {
                'checkpoint': ckpt_name, 'task': task_name,
                'loso_bacc': loso_bacc, 'loso_auc': loso_auc, 'loso_sites': sites_used,
                'cv_bacc': cv_bacc, 'cv_auc': cv_auc,
            }
            all_results.append(result)

            logger.info(
                f"  {ckpt_name:25s} | LOSO bAcc={loso_bacc:.1%} AUC={loso_auc:.3f} ({sites_used} sites)"
                f" | CV bAcc={cv_bacc:.1%} AUC={cv_auc:.3f}"
            )

    # ── Save results ──
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "phase3_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Print summary ──
    print(f"\n\n{'='*80}")
    print(f"★ PHASE 3 END-TO-END FINE-TUNING RESULTS")
    print(f"{'='*80}")
    for r in all_results:
        print(f"  {r['checkpoint']:25s} | {r['task']:12s} | "
              f"LOSO bAcc={r['loso_bacc']:.1%} AUC={r['loso_auc']:.3f} | "
              f"CV bAcc={r['cv_bacc']:.1%} AUC={r['cv_auc']:.3f}")

    return all_results


# ──────────────────── Entry Point ─────────────────────────────

if __name__ == "__main__":
    cfg = Phase3Config()
    print("=" * 60)
    print("Phase 3: End-to-end fine-tuning for diagnosis")
    print(f"  Checkpoint: {cfg.checkpoint_path}")
    print(f"  Model: embed_dim={cfg.embed_dim}, depth={cfg.encoder_depth}")
    print(f"  Encoder LR: {cfg.ft_encoder_lr}, Head LR: {cfg.ft_head_lr}")
    print(f"  Fine-tune epochs: {cfg.ft_epochs}")
    print("=" * 60)
    evaluate_phase3(cfg)
