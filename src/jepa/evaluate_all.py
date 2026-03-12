# src/jepa/evaluate_all.py
"""
Comprehensive evaluation: all methods × all tasks × proper LOSO CV.

Methods compared:
  B1: Raw (no pretraining)          — train classifier on raw fMRI
  B2: ComBat                        — statistical harmonization + classifier
  M1: JEPA vanilla                  — self-supervised only
  M2: JEPA + Adversarial v2         — gradient reversal site removal
  M3: JEPA + Contrastive            — NT-Xent on travelling subjects
  M4: JEPA + CL + VICReg            — contrastive + collapse prevention

Evaluation:
  - Frozen probing: Linear (LogReg) + MLP, LOSO CV
  - End-to-end fine-tuning: unfreeze encoder + head, LOSO + 5-fold CV
  - Metrics: balanced accuracy, AUC, F1 macro, precision, recall
  - Chance levels reported for every task

Usage:
    cd /data/s.dharia-ra/multi-site-problem
    PYTHONPATH=src python -m jepa.evaluate_all
"""

import sys
import os
import json
import logging
import time
import argparse
from pathlib import Path
from copy import deepcopy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score,
    f1_score, precision_score, recall_score,
    confusion_matrix,
)
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.dataset import FMRIDataset
from jepa.model import JEPA

logger = logging.getLogger("EvalAll")


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

class EvalConfig:
    # ── Data ──
    data_dir = "preprocessed_data_fir/roi_timeseries_1600"

    # ── Model architecture (must match all checkpoints) ──
    embed_dim = 400
    encoder_depth = 3
    predictor_depth = 2
    num_heads = 4
    mlp_ratio = 4.0
    dropout = 0.2
    ema_decay = 0.996

    # ── Methods to evaluate ──
    # (name, checkpoint_path_or_None, method_type)
    methods = [
        # Foundation Baselines (No unsupervised pre-training, Transformer from scratch)
        ("Raw (Transformer from scratch)",    None, "scratch"),
        ("ComBat (Transformer from scratch)", None, "combat_scratch"),
        
        # Generative / Contrastive Baselines (Pre-trained Transformers)
        ("SimCLR Baseline",      "checkpoints/baselines_simclr/simclr_final.pt", "jepa"),
        ("Transformer-VAE",      "checkpoints/baselines_vae/vae_final.pt",       "jepa"),
        
        # JEPA Progressions (Ablation)
        ("JEPA Vanilla (No Curriculum)",         "checkpoints/phase2_vanilla_nocurr/model_final.pt", "jepa"), 
        ("JEPA Vanilla (With Curriculum)",       "checkpoints/phase2/model_final.pt",                "jepa"), 
        ("JEPA + Contrastive (No Curriculum)",   "checkpoints/phase2_contrastive_nocurr/model_final.pt", "jepa"), 
        ("JEPA + Contrastive (With Curriculum)", "checkpoints/phase2_contrastive/model_final.pt",        "jepa"),
        
        # JEPA + Adversarial (GRL)
        ("JEPA + Adversarial (No Curriculum)",   "checkpoints/phase2_adversarial_nocurr/model_final.pt", "jepa"),
        ("JEPA + Adversarial (With Curriculum)", "checkpoints/phase2_adversarial/model_final.pt",        "jepa"),

        # JEPA + VICReg (standalone, no contrastive)
        ("JEPA + VICReg (No Curriculum)",        "checkpoints/phase2_vicreg_nocurr/model_final.pt",      "jepa"),
        ("JEPA + VICReg (With Curriculum)",       "checkpoints/phase2_vicreg/model_final.pt",            "jepa"),
    ]

    # ── Diagnosis tasks ──
    # (task_name, hc_diag_code, disorder_diag_code)
    # Use 'ALL' to group all non-HC
    tasks = [
        ("HC vs ALL",       0, "ALL"),
        ("HC vs MDD",       0, 2),
        ("HC vs SCZ",       0, 4),
        ("HC vs ASD",       0, 1),
        ("HC vs Bipolar",   0, 7),
        ("HC vs Pain",      0, 5),
        ("HC vs Stroke",    0, 6),
        ("HC vs Dysthymia", 0, 8),
        ("HC vs Others",    0, 99),
        ("Multiclass (ALL)",0, "MULTI"),
        ("Multiclass (Multi-Site)", 0, "MULTI_FILTERED"),
    ]

    # ── Fine-tuning hyperparams ──
    ft_epochs = 30
    ft_batch_size = 32
    ft_encoder_lr = 1e-5
    ft_head_lr = 1e-3
    ft_weight_decay = 0.01
    head_hidden = 128
    head_dropout = 0.3

    # ── Evaluation ──
    n_cv_folds = 5
    seeds = [42, 123, 456, 777, 999]
    val_fraction = 0.2  # fraction of train set held out for threshold calibration

    # ── Output ──
    save_dir = "results"
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════
# Metrics helper
# ═══════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute all metrics. Returns dict."""
    metrics = {
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 2:
                # Multiclass AUC requires (N, C) probability matrix
                if y_prob.ndim == 2 and y_prob.shape[1] > 1:
                    labels = np.arange(y_prob.shape[1])
                    metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted", labels=labels)
                else:
                    metrics["auc"] = float("nan")
            else:
                metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = float("nan")
    else:
        metrics["auc"] = float("nan")
    return metrics


def chance_level(y):
    """Chance-level balanced accuracy for a label array."""
    n_classes = len(np.unique(y))
    return 1.0 / n_classes if n_classes > 0 else 0.0


def find_optimal_threshold(y_true, y_prob):
    """
    Find the optimal classification threshold using Youden's J statistic.

    J = sensitivity + specificity - 1
    The threshold that maximizes J gives the best trade-off between
    true positive rate and true negative rate.

    Parameters
    ----------
    y_true : array of {0, 1}
    y_prob : array of predicted probabilities for class 1

    Returns
    -------
    optimal_threshold : float
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_j, best_thresh = -1.0, 0.5

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        # Sensitivity (recall of positive class)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        sensitivity = tp / max(tp + fn, 1)
        # Specificity (recall of negative class)
        tn = ((preds == 0) & (y_true == 0)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        specificity = tn / max(tn + fp, 1)

        j = sensitivity + specificity - 1
        if j > best_j:
            best_j = j
            best_thresh = t

    return best_thresh


def aggregate_seed_metrics(seed_metrics_list):
    """
    Aggregate metrics across multiple seeds.
    Returns dict with mean, std, and per-seed values.
    """
    keys = seed_metrics_list[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in seed_metrics_list if not (isinstance(m[k], float) and np.isnan(m[k]))]
        if len(vals) == 0:
            agg[k] = float("nan")
            agg[f"{k}_std"] = float("nan")
        elif isinstance(vals[0], (int, float, np.integer, np.floating)):
            agg[k] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        else:
            agg[k] = vals[0]  # non-numeric, just keep first
    return agg


def compute_per_site_metrics(y_true, y_pred, sites, valid_mask):
    """Compute bAcc per held-out site (only where both classes present)."""
    per_site = {}
    for site in np.unique(sites):
        mask = (sites == site) & valid_mask
        if mask.sum() < 2:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        if len(np.unique(yt)) < 2:
            per_site[str(site)] = {"bacc": float("nan"), "n": int(mask.sum())}
            continue
        per_site[str(site)] = {
            "bacc": float(balanced_accuracy_score(yt, yp)),
            "n": int(mask.sum()),
        }
    return per_site


def compute_topk_accuracy(y_true, y_probs, k=3):
    """Top-k accuracy for multiclass. y_probs: (N, C)."""
    if y_probs is None or y_probs.ndim != 2:
        return float("nan")
    topk = np.argsort(y_probs, axis=1)[:, -k:]
    correct = np.array([y_true[i] in topk[i] for i in range(len(y_true))])
    return float(correct.mean())


# ═══════════════════════════════════════════════════════════════
# Embedding extraction
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_jepa_embeddings(checkpoint_path, X_all, cfg):
    """
    Load JEPA from checkpoint, pass all samples through context encoder,
    return mean-pooled embeddings.

    X_all: np.ndarray (N, 93, 400)
    Returns: np.ndarray (N, embed_dim)
    """
    device = cfg.device
    model = JEPA(
        input_dim=400, max_seq_len=93,
        embed_dim=cfg.embed_dim, encoder_depth=cfg.encoder_depth,
        predictor_depth=cfg.predictor_depth, num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout, ema_decay=cfg.ema_decay,
    ).to(device)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    
    model.eval()

    all_indices = torch.arange(93).to(device)
    embeddings = []
    batch_size = 64

    for i in range(0, len(X_all), batch_size):
        x = torch.tensor(X_all[i:i+batch_size], dtype=torch.float32).to(device)
        emb = model.token_embed(x, all_indices)
        rep = model.context_encoder(emb)
        pooled = rep.mean(dim=1)  # (B, embed_dim)
        embeddings.append(pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def get_raw_features(X_all):
    """Flatten raw fMRI to feature vectors. (N, 93, 400) → (N, 37200)."""
    return X_all.reshape(X_all.shape[0], -1)


def get_combat_features(X_all, sites):
    """
    Apply ComBat harmonization to flattened features.
    Returns harmonized feature matrix (N, 37200).
    """
    try:
        from neuroCombat import neuroCombat
        import pandas as pd
    except ImportError:
        logger.error("neuroCombat not installed! Run: pip install neuroCombat")
        logger.error("Falling back to raw features for ComBat baseline.")
        return get_raw_features(X_all)

    X_flat = get_raw_features(X_all)  # (N, 37200)

    # neuroCombat expects (features, samples) = (37200, N)
    data = X_flat.T.copy()

    # Batch variable (site) — must be a pandas DataFrame
    le = LabelEncoder()
    batch = le.fit_transform(sites)

    covars = pd.DataFrame({"batch": batch})

    logger.info(f"  Running ComBat on {X_flat.shape} ({len(np.unique(batch))} sites)...")
    result = neuroCombat(dat=data, covars=covars, batch_col="batch")
    harmonized = result["data"].T  # back to (N, 37200)

    logger.info(f"  ComBat done. Output shape: {harmonized.shape}")
    return harmonized


# ═══════════════════════════════════════════════════════════════
# Frozen probing (Linear + MLP)
# ═══════════════════════════════════════════════════════════════

def run_frozen_probe_loso_single(X, y, sites, probe_type="linear", class_weight=True,
                                  seed=42, val_fraction=0.2):
    """
    LOSO frozen probing with threshold calibration (single seed).

    For each held-out site:
      - Split train into train_inner / val (stratified)
      - StandardScaler fit on train_inner
      - Train probe on train_inner
      - Find optimal threshold on val via Youden's J
      - Apply calibrated threshold to test

    Returns aggregated metrics dict.
    """
    unique_sites = np.unique(sites)
    n_classes = len(np.unique(y))
    all_preds = np.full(len(y), np.nan)
    if n_classes > 2:
        all_probs = np.full((len(y), n_classes), np.nan)
    else:
        all_probs = np.full(len(y), np.nan)

    for site in unique_sites:
        test_mask = sites == site
        train_mask = ~test_mask

        # Need both classes in train and test
        if len(np.unique(y[test_mask])) < 2:
            continue
        if len(np.unique(y[train_mask])) < 2:
            continue

        X_train_full = X[train_mask]
        y_train_full = y[train_mask]

        # ── Split train into train_inner / val for threshold calibration ──
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
            train_inner_idx, val_idx = next(sss.split(X_train_full, y_train_full))
        except ValueError:
            # If stratification fails (too few samples), use all train data
            train_inner_idx = np.arange(len(y_train_full))
            val_idx = np.arange(len(y_train_full))  # val = train as fallback

        X_train_inner = X_train_full[train_inner_idx]
        y_train_inner = y_train_full[train_inner_idx]
        X_val = X_train_full[val_idx]
        y_val = y_train_full[val_idx]

        # Scale
        scaler = StandardScaler()
        X_train_inner_scaled = scaler.fit_transform(X_train_inner)
        X_val_scaled = scaler.transform(X_val)
        X_test = scaler.transform(X[test_mask])

        if probe_type == "linear":
            clf = LogisticRegression(
                max_iter=2000, random_state=seed,
                class_weight="balanced" if class_weight else None,
            )
        else:  # mlp
            train_counts = np.bincount(y_train_inner)
            can_early_stop = (len(train_counts) >= 2 and np.min(train_counts) >= 2)
            clf = MLPClassifier(
                hidden_layer_sizes=(64, 64),
                activation="relu",
                max_iter=500,
                random_state=seed,
                early_stopping=can_early_stop,
                validation_fraction=0.15 if can_early_stop else 0.1,
            )

        clf.fit(X_train_inner_scaled, y_train_inner)

        if len(np.unique(y_train_full)) > 2:
            test_probs_part = clf.predict_proba(X_test)
            test_preds = clf.predict(X_test)
            all_preds[test_mask] = test_preds

            test_probs_full = np.zeros((len(X_test), n_classes))
            for idx, c in enumerate(clf.classes_):
                test_probs_full[:, c] = test_probs_part[:, idx]
            all_probs[test_mask] = test_probs_full
        else:
            # ── Find optimal threshold on validation set ──
            val_probs = clf.predict_proba(X_val_scaled)[:, 1]
            optimal_threshold = find_optimal_threshold(y_val, val_probs)
    
            # ── Apply calibrated threshold to test set ──
            test_probs = clf.predict_proba(X_test)[:, 1]
            test_preds = (test_probs >= optimal_threshold).astype(int)
    
            all_preds[test_mask] = test_preds
            all_probs[test_mask] = test_probs

    # Aggregate over valid predictions
    valid = ~np.isnan(all_preds)
    preds_info = {
        "all_preds": all_preds, "all_probs": all_probs,
        "y_true": y, "sites": sites, "valid_mask": valid,
    }
    if valid.sum() == 0:
        return ({"bacc": float("nan"), "auc": float("nan"), "f1_macro": float("nan"),
                 "precision": float("nan"), "recall": float("nan"), "n_valid_sites": 0},
                preds_info)

    # For multiclass, all_probs might have rows of NaNs
    prob_arr = all_probs[valid]
    if prob_arr.ndim == 1 and np.all(np.isnan(prob_arr)):
        prob_arr = None
    elif prob_arr.ndim == 2 and np.all(np.isnan(prob_arr)):
        prob_arr = None

    metrics = compute_metrics(y[valid], all_preds[valid].astype(int), prob_arr)

    # Count how many sites were used
    sites_used = 0
    for site in unique_sites:
        mask = sites == site
        if not np.all(np.isnan(all_preds[mask])):
            sites_used += 1
    metrics["n_valid_sites"] = sites_used

    return metrics, preds_info


def run_frozen_probe_loso(X, y, sites, probe_type="linear", class_weight=True,
                          seeds=None, val_fraction=0.2):
    """
    Multi-seed LOSO frozen probing with threshold calibration.

    Runs run_frozen_probe_loso_single() across multiple seeds and
    returns aggregated metrics (mean ± std).
    """
    if seeds is None:
        seeds = [42]

    seed_results = []
    first_preds = None
    for i, seed in enumerate(seeds):
        m, preds = run_frozen_probe_loso_single(X, y, sites, probe_type, class_weight,
                                                 seed=seed, val_fraction=val_fraction)
        seed_results.append(m)
        if i == 0:
            first_preds = preds

    return aggregate_seed_metrics(seed_results), first_preds


def run_frozen_probe_site_cv(X, y_site, probe_type="linear"):
    """
    KFold cross-validation specifically for Site Classification,
    since Leave-One-Site-Out is impossible.
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_preds = np.full(len(y_site), np.nan)
    
    for train_idx, test_idx in kf.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        
        if probe_type == "linear":
            clf = LogisticRegression(max_iter=2000, random_state=42)
        else:
            train_counts = np.bincount(y_site[train_idx])
            can_early_stop = (len(train_counts) >= 2 and np.min(train_counts) >= 2)
            clf = MLPClassifier(
                hidden_layer_sizes=(64, 64),
                activation="relu",
                max_iter=500,
                random_state=42,
                early_stopping=can_early_stop,
                validation_fraction=0.15 if can_early_stop else 0.1,
            )
            
        clf.fit(X_train, y_site[train_idx])
        preds = clf.predict(X_test)
        all_preds[test_idx] = preds

    metrics = compute_metrics(y_site, all_preds.astype(int), None)
    metrics["n_valid_sites"] = len(np.unique(y_site))
    return metrics


# ═══════════════════════════════════════════════════════════════
# End-to-end fine-tuning
# ═══════════════════════════════════════════════════════════════

class DiagnosisHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3, out_dim=2):
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
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FineTuneModel(nn.Module):
    def __init__(self, jepa_model, head):
        super().__init__()
        self.token_embed = jepa_model.token_embed
        self.context_encoder = jepa_model.context_encoder
        self.head = head

    def forward(self, x):
        seq_len = x.size(1)
        idx = torch.arange(seq_len, device=x.device)
        emb = self.token_embed(x, idx)
        rep = self.context_encoder(emb)
        pooled = rep.mean(dim=1)
        return self.head(pooled)


class RawFineTuneModel(nn.Module):
    """MLP classifier on raw flattened fMRI (no JEPA)."""
    def __init__(self, input_dim=37200, hidden_dim=128, dropout=0.3, out_dim=2):
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
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def finetune_one_split(checkpoint_path, cfg, X_train, y_train, X_test, y_test,
                        method_type="jepa", seed=42, n_classes=None):
    """
    Fine-tune and evaluate on one train/test split.
    Returns (preds, probs).
    """
    device = cfg.device
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if n_classes is None:
        n_classes = len(np.unique(y_train))
    out_dim = n_classes if n_classes > 2 else 2

    if method_type == "jepa":
        # Load pretrained JEPA
        jepa = JEPA(
            input_dim=400, max_seq_len=93,
            embed_dim=cfg.embed_dim, encoder_depth=cfg.encoder_depth,
            predictor_depth=cfg.predictor_depth, num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout, ema_decay=cfg.ema_decay,
        ).to(device)
        
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            jepa.load_state_dict(ckpt["model_state_dict"], strict=False)

        flattened_dim = cfg.embed_dim
        head = DiagnosisHead(flattened_dim, cfg.head_hidden, cfg.head_dropout, out_dim=out_dim)
        model = FineTuneModel(jepa, head).to(device)

        encoder_params = list(model.token_embed.parameters()) + list(model.context_encoder.parameters())
        head_params = list(model.head.parameters())
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": cfg.ft_encoder_lr},
            {"params": head_params, "lr": cfg.ft_head_lr},
        ], weight_decay=cfg.ft_weight_decay)

        X_tr_t = torch.tensor(X_train, dtype=torch.float32)
        X_te_t = torch.tensor(X_test, dtype=torch.float32)

    else:  # raw or combat — flattened features
        input_dim = X_train.shape[1] if X_train.ndim == 2 else 93 * 400
        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

        # StandardScaler for raw/combat
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RawFineTuneModel(input_dim=X_train.shape[1],
                                  hidden_dim=cfg.head_hidden,
                                  dropout=cfg.head_dropout, out_dim=out_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                       lr=cfg.ft_head_lr,
                                       weight_decay=cfg.ft_weight_decay)

        X_tr_t = torch.tensor(X_train, dtype=torch.float32)
        X_te_t = torch.tensor(X_test, dtype=torch.float32)

    # Class-balanced loss (inverse frequency weights)
    counts = np.bincount(y_train, minlength=out_dim)
    w = torch.tensor([1.0 / max(c, 1) if c > 0 else 0.0 for c in counts], dtype=torch.float32)
    w = w / w.sum() * (counts > 0).sum()
    criterion = nn.CrossEntropyLoss(weight=w.to(device))

    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=cfg.ft_batch_size, shuffle=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.ft_epochs)

    # Train
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

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_te_t.to(device))
        probs = torch.softmax(logits, dim=1)
        if out_dim == 2:
            probs = probs[:, 1].cpu().numpy()
        else:
            probs = probs.cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    return preds, probs


def run_finetune_loso_single(checkpoint_path, cfg, X_task, y_task, sites_task,
                             method_type="jepa", seed=42):
    """
    Run LOSO fine-tuning evaluation with threshold calibration (single seed).

    For each held-out site:
      - Split train into train_inner/val (80/20 stratified)
      - Fine-tune on train_inner
      - Find optimal threshold on val via Youden's J
      - Apply calibrated threshold to held-out test site

    Returns aggregated metrics dict for one seed.
    """
    unique_sites = np.unique(sites_task)
    n_classes = len(np.unique(y_task))
    all_preds = np.full(len(y_task), np.nan)
    if n_classes > 2:
        all_probs = np.full((len(y_task), n_classes), np.nan)
    else:
        all_probs = np.full(len(y_task), np.nan)
    sites_used = 0

    for site in unique_sites:
        test_mask = sites_task == site
        train_mask = ~test_mask

        if len(np.unique(y_task[test_mask])) < 2:
            continue
        if len(np.unique(y_task[train_mask])) < 2:
            continue

        X_train_full = X_task[train_mask]
        y_train_full = y_task[train_mask]

        # Split train into train_inner / val for threshold calibration
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.val_fraction,
                                         random_state=seed)
            train_inner_idx, val_idx = next(sss.split(X_train_full, y_train_full))
        except ValueError:
            train_inner_idx = np.arange(len(y_train_full))
            val_idx = np.arange(len(y_train_full))

        # Get val probs for threshold calibration
        _, val_probs = finetune_one_split(
            checkpoint_path, cfg,
            X_train_full[train_inner_idx], y_train_full[train_inner_idx],
            X_train_full[val_idx], y_train_full[val_idx],
            method_type=method_type, seed=seed, n_classes=n_classes,
        )

        # Get test probs
        _, test_probs = finetune_one_split(
            checkpoint_path, cfg,
            X_train_full[train_inner_idx], y_train_full[train_inner_idx],
            X_task[test_mask], y_task[test_mask],
            method_type=method_type, seed=seed, n_classes=n_classes,
        )

        if len(np.unique(y_train_full)) > 2:
            test_preds = np.argmax(test_probs, axis=1)
            all_preds[test_mask] = test_preds
            all_probs[test_mask] = test_probs
        else:
            # Find optimal threshold on validation set
            optimal_threshold = find_optimal_threshold(y_train_full[val_idx], val_probs)

            # Apply calibrated threshold to test
            all_preds[test_mask] = (test_probs >= optimal_threshold).astype(int)
            all_probs[test_mask] = test_probs

        sites_used += 1

    valid = ~np.isnan(all_preds)
    preds_info = {
        "all_preds": all_preds, "all_probs": all_probs,
        "y_true": y_task, "sites": sites_task, "valid_mask": valid,
    }
    if valid.sum() == 0:
        return ({"bacc": float("nan"), "auc": float("nan"), "f1_macro": float("nan"),
                 "precision": float("nan"), "recall": float("nan"), "n_valid_sites": 0},
                preds_info)

    # For multiclass, all_probs might have rows of NaNs where valid is False, but we slice with valid
    prob_arr = all_probs[valid]
    if prob_arr.ndim == 1 and np.all(np.isnan(prob_arr)):
        prob_arr = None
    elif prob_arr.ndim == 2 and np.all(np.isnan(prob_arr)):
        prob_arr = None

    metrics = compute_metrics(y_task[valid], all_preds[valid].astype(int), prob_arr)
    metrics["n_valid_sites"] = sites_used
    return metrics, preds_info


def run_finetune_loso(checkpoint_path, cfg, X_task, y_task, sites_task, method_type="jepa"):
    """
    Multi-seed LOSO fine-tuning with threshold calibration.

    Runs run_finetune_loso_single() across cfg.seeds and returns mean +- std.
    """
    seed_results = []
    first_preds = None
    for i, seed in enumerate(cfg.seeds):
        seed_metrics, preds = run_finetune_loso_single(
            checkpoint_path, cfg, X_task, y_task, sites_task,
            method_type=method_type, seed=seed,
        )
        seed_results.append(seed_metrics)
        if i == 0:
            first_preds = preds

    return aggregate_seed_metrics(seed_results), first_preds


# ═══════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_all(cfg: EvalConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = cfg.device

    # ── Load all data ──
    ds = FMRIDataset(cfg.data_dir, "temporal")
    logger.info(f"Loaded {len(ds)} subjects")

    all_ts, meta_list = [], []
    for i in range(len(ds)):
        x, labels = ds[i]
        all_ts.append(x.numpy())
        meta_list.append(labels)

    all_ts = np.array(all_ts)  # (N, 93, 400)
    diags = np.array([m["diag"] for m in meta_list])
    sites = np.array([m["site"] for m in meta_list])

    logger.info(f"Data: {all_ts.shape}, {len(np.unique(sites))} sites, {len(np.unique(diags))} diagnoses")

    # ── Pre-compute embeddings for all methods ──
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Extracting embeddings / features for all methods")
    logger.info("=" * 70)

    method_features = {}

    for method_name, ckpt_path, method_type in cfg.methods:
        logger.info(f"\n  → {method_name}")
        t0 = time.time()

        if method_type == "raw":
            method_features[method_name] = get_raw_features(all_ts)
        elif method_type == "combat":
            method_features[method_name] = get_combat_features(all_ts, sites)
        elif method_type == "jepa":
            method_features[method_name] = extract_jepa_embeddings(ckpt_path, all_ts, cfg)
        elif method_type == "scratch":
            # Pass None as checkpoint to bypass loading
            method_features[method_name] = extract_jepa_embeddings(None, all_ts, cfg)
        elif method_type == "combat_scratch":
            # Run combat on raw features -> Unflatten to 3D -> Pass to JEPA Random Init
            combat_flat = get_combat_features(all_ts, sites)
            combat_3d = combat_flat.reshape(len(all_ts), 93, 400)
            method_features[method_name] = extract_jepa_embeddings(None, combat_3d, cfg)

        dt = time.time() - t0
        shape = method_features[method_name].shape
        logger.info(f"    Shape: {shape} ({dt:.1f}s)")

    # ── Save embeddings for visualization ──
    embed_dir = Path(cfg.save_dir) / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)
    np.save(embed_dir / "sites.npy", sites)
    np.save(embed_dir / "diags.npy", diags)
    for mname, feat in method_features.items():
        safe_name = mname.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        np.save(embed_dir / f"{safe_name}.npy", feat)
    logger.info(f"Embeddings saved to {embed_dir}/ ({len(method_features)} methods)")

    # ── Results storage ──
    all_results = []
    all_preds_store = {}  # (method, eval_type, task) -> preds_info dict

    # ── STEP 2: Site classification probing ──
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Site classification probing (LOSO)")
    logger.info("=" * 70)

    site_le = LabelEncoder()
    y_site = site_le.fit_transform(sites)
    site_chance = chance_level(y_site)
    n_site_classes = len(site_le.classes_)
    logger.info(f"  {n_site_classes} sites, chance = {site_chance:.1%}")

    for method_name, _, _ in cfg.methods:
        X = method_features[method_name]
        for probe_type in ["linear", "mlp"]:
            logger.info(f"\n  {method_name} | {probe_type} probe → site")
            metrics = run_frozen_probe_site_cv(X, y_site, probe_type=probe_type)
            result = {
                "method": method_name,
                "eval_type": f"site_probe_{probe_type}",
                "task": "Site Classification",
                "chance": site_chance,
                **metrics,
            }
            all_results.append(result)
            logger.info(
                f"    bAcc={metrics['bacc']:.1%} (chance={site_chance:.1%}) | "
                f"F1={metrics['f1_macro']:.3f} | "
                f"Sites={metrics['n_valid_sites']}"
            )

    # ── STEP 3: Diagnosis probing + fine-tuning ──
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Diagnosis classification (LOSO)")
    logger.info("=" * 70)

    for task_name, hc_diag, disorder_diag in cfg.tasks:
        # Build task data
        if disorder_diag == "ALL":
            mask = np.ones(len(diags), dtype=bool)
            y_task = (diags != hc_diag).astype(int)
        elif disorder_diag == "MULTI":
            mask = np.ones(len(diags), dtype=bool)
            y_task = LabelEncoder().fit_transform(diags)
        elif disorder_diag == "MULTI_FILTERED":
            # Keep only diagnosis classes that appear at >=2 sites
            from collections import defaultdict
            diag_sites = defaultdict(set)
            for d, st in zip(diags, sites):
                diag_sites[d].add(st)
            valid_classes = sorted([d for d, st_set in diag_sites.items() if len(st_set) >= 2])
            mask = np.isin(diags, valid_classes)
            y_task = LabelEncoder().fit_transform(diags[mask])
        else:
            mask = np.isin(diags, [hc_diag, disorder_diag])
            y_task = (diags[mask] == disorder_diag).astype(int)

        X_task_raw = all_ts[mask]
        sites_task = sites[mask]
        task_chance = chance_level(y_task)
        n_classes = len(np.unique(y_task))

        logger.info(f"\n{'─' * 70}")
        if disorder_diag in ("MULTI", "MULTI_FILTERED"):
            class_counts = {int(c): int((y_task == c).sum()) for c in np.unique(y_task)}
            logger.info(f"TASK: {task_name} | N={len(y_task)} | {n_classes} classes | counts={class_counts} | chance={task_chance:.1%}")
            n_hc = class_counts.get(0, 0)
            n_dis = len(y_task) - n_hc
        else:
            n_hc = (y_task == 0).sum()
            n_dis = (y_task == 1).sum()
            logger.info(f"TASK: {task_name} | N={len(y_task)} (HC={n_hc}, Disease={n_dis}) | chance={task_chance:.1%}")

        if n_dis < 4:
            logger.info(f"  ⚠ Skipping {task_name}: too few disease samples (n={n_dis})")
            continue

        # Skip if disease subjects are all at one site (LOSO will produce 0 valid folds)
        if disorder_diag not in ("ALL", "MULTI", "MULTI_FILTERED"):
            disease_mask = y_task == 1
            disease_sites = set(sites_task[disease_mask])
            if len(disease_sites) < 2:
                logger.info(f"  ⚠ Skipping {task_name}: disease subjects only at {len(disease_sites)} site(s) — LOSO impossible")
                continue

        for method_name, ckpt_path, method_type in cfg.methods:
            # Get features for this task subset
            X_feat_all = method_features[method_name]
            X_feat = X_feat_all[mask]

            # ── 3A: Frozen probing ──
            for probe_type in ["linear", "mlp"]:
                metrics, preds_info = run_frozen_probe_loso(X_feat, y_task, sites_task,
                                                probe_type=probe_type, class_weight=True,
                                                seeds=cfg.seeds, val_fraction=cfg.val_fraction)

                # Collect detailed predictions for later analysis
                detail_key = (method_name, f"frozen_{probe_type}", task_name)
                if preds_info is not None:
                    all_preds_store[detail_key] = preds_info

                result = {
                    "method": method_name,
                    "eval_type": f"frozen_{probe_type}",
                    "task": task_name,
                    "chance": task_chance,
                    "n_hc": int(n_hc), "n_disease": int(n_dis),
                    **metrics,
                }
                all_results.append(result)
                bacc_std = metrics.get('bacc_std', 0)
                auc_std = metrics.get('auc_std', 0)
                logger.info(
                    f"  {method_name:25s} | {probe_type:6s} probe | "
                    f"bAcc={metrics['bacc']:.1%}±{bacc_std:.1%} "
                    f"(Δ={metrics['bacc']-task_chance:+.1%}) | "
                    f"AUC={metrics['auc']:.3f}±{auc_std:.3f} | "
                    f"F1={metrics['f1_macro']:.3f} | "
                    f"Sites={metrics['n_valid_sites']}"
                )

            # ── 3B: End-to-end fine-tuning (LOSO) ──
            if method_type in ["jepa", "scratch"]:
                ft_metrics, ft_preds = run_finetune_loso(
                    ckpt_path, cfg, X_task_raw, y_task, sites_task,
                    method_type="jepa",
                )
            elif method_type == "combat_scratch":
                # ComBat harmonize, reshape to 3D, fine-tune transformer
                combat_flat = get_combat_features(X_task_raw, sites_task)
                combat_3d = combat_flat.reshape(len(X_task_raw), 93, 400)
                ft_metrics, ft_preds = run_finetune_loso(
                    None, cfg, combat_3d, y_task, sites_task,
                    method_type="jepa",
                )
            else:
                # raw — fine-tune an MLP on raw flattened features
                ft_metrics, ft_preds = run_finetune_loso(
                    None, cfg, X_task_raw, y_task, sites_task,
                    method_type="raw",
                )

            # Collect detailed predictions for later analysis
            detail_key = (method_name, "finetune_loso", task_name)
            if ft_preds is not None:
                all_preds_store[detail_key] = ft_preds

            result = {
                "method": method_name,
                "eval_type": "finetune_loso",
                "task": task_name,
                "chance": task_chance,
                "n_hc": int(n_hc), "n_disease": int(n_dis),
                **ft_metrics,
            }
            all_results.append(result)
            bacc_std = ft_metrics.get("bacc_std", float("nan"))
            auc_std = ft_metrics.get("auc_std", float("nan"))
            logger.info(
                f"  {method_name:25s} | finetune | "
                f"bAcc={ft_metrics['bacc']:.1%}±{bacc_std:.1%} "
                f"(Δ={ft_metrics['bacc']-task_chance:+.1%}) | "
                f"AUC={ft_metrics['auc']:.3f}±{auc_std:.3f} | "
                f"F1={ft_metrics['f1_macro']:.3f} | "
                f"Sites={ft_metrics['n_valid_sites']}"
            )

    # ── Save results ──
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results_path = save_dir / "evaluation_results.json"
    # Merge with existing results (for parallel runs)
    existing = []
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing = []
    # Remove old entries for methods we just evaluated
    new_methods = {r["method"] for r in all_results}
    existing = [r for r in existing if r["method"] not in new_methods]
    merged = existing + all_results
    with open(results_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    logger.info(f"\nResults saved: {results_path} ({len(all_results)} new, {len(existing)} existing)")

    # ── Save detailed analysis (confusion matrices, per-site, top-k) ──
    detail_dir = save_dir / "detailed"
    detail_dir.mkdir(parents=True, exist_ok=True)
    detailed_analyses = []

    for (method, eval_type, task), pinfo in all_preds_store.items():
        valid = pinfo["valid_mask"]
        if valid.sum() == 0:
            continue
        y_true_v = pinfo["y_true"][valid]
        y_pred_v = pinfo["all_preds"][valid].astype(int)
        probs_v = pinfo["all_probs"][valid] if pinfo["all_probs"] is not None else None
        sites_v = pinfo["sites"]

        entry = {"method": method, "eval_type": eval_type, "task": task}

        # Per-site metrics
        entry["per_site"] = compute_per_site_metrics(
            pinfo["y_true"], pinfo["all_preds"], pinfo["sites"], valid
        )

        # Confusion matrix
        labels = sorted(np.unique(y_true_v))
        cm = confusion_matrix(y_true_v, y_pred_v, labels=labels)
        entry["confusion_matrix"] = cm.tolist()
        entry["cm_labels"] = [int(l) for l in labels]

        # Top-k accuracy (multiclass only)
        if len(labels) > 2 and probs_v is not None and probs_v.ndim == 2:
            entry["top2_acc"] = compute_topk_accuracy(y_true_v, probs_v, k=2)
            entry["top3_acc"] = compute_topk_accuracy(y_true_v, probs_v, k=3)

        detailed_analyses.append(entry)

    if detailed_analyses:
        detail_path = detail_dir / "detailed_analysis.json"
        with open(detail_path, "w") as f:
            json.dump(detailed_analyses, f, indent=2, default=str)
        logger.info(f"Detailed analysis saved: {detail_path} ({len(detailed_analyses)} entries)")

    # ── Pairwise statistical significance tests ──
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Statistical significance tests (Wilcoxon signed-rank)")
    logger.info("=" * 70)

    stat_results = []
    # Group per-site bAcc by (eval_type, task)
    from itertools import combinations
    task_eval_groups = {}
    for (method, eval_type, task), pinfo in all_preds_store.items():
        valid = pinfo["valid_mask"]
        if valid.sum() == 0:
            continue
        per_site = compute_per_site_metrics(
            pinfo["y_true"], pinfo["all_preds"], pinfo["sites"], valid
        )
        # Collect per-site bAcc as a dict {site: bacc}
        site_baccs = {s: v["bacc"] for s, v in per_site.items()
                      if not (isinstance(v["bacc"], float) and np.isnan(v["bacc"]))}
        key = (eval_type, task)
        if key not in task_eval_groups:
            task_eval_groups[key] = {}
        task_eval_groups[key][method] = site_baccs

    for (eval_type, task), method_site_baccs in task_eval_groups.items():
        methods_list = sorted(method_site_baccs.keys())
        if len(methods_list) < 2:
            continue
        for m1, m2 in combinations(methods_list, 2):
            # Align on shared sites
            shared = sorted(set(method_site_baccs[m1].keys()) & set(method_site_baccs[m2].keys()))
            if len(shared) < 5:
                continue
            vals1 = [method_site_baccs[m1][s] for s in shared]
            vals2 = [method_site_baccs[m2][s] for s in shared]
            try:
                stat, pval = wilcoxon(vals1, vals2)
            except ValueError:
                continue
            mean_diff = np.mean(vals1) - np.mean(vals2)
            entry = {
                "eval_type": eval_type, "task": task,
                "method_1": m1, "method_2": m2,
                "mean_diff": round(float(mean_diff), 4),
                "n_sites": len(shared),
                "wilcoxon_stat": round(float(stat), 4),
                "p_value": round(float(pval), 6),
                "significant_005": pval < 0.05,
                "significant_001": pval < 0.01,
            }
            stat_results.append(entry)
            if pval < 0.05:
                logger.info(
                    f"  {eval_type:15s} | {task:25s} | "
                    f"{m1} vs {m2}: Δ={mean_diff:+.3f}, p={pval:.4f} *"
                )

    if stat_results:
        stat_path = detail_dir / "statistical_tests.json"
        with open(stat_path, "w") as f:
            json.dump(stat_results, f, indent=2, default=str)
        n_sig = sum(1 for s in stat_results if s["significant_005"])
        logger.info(f"Statistical tests saved: {stat_path} ({len(stat_results)} pairs, {n_sig} significant at p<0.05)")

    # ── Print summary table ──
    print_summary_table(all_results, cfg)

    return all_results


def print_summary_table(results, cfg):
    """Pretty-print comparison table."""
    print(f"\n\n{'═' * 100}")
    print("★ COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'═' * 100}")

    # ── Site probe table ──
    print(f"\n{'─' * 80}")
    print("SITE CLASSIFICATION (lower = more site-invariant)")
    print(f"{'─' * 80}")
    print(f"{'Method':30s} | {'Linear bAcc':12s} | {'MLP bAcc':12s} | Chance")

    site_results = [r for r in results if r["task"] == "Site Classification"]
    methods_seen = []
    for r in site_results:
        if r["method"] not in methods_seen:
            methods_seen.append(r["method"])

    for method in methods_seen:
        linear = [r for r in site_results if r["method"] == method and "linear" in r["eval_type"]]
        mlp = [r for r in site_results if r["method"] == method and "mlp" in r["eval_type"]]
        lin_acc = f"{linear[0]['bacc']:.1%}" if linear else "N/A"
        mlp_acc = f"{mlp[0]['bacc']:.1%}" if mlp else "N/A"
        chance = f"{linear[0]['chance']:.1%}" if linear else "?"
        print(f"  {method:28s} | {lin_acc:12s} | {mlp_acc:12s} | {chance}")

    # ── Diagnosis tables ──
    for task_name, _, _ in cfg.tasks:
        task_results = [r for r in results if r["task"] == task_name]
        if not task_results:
            continue

        n_hc = task_results[0].get("n_hc", "?")
        n_dis = task_results[0].get("n_disease", "?")
        chance = task_results[0].get("chance", 0.5)

        print(f"\n{'─' * 100}")
        print(f"{task_name} (HC={n_hc}, Disease={n_dis}) | Chance={chance:.1%}")
        print(f"{'─' * 100}")
        print(f"{'Method':30s} | {'Eval':10s} | {'bAcc (mean±std)':18s} {'Δ':7s} | {'AUC (mean±std)':16s} | {'F1':6s} | Sites")

        for method in methods_seen:
            for eval_type in ["frozen_linear", "frozen_mlp", "finetune_loso"]:
                matches = [r for r in task_results
                           if r["method"] == method and r["eval_type"] == eval_type]
                if not matches:
                    continue
                r = matches[0]
                delta = r["bacc"] - chance if not np.isnan(r["bacc"]) else float("nan")
                eval_label = {"frozen_linear": "LinProbe", "frozen_mlp": "MLPProbe",
                              "finetune_loso": "FineTune"}[eval_type]
                bacc_std = r.get('bacc_std', 0)
                auc_std = r.get('auc_std', 0)
                bacc_str = f"{r['bacc']:.1%}±{bacc_std:.1%}"
                auc_str = f"{r['auc']:.3f}±{auc_std:.3f}"
                print(
                    f"  {method:28s} | {eval_label:10s} | "
                    f"{bacc_str:18s} {delta:+.1%} | "
                    f"{auc_str:16s} | {r['f1_macro']:.3f} | "
                    f"{r.get('n_valid_sites', '?')}"
                )


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate JEPA models and baselines")
    parser.add_argument("--method", type=str, default="all",
                        help="Specific method name to evaluate (e.g. 'SimCLR Baseline'). Defaults to all.")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to use (also selects method shard for multi-GPU parallel runs).")
    parser.add_argument("--n-shards", type=int, default=1,
                        help="Total number of GPU shards (use with --gpu for parallel runs).")
    args = parser.parse_args()

    cfg = EvalConfig()

    # Override device if --gpu is specified
    if args.gpu is not None:
        cfg.device = f"cuda:{args.gpu}"

    # Filter methods if a specific one was requested
    if args.method != "all":
        cfg.methods = [m for m in cfg.methods if m[0] == args.method]
        if not cfg.methods:
            print(f"Error: Method '{args.method}' not found in EvalConfig.methods list.")
            exit(1)
    elif args.n_shards > 1 and args.gpu is not None:
        # Shard methods across GPUs: GPU i gets methods[i::n_shards]
        cfg.methods = cfg.methods[args.gpu::args.n_shards]

    print("=" * 70)
    print(f"Device: {cfg.device} | Methods: {len(cfg.methods)} | "
          f"Shard {args.gpu}/{args.n_shards}" if args.gpu is not None else
          f"Comprehensive Evaluation: All Methods × All Tasks")
    for m in cfg.methods:
        print(f"  • {m[0]}")
    print("=" * 70)

    t_start = time.time()
    evaluate_all(cfg)
    elapsed = time.time() - t_start
    print(f"\nTotal evaluation time: {elapsed/60:.1f} minutes")
