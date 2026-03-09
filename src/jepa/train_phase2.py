# src/jepa/train_phase2.py
"""
Phase 2: Train temporal JEPA on SRPBS 1600 (HC only) with replay buffer.

Curriculum progression:
  Phase 1 taught "same brain, different scanner → same representation"
  Phase 2 scales to 950 healthy controls across 13 sites while preserving
  that invariance via a replay buffer of Phase 1 travelling-subject data.

Key components:
  1. Load Phase 1 pretrained weights
  2. Site-balanced WeightedRandomSampler for HC data
  3. 15% replay buffer from travelling-subject data
  4. Site probe on travelling subjects (gold standard)
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

# ── Make sure our package is importable ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.dataset import FMRIDataset
from jepa.masking import generate_temporal_mask
from jepa.model import JEPA


# ─────────────────────────── Config ───────────────────────────

class Phase2Config:
    """All hyperparameters for Phase 2."""

    # Data
    main_data_dir = "preprocessed_data_fir/roi_timeseries_1600"
    replay_data_dir = "preprocessed_data_fir/roi_timeseries_traveling"
    phase1_checkpoint = "checkpoints/phase1/model_A_phase1.pt"

    # Phase 2 trains on Healthy Controls only
    hc_only = True

    # Model (MUST match Phase 1 — we're loading those weights)
    embed_dim = 400
    encoder_depth = 3
    predictor_depth = 2
    num_heads = 4
    mlp_ratio = 4.0
    dropout = 0.1
    ema_decay = 0.996

    # Training
    epochs = 60
    batch_size = 64
    replay_fraction = 0.15    # 15% of each batch from travelling subjects
    learning_rate = 5e-5      # Lower than Phase 1 — fine-tuning, not from scratch
    weight_decay = 0.05
    warmup_epochs = 5

    # Masking (same recipe as Phase 1)
    temporal_num_blocks = 2
    temporal_block_size = (10, 15)

    # Output
    save_dir = "checkpoints/phase2"
    log_every = 10
    probe_every = 100
    save_every = 20           # Checkpoint every N epochs

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────── HC-Only Dataset (filters by diagnosis) ──────────

class HCOnlyDataset(FMRIDataset):
    """
    Wraps FMRIDataset but keeps only Healthy Control subjects (diag=0).
    This is Phase 2's main training data — 950 subjects from 13 sites.
    """

    def __init__(self, data_dir, tokenization_mode='temporal'):
        super().__init__(data_dir, tokenization_mode)

        # Filter: keep only HC subjects
        hc_files = []
        for ts_path in self.ts_files:
            meta_path = ts_path.replace('_roi_ts.npy', '_meta.json')
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get('diag', -1) == 0:
                hc_files.append(ts_path)

        n_before = len(self.ts_files)
        self.ts_files = hc_files
        print(f"HCOnlyDataset: kept {len(hc_files)}/{n_before} (HC only)")


# ──────────── Site-Balanced Weighted Sampler ───────────────────

def make_site_balanced_sampler(dataset):
    """
    Creates a WeightedRandomSampler where each sample's weight
    is inversely proportional to its site's frequency.

    Effect: sites with fewer subjects get sampled MORE often,
    so the model sees roughly equal numbers from each site per epoch.

    Without this:  UTO (170) dominates, NKN (2) is invisible.
    With this:     each site contributes ~equally per epoch.
    """
    sites = []
    for i in range(len(dataset)):
        ts_path = dataset.ts_files[i]
        meta_path = ts_path.replace('_roi_ts.npy', '_meta.json')
        with open(meta_path) as f:
            meta = json.load(f)
        sites.append(meta['site'])

    site_counts = Counter(sites)

    # Weight = 1 / count  →  rare sites get higher weight
    weights = [1.0 / site_counts[s] for s in sites]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),  # one "epoch" ≈ all data once
        replacement=True,          # required for weighted sampling
    )

    print(f"Site-balanced sampler ({len(site_counts)} sites):")
    for site, count in sorted(site_counts.items()):
        print(f"  {site}: {count} HC, weight={1.0/count:.4f}")

    return sampler


# ─────────── Replay Buffer (cycles through travelling data) ──

class ReplayBuffer:
    """
    Endlessly cycles through the travelling-subject DataLoader.

    Each call to sample() returns one batch from the travelling data.
    When the iterator is exhausted, it restarts automatically.

    This ensures the model keeps "rehearsing" the Phase 1 objective
    (site invariance across same subjects) while learning from new data.
    """

    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self._loader = self._make_loader()
        self._iterator = iter(self._loader)

    def _make_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=0,  # tiny dataset, keep it simple
        )

    def sample(self):
        """Get one batch, recycling the iterator when exhausted."""
        try:
            x, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            x, labels = next(self._iterator)
        return x.to(self.device), labels


# ──────────────────── Collate Function ────────────────────────

def collate_fn(batch):
    """Stack tensors and collect label dicts into a list."""
    tensors, labels_list = zip(*batch)
    tensors = torch.stack(tensors, dim=0)
    return tensors, list(labels_list)


# ─────────── Site Classification Probe (gold standard) ────────

@torch.no_grad()
def extract_embeddings(model, dataset, device):
    """
    Pass all samples through the Context Encoder, average-pool
    to get one embedding vector per scan.
    """
    model.eval()
    embeddings, sites = [], []

    for i in range(len(dataset)):
        x, labels = dataset[i]
        x = x.unsqueeze(0).to(device)
        seq_len = x.size(1)
        all_indices = torch.arange(seq_len).to(device)

        embedded = model.token_embed(x, all_indices)
        repr = model.context_encoder(embedded)
        pooled = repr.flatten(start_dim=1)

        embeddings.append(pooled.cpu().numpy())
        sites.append(labels['site'])

    return np.concatenate(embeddings, axis=0), sites


def run_site_probe(model, dataset, device):
    """
    Logistic Regression on frozen encoder embeddings → site label.
    Evaluated on travelling subjects (same brain, different sites).
    Lower accuracy = more site-invariant = better.
    """
    embeddings, sites = extract_embeddings(model, dataset, device)
    le = LabelEncoder()
    y = le.fit_transform(sites)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import KFold, cross_val_score
    
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, embeddings, y, cv=cv, scoring='accuracy')
    return scores.mean(), len(le.classes_)


# ─────────────── Learning Rate Schedule ───────────────────────

def get_lr(epoch, warmup_epochs, total_epochs, base_lr):
    """Linear warmup → cosine decay."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


# ──────────────────── Main Training Function ──────────────────

def train_phase2(cfg: Phase2Config):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("JEPA-Phase2")
    device = cfg.device
    logger.info(f"Phase 2: Temporal model on {device}")

    # ── 1. Datasets ──
    if cfg.hc_only:
        main_dataset = HCOnlyDataset(cfg.main_data_dir, tokenization_mode='temporal')
    else:
        main_dataset = FMRIDataset(cfg.main_data_dir, tokenization_mode='temporal')

    replay_dataset = FMRIDataset(cfg.replay_data_dir, tokenization_mode='temporal')

    # Probe on travelling subjects — the gold standard for site invariance
    probe_dataset = replay_dataset

    logger.info(f"Main dataset: {len(main_dataset)} HC subjects")
    logger.info(f"Replay dataset: {len(replay_dataset)} travelling scans")

    # ── 2. Site-balanced sampler for main data ──
    sampler = make_site_balanced_sampler(main_dataset)

    main_batch_size = int(cfg.batch_size * (1 - cfg.replay_fraction))
    replay_batch_size = cfg.batch_size - main_batch_size

    main_loader = DataLoader(
        main_dataset,
        batch_size=main_batch_size,
        sampler=sampler,            # balanced by site
        drop_last=True,             # consistent batch size
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    logger.info(
        f"Batch: {main_batch_size} main + {replay_batch_size} replay "
        f"= {cfg.batch_size} | {len(main_loader)} batches/epoch"
    )

    # ── 3. Replay buffer ──
    replay_buffer = ReplayBuffer(replay_dataset, replay_batch_size, device)

    # ── 4. Model + load Phase 1 weights ──
    model = JEPA(
        input_dim=400,
        max_seq_len=93,
        embed_dim=cfg.embed_dim,
        encoder_depth=cfg.encoder_depth,
        predictor_depth=cfg.predictor_depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        ema_decay=cfg.ema_decay,
    ).to(device)

    checkpoint = torch.load(cfg.phase1_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✓ Loaded Phase 1 weights from {cfg.phase1_checkpoint}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    # ── 5. Optimizer (lower LR = fine-tuning) ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # ── 6. Save directory ──
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 7. History tracking ──
    history = {
        'epoch': [],
        'train_loss': [],
        'main_loss': [],
        'replay_loss': [],
        'site_probe_accuracy': [],
        'learning_rate': [],
    }

    # ── 8. Training loop ──
    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses = []
        epoch_main_losses = []
        epoch_replay_losses = []

        # Update learning rate
        lr = get_lr(epoch, cfg.warmup_epochs, cfg.epochs, cfg.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        for batch_idx, (x_main, labels_main) in enumerate(main_loader):
            x_main = x_main.to(device)

            # ── Draw replay batch ──
            x_replay, labels_replay = replay_buffer.sample()

            # ── Concatenate: main + replay ──
            x_combined = torch.cat([x_main, x_replay], dim=0)

            # ── Generate mask (same mask for all samples in batch) ──
            ctx_idx, tgt_idx = generate_temporal_mask(
                seq_len=93,
                num_blocks=cfg.temporal_num_blocks,
                block_size_range=cfg.temporal_block_size,
            )
            ctx_idx = ctx_idx.to(device)
            tgt_idx = tgt_idx.to(device)

            # ── Forward pass ──
            predicted, target_repr = model(x_combined, ctx_idx, tgt_idx)

            # ── Combined loss ──
            loss = F.mse_loss(predicted, target_repr)

            # Track main vs replay loss separately (diagnostic only)
            n_main = x_main.size(0)
            with torch.no_grad():
                main_loss = F.mse_loss(predicted[:n_main], target_repr[:n_main])
                replay_loss = F.mse_loss(predicted[n_main:], target_repr[n_main:])

            # ── Backward ──
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ── EMA update target encoder ──
            model.update_target_encoder()

            epoch_losses.append(loss.item())
            epoch_main_losses.append(main_loss.item())
            epoch_replay_losses.append(replay_loss.item())

            if (batch_idx + 1) % cfg.log_every == 0:
                logger.info(
                    f"Epoch {epoch+1}/{cfg.epochs} | "
                    f"Batch {batch_idx+1}/{len(main_loader)} | "
                    f"Loss: {loss.item():.4f} "
                    f"(main: {main_loss.item():.4f}, replay: {replay_loss.item():.4f})"
                )

        # ── Epoch summary ──
        avg_loss = np.mean(epoch_losses)
        avg_main = np.mean(epoch_main_losses)
        avg_replay = np.mean(epoch_replay_losses)

        # ── Site probe (evaluated on travelling subjects) ──
        site_acc = None
        if (epoch + 1) % cfg.probe_every == 0 or epoch == 0:
            site_acc, n_sites = run_site_probe(model, probe_dataset, device)
            logger.info(
                f"Epoch {epoch+1} | "
                f"Loss: {avg_loss:.4f} (main: {avg_main:.4f}, replay: {avg_replay:.4f}) | "
                f"LR: {lr:.6f} | "
                f"Site Probe: {site_acc:.1%} (chance={1/n_sites:.1%})"
            )
        else:
            logger.info(
                f"Epoch {epoch+1} | "
                f"Loss: {avg_loss:.4f} (main: {avg_main:.4f}, replay: {avg_replay:.4f}) | "
                f"LR: {lr:.6f}"
            )

        # ── Periodic checkpoint ──
        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = save_dir / f"model_A_phase2_ep{epoch+1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'config': vars(cfg),
            }, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

        # ── Record history ──
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['main_loss'].append(avg_main)
        history['replay_loss'].append(avg_replay)
        history['site_probe_accuracy'].append(site_acc)
        history['learning_rate'].append(lr)

    # ── Save final model + history ──
    final_path = save_dir / "model_A_phase2_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(cfg),
        'history': history,
    }, final_path)
    logger.info(f"Final model saved: {final_path}")

    # Standardized name for evaluator
    eval_final_path = save_dir / "model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(cfg),
    }, eval_final_path)
    logger.info(f"Saved evaluation-ready checkpoint: {eval_final_path}")

    hist_path = save_dir / "history_phase2.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


# ──────────────────── Entry Point ─────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_curriculum", action="store_true")
    args = parser.parse_args()

    cfg = Phase2Config()
    if args.no_curriculum:
        cfg.save_dir = "checkpoints/phase2_vanilla_nocurr"
        cfg.phase1_checkpoint = "checkpoints/phase1_nocurr/model_final.pt"
        print("\n--- WARNING: RUNNING Phase 2 Vanilla WITHOUT CURRICULUM (ALL SUBJECTS) ---\n")

    print("=" * 60)
    print("Phase 2: Temporal JEPA on SRPBS 1600 (HC only)")
    print("=" * 60)
    train_phase2(cfg)