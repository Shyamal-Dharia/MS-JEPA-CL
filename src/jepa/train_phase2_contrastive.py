# src/jepa/train_phase2_contrastive.py
"""
Phase 2 + Contrastive: JEPA training with NT-Xent loss on travelling subjects.

The JEPA loss alone didn't push site probe below 34%.
Adding contrastive learning explicitly teaches:
  "same subject at different sites → SAME embedding"
  "different subjects → DIFFERENT embeddings"

This only applies to the replay buffer (travelling subjects),
where we know subject identity across scanner sites.
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
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.dataset import FMRIDataset
from jepa.masking import generate_temporal_mask
from jepa.model import JEPA


# ─────────────────────────── Config ───────────────────────────

class Phase2ContrastiveConfig:
    """Phase 2 config with contrastive loss additions."""

    # Data
    main_data_dir = "preprocessed_data_fir/roi_timeseries_1600"
    replay_data_dir = "preprocessed_data_fir/roi_timeseries_traveling"
    phase1_checkpoint = "checkpoints/phase1/model_A_phase1.pt"

    # Phase 2 trains on Healthy Controls only
    hc_only = True

    # Model (must match Phase 1)
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
    learning_rate = 5e-5
    weight_decay = 0.05
    warmup_epochs = 5

    # Contrastive loss (applied ONLY to replay buffer)
    contrastive_weight = 0.5     # λ: how much contrastive loss contributes
    temperature = 0.1            # τ: sharpness of similarity distribution

    # Masking
    temporal_num_blocks = 2
    temporal_block_size = (10, 15)

    # Output
    save_dir = "checkpoints/phase2_contrastive"
    log_every = 10
    probe_every = 100
    save_every = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────── HC-Only Dataset (filters by diagnosis) ──────────

class HCOnlyDataset(FMRIDataset):
    """
    Wraps FMRIDataset but keeps only Healthy Control subjects (diag=0).
    This is Phase 2's main training data — 950 subjects from 13 sites.
    """

    def __init__(self, data_dir, tokenization_mode='temporal'):
        super().__init__(data_dir, tokenization_mode)

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
    """
    sites = []
    for i in range(len(dataset)):
        ts_path = dataset.ts_files[i]
        meta_path = ts_path.replace('_roi_ts.npy', '_meta.json')
        with open(meta_path) as f:
            meta = json.load(f)
        sites.append(meta['site'])

    site_counts = Counter(sites)
    weights = [1.0 / site_counts[s] for s in sites]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
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
            num_workers=0,
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
    tensors, labels_list = zip(*batch)
    tensors = torch.stack(tensors, dim=0)
    return tensors, list(labels_list)


# ──────────── NT-Xent Contrastive Loss ────────────────────────

def contrastive_loss_ntxent(embeddings, subject_ids, temperature=0.1):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    For each scan in the batch, finds another scan of the SAME subject
    (scanned at a different site) as the positive. All other subjects
    are negatives.

    Parameters
    ----------
    embeddings : Tensor (N, embed_dim)
        L2-normalized mean-pooled encoder outputs for replay batch.
    subject_ids : list of str, length N
        Subject identifier for each scan.
    temperature : float
        Controls sharpness. Lower = harder contrasts.

    Returns
    -------
    loss : scalar Tensor
        Mean NT-Xent loss over all valid anchors.
    n_valid : int
        Number of anchors that had at least one positive pair.
    """
    N = embeddings.size(0)
    if N < 2:
        return torch.tensor(0.0, device=embeddings.device), 0

    # L2 normalize embeddings (cosine similarity via dot product)
    embeddings = F.normalize(embeddings, dim=1)

    # Pairwise cosine similarity matrix: (N, N)
    sim_matrix = torch.mm(embeddings, embeddings.T) / temperature

    # Build positive mask: same subject, different index
    # positive_mask[i, j] = 1 if subject_ids[i] == subject_ids[j] and i != j
    positive_mask = torch.zeros(N, N, dtype=torch.bool, device=embeddings.device)
    for i in range(N):
        for j in range(N):
            if i != j and subject_ids[i] == subject_ids[j]:
                positive_mask[i, j] = True

    # Only compute loss for anchors that have at least one positive
    valid_anchors = positive_mask.any(dim=1)  # (N,)
    n_valid = valid_anchors.sum().item()

    if n_valid == 0:
        return torch.tensor(0.0, device=embeddings.device), 0

    # Mask out self-similarity
    self_mask = torch.eye(N, dtype=torch.bool, device=embeddings.device)

    # Denominator: sum over all j != i
    sim_matrix_no_self = sim_matrix.masked_fill(self_mask, float('-inf'))
    log_denom = torch.logsumexp(sim_matrix_no_self, dim=1)  # (N,)

    # Numerator: log-sum-exp over positive pairs
    sim_positives = sim_matrix.masked_fill(~positive_mask, float('-inf'))
    log_numer = torch.logsumexp(sim_positives, dim=1)  # (N,)

    # Loss per anchor: -log(sum_pos / sum_all) = -(log_numer - log_denom)
    losses = -(log_numer - log_denom)

    # Average over valid anchors only
    loss = losses[valid_anchors].mean()

    return loss, n_valid


# ─────────────── Embedding Extraction ─────────────────────────

def get_pooled_embedding(model, x, device):
    """
    Get mean-pooled encoder embedding for a batch.
    Uses ALL tokens (no masking) for a full representation.

    Parameters
    ----------
    model : JEPA
    x : Tensor (batch, seq_len, input_dim)
    device : str

    Returns
    -------
    Tensor (batch, embed_dim)
    """
    seq_len = x.size(1)
    all_indices = torch.arange(seq_len).to(device)
    embedded = model.token_embed(x, all_indices)
    repr = model.context_encoder(embedded)
    return repr.mean(dim=1)  # (batch, embed_dim)


# ─────────── Site Classification Probe ────────────────────────

@torch.no_grad()
def extract_embeddings(model, dataset, device):
    model.eval()
    embeddings, sites = [], []
    for i in range(len(dataset)):
        x, labels = dataset[i]
        x = x.unsqueeze(0).to(device)
        pooled = get_pooled_embedding(model, x, device)
        embeddings.append(pooled.cpu().numpy())
        sites.append(labels['site'])
    return np.concatenate(embeddings, axis=0), sites


def run_site_probe(model, dataset, device):
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
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


# ──────────────────── Main Training Function ──────────────────

def train_phase2_contrastive(cfg: Phase2ContrastiveConfig):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("JEPA-Phase2-CL")
    device = cfg.device
    logger.info(f"Phase 2 + Contrastive: Temporal model on {device}")

    # ── 1. Datasets ──
    if cfg.hc_only:
        main_dataset = HCOnlyDataset(cfg.main_data_dir, tokenization_mode='temporal')
    else:
        main_dataset = FMRIDataset(cfg.main_data_dir, tokenization_mode='temporal')

    replay_dataset = FMRIDataset(cfg.replay_data_dir, tokenization_mode='temporal')
    probe_dataset = replay_dataset  # Site probe on travelling subjects

    logger.info(f"Main dataset: {len(main_dataset)} HC subjects")
    logger.info(f"Replay dataset: {len(replay_dataset)} travelling scans")

    # ── 2. Site-balanced sampler for main data ──
    sampler = make_site_balanced_sampler(main_dataset)

    main_batch_size = int(cfg.batch_size * (1 - cfg.replay_fraction))
    replay_batch_size = cfg.batch_size - main_batch_size

    main_loader = DataLoader(
        main_dataset,
        batch_size=main_batch_size,
        sampler=sampler,
        drop_last=True,
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
    logger.info(f"Loaded Phase 1 weights from {cfg.phase1_checkpoint}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    # ── 5. Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # ── 6. Save directory ──
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 7. History ──
    history = {
        'epoch': [],
        'train_loss': [],
        'jepa_loss': [],
        'contrastive_loss': [],
        'main_loss': [],
        'replay_loss': [],
        'site_probe_accuracy': [],
        'learning_rate': [],
        'n_valid_pairs': [],
    }

    # ── 8. Training loop ──
    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses, epoch_jepa, epoch_cl = [], [], []
        epoch_main_losses, epoch_replay_losses = [], []
        epoch_pairs = []

        lr = get_lr(epoch, cfg.warmup_epochs, cfg.epochs, cfg.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        for batch_idx, (x_main, labels_main) in enumerate(main_loader):
            x_main = x_main.to(device)

            # ── Draw replay batch (travelling subjects) ──
            x_replay, labels_replay = replay_buffer.sample()

            # ── Concatenate: main + replay ──
            x_combined = torch.cat([x_main, x_replay], dim=0)

            # ── Generate mask ──
            ctx_idx, tgt_idx = generate_temporal_mask(
                seq_len=93,
                num_blocks=cfg.temporal_num_blocks,
                block_size_range=cfg.temporal_block_size,
            )
            ctx_idx = ctx_idx.to(device)
            tgt_idx = tgt_idx.to(device)

            # ── 1. JEPA predictive loss (on combined batch) ──
            predicted, target_repr = model(x_combined, ctx_idx, tgt_idx)
            jepa_loss = F.mse_loss(predicted, target_repr)

            # Track main vs replay JEPA loss separately (diagnostic only)
            n_main = x_main.size(0)
            with torch.no_grad():
                main_loss = F.mse_loss(predicted[:n_main], target_repr[:n_main])
                replay_loss = F.mse_loss(predicted[n_main:], target_repr[n_main:])

            # ── 2. Contrastive NT-Xent loss (ONLY on replay buffer) ──
            # Replay subjects have real cross-site positive pairs
            replay_embeddings = get_pooled_embedding(model, x_replay, device)
            replay_subject_ids = [l['subject'] for l in labels_replay]

            cl_loss, n_valid = contrastive_loss_ntxent(
                replay_embeddings, replay_subject_ids, cfg.temperature
            )

            # ── Total loss ──
            total_loss = jepa_loss + cfg.contrastive_weight * cl_loss

            # ── Backward ──
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target_encoder()

            epoch_losses.append(total_loss.item())
            epoch_jepa.append(jepa_loss.item())
            epoch_cl.append(cl_loss.item())
            epoch_main_losses.append(main_loss.item())
            epoch_replay_losses.append(replay_loss.item())
            epoch_pairs.append(n_valid)

            if (batch_idx + 1) % cfg.log_every == 0:
                logger.info(
                    f"Epoch {epoch+1}/{cfg.epochs} | "
                    f"Batch {batch_idx+1}/{len(main_loader)} | "
                    f"Total: {total_loss.item():.4f} "
                    f"(JEPA: {jepa_loss.item():.4f}, CL: {cl_loss.item():.4f}, "
                    f"pairs: {n_valid})"
                )

        # ── Epoch summary ──
        avg_loss = np.mean(epoch_losses)
        avg_jepa = np.mean(epoch_jepa)
        avg_cl = np.mean(epoch_cl)
        avg_main = np.mean(epoch_main_losses)
        avg_replay = np.mean(epoch_replay_losses)
        avg_pairs = np.mean(epoch_pairs)

        site_acc = None
        if (epoch + 1) % cfg.probe_every == 0 or epoch == 0:
            site_acc, n_sites = run_site_probe(model, probe_dataset, device)
            logger.info(
                f"Epoch {epoch+1} | "
                f"Total: {avg_loss:.4f} (JEPA: {avg_jepa:.4f}, CL: {avg_cl:.4f}) | "
                f"Main: {avg_main:.4f}, Replay: {avg_replay:.4f} | "
                f"Pairs: {avg_pairs:.0f} | "
                f"Site Probe: {site_acc:.1%} (chance={1/n_sites:.1%})"
            )
        else:
            logger.info(
                f"Epoch {epoch+1} | "
                f"Total: {avg_loss:.4f} (JEPA: {avg_jepa:.4f}, CL: {avg_cl:.4f}) | "
                f"Main: {avg_main:.4f}, Replay: {avg_replay:.4f} | "
                f"Pairs: {avg_pairs:.0f}"
            )

        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = save_dir / f"model_A_phase2cl_ep{epoch+1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'config': vars(cfg),
            }, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['jepa_loss'].append(avg_jepa)
        history['contrastive_loss'].append(avg_cl)
        history['main_loss'].append(avg_main)
        history['replay_loss'].append(avg_replay)
        history['site_probe_accuracy'].append(site_acc)
        history['learning_rate'].append(lr)
        history['n_valid_pairs'].append(avg_pairs)

    # ── Save final ──
    final_path = save_dir / "model_A_phase2cl_final.pt"
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

    hist_path = save_dir / "history_phase2cl.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_curriculum", action="store_true")
    args = parser.parse_args()

    cfg = Phase2ContrastiveConfig()
    
    if args.no_curriculum:
        cfg.save_dir = "checkpoints/phase2_contrastive_nocurr"
        cfg.phase1_checkpoint = "checkpoints/phase1_nocurr/model_final.pt"
        print("\n--- WARNING: RUNNING Phase 2 CL WITHOUT CURRICULUM (ALL SUBJECTS) ---\n")

    print("=" * 60)
    print("Phase 2 + Contrastive: Temporal JEPA on SRPBS 1600 (HC)")
    print(f"  Main data: {cfg.main_data_dir}")
    print(f"  Replay data: {cfg.replay_data_dir}")
    print(f"  Phase 1 checkpoint: {cfg.phase1_checkpoint}")
    print(f"  Contrastive weight: {cfg.contrastive_weight}")
    print(f"  Replay fraction: {cfg.replay_fraction}")
    print("=" * 60)

    train_phase2_contrastive(cfg)
