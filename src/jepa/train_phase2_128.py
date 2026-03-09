# src/jepa/train_phase2_128.py
"""
Phase 2 with LARGER model: embed_dim=128, depth=4.
Loads Phase 1 (128-dim) checkpoint, trains on HC data with replay buffer.
"""

import sys
import os
import json
import logging
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


class Phase2Config:
    # Data
    main_data_dir = "preprocessed_data_fir/roi_timeseries_1600"
    replay_data_dir = "preprocessed_data_fir/roi_timeseries_traveling"
    phase1_checkpoint = "checkpoints/phase1_128/model_A_phase1.pt"

    hc_only = True

    # Model — MUST match Phase 1 128-dim
    embed_dim = 128
    encoder_depth = 4
    predictor_depth = 2
    num_heads = 4
    mlp_ratio = 4.0
    dropout = 0.15
    ema_decay = 0.996

    # Training
    epochs = 60
    batch_size = 64
    replay_fraction = 0.15
    learning_rate = 5e-5
    weight_decay = 0.05
    warmup_epochs = 5

    # Masking
    temporal_num_blocks = 2
    temporal_block_size = (10, 15)

    # Output
    save_dir = "checkpoints/phase2_128"
    log_every = 10
    probe_every = 5
    save_every = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"


class HCOnlyDataset(FMRIDataset):
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


def make_site_balanced_sampler(dataset):
    sites = []
    for i in range(len(dataset)):
        ts_path = dataset.ts_files[i]
        meta_path = ts_path.replace('_roi_ts.npy', '_meta.json')
        with open(meta_path) as f:
            meta = json.load(f)
        sites.append(meta['site'])
    site_counts = Counter(sites)
    weights = [1.0 / site_counts[s] for s in sites]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
    print(f"Site-balanced sampler ({len(site_counts)} sites):")
    for site, count in sorted(site_counts.items()):
        print(f"  {site}: {count} HC, weight={1.0/count:.4f}")
    return sampler


class ReplayBuffer:
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self._loader = self._make_loader()
        self._iterator = iter(self._loader)

    def _make_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                          drop_last=False, collate_fn=collate_fn, num_workers=0)

    def sample(self):
        try:
            x, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            x, labels = next(self._iterator)
        return x.to(self.device), labels


def collate_fn(batch):
    tensors, labels_list = zip(*batch)
    tensors = torch.stack(tensors, dim=0)
    return tensors, list(labels_list)


@torch.no_grad()
def extract_embeddings(model, dataset, device):
    model.eval()
    embeddings, sites = [], []
    for i in range(len(dataset)):
        x, labels = dataset[i]
        x = x.unsqueeze(0).to(device)
        all_indices = torch.arange(x.size(1)).to(device)
        embedded = model.token_embed(x, all_indices)
        repr = model.context_encoder(embedded)
        pooled = repr.mean(dim=1)
        embeddings.append(pooled.cpu().numpy())
        sites.append(labels['site'])
    return np.concatenate(embeddings, axis=0), sites


def run_site_probe(model, dataset, device):
    embeddings, sites = extract_embeddings(model, dataset, device)
    le = LabelEncoder()
    y = le.fit_transform(sites)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(embeddings, y)
    return clf.score(embeddings, y), len(le.classes_)


def get_lr(epoch, warmup_epochs, total_epochs, base_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def train_phase2(cfg: Phase2Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("JEPA128-Phase2")
    device = cfg.device
    logger.info(f"Phase 2 (128-dim): Temporal model on {device}")

    main_dataset = HCOnlyDataset(cfg.main_data_dir, tokenization_mode='temporal') if cfg.hc_only else FMRIDataset(cfg.main_data_dir, tokenization_mode='temporal')
    replay_dataset = FMRIDataset(cfg.replay_data_dir, tokenization_mode='temporal')
    probe_dataset = replay_dataset

    sampler = make_site_balanced_sampler(main_dataset)
    main_batch_size = int(cfg.batch_size * (1 - cfg.replay_fraction))
    replay_batch_size = cfg.batch_size - main_batch_size

    main_loader = DataLoader(
        main_dataset, batch_size=main_batch_size, sampler=sampler,
        drop_last=True, collate_fn=collate_fn, num_workers=4,
        pin_memory=(device == "cuda"),
    )
    logger.info(f"Batch: {main_batch_size} main + {replay_batch_size} replay = {cfg.batch_size}")

    replay_buffer = ReplayBuffer(replay_dataset, replay_batch_size, device)

    model = JEPA(
        input_dim=400, max_seq_len=93,
        embed_dim=cfg.embed_dim, encoder_depth=cfg.encoder_depth,
        predictor_depth=cfg.predictor_depth, num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout, ema_decay=cfg.ema_decay,
    ).to(device)

    checkpoint = torch.load(cfg.phase1_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✓ Loaded Phase 1 (128) weights from {cfg.phase1_checkpoint}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'epoch': [], 'train_loss': [], 'main_loss': [], 'replay_loss': [],
        'site_probe_accuracy': [], 'learning_rate': [],
    }

    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses, epoch_main, epoch_replay = [], [], []

        lr = get_lr(epoch, cfg.warmup_epochs, cfg.epochs, cfg.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        for batch_idx, (x_main, labels_main) in enumerate(main_loader):
            x_main = x_main.to(device)
            x_replay, labels_replay = replay_buffer.sample()
            x_combined = torch.cat([x_main, x_replay], dim=0)

            ctx_idx, tgt_idx = generate_temporal_mask(seq_len=93, num_blocks=cfg.temporal_num_blocks, block_size_range=cfg.temporal_block_size)
            ctx_idx, tgt_idx = ctx_idx.to(device), tgt_idx.to(device)

            predicted, target_repr = model(x_combined, ctx_idx, tgt_idx)
            loss = F.mse_loss(predicted, target_repr)

            n_main = x_main.size(0)
            with torch.no_grad():
                main_loss = F.mse_loss(predicted[:n_main], target_repr[:n_main])
                replay_loss = F.mse_loss(predicted[n_main:], target_repr[n_main:])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target_encoder()

            epoch_losses.append(loss.item())
            epoch_main.append(main_loss.item())
            epoch_replay.append(replay_loss.item())

            if (batch_idx + 1) % cfg.log_every == 0:
                logger.info(f"Epoch {epoch+1}/{cfg.epochs} | Batch {batch_idx+1}/{len(main_loader)} | Loss: {loss.item():.4f} (main: {main_loss.item():.4f}, replay: {replay_loss.item():.4f})")

        avg_loss = np.mean(epoch_losses)
        avg_main = np.mean(epoch_main)
        avg_replay = np.mean(epoch_replay)

        site_acc = None
        if (epoch + 1) % cfg.probe_every == 0 or epoch == 0:
            site_acc, n_sites = run_site_probe(model, probe_dataset, device)
            logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} (main: {avg_main:.4f}, replay: {avg_replay:.4f}) | LR: {lr:.6f} | Site Probe: {site_acc:.1%}")
        else:
            logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = save_dir / f"model_A_phase2_ep{epoch+1}.pt"
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch + 1, 'config': vars(cfg)}, ckpt_path)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['main_loss'].append(avg_main)
        history['replay_loss'].append(avg_replay)
        history['site_probe_accuracy'].append(site_acc)
        history['learning_rate'].append(lr)

    final_path = save_dir / "model_A_phase2_final.pt"
    torch.save({'model_state_dict': model.state_dict(), 'config': vars(cfg), 'history': history}, final_path)
    logger.info(f"Final model saved: {final_path}")

    with open(save_dir / "history_phase2_128.json", 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    cfg = Phase2Config()
    print("=" * 60)
    print(f"Phase 2 (128-dim): Temporal JEPA on SRPBS 1600 (HC only)")
    print(f"  embed_dim={cfg.embed_dim}, depth={cfg.encoder_depth}")
    print("=" * 60)
    train_phase2(cfg)
