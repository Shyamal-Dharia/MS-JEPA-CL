# src/jepa/train_phase1_128.py
"""
Phase 1 with LARGER model: embed_dim=128, depth=4.
Everything else identical to train_phase1.py.
"""

import sys
import os
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.dataset import FMRIDataset
from jepa.masking import generate_temporal_mask, generate_spatial_mask
from jepa.model import JEPA

# ─────────────────────────── Config ───────────────────────────

class Config:
    # Data
    data_dir = "preprocessed_data_fir/roi_timeseries_traveling"
    
    # Model — LARGER than 64-dim version
    embed_dim = 128          # was 64
    encoder_depth = 4        # was 3
    predictor_depth = 2
    num_heads = 4
    mlp_ratio = 4.0
    dropout = 0.15           # slightly more dropout for bigger model
    ema_decay = 0.996
    
    # Training
    epochs = 120             # more epochs for larger model
    batch_size = 16
    learning_rate = 1e-4     # slightly lower LR for stability
    weight_decay = 0.05
    warmup_epochs = 15
    
    # Masking
    temporal_num_blocks = 2
    temporal_block_size = (10, 15)
    spatial_num_networks = 2
    
    # Output
    save_dir = "checkpoints/phase1_128"
    log_every = 5
    probe_every = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────── Site Probe ───────────────

@torch.no_grad()
def extract_embeddings(model, dataset, mode, device):
    model.eval()
    embeddings, sites = [], []
    for i in range(len(dataset)):
        x, labels = dataset[i]
        x = x.unsqueeze(0).to(device)
        seq_len = x.size(1)
        all_indices = torch.arange(seq_len).to(device)
        embedded = model.token_embed(x, all_indices)
        repr = model.context_encoder(embedded)
        pooled = repr.mean(dim=1)
        embeddings.append(pooled.cpu().numpy())
        sites.append(labels['site'])
    return np.concatenate(embeddings, axis=0), sites


def run_site_probe(model, dataset, mode, device):
    embeddings, sites = extract_embeddings(model, dataset, mode, device)
    le = LabelEncoder()
    y = le.fit_transform(sites)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(embeddings, y)
    accuracy = clf.score(embeddings, y)
    return accuracy, len(le.classes_)


def get_lr(epoch, warmup_epochs, total_epochs, base_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def collate_fn(batch):
    tensors, labels_list = zip(*batch)
    tensors = torch.stack(tensors, dim=0)
    return tensors, list(labels_list)


def train_one_model(mode: str, cfg: Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(f"JEPA128-{mode}")
    device = cfg.device
    logger.info(f"Training {mode} model (128-dim) on {device}")
    
    dataset = FMRIDataset(cfg.data_dir, tokenization_mode=mode)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        drop_last=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=True if device == "cuda" else False,
    )
    logger.info(f"Dataset: {len(dataset)} scans, {len(loader)} batches/epoch")
    
    if mode == 'temporal':
        input_dim, max_seq_len = 400, 93
    else:
        input_dim, max_seq_len = 93, 400
    
    model = JEPA(
        input_dim=input_dim, max_seq_len=max_seq_len,
        embed_dim=cfg.embed_dim, encoder_depth=cfg.encoder_depth,
        predictor_depth=cfg.predictor_depth, num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout, ema_decay=cfg.ema_decay,
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    history = {'epoch': [], 'train_loss': [], 'site_probe_accuracy': [], 'learning_rate': []}
    
    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses = []
        
        lr = get_lr(epoch, cfg.warmup_epochs, cfg.epochs, cfg.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        for batch_idx, (x, labels) in enumerate(loader):
            x = x.to(device)
            
            if mode == 'temporal':
                ctx_idx, tgt_idx = generate_temporal_mask(
                    seq_len=max_seq_len,
                    num_blocks=cfg.temporal_num_blocks,
                    block_size_range=cfg.temporal_block_size,
                )
            else:
                ctx_idx, tgt_idx = generate_spatial_mask(
                    seq_len=max_seq_len,
                    num_networks_to_mask=cfg.spatial_num_networks,
                )
            
            ctx_idx = ctx_idx.to(device)
            tgt_idx = tgt_idx.to(device)
            
            predicted, target_repr = model(x, ctx_idx, tgt_idx)
            loss = F.mse_loss(predicted, target_repr)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target_encoder()
            
            epoch_losses.append(loss.item())
            
            if (batch_idx + 1) % cfg.log_every == 0:
                logger.info(f"Epoch {epoch+1}/{cfg.epochs} | Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")
        
        avg_loss = np.mean(epoch_losses)
        
        site_acc = None
        if (epoch + 1) % cfg.probe_every == 0 or epoch == 0:
            site_acc, n_sites = run_site_probe(model, dataset, mode, device)
            logger.info(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | LR: {lr:.6f} | Site Probe: {site_acc:.1%} (chance = {1/n_sites:.1%})")
        else:
            logger.info(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | LR: {lr:.6f}")
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['site_probe_accuracy'].append(site_acc)
        history['learning_rate'].append(lr)
    
    model_name = "model_A_phase1.pt" if mode == "temporal" else "model_B_phase1.pt"
    save_path = save_dir / model_name
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(cfg),
        'history': history,
        'mode': mode,
    }, save_path)
    logger.info(f"Saved model to {save_path}")
    
    history_path = save_dir / f"history_{mode}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


if __name__ == "__main__":
    cfg = Config()
    print("=" * 60)
    print(f"Phase 1 (128-dim): Training Model A (Temporal)")
    print(f"  embed_dim={cfg.embed_dim}, depth={cfg.encoder_depth}")
    print(f"  ~{cfg.embed_dim * cfg.encoder_depth * 12}K params estimate")
    print("=" * 60)
    model_A, hist_A = train_one_model('temporal', cfg)
