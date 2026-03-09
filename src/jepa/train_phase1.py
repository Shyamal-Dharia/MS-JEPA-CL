# src/jepa/train_phase1.py
"""
Phase 1: Train JEPA on Travelling Subject data.
Goal: Learn site-invariant brain representations.
"""

import sys
import os
import json
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ── Make sure our package is importable ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.dataset import FMRIDataset
from jepa.masking import generate_temporal_mask, generate_spatial_mask
from jepa.model import JEPA

# ─────────────────────────── Config ───────────────────────────

class Config:
    """All hyperparameters in one place."""
    
    # Data
    data_dir = "preprocessed_data_fir/roi_timeseries_traveling"
    # Model (MUST match Phase 1 — we're loading those weights)
    embed_dim = 400
    encoder_depth = 3
    predictor_depth = 2
    num_heads = 4
    mlp_ratio = 4.0
    dropout = 0.1
    ema_decay = 0.996
    
    # Training
    epochs = 100
    batch_size = 16          # 141 scans → ~9 batches per epoch
    learning_rate = 1.5e-4
    weight_decay = 0.05      # AdamW regularization
    warmup_epochs = 10       # Linear LR warmup for stable early training
    
    # Masking
    temporal_num_blocks = 2
    temporal_block_size = (10, 15)
    spatial_num_networks = 2
    
    # Output
    save_dir = "checkpoints/phase1"
    log_every = 5            # Print loss every N batches
    probe_every = 100          # Run site probe every N epochs

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────── Site Classification Probe ────────────────────

@torch.no_grad()
def extract_embeddings(model, dataset, mode, device):
    """
    Pass all samples through the Context Encoder and average-pool
    to get one embedding vector per scan.
    
    Parameters
    ----------
    model : JEPA
    dataset : FMRIDataset
    mode : str — 'temporal' or 'spatial'
    device : str
    
    Returns
    -------
    embeddings : np.ndarray of shape (num_samples, embed_dim)
    sites : list of site labels
    """
    model.eval()
    embeddings = []
    sites = []
    
    for i in range(len(dataset)):
        x, labels = dataset[i]
        x = x.unsqueeze(0).to(device)      # (1, seq_len, input_dim)
        
        # Use ALL tokens as context (no masking) to get full representation
        seq_len = x.size(1)
        all_indices = torch.arange(seq_len)
        
        # Embed and encode
        embedded = model.token_embed(x, all_indices)
        repr = model.context_encoder(embedded)      # (1, seq_len, embed_dim)
        
        # Average pool across all tokens → (1, embed_dim)
        pooled = repr.flatten(start_dim=1)
        
        embeddings.append(pooled.cpu().numpy())
        sites.append(labels['site'])
    
    return np.concatenate(embeddings, axis=0), sites


def run_site_probe(model, dataset, mode, device):
    """
    Train a Logistic Regression on frozen encoder embeddings 
    to classify scanner site. Reports accuracy.
    
    Lower accuracy = more site-invariant = better!
    """
    embeddings, sites = extract_embeddings(model, dataset, mode, device)
    le = LabelEncoder()
    y = le.fit_transform(sites)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import KFold, cross_val_score
    
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, embeddings, y, cv=cv, scoring='accuracy')
    return scores.mean(), len(le.classes_)


# ─────────────────── Learning Rate Scheduler ──────────────────

def get_lr(epoch, warmup_epochs, total_epochs, base_lr):
    """
    Linear warmup for warmup_epochs, then cosine decay to 0.
    
    This prevents the model from making huge, destructive updates
    in early epochs (warmup) and gradually reduces the learning rate
    as training converges (cosine decay).
    """
    if epoch < warmup_epochs:
        # Linear warmup: 0 → base_lr
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay: base_lr → 0
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


# ──────────────────── Custom Collate Function ─────────────────

def collate_fn(batch):
    """
    DataLoader collects individual samples into a batch.
    
    Default collate can't handle our dict labels, so we write a custom one.
    Stacks tensors and collects label dicts into a list.
    """
    tensors, labels_list = zip(*batch)
    tensors = torch.stack(tensors, dim=0)   # (batch_size, seq_len, input_dim)
    return tensors, list(labels_list)


# ──────────────────── Main Training Function ──────────────────

def train_one_model(mode: str, cfg: Config):
    """
    Train one JEPA model (temporal or spatial) on Phase 1 data.
    
    Parameters
    ----------
    mode : 'temporal' or 'spatial'
    cfg : Config object with all hyperparameters
    """
    
    # ── Setup ──
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(f"JEPA-{mode}")
    
    device = cfg.device
    logger.info(f"Training {mode} model on {device}")
    
    # ── Dataset & DataLoader ──
    dataset = FMRIDataset(cfg.data_dir, tokenization_mode=mode)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,              # Shuffle every epoch — important!
        drop_last=False,           # Keep partial last batch (small dataset)
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
    )
    
    logger.info(f"Dataset: {len(dataset)} scans, {len(loader)} batches/epoch")
    
    # ── Model ──
    if mode == 'temporal':
        input_dim, max_seq_len = 400, 93
    else:
        input_dim, max_seq_len = 93, 400
    
    model = JEPA(
        input_dim=input_dim,
        max_seq_len=max_seq_len,
        embed_dim=cfg.embed_dim,
        encoder_depth=cfg.encoder_depth,
        predictor_depth=cfg.predictor_depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        ema_decay=cfg.ema_decay,
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # ── Optimizer ──
    # AdamW: Adam with decoupled weight decay — standard for transformers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    
    # ── Create save directory ──
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Training history ──
    history = {
        'epoch': [],
        'train_loss': [],
        'site_probe_accuracy': [],
        'learning_rate': [],
    }
    
    # ── Training Loop ──
    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses = []
        
        # Update learning rate
        lr = get_lr(epoch, cfg.warmup_epochs, cfg.epochs, cfg.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        for batch_idx, (x, labels) in enumerate(loader):
            x = x.to(device)   # (batch, seq_len, input_dim)
            
            # ── Generate a NEW random mask for this batch ──
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
            
            # Move indices to device
            ctx_idx = ctx_idx.to(device)
            tgt_idx = tgt_idx.to(device)
            
            # ── Forward pass ──
            predicted, target_repr = model(x, ctx_idx, tgt_idx)
            
            # ── Loss: MSE in latent space ──
            # This is the JEPA objective: make predicted repr match target repr
            loss = F.mse_loss(predicted, target_repr)
            
            # ── Backward pass ──
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # ── EMA update target encoder ──
            model.update_target_encoder()
            
            epoch_losses.append(loss.item())
            
            if (batch_idx + 1) % cfg.log_every == 0:
                logger.info(
                    f"Epoch {epoch+1}/{cfg.epochs} | "
                    f"Batch {batch_idx+1}/{len(loader)} | "
                    f"Loss: {loss.item():.4f}"
                )
        
        # ── Epoch summary ──
        avg_loss = np.mean(epoch_losses)
        
        # ── Site classification probe ──
        site_acc = None
        if (epoch + 1) % cfg.probe_every == 0 or epoch == 0:
            site_acc, n_sites = run_site_probe(model, dataset, mode, device)
            logger.info(
                f"Epoch {epoch+1} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"LR: {lr:.6f} | "
                f"Site Probe: {site_acc:.1%} (chance = {1/n_sites:.1%})"
            )
        else:
            logger.info(
                f"Epoch {epoch+1} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"LR: {lr:.6f}"
            )
        
        # ── Record history ──
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['site_probe_accuracy'].append(site_acc)
        history['learning_rate'].append(lr)
    
    # ── Save final model ──
    model_name = "model_A_phase1.pt" if mode == "temporal" else "model_B_phase1.pt"
    save_path = save_dir / model_name
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(cfg),
        'history': history,
        'mode': mode,
    }, save_path)
    # Final save for evaluate_all compatibility
    final_path = save_dir / "model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(cfg),
        'mode': mode,
    }, final_path)
    logger.info(f"Saved optimal extraction checkpoint: {final_path}")
    
    return model, history


def train_phase1(cfg: Config):
    print("=" * 60)
    print("Phase 1: Training Model A (Temporal)")
    print("=" * 60)
    model_A, hist_A = train_one_model('temporal', cfg)
    
    # print("\n" + "=" * 60)
    # print("Phase 1: Training Model B (Spatial)")
    # print("=" * 60)
    # model_B, hist_B = train_one_model('spatial', cfg)
    
    print("\nPhase 1 complete! Models saved to:", cfg.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: JEPA Predictive Pre-training")
    parser.add_argument("--no_curriculum", action="store_true",
                        help="If set, train directly on all 1626 subjects instead of just travelling subjects.")
    args = parser.parse_args()

    cfg = Config()
    
    if args.no_curriculum:
        cfg.data_dir = "preprocessed_data_fir/roi_timeseries_1600"
        cfg.save_dir = "checkpoints/phase1_nocurr"
        cfg.batch_size = 64  # Increase batch size for 1600 subjects
        cfg.log_every = 10
        print("\n--- WARNING: RUNNING Phase 1 WITHOUT CURRICULUM (ALL SUBJECTS) ---\n")

    train_phase1(cfg)