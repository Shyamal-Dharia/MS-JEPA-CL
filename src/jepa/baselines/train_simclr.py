# src/jepa/baselines/train_simclr.py
"""
SimCLR Baseline for Multi-Site fMRI.

This script trains a standard SimCLR contrastive model on the fMRI data,
using the EXACT SAME TransformerEncoder as the JEPA model. 
This provides a rigorous ablation: does the predictive JEPA architecture
perform better than standard contrastive learning?

Requirements:
- Shared data augmentation (from dataset.py)
- Projection head appended to TransformerEncoder
- NT-Xent Loss on 2 augmented views
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from jepa.dataset import FMRIDataset
from jepa.model import TokenEmbedding, TransformerEncoder
from jepa.train_phase2_contrastive import contrastive_loss_ntxent

# ─────────────────────────── Config ───────────────────────────

class SimCLRConfig:
    data_dir = "preprocessed_data_fir/roi_timeseries_1600"
    
    # Model (must match JEPA Phase 1 exactly)
    embed_dim = 400
    encoder_depth = 3
    num_heads = 4
    mlp_ratio = 4.0
    dropout = 0.1
    
    # Training
    epochs = 100
    batch_size = 128
    learning_rate = 1e-4
    weight_decay = 0.05
    
    # SimCLR specifics
    temperature = 0.1
    proj_dim = 128
    
    # Output
    save_dir = "checkpoints/baselines_simclr"
    log_every = 10
    save_every = 20
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────── Model ───────────────────────────

class SimCLRModel(nn.Module):
    def __init__(self, input_dim=400, max_seq_len=93, embed_dim=400, encoder_depth=3,
                 num_heads=4, mlp_ratio=4.0, dropout=0.1, proj_dim=128):
        super().__init__()
        # Same encoder as JEPA
        self.token_embed = TokenEmbedding(input_dim, embed_dim, max_seq_len)
        self.context_encoder = TransformerEncoder(
            embed_dim, num_heads, encoder_depth, mlp_ratio, dropout
        )
        
        # SimCLR projection head: Linear -> GELU -> Linear
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim)
        )
        
    def forward(self, x, indices=None):
        """Returns the L2-normalized projection z for contrastive loss"""
        emb = self.token_embed(x, indices)
        rep = self.context_encoder(emb)
        pooled = rep.mean(dim=1)  # Global average pooling
        z = self.projector(pooled)
        return z

# ─────────────────────────── Training ───────────────────────────

def train_simclr(cfg):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=save_dir / "train_simclr.log",
        filemode="w", level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    logger.info("============== Baseline: SimCLR ==============")
    logger.info(f"Device: {cfg.device}")
    
    # Load dataset with 2 augmented views
    dataset = FMRIDataset(cfg.data_dir, tokenization_mode="temporal", augment=True, num_views=2)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    logger.info(f"Dataset loaded: {len(dataset)} subjects, batch_size={cfg.batch_size}")
    
    # Init Model
    model = SimCLRModel(
        embed_dim=cfg.embed_dim, encoder_depth=cfg.encoder_depth,
        num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout, proj_dim=cfg.proj_dim
    ).to(cfg.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(loader))
    
    # Save config
    config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith("__")}
    with open(save_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
        
    # All tokens are visible in SimCLR
    indices = torch.arange(93, device=cfg.device)
    
    logger.info("Starting SimCLR training...")
    t0 = time.time()
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (views, _) in enumerate(loader):
            # views is a list of [view1, view2]
            view1, view2 = views[0].to(cfg.device), views[1].to(cfg.device)
            B = view1.size(0)
            
            optimizer.zero_grad()
            
            # Extract projections
            z1 = model(view1, indices)
            z2 = model(view2, indices)
            
            # Combine into single batch (2B, proj_dim)
            z = torch.cat([z1, z2], dim=0)
            
            # Pseudo subject IDs: match view1 and view2
            pseudo_ids = list(range(B)) * 2
            
            loss, n_valid = contrastive_loss_ntxent(z, pseudo_ids, temperature=cfg.temperature)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % cfg.log_every == 0:
                logger.info(f"Epoch {epoch}/{cfg.epochs} | Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        avg_loss = total_loss / len(loader)
        logger.info(f"==> Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} | Time: {(time.time() - t0)/60:.1f}m")
        
        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            ckpt_path = save_dir / f"simclr_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config_dict,
                "loss": avg_loss,
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")
            
    # Final save
    final_path = save_dir / "simclr_final.pt"
    # To be directly compatible with evaluate_all checkpoint extraction:
    # We save a format mimicking JEPA context_encoder
    eval_state_dict = {}
    for k, v in model.state_dict().items():
        if k.startswith('token_embed.') or k.startswith('context_encoder.'):
            eval_state_dict[k] = v
    
    torch.save({
        "epoch": cfg.epochs,
        "model_state_dict": eval_state_dict,
        "config": config_dict,
    }, final_path)
    logger.info(f"Saved optimal extraction checkpoint: {final_path}")
    logger.info("Done.")

if __name__ == "__main__":
    train_simclr(SimCLRConfig())
