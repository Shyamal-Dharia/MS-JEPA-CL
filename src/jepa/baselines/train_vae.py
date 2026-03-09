# src/jepa/baselines/train_vae.py
"""
Transformer-VAE Baseline for Multi-Site fMRI.

This script trains a Variational Autoencoder (VAE) on the fMRI data.
Crucially, it uses the EXACT SAME TransformerEncoder architecture as
the JEPA model for fair comparison.

The encoder compresses the (93, 400) fMRI scan to a single 400-dim
latent vector z (via mu and logvar). The decoder (also a Transformer)
attempts to reconstruct the (93, 400) input.

This proves that predicting in latent space (JEPA) is superior to
reconstructing in input space via a latent bottleneck (VAE).
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

# ─────────────────────────── Config ───────────────────────────

class VAEConfig:
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
    
    # VAE specifics
    beta = 0.5  # KL-divergence weight
    
    # Output
    save_dir = "checkpoints/baselines_vae"
    log_every = 10
    save_every = 20
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────── Model ───────────────────────────

class TransformerVAE(nn.Module):
    def __init__(self, input_dim=400, max_seq_len=93, embed_dim=400, encoder_depth=3,
                 num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        # ENCODER: Same as JEPA's ContextEncoder
        self.encoder_embed = TokenEmbedding(input_dim, embed_dim, max_seq_len)
        self.encoder = TransformerEncoder(
            embed_dim, num_heads, encoder_depth, mlp_ratio, dropout
        )
        
        # LATENT PROJECTION
        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_logvar = nn.Linear(embed_dim, embed_dim)
        
        # DECODER
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
        self.decoder = TransformerEncoder(
            embed_dim, num_heads, encoder_depth, mlp_ratio, dropout
        )
        self.decoder_proj = nn.Linear(embed_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x, indices=None):
        B, seq_len, _ = x.shape
        
        # 1. Encode
        emb = self.encoder_embed(x, indices)
        enc_out = self.encoder(emb)
        pooled = enc_out.mean(dim=1)  # (B, embed_dim)
        
        # 2. Latent Step (mu, log_var)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        z = self.reparameterize(mu, logvar)  # (B, embed_dim)
        
        # 3. Decode
        # Expand z to sequence length and add positional embedding
        z_seq = z.unsqueeze(1).expand(B, seq_len, -1)
        if indices is not None:
            pos = self.decoder_pos_embed[:, indices, :]
        else:
            pos = self.decoder_pos_embed[:, :seq_len, :]
            
        dec_input = z_seq + pos
        dec_out = self.decoder(dec_input)
        recon_x = self.decoder_proj(dec_out)
        
        return recon_x, mu, logvar

# ─────────────────────────── Training ───────────────────────────

def train_vae(cfg):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=save_dir / "train_vae.log",
        filemode="w", level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    logger.info("============== Baseline: Transformer-VAE ==============")
    logger.info(f"Device: {cfg.device}")
    
    # Standard dataset (no augmentation needed for VAE typically, but we use pure raw scans)
    dataset = FMRIDataset(cfg.data_dir, tokenization_mode="temporal", augment=False)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    logger.info(f"Dataset loaded: {len(dataset)} subjects, batch_size={cfg.batch_size}")
    
    # Init Model
    model = TransformerVAE(
        embed_dim=cfg.embed_dim, encoder_depth=cfg.encoder_depth,
        num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout
    ).to(cfg.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(loader))
    
    # Save config
    config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith("__")}
    with open(save_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
        
    indices = torch.arange(93, device=cfg.device)
    
    logger.info("Starting VAE training...")
    t0 = time.time()
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(cfg.device)
            optimizer.zero_grad()
            
            recon_x, mu, logvar = model(x, indices)
            
            # Reconstruction Loss (MSE)
            recon_loss = F.mse_loss(recon_x, x, reduction='mean')
            
            # KL Divergence Loss
            # Form: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            # We mean across batch, sum across latent dimensions
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            
            loss = recon_loss + cfg.beta * kld_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_recon_loss += recon_loss.item()
            total_kl_loss += kld_loss.item()
            
            if (batch_idx + 1) % cfg.log_every == 0:
                logger.info(f"Epoch {epoch}/{cfg.epochs} | Batch {batch_idx+1}/{len(loader)} | "
                            f"Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, KL: {kld_loss.item():.4f}) | "
                            f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        avg_recon = total_recon_loss / len(loader)
        avg_kl = total_kl_loss / len(loader)
        logger.info(f"==> Epoch {epoch} complete | Avg Recon: {avg_recon:.4f} | Avg KL: {avg_kl:.4f} | Time: {(time.time() - t0)/60:.1f}m")
        
        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            ckpt_path = save_dir / f"vae_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config_dict,
                "loss": avg_recon + cfg.beta * avg_kl,
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")
            
    # Final save - repackage for evaluate_all pipeline
    final_path = save_dir / "vae_final.pt"
    # To be directly compatible with evaluate_all checkpoint extraction:
    # Rename VAE encoder weights to match JEPA ContextEncoder exactly.
    eval_state_dict = {}
    for k, v in model.state_dict().items():
        if k.startswith('encoder_embed.'):
            eval_state_dict[k.replace('encoder_embed.', 'token_embed.')] = v
        elif k.startswith('encoder.'):
            eval_state_dict[k.replace('encoder.', 'context_encoder.')] = v

    torch.save({
        "epoch": cfg.epochs,
        "model_state_dict": eval_state_dict,
        "config": config_dict,
    }, final_path)
    logger.info(f"Saved optimal extraction checkpoint: {final_path}")
    logger.info("Done.")

if __name__ == "__main__":
    train_vae(VAEConfig())
