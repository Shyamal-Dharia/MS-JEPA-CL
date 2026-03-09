# src/jepa/train_phase2_advv2.py
"""
Phase 2 + Adversarial v2: Multi-step discriminator + stronger head.

v1 collapsed: the site classifier fell to 6% chance while the independent
site probe stayed at 34%. The classifier gave up, reversed gradients
became ~zero, and the encoder never had to remove site info.

v2 fixes:
  1. SEPARATE optimizers for JEPA and discriminator
  2. K=5 discriminator steps per 1 encoder step (like GAN training)
  3. Stronger discriminator (128-dim hidden, BatchNorm, deeper)
  4. Higher LR for discriminator (10x encoder) to stay competitive

The discriminator trains on DETACHED embeddings → it can't collapse
because its loss is a normal classification problem (no reversed grads).
The encoder then trains with GRL → it must actively fool a STRONG
classifier that's kept up-to-date.
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


# ─────────────────────────── Config ───────────────────────────

class Phase2AdvV2Config:
    """Phase 2 Adversarial v2: multi-step discriminator."""

    # Data
    main_data_dir = "preprocessed_data_fir/roi_timeseries_1600"
    replay_data_dir = "preprocessed_data_fir/roi_timeseries_traveling"
    phase1_checkpoint = "checkpoints/phase1/model_A_phase1.pt"

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
    epochs = 80              # more epochs — need time after alpha ramp
    batch_size = 64
    replay_fraction = 0.15
    learning_rate = 5e-5     # encoder LR (same as before)
    weight_decay = 0.05
    warmup_epochs = 5

    # Adversarial v2 (CHANGED)
    adv_weight = 1.0         # λ: weight of adversarial loss on encoder
    disc_hidden_dim = 128    # was 32 — 4x bigger
    disc_lr = 5e-4           # 10x higher than encoder — must stay competitive
    disc_steps = 5           # K discriminator updates per encoder update
    alpha_max = 1.0          # max GRL strength
    alpha_warmup = 10        # epochs before GRL kicks in (let disc train first)

    # Masking
    temporal_num_blocks = 2
    temporal_block_size = (10, 15)

    # Output
    save_dir = "checkpoints/phase2_adversarial"
    log_every = 10
    probe_every = 100
    save_every = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────── Gradient Reversal Layer ──────────────────────────

class GradientReversal(torch.autograd.Function):
    """Same as v1: flip gradient sign, scaled by alpha."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class SiteClassifierV2(nn.Module):
    """
    Stronger site classifier (v2).

    v1 had:  Linear(64, 32) → ReLU → Dropout → Linear(32, 16)
    v2 has:  Linear(64, 128) → BN → ReLU → Dropout →
             Linear(128, 64) → BN → ReLU → Dropout →
             Linear(64, num_sites)

    Key additions:
    - BatchNorm: stabilizes discriminator training, prevents mode collapse
    - Deeper: 3 layers vs 2 — more capacity to detect subtle site patterns
    - Wider: 128 hidden dim → harder for encoder to fool
    """
    def __init__(self, embed_dim, num_sites, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_sites),
        )

    def forward(self, x, alpha=None):
        """
        If alpha is provided → apply GRL (encoder training step).
        If alpha is None → no GRL (discriminator training step).
        """
        if alpha is not None:
            x = GradientReversal.apply(x, alpha)
        return self.net(x)


# ─────────── Alpha Schedule ──────────────────────────────────

def get_alpha(epoch, total_epochs, alpha_max=1.0, warmup=10):
    """
    Delayed sigmoid ramp-up.

    Epoch 0 → warmup: α = 0  (discriminator trains freely, no reversal)
    warmup → end:     sigmoid ramp from 0 → alpha_max

    The warmup lets the discriminator build competence BEFORE the encoder
    starts trying to fool it. Without this, both start from scratch and
    the discriminator can't keep up.
    """
    if epoch < warmup:
        return 0.0
    p = (epoch - warmup) / (total_epochs - warmup)
    return float(alpha_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


# ──────────── HC-Only Dataset ─────────────────────────────────

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


# ──────────── Site-Balanced Sampler ───────────────────────────

def make_site_balanced_sampler(dataset):
    sites = []
    for i in range(len(dataset)):
        meta_path = dataset.ts_files[i].replace('_roi_ts.npy', '_meta.json')
        with open(meta_path) as f:
            sites.append(json.load(f)['site'])

    site_counts = Counter(sites)
    weights = [1.0 / site_counts[s] for s in sites]
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    print(f"Site-balanced sampler ({len(site_counts)} sites):")
    for site, count in sorted(site_counts.items()):
        print(f"  {site}: {count} HC, weight={1.0/count:.4f}")
    return sampler


# ─────────── Replay Buffer ───────────────────────────────────

class ReplayBuffer:
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self._loader = self._make_loader()
        self._iterator = iter(self._loader)

    def _make_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          shuffle=True, drop_last=False,
                          collate_fn=collate_fn, num_workers=0)

    def sample(self):
        try:
            x, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._loader)
            x, labels = next(self._iterator)
        return x.to(self.device), labels


def collate_fn(batch):
    tensors, labels_list = zip(*batch)
    return torch.stack(tensors, dim=0), list(labels_list)


# ─────────── Build Site Label Encoder ─────────────────────────

def build_site_encoder(main_data_dir, replay_data_dir):
    import glob
    all_sites = set()
    for d in [main_data_dir, replay_data_dir]:
        for m in glob.glob(os.path.join(d, '*_meta.json')):
            all_sites.add(json.load(open(m))['site'])

    le = LabelEncoder()
    le.fit(sorted(all_sites))
    print(f"Site encoder: {len(le.classes_)} sites → {list(le.classes_)}")
    return le


# ─────────── Site Classification Probe (gold standard) ────────

@torch.no_grad()
def extract_embeddings(model, dataset, device):
    model.eval()
    embeddings, sites = [], []
    for i in range(len(dataset)):
        x, labels = dataset[i]
        x = x.unsqueeze(0).to(device)
        # Pass all replay through encoder to get representations
        all_indices = torch.arange(93).to(device)
        embedded = model.token_embed(x, all_indices)
        repr = model.context_encoder(embedded)
        pooled = repr.mean(dim=1)
        embeddings.append(pooled.cpu().numpy())
        sites.append(labels['site'])
    return np.concatenate(embeddings, axis=0), sites


def run_site_probe(model, dataset, device):
    """
    Independent Logistic Regression probe on travelling subjects.
    This is separate from the adversarial head — it's our unbiased metric.
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
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


# ─────────── Pooled Embedding Helper ─────────────────────────

def get_pooled_embedding(model, x, device):
    """
    Pass x through encoder (full sequence, no masking) → mean pool.
    Returns (batch, embed_dim) tensor attached to model's graph.
    """
    seq_len = x.size(1)
    all_indices = torch.arange(seq_len, device=device)
    embedded = model.token_embed(x, all_indices)
    repr_full = model.context_encoder(embedded)   # (batch, 93, 64)
    return repr_full.mean(dim=1)                            # (batch, embed_dim)


# ──────────────────── Main Training Function ──────────────────

def train_phase2_adversarial_v2(cfg: Phase2AdvV2Config):

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("JEPA-Phase2-ADVv2")
    device = cfg.device
    logger.info(f"Phase 2 + Adversarial v2 on {device}")
    logger.info(f"Key v2 changes: disc_steps={cfg.disc_steps}, "
                f"disc_hidden={cfg.disc_hidden_dim}, disc_lr={cfg.disc_lr}, "
                f"alpha_warmup={cfg.alpha_warmup}")

    # ── 1. Datasets ──
    if cfg.hc_only:
        main_dataset = HCOnlyDataset(cfg.main_data_dir, tokenization_mode='temporal')
    else:
        main_dataset = FMRIDataset(cfg.main_data_dir, tokenization_mode='temporal')

    replay_dataset = FMRIDataset(cfg.replay_data_dir, tokenization_mode='temporal')
    probe_dataset = replay_dataset

    logger.info(f"Main: {len(main_dataset)} | Replay: {len(replay_dataset)}")

    # ── 2. Site encoder ──
    site_encoder = build_site_encoder(cfg.main_data_dir, cfg.replay_data_dir)
    num_sites = len(site_encoder.classes_)

    # ── 3. Sampler & loader ──
    sampler = make_site_balanced_sampler(main_dataset)
    main_batch_size = int(cfg.batch_size * (1 - cfg.replay_fraction))
    replay_batch_size = cfg.batch_size - main_batch_size

    main_loader = DataLoader(
        main_dataset, batch_size=main_batch_size, sampler=sampler,
        drop_last=True, collate_fn=collate_fn, num_workers=4,
        pin_memory=(device == "cuda"),
    )

    logger.info(f"Batch: {main_batch_size} main + {replay_batch_size} replay")

    # ── 4. Replay buffer ──
    replay_buffer = ReplayBuffer(replay_dataset, replay_batch_size, device)

    # ── 5. JEPA model + Phase 1 weights ──
    model = JEPA(
        input_dim=400, max_seq_len=93,
        embed_dim=cfg.embed_dim, encoder_depth=cfg.encoder_depth,
        predictor_depth=cfg.predictor_depth, num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout, ema_decay=cfg.ema_decay,
    ).to(device)

    checkpoint = torch.load(cfg.phase1_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded Phase 1 from {cfg.phase1_checkpoint}")

    # ── 6. Stronger site classifier (v2) ──
    flattened_dim = cfg.embed_dim
    site_classifier = SiteClassifierV2(
        embed_dim=flattened_dim,
        num_sites=num_sites,
        hidden_dim=cfg.disc_hidden_dim,
    ).to(device)

    jepa_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    disc_params = sum(p.numel() for p in site_classifier.parameters() if p.requires_grad)
    logger.info(f"JEPA params: {jepa_params:,} | Disc params: {disc_params:,}")

    # ── 7. SEPARATE optimizers (key v2 change) ──
    optimizer_jepa = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    optimizer_disc = torch.optim.AdamW(
        site_classifier.parameters(),
        lr=cfg.disc_lr,
        weight_decay=1e-4,      # lighter weight decay for disc
    )

    # ── 8. Save directory ──
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 9. History ──
    history = {
        'epoch': [], 'train_loss': [], 'jepa_loss': [], 'adv_loss': [],
        'main_loss': [], 'replay_loss': [],
        'site_probe_accuracy': [], 'disc_accuracy': [],
        'learning_rate': [], 'alpha': [], 'disc_loss': [],
    }

    # ── 10. Training loop ──
    for epoch in range(cfg.epochs):
        model.train()
        site_classifier.train()

        epoch_total, epoch_jepa, epoch_adv = [], [], []
        epoch_main, epoch_replay, epoch_disc = [], [], []
        epoch_disc_correct, epoch_disc_total = 0, 0

        # LR for encoder (cosine)
        lr = get_lr(epoch, cfg.warmup_epochs, cfg.epochs, cfg.learning_rate)
        for pg in optimizer_jepa.param_groups:
            pg['lr'] = lr

        # GRL strength: 0 during warmup, then ramps up
        alpha = get_alpha(epoch, cfg.epochs, cfg.alpha_max, cfg.alpha_warmup)

        for batch_idx, (x_main, labels_main) in enumerate(main_loader):
            x_main = x_main.to(device)

            # ── Draw replay batch ──
            x_replay, labels_replay = replay_buffer.sample()
            x_combined = torch.cat([x_main, x_replay], dim=0)

            # ── Get site labels for combined batch ──
            all_labels = labels_main + labels_replay
            site_names = [lbl['site'] for lbl in all_labels]
            site_targets = torch.tensor(
                site_encoder.transform(site_names), dtype=torch.long
            ).to(device)

            # ════════════════════════════════════════════════
            # STEP A: Train discriminator K times (DETACHED)
            # ════════════════════════════════════════════════
            # The discriminator sees embeddings but can't affect
            # the encoder — this keeps it strong and competitive.

            disc_loss_accum = 0.0
            for _ in range(cfg.disc_steps):
                with torch.no_grad():
                    pooled_detached = get_pooled_embedding(model, x_combined, device)

                # No GRL here → normal classification training
                site_logits_d = site_classifier(pooled_detached, alpha=None)
                disc_loss = F.cross_entropy(site_logits_d, site_targets)

                optimizer_disc.zero_grad()
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(site_classifier.parameters(), max_norm=1.0)
                optimizer_disc.step()

                disc_loss_accum += disc_loss.item()

            avg_disc_loss = disc_loss_accum / cfg.disc_steps

            # Track discriminator accuracy (after K updates)
            with torch.no_grad():
                pooled_check = get_pooled_embedding(model, x_combined, device)
                logits_check = site_classifier(pooled_check, alpha=None)
                disc_preds = logits_check.argmax(dim=1)
                epoch_disc_correct += (disc_preds == site_targets).sum().item()
                epoch_disc_total += len(site_targets)

            # ════════════════════════════════════════════════
            # STEP B: Train encoder (JEPA + GRL adversarial)
            # ════════════════════════════════════════════════
            # Now the discriminator is strong → the GRL provides
            # meaningful reversed gradients to the encoder.

            # B1: JEPA masked prediction
            ctx_idx, tgt_idx = generate_temporal_mask(
                seq_len=93,
                num_blocks=cfg.temporal_num_blocks,
                block_size_range=cfg.temporal_block_size,
            )
            ctx_idx = ctx_idx.to(device)
            tgt_idx = tgt_idx.to(device)

            predicted, target_repr = model(x_combined, ctx_idx, tgt_idx)
            jepa_loss = F.mse_loss(predicted, target_repr)

            # Track main vs replay JEPA loss
            n_main = x_main.size(0)
            with torch.no_grad():
                main_loss_val = F.mse_loss(predicted[:n_main], target_repr[:n_main])
                replay_loss_val = F.mse_loss(predicted[n_main:], target_repr[n_main:])

            # B2: Adversarial — now WITH GRL (gradients reversed)
            pooled = get_pooled_embedding(model, x_combined, device)
            site_logits = site_classifier(pooled, alpha=alpha)  # GRL applied!
            adv_loss = F.cross_entropy(site_logits, site_targets)

            # Total encoder loss
            total_loss = jepa_loss + cfg.adv_weight * adv_loss

            # Encoder update only
            optimizer_jepa.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_jepa.step()
            model.update_target_encoder()

            # ── Logging ──
            epoch_total.append(total_loss.item())
            epoch_jepa.append(jepa_loss.item())
            epoch_adv.append(adv_loss.item())
            epoch_main.append(main_loss_val.item())
            epoch_replay.append(replay_loss_val.item())
            epoch_disc.append(avg_disc_loss)

            if (batch_idx + 1) % cfg.log_every == 0:
                logger.info(
                    f"E{epoch+1} B{batch_idx+1}/{len(main_loader)} | "
                    f"JEPA: {jepa_loss.item():.4f} ADV: {adv_loss.item():.4f} "
                    f"Disc: {avg_disc_loss:.4f} | α:{alpha:.3f}"
                )

        # ── Epoch summary ──
        avg_total = np.mean(epoch_total)
        avg_jepa = np.mean(epoch_jepa)
        avg_adv = np.mean(epoch_adv)
        avg_main = np.mean(epoch_main)
        avg_replay = np.mean(epoch_replay)
        avg_disc_epoch = np.mean(epoch_disc)
        disc_acc = epoch_disc_correct / max(epoch_disc_total, 1)

        site_acc = None
        if (epoch + 1) % cfg.probe_every == 0 or epoch == 0:
            site_acc, n_sites_probe = run_site_probe(model, probe_dataset, device)
            logger.info(
                f"Epoch {epoch+1}/{cfg.epochs} | "
                f"Total: {avg_total:.4f} (JEPA: {avg_jepa:.4f}, ADV: {avg_adv:.4f}) | "
                f"Disc loss: {avg_disc_epoch:.4f}, Disc acc: {disc_acc:.1%} | "
                f"α: {alpha:.3f} | "
                f"SITE PROBE: {site_acc:.1%} (chance={1/n_sites_probe:.1%})"
            )
        else:
            logger.info(
                f"Epoch {epoch+1}/{cfg.epochs} | "
                f"Total: {avg_total:.4f} (JEPA: {avg_jepa:.4f}, ADV: {avg_adv:.4f}) | "
                f"Disc loss: {avg_disc_epoch:.4f}, Disc acc: {disc_acc:.1%} | "
                f"α: {alpha:.3f}"
            )

        if (epoch + 1) % cfg.save_every == 0:
            ckpt = save_dir / f"model_A_phase2advv2_ep{epoch+1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'site_classifier_state_dict': site_classifier.state_dict(),
                'optimizer_jepa_state_dict': optimizer_jepa.state_dict(),
                'optimizer_disc_state_dict': optimizer_disc.state_dict(),
                'epoch': epoch + 1,
                'config': vars(cfg),
            }, ckpt)
            logger.info(f"Checkpoint: {ckpt}")

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_total)
        history['jepa_loss'].append(avg_jepa)
        history['adv_loss'].append(avg_adv)
        history['disc_loss'].append(avg_disc_epoch)
        history['main_loss'].append(avg_main)
        history['replay_loss'].append(avg_replay)
        history['site_probe_accuracy'].append(site_acc)
        history['disc_accuracy'].append(disc_acc)
        history['learning_rate'].append(lr)
        history['alpha'].append(alpha)

    # ── Save final ──
    final = save_dir / "model_A_phase2advv2_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'site_classifier_state_dict': site_classifier.state_dict(),
        'config': vars(cfg),
        'history': history,
    }, final)
    logger.info(f"Final model: {final}")

    # Standardized name for evaluator
    eval_final_path = save_dir / "model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(cfg),
    }, eval_final_path)
    logger.info(f"Saved evaluation-ready checkpoint: {eval_final_path}")

    hist_path = save_dir / "history_phase2advv2.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_curriculum", action="store_true")
    args = parser.parse_args()

    cfg = Phase2AdvV2Config()

    if args.no_curriculum:
        cfg.save_dir = "checkpoints/phase2_adversarial_nocurr"
        cfg.phase1_checkpoint = "checkpoints/phase1_nocurr/model_final.pt"
        print("\n--- WARNING: RUNNING Phase 2 ADV WITHOUT CURRICULUM ---\n")

    print("=" * 60)
    print("Phase 2 + Adversarial v2: Multi-step Disc + Stronger Head")
    print("=" * 60)
    print(f"  disc_steps:     {cfg.disc_steps} (K updates per encoder step)")
    print(f"  disc_hidden:    {cfg.disc_hidden_dim}")
    print(f"  disc_lr:        {cfg.disc_lr}")
    print(f"  alpha_warmup:   {cfg.alpha_warmup} epochs")
    print(f"  epochs:         {cfg.epochs}")
    print("=" * 60)
    train_phase2_adversarial_v2(cfg)