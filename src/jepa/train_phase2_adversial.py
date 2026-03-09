# src/jepa/train_phase2_adversarial.py
"""
Phase 2 + Adversarial: JEPA training with Gradient Reversal Layer.

The JEPA loss alone leaves 34% site classification accuracy.
Adversarial training with a Gradient Reversal Layer (GRL) actively
pushes site information OUT of the encoder embeddings.

Architecture:
  Input → [Context Encoder] → embedding
                                │
                    ┌───────────┼───────────┐
                    │                       │
              [Predictor]          [GRL → Site Classifier]
              (JEPA loss)          (adversarial loss)

The GRL flips gradient sign:
  - Site classifier learns to PREDICT site (normal gradients)
  - Encoder learns to FOOL the classifier (reversed gradients)
  → Encoder removes site information while preserving brain dynamics.

Key advantage over contrastive: uses ALL 950+ HC subjects' site labels,
not just the 9 travelling subjects.
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

class Phase2AdvConfig:
    """Phase 2 config with adversarial training."""

    # Data
    main_data_dir = "preprocessed_data_fir/roi_timeseries_1600"
    replay_data_dir = "preprocessed_data_fir/roi_timeseries_traveling"
    phase1_checkpoint = "checkpoints/phase1/model_A_phase1.pt"

    hc_only = True

    # Model (must match Phase 1)
    embed_dim = 64
    encoder_depth = 3
    predictor_depth = 2
    num_heads = 4
    mlp_ratio = 4.0
    dropout = 0.1
    ema_decay = 0.996

    # Training
    epochs = 60
    batch_size = 64
    replay_fraction = 0.15
    learning_rate = 5e-5
    weight_decay = 0.05
    warmup_epochs = 5

    # Adversarial (NEW)
    adv_weight = 1.0             # λ: weight of adversarial loss
    adv_hidden_dim = 32          # site classifier hidden layer size
    alpha_max = 1.0              # maximum GRL strength

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
    """
    Forward:  identity — passes embeddings through unchanged.
    Backward: NEGATES the gradient, scaled by alpha.

    This is the key trick from "Domain-Adversarial Training of Neural Networks"
    (Ganin et al., 2016). The encoder receives -α * ∂L_site/∂θ_encoder,
    which pushes it to MAXIMIZE the site classifier's loss, i.e.,
    make site prediction as hard as possible.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class SiteClassifier(nn.Module):
    """
    Small MLP that predicts scanner site from encoder embeddings.

    Sits behind the GRL — its own weights get NORMAL gradients
    (it learns to classify site), but the ENCODER gets REVERSED
    gradients (it learns to fool this classifier).
    """
    def __init__(self, embed_dim, num_sites, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_sites),
        )

    def forward(self, x, alpha=1.0):
        """
        x: (batch, embed_dim) — pooled encoder embeddings
        alpha: GRL strength (0 = no reversal, 1 = full reversal)
        """
        reversed_x = GradientReversal.apply(x, alpha)
        return self.net(reversed_x)  # (batch, num_sites) logits


# ─────────── Alpha Schedule ──────────────────────────────────

def get_alpha(epoch, total_epochs, alpha_max=1.0):
    """
    Sigmoid ramp-up for GRL strength.

    Epoch 0:  α ≈ 0   → adversarial off, let JEPA learn freely
    Mid:      α ≈ 0.5 → gradual site removal
    Final:    α ≈ 1.0 → full adversarial pressure

    From Ganin et al.: α = 2/(1 + exp(-10p)) - 1, where p = epoch/total
    """
    p = epoch / total_epochs
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
    """
    Build a consistent site → integer mapping across BOTH datasets.
    Returns a sklearn LabelEncoder fitted on all unique sites.
    """
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
        seq_len = x.size(1)
        all_indices = torch.arange(seq_len).to(device)
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


# ──────────────────── Main Training Function ──────────────────

def train_phase2_adversarial(cfg: Phase2AdvConfig):

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("JEPA-Phase2-ADV")
    device = cfg.device
    logger.info(f"Phase 2 + Adversarial: Temporal model on {device}")

    # ── 1. Datasets ──
    if cfg.hc_only:
        main_dataset = HCOnlyDataset(cfg.main_data_dir, tokenization_mode='temporal')
    else:
        main_dataset = FMRIDataset(cfg.main_data_dir, tokenization_mode='temporal')

    replay_dataset = FMRIDataset(cfg.replay_data_dir, tokenization_mode='temporal')
    probe_dataset = replay_dataset

    logger.info(f"Main: {len(main_dataset)} | Replay: {len(replay_dataset)}")

    # ── 2. Site encoder (consistent mapping across both datasets) ──
    site_encoder = build_site_encoder(cfg.main_data_dir, cfg.replay_data_dir)
    num_sites = len(site_encoder.classes_)

    # ── 3. Sampler & loaders ──
    sampler = make_site_balanced_sampler(main_dataset)
    main_batch_size = int(cfg.batch_size * (1 - cfg.replay_fraction))
    replay_batch_size = cfg.batch_size - main_batch_size

    main_loader = DataLoader(
        main_dataset, batch_size=main_batch_size, sampler=sampler,
        drop_last=True, collate_fn=collate_fn, num_workers=4,
        pin_memory=(device == "cuda"),
    )

    logger.info(f"Batch: {main_batch_size} main + {replay_batch_size} replay = {cfg.batch_size}")

    # ── 4. Replay buffer ──
    replay_buffer = ReplayBuffer(replay_dataset, replay_batch_size, device)

    # ── 5. Model + load Phase 1 weights ──
    model = JEPA(
        input_dim=400, max_seq_len=93,
        embed_dim=cfg.embed_dim, encoder_depth=cfg.encoder_depth,
        predictor_depth=cfg.predictor_depth, num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout, ema_decay=cfg.ema_decay,
    ).to(device)

    checkpoint = torch.load(cfg.phase1_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded Phase 1 weights from {cfg.phase1_checkpoint}")

    # ── 6. Site classifier (NEW) ──
    flattened_dim = cfg.embed_dim * 93
    site_classifier = SiteClassifier(
        embed_dim=flattened_dim,
        num_sites=num_sites,
        hidden_dim=cfg.adv_hidden_dim,
    ).to(device)

    trainable_jepa = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_adv = sum(p.numel() for p in site_classifier.parameters() if p.requires_grad)
    logger.info(f"JEPA params: {trainable_jepa:,} | Site classifier params: {trainable_adv:,}")

    # ── 7. Optimizer (includes BOTH model and site classifier params) ──
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(site_classifier.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # ── 8. Save directory ──
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 9. History ──
    history = {
        'epoch': [], 'train_loss': [], 'jepa_loss': [], 'adv_loss': [],
        'main_loss': [], 'replay_loss': [],
        'site_probe_accuracy': [], 'adv_site_accuracy': [],
        'learning_rate': [], 'alpha': [],
    }

    # ── 10. Training loop ──
    for epoch in range(cfg.epochs):
        model.train()
        site_classifier.train()

        epoch_total, epoch_jepa, epoch_adv = [], [], []
        epoch_main, epoch_replay = [], []
        epoch_adv_correct, epoch_adv_total = 0, 0

        lr = get_lr(epoch, cfg.warmup_epochs, cfg.epochs, cfg.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # GRL strength ramps up over training
        alpha = get_alpha(epoch, cfg.epochs, cfg.alpha_max)

        for batch_idx, (x_main, labels_main) in enumerate(main_loader):
            x_main = x_main.to(device)

            # ── Draw replay batch ──
            x_replay, labels_replay = replay_buffer.sample()

            # ── Concatenate for JEPA ──
            x_combined = torch.cat([x_main, x_replay], dim=0)

            # ── JEPA forward (masked prediction) ──
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

            # ── Adversarial forward (full sequence, no masking) ──
            # Pass ALL tokens through encoder, then flatten → (batch, embed_dim * seq_len)
            seq_len = x_combined.size(1)
            all_indices = torch.arange(seq_len).to(device)
            embedded_full = model.token_embed(x_combined, all_indices)
            repr_full = model.context_encoder(embedded_full)   # (batch, 93, 64)
            pooled = repr_full.flatten(start_dim=1)            # (batch, 93 * 64)

            # Get site labels for entire batch
            all_labels = labels_main + labels_replay
            site_names = [lbl['site'] for lbl in all_labels]
            site_targets = torch.tensor(
                site_encoder.transform(site_names), dtype=torch.long
            ).to(device)

            # Site classifier with GRL
            site_logits = site_classifier(pooled, alpha=alpha)
            adv_loss = F.cross_entropy(site_logits, site_targets)

            # Track adversarial accuracy
            with torch.no_grad():
                adv_preds = site_logits.argmax(dim=1)
                epoch_adv_correct += (adv_preds == site_targets).sum().item()
                epoch_adv_total += len(site_targets)

            # ── Total loss ──
            total_loss = jepa_loss + cfg.adv_weight * adv_loss

            # ── Backward ──
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target_encoder()

            epoch_total.append(total_loss.item())
            epoch_jepa.append(jepa_loss.item())
            epoch_adv.append(adv_loss.item())
            epoch_main.append(main_loss_val.item())
            epoch_replay.append(replay_loss_val.item())

            if (batch_idx + 1) % cfg.log_every == 0:
                logger.info(
                    f"Epoch {epoch+1}/{cfg.epochs} | "
                    f"Batch {batch_idx+1}/{len(main_loader)} | "
                    f"Total: {total_loss.item():.4f} "
                    f"(JEPA: {jepa_loss.item():.4f}, ADV: {adv_loss.item():.4f}) | "
                    f"α: {alpha:.3f}"
                )

        # ── Epoch summary ──
        avg_total = np.mean(epoch_total)
        avg_jepa = np.mean(epoch_jepa)
        avg_adv = np.mean(epoch_adv)
        avg_main = np.mean(epoch_main)
        avg_replay = np.mean(epoch_replay)
        adv_acc = epoch_adv_correct / max(epoch_adv_total, 1)

        site_acc = None
        if (epoch + 1) % cfg.probe_every == 0 or epoch == 0:
            site_acc, n_sites_probe = run_site_probe(model, probe_dataset, device)
            logger.info(
                f"Epoch {epoch+1} | "
                f"Total: {avg_total:.4f} (JEPA: {avg_jepa:.4f}, ADV: {avg_adv:.4f}) | "
                f"α: {alpha:.3f} | "
                f"ADV acc: {adv_acc:.1%} | "
                f"Site Probe: {site_acc:.1%} (chance={1/n_sites_probe:.1%})"
            )
        else:
            logger.info(
                f"Epoch {epoch+1} | "
                f"Total: {avg_total:.4f} (JEPA: {avg_jepa:.4f}, ADV: {avg_adv:.4f}) | "
                f"α: {alpha:.3f} | ADV acc: {adv_acc:.1%}"
            )

        if (epoch + 1) % cfg.save_every == 0:
            ckpt = save_dir / f"model_A_phase2adv_ep{epoch+1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'site_classifier_state_dict': site_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'config': vars(cfg),
            }, ckpt)
            logger.info(f"Checkpoint: {ckpt}")

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_total)
        history['jepa_loss'].append(avg_jepa)
        history['adv_loss'].append(avg_adv)
        history['main_loss'].append(avg_main)
        history['replay_loss'].append(avg_replay)
        history['site_probe_accuracy'].append(site_acc)
        history['adv_site_accuracy'].append(adv_acc)
        history['learning_rate'].append(lr)
        history['alpha'].append(alpha)

    # ── Save final ──
    final = save_dir / "model_A_phase2adv_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'site_classifier_state_dict': site_classifier.state_dict(),
        'config': vars(cfg),
        'history': history,
    }, final)
    logger.info(f"Final model: {final}")

    hist_path = save_dir / "history_phase2adv.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    cfg = Phase2AdvConfig()
    print("=" * 60)
    print("Phase 2 + Adversarial: Temporal JEPA + GRL")
    print("=" * 60)
    train_phase2_adversarial(cfg)