"""Unified JEPA trainer for the jepa_fusion branch.

Supports:
- temporal (basic, conv_net, rotary_conv_net) and st_roi_patch model families
- pooled and curriculum training modes
- VICReg, global loss, and replay alignment regularisation
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..jepa_clean.data import WindowedFMRIDataset, make_subject_id
from ..jepa_clean.masking import sample_temporal_mask
from ..jepa_clean.model import JEPA
from ..jepa_clean.probe import run_site_probe
from ..jepa_clean.repro import make_torch_generator, make_worker_init_fn, seed_everything
from ..jepa_clean.train_global_curriculum import (
    DatasetBalancedBatchSampler,
    ReplayBuffer,
    collate_fn,
    get_lr,
    replay_pair_alignment_loss,
    vicreg_regularization,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class JEPAFusionConfig:
    main_data_dirs: list[str] = field(
        default_factory=lambda: [
            "preprocessed_data_fir/full/roi_timeseries_1600_full",
            "preprocessed_data_fir/full/roi_timeseries_abide_full",
            "preprocessed_data_fir/full/roi_timeseries_abide2_full",
        ]
    )
    travel_data_dir: str = "preprocessed_data_fir/full/roi_timeseries_traveling_full"

    # Mode control
    training_mode: str = "pooled"         # "pooled" or "curriculum"
    model_family: str = "temporal"        # "temporal" or "st_roi_patch"
    temporal_variant: str = "basic"       # "basic", "conv_net", "rotary_conv_net"
    mask_profile: str = "temporal_default"

    # Architecture
    input_dim: int = 400
    target_timepoints: int = 45
    embed_dim: int = 400
    encoder_depth: int = 8
    predictor_depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    ema_decay: float = 0.996
    pos_embed_mode: str = "sincos"
    use_conv_stem: bool = False
    conv_kernel_size: int = 3
    use_network_embeddings: bool = False

    # ST-ROI-patch specific
    st_time_patch: int = 5
    st_roi_group_size: int = 40
    st_reorder_mode: str = "hemi_network"

    # Windowing
    window_seconds: float = 90.0
    window_stride_seconds: float = 30.0

    # Sampling
    pooled_sampling: str = "random"
    phase1_sampling: str = "sliding"
    phase2_main_sampling: str = "random"
    phase2_replay_sampling: str = "sliding"

    # Training schedule
    pooled_epochs: int = 60
    phase1_epochs: int = 20
    phase2_epochs: int = 40
    batch_size: int = 32
    replay_fraction: float = 0.2
    pooled_lr: float = 0.0001
    phase1_lr: float = 0.0001
    phase2_lr: float = 0.00001
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    steps_per_epoch: int | None = None
    max_main_scans: int | None = None
    max_travel_scans: int | None = None

    # Losses
    vicreg_weight: float = 0.1
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 1.0
    vicreg_var_target: float = 1.0
    global_loss_weight: float = 0.0
    replay_alignment_weight: float = 0.0
    replay_pair_sampling: bool = False

    # Adversarial (optional)
    adversarial_site_weight: float = 0.0

    # Masking
    mask_num_time_blocks: int = 2
    mask_time_block_size_min: int = 6
    mask_time_block_size_max: int = 10
    st_mask_num_time_blocks: int = 2
    st_mask_time_block_size: int = 2
    st_mask_num_roi_blocks: int = 2
    st_mask_roi_block_size: int = 3

    # I/O
    save_dir: str = "checkpoints/jepa_fusion"
    seed: int = 42
    deterministic: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_logging(log_file: str | None = None):
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def build_model(cfg: JEPAFusionConfig) -> JEPA:
    """Build model from config, applying model-family-specific settings."""
    pos_embed_mode = cfg.pos_embed_mode
    use_conv_stem = cfg.use_conv_stem
    use_network_embeddings = cfg.use_network_embeddings

    if cfg.model_family == "temporal":
        if cfg.temporal_variant == "basic":
            pos_embed_mode = "sincos"
            use_conv_stem = False
            use_network_embeddings = False
        elif cfg.temporal_variant == "conv_net":
            pos_embed_mode = "sincos"
            use_conv_stem = True
            use_network_embeddings = True
        elif cfg.temporal_variant == "rotary_conv_net":
            pos_embed_mode = "rotary"
            use_conv_stem = True
            use_network_embeddings = True

    return JEPA(
        input_dim=cfg.input_dim,
        max_seq_len=cfg.target_timepoints,
        embed_dim=cfg.embed_dim,
        encoder_depth=cfg.encoder_depth,
        predictor_depth=cfg.predictor_depth,
        num_heads=cfg.num_heads,
        pos_embed_mode=pos_embed_mode,
        use_conv_stem=use_conv_stem,
        conv_kernel_size=cfg.conv_kernel_size,
        use_network_embeddings=use_network_embeddings,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        ema_decay=cfg.ema_decay,
    )


def complement_indices(seq_len: int, target_indices: torch.LongTensor, device: torch.device):
    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    keep_mask[target_indices.to(device)] = False
    return torch.arange(seq_len, device=device)[keep_mask]


def mask_batch(seq_len: int, cfg: JEPAFusionConfig, device: torch.device):
    mask = sample_temporal_mask(
        seq_len=seq_len,
        num_time_blocks=cfg.mask_num_time_blocks,
        time_block_size=(cfg.mask_time_block_size_min, cfg.mask_time_block_size_max),
    )
    target_indices = mask["target_indices"].to(device)
    context_indices = complement_indices(seq_len, target_indices, device)
    return context_indices, target_indices


def save_checkpoint(path: Path, model, cfg: JEPAFusionConfig, history, epoch: int):
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "model_state_dict": state_dict,
            "config": asdict(cfg),
            "history": history,
            "epoch": epoch,
        },
        path,
    )


def maybe_run_site_probe(model, dataset, device, seed, logger):
    try:
        return run_site_probe(model, dataset, device, seed=seed)
    except ValueError as exc:
        logger.warning("Skipping site probe: %s", exc)
        return None


def dataset_site_counts(dataset: WindowedFMRIDataset) -> dict:
    counts = {}
    for record in dataset.records:
        key = f"{record.dataset}:{record.site}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_epoch(model, loader, optimizer, cfg, device, replay_buffer=None):
    model.train()
    total_losses, jepa_losses, global_losses = [], [], []
    vic_losses, replay_alignment_losses = [], []
    main_losses, replay_losses = [], []

    for batch_idx, (x_main, labels_main) in enumerate(loader):
        x_main = x_main.to(device)
        replay_pair_ids = None
        replay_batch_size = 0

        if replay_buffer is not None:
            x_replay, labels_replay, replay_pair_ids = replay_buffer.sample()
            x_batch = torch.cat([x_main, x_replay], dim=0)
            n_main = x_main.size(0)
            replay_batch_size = x_replay.size(0)
        else:
            x_batch = x_main
            n_main = x_batch.size(0)

        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        context_indices, target_indices = mask_batch(
            seq_len=raw_model.num_tokens, cfg=cfg, device=x_batch.device,
        )

        ctx_batched = context_indices.unsqueeze(0).expand(x_batch.size(0), -1)
        tgt_batched = target_indices.unsqueeze(0).expand(x_batch.size(0), -1)

        predicted, target, pooled, pred_global, target_global = model(
            x_batch, ctx_batched, tgt_batched, return_global=True,
        )
        jepa_loss = F.mse_loss(predicted, target)
        global_loss = F.mse_loss(pred_global, target_global)
        vic_loss, _, _ = vicreg_regularization(
            pooled,
            var_weight=cfg.vicreg_var_weight,
            cov_weight=cfg.vicreg_cov_weight,
            var_target=cfg.vicreg_var_target,
        )
        replay_alignment = pooled.new_zeros(())
        if (
            replay_buffer is not None
            and cfg.replay_alignment_weight > 0.0
            and replay_batch_size > 0
            and replay_pair_ids is not None
        ):
            replay_embeddings = pooled[n_main : n_main + replay_batch_size]
            replay_alignment = replay_pair_alignment_loss(replay_embeddings, replay_pair_ids)

        total_loss = (
            jepa_loss
            + cfg.global_loss_weight * global_loss
            + cfg.vicreg_weight * vic_loss
            + cfg.replay_alignment_weight * replay_alignment
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if isinstance(model, torch.nn.DataParallel):
            model.module.update_target_encoder()
        else:
            model.update_target_encoder()

        total_losses.append(total_loss.item())
        jepa_losses.append(jepa_loss.item())
        global_losses.append(global_loss.item())
        vic_losses.append(vic_loss.item())
        replay_alignment_losses.append(replay_alignment.item())

        if replay_buffer is not None:
            with torch.no_grad():
                main_losses.append(F.mse_loss(predicted[:n_main], target[:n_main]).item())
                replay_losses.append(F.mse_loss(predicted[n_main:], target[n_main:]).item())

        if cfg.steps_per_epoch is not None and (batch_idx + 1) >= cfg.steps_per_epoch:
            break

    metrics = {
        "train_loss": float(np.mean(total_losses)),
        "jepa_loss": float(np.mean(jepa_losses)),
        "global_loss": float(np.mean(global_losses)),
        "vicreg_loss": float(np.mean(vic_losses)),
        "replay_alignment_loss": float(np.mean(replay_alignment_losses)),
    }
    if replay_buffer is not None:
        metrics["main_loss"] = float(np.mean(main_losses)) if main_losses else None
        metrics["replay_loss"] = float(np.mean(replay_losses)) if replay_losses else None
    return metrics


# ---------------------------------------------------------------------------
# Pooled training
# ---------------------------------------------------------------------------


def train_pooled(cfg: JEPAFusionConfig, logger):
    device = cfg.device
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_dirs = list(cfg.main_data_dirs)
    dataset = WindowedFMRIDataset(
        all_dirs,
        window_seconds=cfg.window_seconds,
        stride_seconds=cfg.window_stride_seconds,
        target_timepoints=cfg.target_timepoints,
        sampling=cfg.pooled_sampling,
        exclude_unknown_diagnoses=True,
        max_scans=cfg.max_main_scans,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=make_worker_init_fn(cfg.seed + 1),
        generator=make_torch_generator(cfg.seed + 1),
    )

    model = build_model(cfg).to(device)
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.pooled_lr, weight_decay=cfg.weight_decay)

    logger.info(
        "Pooled training | scans=%d | dataset_site_counts=%s",
        len(dataset), dataset_site_counts(dataset),
    )

    history = []
    for epoch in range(cfg.pooled_epochs):
        lr = get_lr(epoch, cfg.warmup_epochs, cfg.pooled_epochs, cfg.pooled_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        metrics = run_epoch(model, loader, optimizer, cfg, device)
        metrics["epoch"] = epoch + 1
        metrics["lr"] = lr
        history.append(metrics)

        logger.info(
            "Pooled Epoch %d | loss=%.4f jepa=%.4f global=%.4f vic=%.4f align=%.4f",
            epoch + 1,
            metrics["train_loss"],
            metrics["jepa_loss"],
            metrics["global_loss"],
            metrics["vicreg_loss"],
            metrics["replay_alignment_loss"],
        )

    ckpt_path = save_dir / "pooled_final.pt"
    save_checkpoint(ckpt_path, model, cfg, history, cfg.pooled_epochs)
    with open(save_dir / "pooled_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved pooled checkpoint: %s", ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# Curriculum training (Phase 1 + Phase 2)
# ---------------------------------------------------------------------------


def train_phase1(cfg: JEPAFusionConfig, logger):
    device = cfg.device
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = WindowedFMRIDataset(
        cfg.travel_data_dir,
        window_seconds=cfg.window_seconds,
        stride_seconds=cfg.window_stride_seconds,
        target_timepoints=cfg.target_timepoints,
        sampling=cfg.phase1_sampling,
        exclude_unknown_diagnoses=True,
        max_scans=cfg.max_travel_scans,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=make_worker_init_fn(cfg.seed + 11),
        generator=make_torch_generator(cfg.seed + 11),
    )

    model = build_model(cfg).to(device)
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.phase1_lr, weight_decay=cfg.weight_decay)

    logger.info(
        "Phase 1 | travel scans=%d | dataset_site_counts=%s",
        len(dataset), dataset_site_counts(dataset),
    )

    history = []
    for epoch in range(cfg.phase1_epochs):
        lr = get_lr(epoch, cfg.warmup_epochs, cfg.phase1_epochs, cfg.phase1_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        metrics = run_epoch(model, loader, optimizer, cfg, device)
        metrics["epoch"] = epoch + 1
        metrics["lr"] = lr
        history.append(metrics)

        logger.info(
            "Phase 1 Epoch %d | loss=%.4f jepa=%.4f global=%.4f vic=%.4f align=%.4f",
            epoch + 1,
            metrics["train_loss"],
            metrics["jepa_loss"],
            metrics["global_loss"],
            metrics["vicreg_loss"],
            metrics["replay_alignment_loss"],
        )

    ckpt_path = save_dir / "phase1_final.pt"
    save_checkpoint(ckpt_path, model, cfg, history, cfg.phase1_epochs)
    with open(save_dir / "phase1_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved phase 1 checkpoint: %s", ckpt_path)
    return ckpt_path


def train_phase2(cfg: JEPAFusionConfig, logger, phase1_checkpoint: Path):
    device = cfg.device
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    main_dataset = WindowedFMRIDataset(
        list(cfg.main_data_dirs),
        window_seconds=cfg.window_seconds,
        stride_seconds=cfg.window_stride_seconds,
        target_timepoints=cfg.target_timepoints,
        sampling=cfg.phase2_main_sampling,
        exclude_unknown_diagnoses=True,
        max_scans=cfg.max_main_scans,
    )
    replay_dataset = WindowedFMRIDataset(
        cfg.travel_data_dir,
        window_seconds=cfg.window_seconds,
        stride_seconds=cfg.window_stride_seconds,
        target_timepoints=cfg.target_timepoints,
        sampling=cfg.phase2_replay_sampling,
        exclude_unknown_diagnoses=True,
        max_scans=cfg.max_travel_scans,
    )

    main_batch = max(1, int(cfg.batch_size * (1.0 - cfg.replay_fraction)))
    replay_batch = max(1, cfg.batch_size - main_batch)

    batch_sampler = DatasetBalancedBatchSampler(main_dataset, batch_size=main_batch, seed=cfg.seed + 21)
    main_loader = DataLoader(
        main_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=make_worker_init_fn(cfg.seed + 22),
    )
    replay_buffer = ReplayBuffer(
        replay_dataset, replay_batch, device,
        seed=cfg.seed + 23, pair_sampling=cfg.replay_pair_sampling,
    )

    model = build_model(cfg).to(device)
    checkpoint = torch.load(phase1_checkpoint, map_location=device, weights_only=False)
    raw = model.module if isinstance(model, torch.nn.DataParallel) else model
    raw.load_state_dict(checkpoint["model_state_dict"])

    if device == "cuda" and torch.cuda.device_count() > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.phase2_lr, weight_decay=cfg.weight_decay)

    logger.info(
        "Phase 2 | main scans=%d replay scans=%d | pair_sampling=%s global_w=%.3f align_w=%.3f | "
        "main dataset_site_counts=%s",
        len(main_dataset), len(replay_dataset),
        cfg.replay_pair_sampling, cfg.global_loss_weight, cfg.replay_alignment_weight,
        dataset_site_counts(main_dataset),
    )

    history = []
    for epoch in range(cfg.phase2_epochs):
        lr = get_lr(epoch, cfg.warmup_epochs, cfg.phase2_epochs, cfg.phase2_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        metrics = run_epoch(model, main_loader, optimizer, cfg, device, replay_buffer=replay_buffer)
        metrics["epoch"] = epoch + 1
        metrics["lr"] = lr
        history.append(metrics)

        logger.info(
            "Phase 2 Epoch %d | loss=%.4f jepa=%.4f global=%.4f vic=%.4f align=%.4f | main=%.4f replay=%.4f",
            epoch + 1,
            metrics["train_loss"],
            metrics["jepa_loss"],
            metrics["global_loss"],
            metrics["vicreg_loss"],
            metrics["replay_alignment_loss"],
            metrics.get("main_loss", 0.0),
            metrics.get("replay_loss", 0.0),
        )

    ckpt_path = save_dir / "phase2_final.pt"
    save_checkpoint(ckpt_path, model, cfg, history, cfg.phase2_epochs)
    with open(save_dir / "phase2_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved phase 2 checkpoint: %s", ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Unified JEPA trainer for jepa_fusion")
    p.add_argument("--model-family", choices=["temporal", "st_roi_patch"], default=None)
    p.add_argument("--temporal-variant", choices=["basic", "conv_net", "rotary_conv_net"], default=None)
    p.add_argument("--training-mode", choices=["pooled", "curriculum"], default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--log-file", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--pooled-epochs", type=int, default=None)
    p.add_argument("--phase1-epochs", type=int, default=None)
    p.add_argument("--phase2-epochs", type=int, default=None)
    p.add_argument("--pooled-lr", type=float, default=None)
    p.add_argument("--phase1-lr", type=float, default=None)
    p.add_argument("--phase2-lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--replay-fraction", type=float, default=None)
    p.add_argument("--global-loss-weight", type=float, default=None)
    p.add_argument("--vicreg-weight", type=float, default=None)
    p.add_argument("--replay-alignment-weight", type=float, default=None)
    p.add_argument("--replay-pair-sampling", action="store_true")
    p.add_argument("--adversarial-site-weight", type=float, default=None)
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--encoder-depth", type=int, default=None)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--max-main-scans", type=int, default=None)
    p.add_argument("--max-travel-scans", type=int, default=None)
    p.add_argument("--cuda-visible-devices", default=None, help="CUDA_VISIBLE_DEVICES override")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = JEPAFusionConfig()

    # Apply CLI overrides
    if args.cuda_visible_devices is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.model_family is not None:
        cfg.model_family = args.model_family
    if args.temporal_variant is not None:
        cfg.temporal_variant = args.temporal_variant
    if args.training_mode is not None:
        cfg.training_mode = args.training_mode
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.deterministic:
        cfg.deterministic = True
    if args.pooled_epochs is not None:
        cfg.pooled_epochs = args.pooled_epochs
    if args.phase1_epochs is not None:
        cfg.phase1_epochs = args.phase1_epochs
    if args.phase2_epochs is not None:
        cfg.phase2_epochs = args.phase2_epochs
    if args.pooled_lr is not None:
        cfg.pooled_lr = args.pooled_lr
    if args.phase1_lr is not None:
        cfg.phase1_lr = args.phase1_lr
    if args.phase2_lr is not None:
        cfg.phase2_lr = args.phase2_lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.replay_fraction is not None:
        cfg.replay_fraction = args.replay_fraction
    if args.global_loss_weight is not None:
        cfg.global_loss_weight = args.global_loss_weight
    if args.vicreg_weight is not None:
        cfg.vicreg_weight = args.vicreg_weight
    if args.replay_alignment_weight is not None:
        cfg.replay_alignment_weight = args.replay_alignment_weight
    if args.replay_pair_sampling:
        cfg.replay_pair_sampling = True
    if args.adversarial_site_weight is not None:
        cfg.adversarial_site_weight = args.adversarial_site_weight
    if args.embed_dim is not None:
        cfg.embed_dim = args.embed_dim
    if args.encoder_depth is not None:
        cfg.encoder_depth = args.encoder_depth
    if args.steps_per_epoch is not None:
        cfg.steps_per_epoch = args.steps_per_epoch
    if args.max_main_scans is not None:
        cfg.max_main_scans = args.max_main_scans
    if args.max_travel_scans is not None:
        cfg.max_travel_scans = args.max_travel_scans

    # Apply model-family-specific defaults
    if cfg.model_family == "temporal":
        if cfg.temporal_variant == "rotary_conv_net":
            cfg.pos_embed_mode = "rotary"
            cfg.use_conv_stem = True
            cfg.use_network_embeddings = True
        elif cfg.temporal_variant == "conv_net":
            cfg.pos_embed_mode = "sincos"
            cfg.use_conv_stem = True
            cfg.use_network_embeddings = True
        else:  # basic
            cfg.pos_embed_mode = "sincos"
            cfg.use_conv_stem = False
            cfg.use_network_embeddings = False

    setup_logging(args.log_file)
    lgr = logging.getLogger("JEPA-Fusion")
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    lgr.info("Config: %s", json.dumps(asdict(cfg), indent=2))

    if cfg.training_mode == "pooled":
        train_pooled(cfg, lgr)
    elif cfg.training_mode == "curriculum":
        phase1_ckpt = train_phase1(cfg, lgr)
        train_phase2(cfg, lgr, phase1_ckpt)
    else:
        raise ValueError(f"Unsupported training_mode: {cfg.training_mode}")


if __name__ == "__main__":
    main()
