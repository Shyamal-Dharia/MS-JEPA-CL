from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from ..jepa_clean.repro import make_torch_generator, make_worker_init_fn, seed_everything
    from .losses import conditional_site_invariance_loss
    from .masking import complement_indices, sample_temporal_mask
    from .model_fourier_patch import FourierPatchJEPA, build_model_from_config
    from .train_jepa import (
        MAIN_DATASET_DIRS,
        MAIN_DATASET_ORDER,
        MAIN_DATA_DIRS,
        TRAVEL_DATA_DIR,
        ReplayBuffer,
        build_dataset,
        collate_fn,
        get_lr,
        infonce_loss,
        make_dataset_site_sampler,
        parse_dataset_keys,
        replay_pair_alignment_loss,
        resolve_main_data_dirs,
        save_checkpoint,
        save_history,
        setup_logging,
        summarize_records,
        vicreg_regularization,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_clean.repro import make_torch_generator, make_worker_init_fn, seed_everything
    from src.jepa_fusion.losses import conditional_site_invariance_loss
    from src.jepa_fusion.masking import complement_indices, sample_temporal_mask
    from src.jepa_fusion.model_fourier_patch import FourierPatchJEPA, build_model_from_config
    from src.jepa_fusion.train_jepa import (
        MAIN_DATASET_DIRS,
        MAIN_DATASET_ORDER,
        MAIN_DATA_DIRS,
        TRAVEL_DATA_DIR,
        ReplayBuffer,
        build_dataset,
        collate_fn,
        get_lr,
        infonce_loss,
        make_dataset_site_sampler,
        parse_dataset_keys,
        replay_pair_alignment_loss,
        resolve_main_data_dirs,
        save_checkpoint,
        save_history,
        setup_logging,
        summarize_records,
        vicreg_regularization,
    )


@dataclass
class FourierPatchJEPAConfig:
    main_dataset_keys: list[str] = field(default_factory=lambda: list(MAIN_DATASET_ORDER))
    main_data_dirs: list[str] = field(default_factory=lambda: list(MAIN_DATA_DIRS))
    travel_data_dir: str = TRAVEL_DATA_DIR

    training_mode: str = "pooled"

    input_dim: int = 400
    target_timepoints: int = 45
    time_patch: int = 5
    embed_dim: int = 400
    encoder_depth: int = 8
    predictor_depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    ema_decay: float = 0.996
    pos_embed_mode: str = "sincos"

    fourier_freqs: int = 4
    fourier_increment_time: float = 0.1
    fourier_margin: float = 0.4
    context_gaussian_noise_std: float = 0.0

    window_seconds: float = 90.0
    window_stride_seconds: float = 30.0
    pooled_sampling: str = "random"
    phase1_sampling: str = "sliding"
    phase2_main_sampling: str = "random"
    phase2_replay_sampling: str = "sliding"

    pooled_epochs: int = 60
    phase1_epochs: int = 20
    phase2_epochs: int = 40
    batch_size: int = 32
    replay_fraction: float = 0.2
    pooled_lr: float = 1e-4
    phase1_lr: float = 1e-4
    phase2_lr: float = 1e-5
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    steps_per_epoch: int | None = None
    max_main_scans: int | None = None
    max_travel_scans: int | None = None

    vicreg_weight: float = 0.2
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 1.0
    vicreg_var_target: float = 1.0
    replay_alignment_weight: float = 0.2
    replay_pair_sampling: bool = False
    infonce_weight: float = 0.0
    infonce_temperature: float = 0.07
    conditional_invariance_weight: float = 0.0
    conditional_repulsion_weight: float = 0.0
    conditional_sigma_age: float = 5.0
    conditional_eta: float = 0.5
    conditional_margin: float = 1.0
    conditional_eps: float = 1e-8

    mask_num_patch_blocks: int = 1
    mask_patch_block_size_min: int = 1
    mask_patch_block_size_max: int = 2

    save_dir: str = "checkpoints/jepa_fusion/fourier_patch_default"
    seed: int = 42
    deterministic: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_data_parallel: bool = False


def build_model(cfg: FourierPatchJEPAConfig):
    return build_model_from_config(asdict(cfg))


def sample_mask(cfg: FourierPatchJEPAConfig, raw_model: FourierPatchJEPA, device: torch.device):
    mask = sample_temporal_mask(
        seq_len=raw_model.num_tokens,
        num_time_blocks=cfg.mask_num_patch_blocks,
        time_block_size=(cfg.mask_patch_block_size_min, cfg.mask_patch_block_size_max),
    )
    target_indices = mask.target_indices.to(device)
    context_indices = complement_indices(raw_model.num_tokens, target_indices, device=device)
    return context_indices, target_indices


def apply_context_noise(x: torch.Tensor, cfg: FourierPatchJEPAConfig):
    if cfg.context_gaussian_noise_std <= 0.0:
        return x
    noise = torch.randn_like(x) * float(cfg.context_gaussian_noise_std)
    return x + noise


def _encode_conditional_metadata(labels, device: torch.device):
    valid_indices = []
    site_values = []
    age_values = []
    sex_values = []

    for idx, meta in enumerate(labels):
        raw_sex = str(meta.get("sex", "") or "").strip().upper()
        if raw_sex not in {"M", "F"}:
            continue
        try:
            age_value = float(meta.get("age", -1))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(age_value) or age_value < 0:
            continue

        valid_indices.append(idx)
        site_values.append(str(meta.get("site_raw", meta.get("site", ""))))
        age_values.append(age_value)
        sex_values.append(1 if raw_sex == "F" else 0)

    if len(valid_indices) < 2:
        return None

    site_to_idx = {site_name: idx for idx, site_name in enumerate(sorted(set(site_values)))}
    return (
        torch.as_tensor(valid_indices, device=device, dtype=torch.long),
        torch.as_tensor([site_to_idx[site_name] for site_name in site_values], device=device, dtype=torch.long),
        torch.as_tensor(age_values, device=device, dtype=torch.float32),
        torch.as_tensor(sex_values, device=device, dtype=torch.long),
    )


def run_epoch(
    model,
    loader,
    optimizer,
    cfg: FourierPatchJEPAConfig,
    device: str,
    replay_buffer: ReplayBuffer | None = None,
):
    model.train()
    total_losses = []
    jepa_losses = []
    vic_losses = []
    vic_var_losses = []
    vic_cov_losses = []
    replay_alignment_losses = []
    infonce_losses = []
    conditional_losses = []
    repulsion_losses = []
    main_losses = []
    replay_losses = []

    for batch_idx, (x_main, labels_main) in enumerate(loader):
        x_main = x_main.to(device)
        replay_pair_ids = None
        replay_batch_size = 0

        if replay_buffer is not None:
            x_replay, labels_replay, replay_pair_ids = replay_buffer.sample()
            x_batch = torch.cat([x_main, x_replay], dim=0)
            labels_batch = list(labels_main) + list(labels_replay)
            n_main = x_main.size(0)
            replay_batch_size = x_replay.size(0)
        else:
            x_batch = x_main
            labels_batch = list(labels_main)
            n_main = x_batch.size(0)

        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        context_indices, target_indices = sample_mask(cfg, raw_model, x_batch.device)
        ctx_batched = context_indices.unsqueeze(0).expand(x_batch.size(0), -1)
        tgt_batched = target_indices.unsqueeze(0).expand(x_batch.size(0), -1)

        context_x = apply_context_noise(x_batch, cfg)
        predicted, target, pooled = model(
            x_batch,
            ctx_batched,
            tgt_batched,
            context_x=context_x,
            target_x=x_batch,
        )
        jepa_loss = F.mse_loss(predicted, target)
        vic_loss, vic_var, vic_cov = vicreg_regularization(
            pooled,
            var_weight=cfg.vicreg_var_weight,
            cov_weight=cfg.vicreg_cov_weight,
            var_target=cfg.vicreg_var_target,
        )
        replay_alignment = pooled.new_zeros(())
        nce_loss = pooled.new_zeros(())
        cond_loss = pooled.new_zeros(())
        rep_loss = pooled.new_zeros(())
        if replay_buffer is not None and replay_batch_size > 0 and replay_pair_ids is not None:
            replay_embeddings = pooled[n_main : n_main + replay_batch_size]
            if cfg.replay_alignment_weight > 0.0:
                replay_alignment = replay_pair_alignment_loss(replay_embeddings, replay_pair_ids)
            if cfg.infonce_weight > 0.0:
                nce_loss = infonce_loss(
                    replay_embeddings,
                    replay_pair_ids,
                    temperature=cfg.infonce_temperature,
                )

        if cfg.conditional_invariance_weight > 0.0 or cfg.conditional_repulsion_weight > 0.0:
            conditional_meta = _encode_conditional_metadata(labels_batch, pooled.device)
            if conditional_meta is not None:
                valid_idx, site_tensor, age_tensor, sex_tensor = conditional_meta
                _loss_total, cond_loss, rep_loss = conditional_site_invariance_loss(
                    core_loss=jepa_loss.new_zeros(()),
                    z=pooled.index_select(0, valid_idx),
                    site=site_tensor,
                    age=age_tensor,
                    sex=sex_tensor,
                    lambda_cond=cfg.conditional_invariance_weight,
                    lambda_rep=cfg.conditional_repulsion_weight,
                    sigma_age=cfg.conditional_sigma_age,
                    eta=cfg.conditional_eta,
                    margin=cfg.conditional_margin,
                    eps=cfg.conditional_eps,
                )

        total_loss = (
            jepa_loss
            + (cfg.vicreg_weight * vic_loss)
            + (cfg.replay_alignment_weight * replay_alignment)
            + (cfg.infonce_weight * nce_loss)
            + (cfg.conditional_invariance_weight * cond_loss)
            + (cfg.conditional_repulsion_weight * rep_loss)
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
        vic_losses.append(vic_loss.item())
        vic_var_losses.append(vic_var.item())
        vic_cov_losses.append(vic_cov.item())
        replay_alignment_losses.append(replay_alignment.item())
        infonce_losses.append(nce_loss.item())
        conditional_losses.append(cond_loss.item())
        repulsion_losses.append(rep_loss.item())

        if replay_buffer is not None:
            with torch.no_grad():
                main_losses.append(F.mse_loss(predicted[:n_main], target[:n_main]).item())
                replay_losses.append(F.mse_loss(predicted[n_main:], target[n_main:]).item())

        if cfg.steps_per_epoch is not None and (batch_idx + 1) >= cfg.steps_per_epoch:
            break

    metrics = {
        "train_loss": float(np.mean(total_losses)),
        "jepa_loss": float(np.mean(jepa_losses)),
        "vicreg_loss": float(np.mean(vic_losses)),
        "vicreg_var_loss": float(np.mean(vic_var_losses)),
        "vicreg_cov_loss": float(np.mean(vic_cov_losses)),
        "replay_alignment_loss": float(np.mean(replay_alignment_losses)),
        "infonce_loss": float(np.mean(infonce_losses)),
        "conditional_invariance_loss": float(np.mean(conditional_losses)),
        "conditional_repulsion_loss": float(np.mean(repulsion_losses)),
    }
    if replay_buffer is not None:
        metrics["main_loss"] = float(np.mean(main_losses)) if main_losses else None
        metrics["replay_loss"] = float(np.mean(replay_losses)) if replay_losses else None
    return metrics


def train_pooled(cfg: FourierPatchJEPAConfig, logger: logging.Logger):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pooled_dirs = list(cfg.main_data_dirs) + [cfg.travel_data_dir]
    dataset = build_dataset(
        pooled_dirs,
        sampling=cfg.pooled_sampling,
        cfg=cfg,
        max_scans=cfg.max_main_scans,
    )
    sampler = make_dataset_site_sampler(dataset, seed=cfg.seed + 11)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=make_worker_init_fn(cfg.seed + 12),
    )

    pair_buffer = None
    if cfg.replay_alignment_weight > 0.0 or cfg.infonce_weight > 0.0:
        travel_dataset = build_dataset(
            cfg.travel_data_dir,
            sampling="sliding",
            cfg=cfg,
            max_scans=cfg.max_travel_scans,
        )
        pair_batch = max(1, int(cfg.batch_size * cfg.replay_fraction))
        pair_buffer = ReplayBuffer(
            travel_dataset,
            batch_size=pair_batch,
            device=cfg.device,
            seed=cfg.seed + 14,
            pair_sampling=True,
        )
        logger.info(
            "Pooled pair buffer | travel scans=%d pair_batch=%d",
            len(travel_dataset),
            pair_batch,
        )

    model = build_model(cfg).to(cfg.device)
    if cfg.device == "cuda" and torch.cuda.device_count() > 1 and cfg.use_data_parallel:
        logger.info("Using %d GPUs via DataParallel", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.pooled_lr, weight_decay=cfg.weight_decay)
    history = []
    logger.info(
        "Pooled training | scans=%d | dataset_site_counts=%s",
        len(dataset),
        summarize_records(dataset.records),
    )
    for epoch in range(cfg.pooled_epochs):
        lr = get_lr(epoch, cfg.warmup_epochs, cfg.pooled_epochs, cfg.pooled_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        metrics = run_epoch(model, loader, optimizer, cfg, cfg.device, replay_buffer=pair_buffer)
        metrics["epoch"] = epoch + 1
        metrics["lr"] = lr
        history.append(metrics)
        logger.info(
            "Pooled Epoch %d | loss=%.4f jepa=%.4f vic=%.4f align=%.4f nce=%.4f cond=%.4f rep=%.4f",
            epoch + 1,
            metrics["train_loss"],
            metrics["jepa_loss"],
            metrics["vicreg_loss"],
            metrics["replay_alignment_loss"],
            metrics["infonce_loss"],
            metrics["conditional_invariance_loss"],
            metrics["conditional_repulsion_loss"],
        )

    ckpt_path = save_dir / "pooled_final.pt"
    save_checkpoint(model, cfg, history, ckpt_path)
    save_history(history, save_dir / "pooled_history.json")
    logger.info("Saved pooled checkpoint: %s", ckpt_path)
    return ckpt_path


def train_curriculum(cfg: FourierPatchJEPAConfig, logger: logging.Logger):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    phase1_dataset = build_dataset(
        cfg.travel_data_dir,
        sampling=cfg.phase1_sampling,
        cfg=cfg,
        max_scans=cfg.max_travel_scans,
    )
    phase2_main_dataset = build_dataset(
        cfg.main_data_dirs,
        sampling=cfg.phase2_main_sampling,
        cfg=cfg,
        max_scans=cfg.max_main_scans,
    )
    phase2_replay_dataset = build_dataset(
        cfg.travel_data_dir,
        sampling=cfg.phase2_replay_sampling,
        cfg=cfg,
        max_scans=cfg.max_travel_scans,
    )
    phase1_loader = DataLoader(
        phase1_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=make_worker_init_fn(cfg.seed + 21),
        generator=make_torch_generator(cfg.seed + 21),
    )

    model = build_model(cfg).to(cfg.device)
    if cfg.device == "cuda" and torch.cuda.device_count() > 1 and cfg.use_data_parallel:
        logger.info("Using %d GPUs via DataParallel", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.phase1_lr, weight_decay=cfg.weight_decay)
    phase1_history = []
    logger.info(
        "Phase 1 | travel scans=%d | dataset_site_counts=%s",
        len(phase1_dataset),
        summarize_records(phase1_dataset.records),
    )
    for epoch in range(cfg.phase1_epochs):
        lr = get_lr(epoch, cfg.warmup_epochs, cfg.phase1_epochs, cfg.phase1_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        metrics = run_epoch(model, phase1_loader, optimizer, cfg, cfg.device)
        metrics["epoch"] = epoch + 1
        metrics["lr"] = lr
        phase1_history.append(metrics)
        logger.info(
            "Phase 1 Epoch %d | loss=%.4f jepa=%.4f vic=%.4f align=%.4f nce=%.4f cond=%.4f rep=%.4f",
            epoch + 1,
            metrics["train_loss"],
            metrics["jepa_loss"],
            metrics["vicreg_loss"],
            metrics["replay_alignment_loss"],
            metrics["infonce_loss"],
            metrics["conditional_invariance_loss"],
            metrics["conditional_repulsion_loss"],
        )

    phase1_ckpt = save_dir / "phase1_final.pt"
    save_checkpoint(model, cfg, phase1_history, phase1_ckpt, extra={"phase": "phase1"})
    save_history(phase1_history, save_dir / "phase1_history.json")
    logger.info("Saved phase 1 checkpoint: %s", phase1_ckpt)

    main_batch = max(1, int(cfg.batch_size * (1.0 - cfg.replay_fraction)))
    replay_batch = max(1, cfg.batch_size - main_batch)
    main_sampler = make_dataset_site_sampler(phase2_main_dataset, seed=cfg.seed + 31)
    phase2_main_loader = DataLoader(
        phase2_main_dataset,
        batch_size=main_batch,
        sampler=main_sampler,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=make_worker_init_fn(cfg.seed + 32),
    )
    replay_buffer = None
    if replay_batch > 0 and cfg.replay_fraction > 0.0:
        replay_buffer = ReplayBuffer(
            phase2_replay_dataset,
            batch_size=replay_batch,
            device=cfg.device,
            seed=cfg.seed + 33,
            pair_sampling=(cfg.replay_pair_sampling or cfg.infonce_weight > 0.0),
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.phase2_lr, weight_decay=cfg.weight_decay)
    phase2_history = []
    logger.info(
        "Phase 2 | main scans=%d replay scans=%d | pair_sampling=%s align_w=%.3f | main dataset_site_counts=%s",
        len(phase2_main_dataset),
        len(phase2_replay_dataset),
        cfg.replay_pair_sampling,
        cfg.replay_alignment_weight,
        summarize_records(phase2_main_dataset.records),
    )
    for epoch in range(cfg.phase2_epochs):
        lr = get_lr(epoch, cfg.warmup_epochs, cfg.phase2_epochs, cfg.phase2_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        metrics = run_epoch(
            model=model,
            loader=phase2_main_loader,
            optimizer=optimizer,
            cfg=cfg,
            device=cfg.device,
            replay_buffer=replay_buffer,
        )
        metrics["epoch"] = epoch + 1
        metrics["lr"] = lr
        phase2_history.append(metrics)
        logger.info(
            "Phase 2 Epoch %d | loss=%.4f jepa=%.4f vic=%.4f align=%.4f nce=%.4f cond=%.4f rep=%.4f | main=%s replay=%s",
            epoch + 1,
            metrics["train_loss"],
            metrics["jepa_loss"],
            metrics["vicreg_loss"],
            metrics["replay_alignment_loss"],
            metrics["infonce_loss"],
            metrics["conditional_invariance_loss"],
            metrics["conditional_repulsion_loss"],
            "n/a" if metrics.get("main_loss") is None else f"{metrics['main_loss']:.4f}",
            "n/a" if metrics.get("replay_loss") is None else f"{metrics['replay_loss']:.4f}",
        )

    phase2_ckpt = save_dir / "phase2_final.pt"
    save_checkpoint(model, cfg, phase2_history, phase2_ckpt, extra={"phase": "phase2"})
    save_history(phase2_history, save_dir / "phase2_history.json")
    logger.info("Saved phase 2 checkpoint: %s", phase2_ckpt)
    return phase1_ckpt, phase2_ckpt


def parse_args():
    parser = argparse.ArgumentParser(description="Train 9-patch Fourier JEPA on raw ROI timeseries.")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--main-datasets",
        default=None,
        help="Comma-separated main dataset keys from {srpbs,abide1,abide2}. Default uses all three.",
    )
    parser.add_argument("--travel-data-dir", default=None)
    parser.add_argument("--training-mode", choices=["pooled", "curriculum"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pooled-epochs", type=int, default=None)
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--phase2-epochs", type=int, default=None)
    parser.add_argument("--pooled-lr", type=float, default=None)
    parser.add_argument("--phase1-lr", type=float, default=None)
    parser.add_argument("--phase2-lr", type=float, default=None)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=None)
    parser.add_argument("--target-timepoints", type=int, default=None)
    parser.add_argument("--time-patch", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--encoder-depth", type=int, default=None)
    parser.add_argument("--predictor-depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--fourier-freqs", type=int, default=None)
    parser.add_argument("--fourier-increment-time", type=float, default=None)
    parser.add_argument("--fourier-margin", type=float, default=None)
    parser.add_argument("--context-gaussian-noise-std", type=float, default=None)
    parser.add_argument("--vicreg-weight", type=float, default=None)
    parser.add_argument("--replay-alignment-weight", type=float, default=None)
    parser.add_argument("--replay-fraction", type=float, default=None)
    parser.add_argument("--replay-pair-sampling", action="store_true")
    parser.add_argument("--infonce-weight", type=float, default=None)
    parser.add_argument("--infonce-temperature", type=float, default=None)
    parser.add_argument("--conditional-invariance-weight", type=float, default=None)
    parser.add_argument("--conditional-repulsion-weight", type=float, default=None)
    parser.add_argument("--conditional-sigma-age", type=float, default=None)
    parser.add_argument("--conditional-eta", type=float, default=None)
    parser.add_argument("--conditional-margin", type=float, default=None)
    parser.add_argument("--mask-num-patch-blocks", type=int, default=None)
    parser.add_argument("--mask-patch-block-min", type=int, default=None)
    parser.add_argument("--mask-patch-block-max", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-main-scans", type=int, default=None)
    parser.add_argument("--max-travel-scans", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = FourierPatchJEPAConfig()
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.device is not None:
        cfg.device = args.device
    if args.main_datasets is not None:
        cfg.main_dataset_keys = parse_dataset_keys(args.main_datasets)
        cfg.main_data_dirs = resolve_main_data_dirs(cfg.main_dataset_keys)
    if args.travel_data_dir is not None:
        cfg.travel_data_dir = args.travel_data_dir
    if args.training_mode is not None:
        cfg.training_mode = args.training_mode
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
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
    if args.window_seconds is not None:
        cfg.window_seconds = args.window_seconds
    if args.window_stride_seconds is not None:
        cfg.window_stride_seconds = args.window_stride_seconds
    if args.target_timepoints is not None:
        cfg.target_timepoints = args.target_timepoints
    if args.time_patch is not None:
        cfg.time_patch = args.time_patch
    if args.embed_dim is not None:
        cfg.embed_dim = args.embed_dim
    if args.encoder_depth is not None:
        cfg.encoder_depth = args.encoder_depth
    if args.predictor_depth is not None:
        cfg.predictor_depth = args.predictor_depth
    if args.num_heads is not None:
        cfg.num_heads = args.num_heads
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.fourier_freqs is not None:
        cfg.fourier_freqs = args.fourier_freqs
    if args.fourier_increment_time is not None:
        cfg.fourier_increment_time = args.fourier_increment_time
    if args.fourier_margin is not None:
        cfg.fourier_margin = args.fourier_margin
    if args.context_gaussian_noise_std is not None:
        cfg.context_gaussian_noise_std = args.context_gaussian_noise_std
    if args.vicreg_weight is not None:
        cfg.vicreg_weight = args.vicreg_weight
    if args.replay_alignment_weight is not None:
        cfg.replay_alignment_weight = args.replay_alignment_weight
    if args.replay_fraction is not None:
        cfg.replay_fraction = args.replay_fraction
    if args.replay_pair_sampling:
        cfg.replay_pair_sampling = True
    if args.infonce_weight is not None:
        cfg.infonce_weight = args.infonce_weight
    if args.infonce_temperature is not None:
        cfg.infonce_temperature = args.infonce_temperature
    if args.conditional_invariance_weight is not None:
        cfg.conditional_invariance_weight = args.conditional_invariance_weight
    if args.conditional_repulsion_weight is not None:
        cfg.conditional_repulsion_weight = args.conditional_repulsion_weight
    if args.conditional_sigma_age is not None:
        cfg.conditional_sigma_age = args.conditional_sigma_age
    if args.conditional_eta is not None:
        cfg.conditional_eta = args.conditional_eta
    if args.conditional_margin is not None:
        cfg.conditional_margin = args.conditional_margin
    if args.mask_num_patch_blocks is not None:
        cfg.mask_num_patch_blocks = args.mask_num_patch_blocks
    if args.mask_patch_block_min is not None:
        cfg.mask_patch_block_size_min = args.mask_patch_block_min
    if args.mask_patch_block_max is not None:
        cfg.mask_patch_block_size_max = args.mask_patch_block_max
    if args.steps_per_epoch is not None:
        cfg.steps_per_epoch = args.steps_per_epoch
    if args.max_main_scans is not None:
        cfg.max_main_scans = args.max_main_scans
    if args.max_travel_scans is not None:
        cfg.max_travel_scans = args.max_travel_scans
    if args.seed is not None:
        cfg.seed = args.seed
    if args.deterministic:
        cfg.deterministic = True
    if args.non_deterministic:
        cfg.deterministic = False
    if args.data_parallel:
        cfg.use_data_parallel = True

    if cfg.target_timepoints % cfg.time_patch != 0:
        raise ValueError(
            f"target_timepoints={cfg.target_timepoints} must be divisible by time_patch={cfg.time_patch}"
        )

    if args.quick:
        cfg.pooled_epochs = 2
        cfg.phase1_epochs = 1
        cfg.phase2_epochs = 1
        cfg.batch_size = min(cfg.batch_size, 8)
        cfg.steps_per_epoch = 2
        cfg.max_main_scans = 64 if cfg.max_main_scans is None else min(cfg.max_main_scans, 64)
        cfg.max_travel_scans = 64 if cfg.max_travel_scans is None else min(cfg.max_travel_scans, 64)
        if args.save_dir is None:
            cfg.save_dir = f"checkpoints/jepa_fusion/quick_fourier_patch_{cfg.training_mode}"

    setup_logging(args.log_file)
    logger = logging.getLogger("train_jepa_fourier_patch")
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    logger.info("FourierPatchJEPA config: %s", asdict(cfg))

    if cfg.training_mode == "pooled":
        train_pooled(cfg, logger)
    elif cfg.training_mode == "curriculum":
        train_curriculum(cfg, logger)
    else:
        raise ValueError(f"Unsupported training mode: {cfg.training_mode}")


if __name__ == "__main__":
    main()
