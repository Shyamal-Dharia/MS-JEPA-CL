from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from ..jepa_clean.repro import make_torch_generator, make_worker_init_fn, seed_everything
    from .masking import complement_indices, sample_temporal_mask
    from .model import FusionJEPA
    from .train_jepa import (
        FusionJEPAConfig,
        build_dataset,
        get_lr,
        parse_dataset_keys,
        resolve_architecture,
        resolve_main_data_dirs,
        save_history,
        setup_logging,
        summarize_records,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_clean.repro import make_torch_generator, make_worker_init_fn, seed_everything
    from src.jepa_fusion.masking import complement_indices, sample_temporal_mask
    from src.jepa_fusion.model import FusionJEPA
    from src.jepa_fusion.train_jepa import (
        FusionJEPAConfig,
        build_dataset,
        get_lr,
        parse_dataset_keys,
        resolve_architecture,
        resolve_main_data_dirs,
        save_history,
        setup_logging,
        summarize_records,
    )


class MaskedReconstructionSSL(nn.Module):
    def __init__(self, encoder: FusionJEPA, input_dim: int):
        super().__init__()
        self.encoder = encoder
        embed_dim = int(encoder.token_embed.embed_dim)
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, context_indices: torch.Tensor, target_indices: torch.Tensor):
        context_repr = self.encoder._encode_branch_tokens(
            self.encoder.token_embed,
            self.encoder.context_encoder,
            x,
            indices=context_indices,
        )
        predicted = self.encoder.predictor(context_repr, context_indices, target_indices)
        return self.reconstruction_head(predicted)


def collate_fn(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), list(labels)


def sample_recon_mask(cfg: FusionJEPAConfig, model: FusionJEPA, device: torch.device):
    if model.model_family != "temporal":
        raise ValueError("Masked reconstruction SSL currently supports model_family=temporal only.")
    mask = sample_temporal_mask(
        seq_len=model.num_tokens,
        num_time_blocks=cfg.mask_num_time_blocks,
        time_block_size=(cfg.mask_time_block_size_min, cfg.mask_time_block_size_max),
    )
    target_indices = mask.target_indices.to(device)
    context_indices = complement_indices(model.num_tokens, target_indices, device=device)
    return context_indices, target_indices


def run_epoch(
    model: MaskedReconstructionSSL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: FusionJEPAConfig,
    device: str,
):
    model.train()
    losses = []
    mae_losses = []
    for batch_idx, (x, _labels) in enumerate(loader):
        x = x.to(device)
        context_indices, target_indices = sample_recon_mask(cfg, model.encoder, x.device)
        recon = model(x, context_indices, target_indices)
        target = x[:, target_indices, :]
        loss = F.mse_loss(recon, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(float(loss.item()))
        mae_losses.append(float((recon.detach() - target).abs().mean().item()))
        if cfg.steps_per_epoch is not None and (batch_idx + 1) >= cfg.steps_per_epoch:
            break

    return {
        "train_loss": float(np.mean(losses)),
        "masked_recon_mse": float(np.mean(losses)),
        "masked_recon_mae": float(np.mean(mae_losses)),
    }


def save_encoder_checkpoint(
    ssl_model: MaskedReconstructionSSL,
    cfg: FusionJEPAConfig,
    history: list[dict],
    path: Path,
):
    encoder = ssl_model.encoder
    payload = {
        "model_state_dict": encoder.state_dict(),
        "config": {
            **asdict(cfg),
            "ssl_objective": "masked_reconstruction",
            "checkpoint_name": "masked_recon_final.pt",
        },
        "history": history,
        "discarded_heads": ["reconstruction_head"],
    }
    torch.save(payload, path)


def train(cfg: FusionJEPAConfig, logger: logging.Logger):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(
        cfg.main_data_dirs,
        sampling=cfg.pooled_sampling,
        cfg=cfg,
        max_scans=cfg.max_main_scans,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=make_worker_init_fn(cfg.seed + 12),
        generator=make_torch_generator(cfg.seed + 12),
    )

    encoder = FusionJEPA(
        model_family=cfg.model_family,
        input_dim=cfg.input_dim,
        target_timepoints=cfg.target_timepoints,
        embed_dim=cfg.embed_dim,
        encoder_depth=cfg.encoder_depth,
        predictor_depth=cfg.predictor_depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        ema_decay=cfg.ema_decay,
        pos_embed_mode=cfg.pos_embed_mode,
        use_conv_stem=cfg.use_conv_stem,
        conv_kernel_size=cfg.conv_kernel_size,
        use_network_embeddings=cfg.use_network_embeddings,
        st_time_patch=cfg.st_time_patch,
        st_roi_group_size=cfg.st_roi_group_size,
        st_reorder_mode=cfg.st_reorder_mode,
    )
    ssl_model = MaskedReconstructionSSL(encoder, input_dim=cfg.input_dim).to(cfg.device)
    if cfg.device == "cuda" and torch.cuda.device_count() > 1 and cfg.use_data_parallel:
        logger.info("Using %d GPUs via DataParallel", torch.cuda.device_count())
        ssl_model = torch.nn.DataParallel(ssl_model)

    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=cfg.pooled_lr, weight_decay=cfg.weight_decay)
    history = []
    logger.info(
        "Masked Recon. SSL | scans=%d | dataset_site_counts=%s",
        len(dataset),
        summarize_records(dataset.records),
    )
    for epoch in range(cfg.pooled_epochs):
        lr = get_lr(epoch, cfg.warmup_epochs, cfg.pooled_epochs, cfg.pooled_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        metrics = run_epoch(ssl_model, loader, optimizer, cfg, cfg.device)
        metrics["epoch"] = epoch + 1
        metrics["lr"] = lr
        history.append(metrics)
        logger.info(
            "Epoch %d | mse=%.5f mae=%.5f lr=%.3e",
            epoch + 1,
            metrics["masked_recon_mse"],
            metrics["masked_recon_mae"],
            lr,
        )

    raw_ssl_model = ssl_model.module if isinstance(ssl_model, torch.nn.DataParallel) else ssl_model
    ckpt_path = save_dir / "masked_recon_final.pt"
    save_encoder_checkpoint(raw_ssl_model, cfg, history, ckpt_path)
    save_history(history, save_dir / "masked_recon_history.json")
    logger.info("Saved masked reconstruction encoder checkpoint: %s", ckpt_path)
    return ckpt_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Masked Recon. SSL baseline.")
    parser.add_argument("--save-dir", default="checkpoints/jepa_fusion/masked_recon_ssl")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--main-datasets", default="srpbs,abide1,abide2")
    parser.add_argument("--temporal-variant", choices=["basic", "conv_net", "rotary_conv_net"], default="rotary_conv_net")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=None)
    parser.add_argument("--target-timepoints", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-main-scans", type=int, default=None)
    parser.add_argument("--mask-num-time-blocks", type=int, default=None)
    parser.add_argument("--mask-time-block-size-min", type=int, default=None)
    parser.add_argument("--mask-time-block-size-max", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = FusionJEPAConfig()
    cfg.training_mode = "pooled"
    cfg.model_family = "temporal"
    cfg.temporal_variant = args.temporal_variant
    cfg.main_dataset_keys = parse_dataset_keys(args.main_datasets)
    cfg.main_data_dirs = resolve_main_data_dirs(cfg.main_dataset_keys)
    cfg.save_dir = args.save_dir
    cfg.seed = int(args.seed)
    if args.device is not None:
        cfg.device = args.device

    resolve_architecture(cfg)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.pooled_epochs = args.epochs
    if args.lr is not None:
        cfg.pooled_lr = args.lr
    if args.window_seconds is not None:
        cfg.window_seconds = args.window_seconds
    if args.window_stride_seconds is not None:
        cfg.window_stride_seconds = args.window_stride_seconds
    if args.target_timepoints is not None:
        cfg.target_timepoints = args.target_timepoints
    if args.steps_per_epoch is not None:
        cfg.steps_per_epoch = args.steps_per_epoch
    if args.max_main_scans is not None:
        cfg.max_main_scans = args.max_main_scans
    if args.mask_num_time_blocks is not None:
        cfg.mask_num_time_blocks = args.mask_num_time_blocks
    if args.mask_time_block_size_min is not None:
        cfg.mask_time_block_size_min = args.mask_time_block_size_min
    if args.mask_time_block_size_max is not None:
        cfg.mask_time_block_size_max = args.mask_time_block_size_max
    if args.deterministic:
        cfg.deterministic = True
    if args.non_deterministic:
        cfg.deterministic = False
    if args.data_parallel:
        cfg.use_data_parallel = True
    if args.quick:
        cfg.pooled_epochs = 1
        cfg.batch_size = min(cfg.batch_size, 8)
        cfg.steps_per_epoch = cfg.steps_per_epoch or 2
        cfg.max_main_scans = cfg.max_main_scans or 48

    setup_logging(args.log_file)
    logger = logging.getLogger("Masked-Recon-SSL")
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    logger.info("Config: %s", json.dumps(asdict(cfg), indent=2))
    train(cfg, logger)


if __name__ == "__main__":
    main()
