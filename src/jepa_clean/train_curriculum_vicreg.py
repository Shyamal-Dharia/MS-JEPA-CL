import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import WindowedFMRIDataset
from .masking import (
    sample_temporal_mask,
)
from .model import JEPA
from .probe import run_site_probe
from .repro import make_torch_generator, make_worker_init_fn, seed_everything


@dataclass
class CurriculumVICRegConfig:
    main_data_dirs: list[str] = field(
        default_factory=lambda: ["preprocessed_data_fir/full/roi_timeseries_1600_full"]
    )
    replay_data_dir: str = "preprocessed_data_fir/full/roi_timeseries_traveling_full"
    abide_data_dir: str = "preprocessed_data_fir/full/roi_timeseries_abide_full"
    abide2_data_dir: str = "preprocessed_data_fir/full/roi_timeseries_abide2_full"
    include_abide: bool = True
    include_abide2: bool = False
    hc_only: bool = False

    window_seconds: float = 90.0
    window_stride_seconds: float = 30.0
    target_timepoints: int = 45
    phase1_sampling: str = "sliding"
    phase2_main_sampling: str = "random"
    phase2_replay_sampling: str = "sliding"
    probe_sampling: str = "center"

    embed_dim: int = 400
    encoder_depth: int = 16
    predictor_depth: int = 2
    num_heads: int = 8
    pos_embed_mode: str = "sincos"
    use_conv_stem: bool = True
    conv_kernel_size: int = 3
    use_roi_embeddings: bool = True
    use_network_embeddings: bool = True
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    ema_decay: float = 0.996

    phase1_epochs: int = 20
    phase2_epochs: int = 40
    batch_size: int = 32
    replay_fraction: float = 0.2
    phase1_lr: float = 0.0001
    phase2_lr: float = 0.00001
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    steps_per_epoch: int | None = None
    max_main_scans: int | None = None
    max_replay_scans: int | None = None

    vicreg_weight: float = 0.1
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 1.0
    vicreg_var_target: float = 1.0

    mask_mode: str = "temporal"
    mask_num_time_blocks: int = 2
    mask_time_block_size_min: int = 6
    mask_time_block_size_max: int = 10
    mask_num_networks_min: int = 1
    mask_num_networks_max: int = 2

    probe_every: int = 1
    save_dir: str = "checkpoints/jepa_clean_curriculum_vicreg"
    seed: int = 42
    deterministic: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def setup_logging(log_file: str | None = None):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def collate_fn(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), list(labels)


def summarize_dataset_records(records):
    counts = {}
    for record in records:
        counts[record.dataset] = counts.get(record.dataset, 0) + 1
    return dict(sorted(counts.items()))


def make_site_balanced_sampler(dataset: WindowedFMRIDataset, seed: int):
    site_counts = {}
    for record in dataset.records:
        site_counts[record.site] = site_counts.get(record.site, 0) + 1
    weights = [1.0 / site_counts[record.site] for record in dataset.records]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=make_torch_generator(seed),
    )


class ReplayBuffer:
    def __init__(self, dataset, batch_size, device, seed: int):
        self.device = device
        self.dataset = dataset
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            collate_fn=collate_fn,
            worker_init_fn=make_worker_init_fn(seed),
            generator=make_torch_generator(seed),
        )
        self.iterator = iter(self.loader)

    def sample(self):
        try:
            x, labels = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            x, labels = next(self.iterator)
        return x.to(self.device), labels


def get_lr(epoch: int, warmup_epochs: int, total_epochs: int, base_lr: float):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def vicreg_regularization(embeddings, var_weight=1.0, cov_weight=1.0, var_target=1.0):
    n, d = embeddings.shape
    std = embeddings.std(dim=0)
    var_loss = torch.relu(var_target - std).mean()

    centered = embeddings - embeddings.mean(dim=0)
    cov = (centered.T @ centered) / max(1, n - 1)
    cov.fill_diagonal_(0)
    cov_loss = cov.pow(2).sum() / d

    total = var_weight * var_loss + cov_weight * cov_loss
    return total, var_loss, cov_loss


def build_model(cfg: CurriculumVICRegConfig):
    return JEPA(
        input_dim=400,
        max_seq_len=cfg.target_timepoints,
        embed_dim=cfg.embed_dim,
        encoder_depth=cfg.encoder_depth,
        predictor_depth=cfg.predictor_depth,
        num_heads=cfg.num_heads,
        pos_embed_mode=cfg.pos_embed_mode,
        use_conv_stem=cfg.use_conv_stem,
        conv_kernel_size=cfg.conv_kernel_size,
        use_roi_embeddings=cfg.use_roi_embeddings,
        use_network_embeddings=cfg.use_network_embeddings,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        ema_decay=cfg.ema_decay,
    )


def complement_indices(seq_len: int, target_indices: torch.LongTensor, device: torch.device):
    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    keep_mask[target_indices.to(device)] = False
    return torch.arange(seq_len, device=device)[keep_mask]


def mask_batch(seq_len: int, cfg: CurriculumVICRegConfig, device: torch.device):

    if cfg.mask_mode == "temporal":
        mask = sample_temporal_mask(
            seq_len=seq_len,
            num_time_blocks=cfg.mask_num_time_blocks,
            time_block_size=(cfg.mask_time_block_size_min, cfg.mask_time_block_size_max),
        )
        target_indices = mask["target_indices"].to(device)
        context_indices = complement_indices(seq_len=seq_len, target_indices=target_indices, device=device)
        return context_indices, target_indices, mask

    if cfg.mask_mode == "spatiotemporal":
        raise ValueError(
            "Strict I-JEPA context/target splitting is currently implemented only for temporal masking. "
            "With temporal tokens, spatio-temporal masking would require a different tokenization scheme."
        )

    raise ValueError(f"Unsupported mask_mode: {cfg.mask_mode}")


def save_history(history, path: Path):
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def run_epoch(model, loader, optimizer, cfg, device, replay_buffer=None):
    model.train()
    total_losses = []
    jepa_losses = []
    vic_losses = []
    vic_var_losses = []
    vic_cov_losses = []
    main_losses = []
    replay_losses = []

    for batch_idx, (x_main, labels_main) in enumerate(loader):
        x_main = x_main.to(device)

        if replay_buffer is not None:
            x_replay, labels_replay = replay_buffer.sample()
            x_batch = torch.cat([x_main, x_replay], dim=0)
            labels_batch = list(labels_main) + list(labels_replay)
            n_main = x_main.size(0)
        else:
            x_batch = x_main
            labels_batch = list(labels_main)
            n_main = x_batch.size(0)

        context_indices, target_indices, _ = mask_batch(
            seq_len=model.module.num_tokens if isinstance(model, torch.nn.DataParallel) else model.num_tokens,
            cfg=cfg,
            device=x_batch.device,
        )

        # Expand indices with a batch dimension so DataParallel evenly chunks them to all GPUs
        # instead of arbitrarily chunking the sequence of indices!
        ctx_batched = context_indices.unsqueeze(0).expand(x_batch.size(0), -1)
        tgt_batched = target_indices.unsqueeze(0).expand(x_batch.size(0), -1)

        # Use unified forward which is properly routed through DataParallel across GPUs
        predicted, target, pooled = model(x_batch, ctx_batched, tgt_batched)
        jepa_loss = F.mse_loss(predicted, target)
        vic_loss, vic_var, vic_cov = vicreg_regularization(
            pooled,
            var_weight=cfg.vicreg_var_weight,
            cov_weight=cfg.vicreg_cov_weight,
            var_target=cfg.vicreg_var_target,
        )
        total_loss = jepa_loss + cfg.vicreg_weight * vic_loss

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

        if replay_buffer is not None:
            with torch.no_grad():
                main_pred = predicted[:n_main]
                main_tgt = target[:n_main]
                replay_pred = predicted[n_main:]
                replay_tgt = target[n_main:]
                main_losses.append(F.mse_loss(main_pred, main_tgt).item())
                replay_losses.append(F.mse_loss(replay_pred, replay_tgt).item())

        if cfg.steps_per_epoch is not None and (batch_idx + 1) >= cfg.steps_per_epoch:
            break

    metrics = {
        "train_loss": float(np.mean(total_losses)),
        "jepa_loss": float(np.mean(jepa_losses)),
        "vicreg_loss": float(np.mean(vic_losses)),
        "vicreg_var_loss": float(np.mean(vic_var_losses)),
        "vicreg_cov_loss": float(np.mean(vic_cov_losses)),
    }
    if replay_buffer is not None:
        metrics["main_loss"] = float(np.mean(main_losses)) if main_losses else None
        metrics["replay_loss"] = float(np.mean(replay_losses)) if replay_losses else None
    return metrics


def train_phase1(cfg: CurriculumVICRegConfig, logger):
    device = cfg.device
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = WindowedFMRIDataset(
        cfg.replay_data_dir,
        window_seconds=cfg.window_seconds,
        stride_seconds=cfg.window_stride_seconds,
        target_timepoints=cfg.target_timepoints,
        sampling=cfg.phase1_sampling,
        max_scans=cfg.max_replay_scans,
    )
    probe_dataset = WindowedFMRIDataset(
        cfg.replay_data_dir,
        window_seconds=cfg.window_seconds,
        stride_seconds=cfg.window_stride_seconds,
        target_timepoints=cfg.target_timepoints,
        sampling=cfg.probe_sampling,
        max_scans=cfg.max_replay_scans,
    )
    loader = DataLoader(
        train_dataset,
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
        logger.info(f"Using {torch.cuda.device_count()} GPUs for Phase 1 via DataParallel")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.phase1_lr, weight_decay=cfg.weight_decay
    )

    history = []
    logger.info(
        "Phase 1 | travelling scans=%d | target_timepoints=%d",
        len(train_dataset),
        cfg.target_timepoints,
    )
    for epoch in range(cfg.phase1_epochs):
        lr = get_lr(epoch, cfg.warmup_epochs, cfg.phase1_epochs, cfg.phase1_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        metrics = run_epoch(model, loader, optimizer, cfg, device)
        metrics["epoch"] = epoch + 1
        metrics["lr"] = lr

        if (epoch + 1) % cfg.probe_every == 0:
            probe = run_site_probe(model, probe_dataset, device, seed=cfg.seed)
            metrics["site_probe_accuracy"] = probe["accuracy"]
            metrics["site_probe_balanced_accuracy"] = probe["balanced_accuracy"]
            metrics["site_probe_majority_accuracy"] = probe["majority_accuracy"]
            metrics["site_probe_majority_balanced_accuracy"] = probe["majority_balanced_accuracy"]
            logger.info(
                "Phase 1 Epoch %d | loss=%.4f jepa=%.4f vic=%.4f | "
                "site bAcc=%.1f%% acc=%.1f%% majority_acc=%.1f%%",
                epoch + 1,
                metrics["train_loss"],
                metrics["jepa_loss"],
                metrics["vicreg_loss"],
                100.0 * probe["balanced_accuracy"],
                100.0 * probe["accuracy"],
                100.0 * probe["majority_accuracy"],
            )
        else:
            logger.info(
                "Phase 1 Epoch %d | loss=%.4f jepa=%.4f vic=%.4f",
                epoch + 1,
                metrics["train_loss"],
                metrics["jepa_loss"],
                metrics["vicreg_loss"],
            )
        history.append(metrics)

    ckpt_path = save_dir / "phase1_final.pt"
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "model_state_dict": state_dict,
            "config": asdict(cfg),
            "history": history,
        },
        ckpt_path,
    )
    save_history(history, save_dir / "phase1_history.json")
    logger.info("Saved Phase 1 checkpoint: %s", ckpt_path)
    return ckpt_path


def train_phase2(cfg: CurriculumVICRegConfig, logger, phase1_checkpoint: Path):
    device = cfg.device
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    main_dirs = list(cfg.main_data_dirs)
    if cfg.include_abide:
        main_dirs.append(cfg.abide_data_dir)
    if cfg.include_abide2:
        main_dirs.append(cfg.abide2_data_dir)

    diag_filter = 0 if cfg.hc_only else None
    main_dataset = WindowedFMRIDataset(
        main_dirs,
        window_seconds=cfg.window_seconds,
        stride_seconds=cfg.window_stride_seconds,
        target_timepoints=cfg.target_timepoints,
        sampling=cfg.phase2_main_sampling,
        diag_filter=diag_filter,
        max_scans=cfg.max_main_scans,
    )
    replay_dataset = WindowedFMRIDataset(
        cfg.replay_data_dir,
        window_seconds=cfg.window_seconds,
        stride_seconds=cfg.window_stride_seconds,
        target_timepoints=cfg.target_timepoints,
        sampling=cfg.phase2_replay_sampling,
        max_scans=cfg.max_replay_scans,
    )
    probe_dataset = WindowedFMRIDataset(
        cfg.replay_data_dir,
        window_seconds=cfg.window_seconds,
        stride_seconds=cfg.window_stride_seconds,
        target_timepoints=cfg.target_timepoints,
        sampling=cfg.probe_sampling,
        max_scans=cfg.max_replay_scans,
    )

    main_batch = max(1, int(cfg.batch_size * (1.0 - cfg.replay_fraction)))
    replay_batch = max(1, cfg.batch_size - main_batch)
    sampler = make_site_balanced_sampler(main_dataset, seed=cfg.seed + 21)
    main_loader = DataLoader(
        main_dataset,
        batch_size=main_batch,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=make_worker_init_fn(cfg.seed + 22),
    )
    replay_buffer = ReplayBuffer(replay_dataset, replay_batch, device, seed=cfg.seed + 23)

    model = build_model(cfg).to(device)
    checkpoint = torch.load(phase1_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if device == "cuda" and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for Phase 2 via DataParallel")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.phase2_lr, weight_decay=cfg.weight_decay
    )

    history = []
    logger.info(
        "Phase 2 | main scans=%d replay scans=%d | hc_only=%s include_abide=%s include_abide2=%s",
        len(main_dataset),
        len(replay_dataset),
        cfg.hc_only,
        cfg.include_abide,
        cfg.include_abide2,
    )
    logger.info(
        "Phase 2 dataset counts | main=%s replay=%s",
        summarize_dataset_records(main_dataset.records),
        summarize_dataset_records(replay_dataset.records),
    )
    for epoch in range(cfg.phase2_epochs):
        lr = get_lr(epoch, cfg.warmup_epochs, cfg.phase2_epochs, cfg.phase2_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        metrics = run_epoch(
            model=model,
            loader=main_loader,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            replay_buffer=replay_buffer,
        )
        metrics["epoch"] = epoch + 1
        metrics["lr"] = lr

        if (epoch + 1) % cfg.probe_every == 0:
            probe = run_site_probe(model, probe_dataset, device, seed=cfg.seed)
            metrics["site_probe_accuracy"] = probe["accuracy"]
            metrics["site_probe_balanced_accuracy"] = probe["balanced_accuracy"]
            metrics["site_probe_majority_accuracy"] = probe["majority_accuracy"]
            metrics["site_probe_majority_balanced_accuracy"] = probe["majority_balanced_accuracy"]
            logger.info(
                "Phase 2 Epoch %d | loss=%.4f jepa=%.4f vic=%.4f | "
                "main=%.4f replay=%.4f | site bAcc=%.1f%% acc=%.1f%% majority_acc=%.1f%%",
                epoch + 1,
                metrics["train_loss"],
                metrics["jepa_loss"],
                metrics["vicreg_loss"],
                metrics["main_loss"],
                metrics["replay_loss"],
                100.0 * probe["balanced_accuracy"],
                100.0 * probe["accuracy"],
                100.0 * probe["majority_accuracy"],
            )
        else:
            logger.info(
                "Phase 2 Epoch %d | loss=%.4f jepa=%.4f vic=%.4f | main=%.4f replay=%.4f",
                epoch + 1,
                metrics["train_loss"],
                metrics["jepa_loss"],
                metrics["vicreg_loss"],
                metrics["main_loss"],
                metrics["replay_loss"],
            )
        history.append(metrics)

    ckpt_path = save_dir / "phase2_final.pt"
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "model_state_dict": state_dict,
            "config": asdict(cfg),
            "history": history,
        },
        ckpt_path,
    )
    save_history(history, save_dir / "phase2_history.json")
    logger.info("Saved Phase 2 checkpoint: %s", ckpt_path)
    return ckpt_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean curriculum JEPA + VICReg trainer with spatio-temporal masking"
    )
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--exclude-abide", action="store_true")
    parser.add_argument("--exclude-abide2", action="store_true")
    parser.add_argument("--hc-only", action="store_true")
    parser.add_argument("--include-abide", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--include-abide2", action="store_true")
    parser.add_argument("--no-hc-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--phase2-epochs", type=int, default=None)
    parser.add_argument("--phase1-lr", type=float, default=None)
    parser.add_argument("--phase2-lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--predictor-depth", type=int, default=None)
    parser.add_argument("--pos-embed-mode", choices=["learned", "sincos", "rotary"], default=None)
    parser.add_argument("--conv-stem", action="store_true")
    parser.add_argument("--no-conv-stem", action="store_true")
    parser.add_argument("--conv-kernel-size", type=int, default=None)
    parser.add_argument("--roi-embeddings", action="store_true")
    parser.add_argument("--no-roi-embeddings", action="store_true")
    parser.add_argument("--network-embeddings", action="store_true")
    parser.add_argument("--no-network-embeddings", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=None)
    parser.add_argument("--target-timepoints", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-main-scans", type=int, default=None)
    parser.add_argument("--max-replay-scans", type=int, default=None)
    parser.add_argument("--mask-mode", choices=["temporal", "spatiotemporal"], default=None)
    parser.add_argument("--phase1-sampling", choices=["random", "center", "sliding"], default=None)
    parser.add_argument("--phase2-main-sampling", choices=["random", "center", "sliding"], default=None)
    parser.add_argument("--phase2-replay-sampling", choices=["random", "center", "sliding"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = CurriculumVICRegConfig()

    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.device is not None:
        cfg.device = args.device
    if args.exclude_abide:
        cfg.include_abide = False
    if args.exclude_abide2:
        cfg.include_abide2 = False
    if args.include_abide:
        cfg.include_abide = True
    if args.include_abide2:
        cfg.include_abide2 = True
    if args.hc_only:
        cfg.hc_only = True
    if args.no_hc_only:
        cfg.hc_only = False
    if args.phase1_epochs is not None:
        cfg.phase1_epochs = args.phase1_epochs
    if args.phase2_epochs is not None:
        cfg.phase2_epochs = args.phase2_epochs
    if args.phase1_lr is not None:
        cfg.phase1_lr = args.phase1_lr
    if args.phase2_lr is not None:
        cfg.phase2_lr = args.phase2_lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.embed_dim is not None:
        cfg.embed_dim = args.embed_dim
    if args.predictor_depth is not None:
        cfg.predictor_depth = args.predictor_depth
    if args.pos_embed_mode is not None:
        cfg.pos_embed_mode = args.pos_embed_mode
    if args.conv_stem:
        cfg.use_conv_stem = True
    if args.no_conv_stem:
        cfg.use_conv_stem = False
    if args.conv_kernel_size is not None:
        cfg.conv_kernel_size = args.conv_kernel_size
    if args.roi_embeddings:
        cfg.use_roi_embeddings = True
    if args.no_roi_embeddings:
        cfg.use_roi_embeddings = False
    if args.network_embeddings:
        cfg.use_network_embeddings = True
    if args.no_network_embeddings:
        cfg.use_network_embeddings = False
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
    if args.max_replay_scans is not None:
        cfg.max_replay_scans = args.max_replay_scans
    if args.mask_mode is not None:
        cfg.mask_mode = args.mask_mode
    if args.phase1_sampling is not None:
        cfg.phase1_sampling = args.phase1_sampling
    if args.phase2_main_sampling is not None:
        cfg.phase2_main_sampling = args.phase2_main_sampling
    if args.phase2_replay_sampling is not None:
        cfg.phase2_replay_sampling = args.phase2_replay_sampling
    if args.seed is not None:
        cfg.seed = args.seed
    if args.deterministic:
        cfg.deterministic = True
    if args.non_deterministic:
        cfg.deterministic = False

    if args.quick:
        cfg.phase1_epochs = 1
        cfg.phase2_epochs = 1
        cfg.batch_size = min(cfg.batch_size, 8)
        cfg.steps_per_epoch = cfg.steps_per_epoch or 4
        cfg.max_main_scans = cfg.max_main_scans or 96
        cfg.max_replay_scans = cfg.max_replay_scans or 64
        cfg.probe_every = 1
        if args.save_dir is None:
            cfg.save_dir = "checkpoints/jepa_clean_curriculum_vicreg_quick"

    setup_logging(args.log_file)
    logger = logging.getLogger("JEPA-Clean")
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    logger.info("Config: %s", json.dumps(asdict(cfg), indent=2))
    phase1_ckpt = train_phase1(cfg, logger)
    train_phase2(cfg, logger, phase1_ckpt)


if __name__ == "__main__":
    main()
