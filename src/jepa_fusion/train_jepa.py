from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from ..jepa_clean.data import WindowedFMRIDataset, load_scan_records, make_subject_id
    from ..jepa_clean.repro import make_torch_generator, make_worker_init_fn, seed_everything
    from .masking import complement_indices, sample_rectangular_patch_mask, sample_temporal_mask
    from .model import FusionJEPA, build_model_from_config
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_clean.data import WindowedFMRIDataset, load_scan_records, make_subject_id
    from src.jepa_clean.repro import make_torch_generator, make_worker_init_fn, seed_everything
    from src.jepa_fusion.masking import complement_indices, sample_rectangular_patch_mask, sample_temporal_mask
    from src.jepa_fusion.model import FusionJEPA, build_model_from_config


MAIN_DATASET_DIRS = {
    "srpbs": "preprocessed_data_fir/full/roi_timeseries_1600_full",
    "abide1": "preprocessed_data_fir/full/roi_timeseries_abide_full",
    "abide2": "preprocessed_data_fir/full/roi_timeseries_abide2_full",
}
MAIN_DATASET_ORDER = ("srpbs", "abide1", "abide2")
MAIN_DATA_DIRS = [MAIN_DATASET_DIRS[key] for key in MAIN_DATASET_ORDER]
TRAVEL_DATA_DIR = "preprocessed_data_fir/full/roi_timeseries_traveling_full"


@dataclass
class FusionJEPAConfig:
    main_dataset_keys: list[str] = field(default_factory=lambda: list(MAIN_DATASET_ORDER))
    main_data_dirs: list[str] = field(default_factory=lambda: list(MAIN_DATA_DIRS))
    travel_data_dir: str = TRAVEL_DATA_DIR

    training_mode: str = "pooled"
    model_family: str = "temporal"
    temporal_variant: str = "conv_net"
    mask_profile: str = "temporal_default"

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
    use_conv_stem: bool = True
    conv_kernel_size: int = 3
    use_network_embeddings: bool = True

    st_time_patch: int = 5
    st_roi_group_size: int = 40
    st_reorder_mode: str = "hemi_network"

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

    vicreg_weight: float = 0.1
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 1.0
    vicreg_var_target: float = 1.0
    global_loss_weight: float = 0.0
    replay_alignment_weight: float = 0.0
    replay_pair_sampling: bool = False
    adversarial_site_weight: float = 0.0
    site_adversary_hidden_dim: int = 256
    infonce_weight: float = 0.0
    infonce_temperature: float = 0.07
    site_grl_lambda: float = 1.0
    num_site_classes: int = 0
    site_class_names: list[str] = field(default_factory=list)

    mask_num_time_blocks: int = 2
    mask_time_block_size_min: int = 6
    mask_time_block_size_max: int = 10
    st_mask_num_time_blocks: int = 2
    st_mask_time_block_size: int = 2
    st_mask_num_roi_blocks: int = 2
    st_mask_roi_block_size: int = 3

    save_dir: str = "checkpoints/jepa_fusion/default"
    seed: int = 42
    deterministic: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_data_parallel: bool = False


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


def collate_fn(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), list(labels)


def summarize_records(records):
    counts = {}
    for record in records:
        key = f"{record.dataset}:{record.site}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def parse_dataset_keys(text: str):
    keys = []
    for item in text.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in MAIN_DATASET_DIRS:
            raise ValueError(
                f"Unsupported dataset key: {key!r}. Expected one of {sorted(MAIN_DATASET_DIRS)}."
            )
        keys.append(key)
    if not keys:
        raise ValueError("At least one main dataset key is required.")
    deduped = []
    seen = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def resolve_main_data_dirs(dataset_keys: list[str]):
    return [MAIN_DATASET_DIRS[key] for key in dataset_keys]


def make_dataset_site_key(dataset_name: str, site_name: str):
    return f"{dataset_name}:{site_name}"


def configure_site_classes(cfg: FusionJEPAConfig, records):
    if cfg.adversarial_site_weight <= 0.0:
        cfg.num_site_classes = 0
        cfg.site_class_names = []
        return
    site_names = sorted({make_dataset_site_key(record.dataset, record.site) for record in records})
    cfg.site_class_names = list(site_names)
    cfg.num_site_classes = len(site_names)


def encode_site_targets(labels, site_to_index: dict[str, int], device: str):
    targets = [site_to_index[make_dataset_site_key(label["dataset"], label["site"])] for label in labels]
    return torch.as_tensor(targets, dtype=torch.long, device=device)


def collect_record_union(*record_lists):
    by_scan = {}
    for records in record_lists:
        for record in records:
            by_scan[(record.dataset, record.scan_id)] = record
    return list(by_scan.values())


def resolve_architecture(cfg: FusionJEPAConfig):
    if cfg.model_family == "st_roi_patch":
        cfg.pos_embed_mode = "sincos"
        cfg.use_conv_stem = False
        cfg.use_network_embeddings = False
        cfg.mask_profile = "st_rect_moderate"
        return

    variant_defaults = {
        "basic": ("sincos", False, False),
        "conv_net": ("sincos", True, True),
        "rotary_conv_net": ("rotary", True, True),
    }
    if cfg.temporal_variant not in variant_defaults:
        raise ValueError(f"Unsupported temporal_variant: {cfg.temporal_variant}")
    pos_embed_mode, use_conv_stem, use_network_embeddings = variant_defaults[cfg.temporal_variant]
    cfg.pos_embed_mode = pos_embed_mode
    cfg.use_conv_stem = use_conv_stem
    cfg.use_network_embeddings = use_network_embeddings
    cfg.mask_profile = "temporal_default"


def build_model(cfg: FusionJEPAConfig):
    return build_model_from_config(asdict(cfg))


def make_dataset_site_sampler(dataset: WindowedFMRIDataset, seed: int):
    if dataset.window_records is not None:
        raise ValueError("Dataset-site sampler expects a per-scan dataset, not a sliding-window dataset.")
    counts = {}
    for record in dataset.records:
        key = (record.dataset, record.site)
        counts[key] = counts.get(key, 0) + 1
    weights = [1.0 / counts[(record.dataset, record.site)] for record in dataset.records]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=make_torch_generator(seed),
    )


class ReplayBuffer:
    def __init__(
        self,
        dataset: WindowedFMRIDataset,
        batch_size: int,
        device: str,
        seed: int,
        pair_sampling: bool = False,
    ):
        self.device = device
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.pair_sampling = bool(pair_sampling)
        self.rng = np.random.default_rng(seed)
        self.all_indices = np.arange(len(dataset), dtype=int)

        if self.pair_sampling:
            subject_site_to_indices = defaultdict(lambda: defaultdict(list))
            for idx in range(len(dataset)):
                if dataset.window_records is None:
                    record = dataset.records[idx]
                else:
                    record = dataset.window_records[idx].record
                subject_site_to_indices[make_subject_id(record)][record.site].append(idx)

            self.valid_pair_subjects = [
                subject_id
                for subject_id, site_to_indices in subject_site_to_indices.items()
                if len(site_to_indices) >= 2
            ]
            self.subject_site_to_indices = {
                subject_id: {site: np.asarray(indices, dtype=int) for site, indices in site_to_indices.items()}
                for subject_id, site_to_indices in subject_site_to_indices.items()
            }
            if self.valid_pair_subjects:
                self.loader = None
                self.iterator = None
            else:
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
        else:
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
            self.valid_pair_subjects = []
            self.subject_site_to_indices = {}

    def sample(self):
        if self.pair_sampling and self.valid_pair_subjects:
            indices = []
            pair_ids = []
            n_pairs = self.batch_size // 2
            chosen_subjects = self.rng.choice(self.valid_pair_subjects, size=n_pairs, replace=True)
            for pair_id, subject_id in enumerate(chosen_subjects.tolist()):
                site_to_indices = self.subject_site_to_indices[subject_id]
                sites = self.rng.choice(list(site_to_indices.keys()), size=2, replace=False)
                for site in sites.tolist():
                    indices.append(int(self.rng.choice(site_to_indices[site])))
                    pair_ids.append(pair_id)

            while len(indices) < self.batch_size:
                indices.append(int(self.rng.choice(self.all_indices)))
                pair_ids.append(-1)

            batch = [self.dataset[idx] for idx in indices]
            x, labels = collate_fn(batch)
            return x.to(self.device), labels, pair_ids

        try:
            x, labels = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            x, labels = next(self.iterator)
        return x.to(self.device), labels, None


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


def replay_pair_alignment_loss(embeddings: torch.Tensor, pair_ids):
    if pair_ids is None or embeddings.size(0) < 2:
        return embeddings.new_zeros(())

    pair_ids = torch.as_tensor(pair_ids, device=embeddings.device, dtype=torch.long)
    valid_pair_ids = torch.unique(pair_ids[pair_ids >= 0])
    if valid_pair_ids.numel() == 0:
        return embeddings.new_zeros(())

    normalized = F.normalize(embeddings, dim=-1)
    losses = []
    for pair_id in valid_pair_ids:
        idx = torch.nonzero(pair_ids == pair_id, as_tuple=False).flatten()
        if idx.numel() != 2:
            continue
        similarity = F.cosine_similarity(
            normalized[idx[0]].unsqueeze(0),
            normalized[idx[1]].unsqueeze(0),
            dim=-1,
        )
        losses.append(1.0 - similarity.mean())

    if not losses:
        return embeddings.new_zeros(())
    return torch.stack(losses).mean()


def infonce_loss(embeddings: torch.Tensor, pair_ids, temperature: float = 0.07):
    """InfoNCE contrastive loss on traveling-subject cross-site pairs.

    For each pair (anchor, positive) from the same subject at different sites,
    all other embeddings in the batch serve as negatives.
    """
    if pair_ids is None or embeddings.size(0) < 2:
        return embeddings.new_zeros(())

    pair_ids = torch.as_tensor(pair_ids, device=embeddings.device, dtype=torch.long)
    valid_mask = pair_ids >= 0
    if valid_mask.sum() < 2:
        return embeddings.new_zeros(())

    normalized = F.normalize(embeddings, dim=-1)
    sim_matrix = normalized @ normalized.T / temperature

    losses = []
    valid_pair_ids = torch.unique(pair_ids[valid_mask])
    for pid in valid_pair_ids:
        idx = torch.nonzero(pair_ids == pid, as_tuple=False).flatten()
        if idx.numel() != 2:
            continue
        for anchor_pos in range(2):
            anchor = idx[anchor_pos]
            positive = idx[1 - anchor_pos]
            numerator = sim_matrix[anchor, positive]
            mask = torch.ones(embeddings.size(0), dtype=torch.bool, device=embeddings.device)
            mask[anchor] = False
            denominator = torch.logsumexp(sim_matrix[anchor][mask], dim=0)
            losses.append(denominator - numerator)

    if not losses:
        return embeddings.new_zeros(())
    return torch.stack(losses).mean()


def sample_mask(cfg: FusionJEPAConfig, raw_model: FusionJEPA, device: torch.device):
    if cfg.mask_profile == "temporal_default":
        mask = sample_temporal_mask(
            seq_len=raw_model.num_tokens,
            num_time_blocks=cfg.mask_num_time_blocks,
            time_block_size=(cfg.mask_time_block_size_min, cfg.mask_time_block_size_max),
        )
        target_indices = mask.target_indices.to(device)
    elif cfg.mask_profile == "st_rect_moderate":
        if raw_model.model_family != "st_roi_patch":
            raise ValueError("st_rect_moderate mask_profile requires model_family=st_roi_patch")
        tokenizer = raw_model.token_embed
        mask = sample_rectangular_patch_mask(
            num_time_patches=tokenizer.num_time_patches,
            num_roi_groups=tokenizer.num_roi_groups,
            num_time_blocks=cfg.st_mask_num_time_blocks,
            time_block_size=cfg.st_mask_time_block_size,
            num_roi_blocks=cfg.st_mask_num_roi_blocks,
            roi_block_size=cfg.st_mask_roi_block_size,
        )
        target_indices = mask.target_indices.to(device)
    else:
        raise ValueError(f"Unsupported mask_profile: {cfg.mask_profile}")

    context_indices = complement_indices(raw_model.num_tokens, target_indices, device=device)
    return context_indices, target_indices


def run_epoch(
    model,
    loader,
    optimizer,
    cfg: FusionJEPAConfig,
    device: str,
    replay_buffer: ReplayBuffer | None = None,
    site_to_index: dict[str, int] | None = None,
):
    model.train()
    total_losses = []
    jepa_losses = []
    global_losses = []
    vic_losses = []
    vic_var_losses = []
    vic_cov_losses = []
    replay_alignment_losses = []
    infonce_losses = []
    site_adv_losses = []
    site_adv_accs = []
    main_losses = []
    replay_losses = []

    use_site_adversary = site_to_index is not None and cfg.adversarial_site_weight > 0.0

    for batch_idx, (x_main, labels_main) in enumerate(loader):
        x_main = x_main.to(device)
        replay_pair_ids = None
        replay_batch_size = 0
        labels_replay = []
        if replay_buffer is not None:
            x_replay, labels_replay, replay_pair_ids = replay_buffer.sample()
            x_batch = torch.cat([x_main, x_replay], dim=0)
            n_main = x_main.size(0)
            replay_batch_size = x_replay.size(0)
        else:
            x_batch = x_main
            n_main = x_batch.size(0)

        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        context_indices, target_indices = sample_mask(cfg, raw_model, x_batch.device)
        ctx_batched = context_indices.unsqueeze(0).expand(x_batch.size(0), -1)
        tgt_batched = target_indices.unsqueeze(0).expand(x_batch.size(0), -1)

        model_output = model(
            x_batch,
            ctx_batched,
            tgt_batched,
            return_global=True,
            return_site_logits=use_site_adversary,
        )
        if use_site_adversary:
            predicted, target, pooled, pred_global, target_global, site_logits = model_output
        else:
            predicted, target, pooled, pred_global, target_global = model_output
            site_logits = None
        jepa_loss = F.mse_loss(predicted, target)
        global_loss = F.mse_loss(pred_global, target_global)
        vic_loss, vic_var, vic_cov = vicreg_regularization(
            pooled,
            var_weight=cfg.vicreg_var_weight,
            cov_weight=cfg.vicreg_cov_weight,
            var_target=cfg.vicreg_var_target,
        )
        replay_alignment = pooled.new_zeros(())
        nce_loss = pooled.new_zeros(())
        if (
            replay_buffer is not None
            and replay_batch_size > 0
            and replay_pair_ids is not None
        ):
            replay_embeddings = pooled[n_main : n_main + replay_batch_size]
            if cfg.replay_alignment_weight > 0.0:
                replay_alignment = replay_pair_alignment_loss(replay_embeddings, replay_pair_ids)
            if cfg.infonce_weight > 0.0:
                nce_loss = infonce_loss(replay_embeddings, replay_pair_ids, temperature=cfg.infonce_temperature)
        site_adv_loss = pooled.new_zeros(())
        if site_logits is not None:
            site_targets = encode_site_targets(list(labels_main) + list(labels_replay), site_to_index, device)
            site_adv_loss = F.cross_entropy(site_logits, site_targets)
            with torch.no_grad():
                site_pred = site_logits.argmax(dim=1)
                site_adv_accs.append(float((site_pred == site_targets).float().mean().item()))

        total_loss = (
            jepa_loss
            + (cfg.global_loss_weight * global_loss)
            + (cfg.vicreg_weight * vic_loss)
            + (cfg.replay_alignment_weight * replay_alignment)
            + (cfg.infonce_weight * nce_loss)
            + (cfg.adversarial_site_weight * site_adv_loss)
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
        vic_var_losses.append(vic_var.item())
        vic_cov_losses.append(vic_cov.item())
        replay_alignment_losses.append(replay_alignment.item())
        infonce_losses.append(nce_loss.item())
        site_adv_losses.append(site_adv_loss.item())

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
        "vicreg_var_loss": float(np.mean(vic_var_losses)),
        "vicreg_cov_loss": float(np.mean(vic_cov_losses)),
        "replay_alignment_loss": float(np.mean(replay_alignment_losses)),
        "infonce_loss": float(np.mean(infonce_losses)),
        "site_adversary_loss": float(np.mean(site_adv_losses)),
        "site_adversary_acc": float(np.mean(site_adv_accs)) if site_adv_accs else None,
    }
    if replay_buffer is not None:
        metrics["main_loss"] = float(np.mean(main_losses)) if main_losses else None
        metrics["replay_loss"] = float(np.mean(replay_losses)) if replay_losses else None
    return metrics


def save_history(history, path: Path):
    with path.open("w") as handle:
        json.dump(history, handle, indent=2)


def save_checkpoint(model, cfg: FusionJEPAConfig, history, path: Path, extra: dict | None = None):
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    payload = {
        "model_state_dict": state_dict,
        "config": asdict(cfg),
        "history": history,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def build_dataset(
    data_dirs,
    *,
    sampling: str,
    cfg: FusionJEPAConfig,
    max_scans: int | None,
):
    return WindowedFMRIDataset(
        data_dirs,
        window_seconds=cfg.window_seconds,
        stride_seconds=cfg.window_stride_seconds,
        target_timepoints=cfg.target_timepoints,
        sampling=sampling,
        max_scans=max_scans,
    )


def train_pooled(cfg: FusionJEPAConfig, logger: logging.Logger):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pooled_dirs = list(cfg.main_data_dirs) + [cfg.travel_data_dir]
    dataset = build_dataset(
        pooled_dirs,
        sampling=cfg.pooled_sampling,
        cfg=cfg,
        max_scans=cfg.max_main_scans,
    )
    configure_site_classes(cfg, dataset.records)
    site_to_index = {name: idx for idx, name in enumerate(cfg.site_class_names)}
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

    # Build a pair-sampling buffer from traveling subjects for InfoNCE / alignment
    pair_buffer = None
    needs_pair_buffer = cfg.infonce_weight > 0.0 or cfg.replay_alignment_weight > 0.0
    if needs_pair_buffer:
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
    elif cfg.device == "cuda" and torch.cuda.device_count() > 1:
        logger.info(
            "Visible GPUs: %d | single-GPU mode active because DataParallel is disabled",
            torch.cuda.device_count(),
        )

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
        if site_to_index:
            metrics = run_epoch(model, loader, optimizer, cfg, cfg.device, replay_buffer=pair_buffer, site_to_index=site_to_index)
        else:
            metrics = run_epoch(model, loader, optimizer, cfg, cfg.device, replay_buffer=pair_buffer)
        metrics["epoch"] = epoch + 1
        metrics["lr"] = lr
        history.append(metrics)
        logger.info(
            "Pooled Epoch %d | loss=%.4f jepa=%.4f global=%.4f vic=%.4f align=%.4f nce=%.4f adv=%.4f",
            epoch + 1,
            metrics["train_loss"],
            metrics["jepa_loss"],
            metrics["global_loss"],
            metrics["vicreg_loss"],
            metrics["replay_alignment_loss"],
            metrics["infonce_loss"],
            metrics["site_adversary_loss"],
        )

    ckpt_path = save_dir / "pooled_final.pt"
    save_checkpoint(model, cfg, history, ckpt_path)
    save_history(history, save_dir / "pooled_history.json")
    logger.info("Saved pooled checkpoint: %s", ckpt_path)
    return ckpt_path


def train_curriculum(cfg: FusionJEPAConfig, logger: logging.Logger):
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
    configure_site_classes(
        cfg,
        collect_record_union(phase1_dataset.records, phase2_main_dataset.records, phase2_replay_dataset.records),
    )
    site_to_index = {name: idx for idx, name in enumerate(cfg.site_class_names)}
    use_pair_losses = cfg.replay_alignment_weight > 0.0 or cfg.infonce_weight > 0.0
    phase1_main_batch = cfg.batch_size
    phase1_pair_buffer = None
    if use_pair_losses and cfg.replay_fraction > 0.0:
        phase1_main_batch = max(1, int(cfg.batch_size * (1.0 - cfg.replay_fraction)))
        phase1_replay_batch = max(1, cfg.batch_size - phase1_main_batch)
        phase1_pair_buffer = ReplayBuffer(
            phase1_dataset,
            batch_size=phase1_replay_batch,
            device=cfg.device,
            seed=cfg.seed + 22,
            pair_sampling=True,
        )
    else:
        phase1_replay_batch = 0

    phase1_loader = DataLoader(
        phase1_dataset,
        batch_size=phase1_main_batch,
        shuffle=True,
        drop_last=phase1_pair_buffer is not None,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=make_worker_init_fn(cfg.seed + 21),
        generator=make_torch_generator(cfg.seed + 21),
    )

    model = build_model(cfg).to(cfg.device)
    if cfg.device == "cuda" and torch.cuda.device_count() > 1 and cfg.use_data_parallel:
        logger.info("Using %d GPUs via DataParallel", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
    elif cfg.device == "cuda" and torch.cuda.device_count() > 1:
        logger.info(
            "Visible GPUs: %d | single-GPU mode active because DataParallel is disabled",
            torch.cuda.device_count(),
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.phase1_lr, weight_decay=cfg.weight_decay)
    phase1_history = []
    logger.info(
        "Phase 1 | travel scans=%d | dataset_site_counts=%s | n_site_classes=%d adv_w=%.3f pair_sampling=%s align_w=%.3f nce_w=%.3f main_batch=%d replay_batch=%d",
        len(phase1_dataset),
        summarize_records(phase1_dataset.records),
        cfg.num_site_classes,
        cfg.adversarial_site_weight,
        phase1_pair_buffer is not None,
        cfg.replay_alignment_weight,
        cfg.infonce_weight,
        phase1_main_batch,
        phase1_replay_batch,
    )
    for epoch in range(cfg.phase1_epochs):
        lr = get_lr(epoch, cfg.warmup_epochs, cfg.phase1_epochs, cfg.phase1_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        metrics = run_epoch(
            model,
            phase1_loader,
            optimizer,
            cfg,
            cfg.device,
            replay_buffer=phase1_pair_buffer,
            site_to_index=site_to_index or None,
        )
        metrics["epoch"] = epoch + 1
        metrics["lr"] = lr
        phase1_history.append(metrics)
        logger.info(
            "Phase 1 Epoch %d | loss=%.4f jepa=%.4f global=%.4f vic=%.4f align=%.4f nce=%.4f adv=%.4f",
            epoch + 1,
            metrics["train_loss"],
            metrics["jepa_loss"],
            metrics["global_loss"],
            metrics["vicreg_loss"],
            metrics["replay_alignment_loss"],
            metrics["infonce_loss"],
            metrics["site_adversary_loss"],
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
            pair_sampling=cfg.replay_pair_sampling,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.phase2_lr, weight_decay=cfg.weight_decay)
    phase2_history = []
    logger.info(
        "Phase 2 | main scans=%d replay scans=%d | pair_sampling=%s global_w=%.3f align_w=%.3f | main dataset_site_counts=%s",
        len(phase2_main_dataset),
        len(phase2_replay_dataset),
        cfg.replay_pair_sampling,
        cfg.global_loss_weight,
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
            site_to_index=site_to_index or None,
        )
        metrics["epoch"] = epoch + 1
        metrics["lr"] = lr
        phase2_history.append(metrics)
        logger.info(
            "Phase 2 Epoch %d | loss=%.4f jepa=%.4f global=%.4f vic=%.4f align=%.4f nce=%.4f adv=%.4f | main=%s replay=%s",
            epoch + 1,
            metrics["train_loss"],
            metrics["jepa_loss"],
            metrics["global_loss"],
            metrics["vicreg_loss"],
            metrics["replay_alignment_loss"],
            metrics["infonce_loss"],
            metrics["site_adversary_loss"],
            "n/a" if metrics.get("main_loss") is None else f"{metrics['main_loss']:.4f}",
            "n/a" if metrics.get("replay_loss") is None else f"{metrics['replay_loss']:.4f}",
        )

    phase2_ckpt = save_dir / "phase2_final.pt"
    save_checkpoint(model, cfg, phase2_history, phase2_ckpt, extra={"phase": "phase2"})
    save_history(phase2_history, save_dir / "phase2_history.json")
    logger.info("Saved phase 2 checkpoint: %s", phase2_ckpt)
    return phase1_ckpt, phase2_ckpt


def parse_args():
    parser = argparse.ArgumentParser(description="Train clean JEPA models for the jepa_fusion branch.")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--main-datasets",
        default=None,
        help="Comma-separated main dataset keys from {srpbs,abide1,abide2}. Default uses all three.",
    )
    parser.add_argument(
        "--travel-data-dir",
        default=None,
        help="Optional override for the travelling-subject ROI directory.",
    )
    parser.add_argument("--training-mode", choices=["pooled", "curriculum"], default=None)
    parser.add_argument("--model-family", choices=["temporal", "st_roi_patch"], default=None)
    parser.add_argument("--temporal-variant", choices=["basic", "conv_net", "rotary_conv_net"], default=None)
    parser.add_argument("--mask-profile", choices=["temporal_default", "st_rect_moderate"], default=None)
    parser.add_argument("--pos-embed-mode", choices=["sincos", "rotary"], default=None)
    parser.add_argument("--use-conv-stem", action="store_true")
    parser.add_argument("--no-conv-stem", action="store_true")
    parser.add_argument("--conv-kernel-size", type=int, default=None)
    parser.add_argument("--use-network-embeddings", action="store_true")
    parser.add_argument("--no-network-embeddings", action="store_true")
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
    parser.add_argument("--st-time-patch", type=int, default=None)
    parser.add_argument("--st-roi-group-size", type=int, default=None)
    parser.add_argument("--global-loss-weight", type=float, default=None)
    parser.add_argument("--replay-alignment-weight", type=float, default=None)
    parser.add_argument("--replay-pair-sampling", action="store_true")
    parser.add_argument("--infonce-weight", type=float, default=None)
    parser.add_argument("--infonce-temperature", type=float, default=None)
    parser.add_argument("--adversarial-site-weight", type=float, default=None)
    parser.add_argument("--site-adversary-hidden-dim", type=int, default=None)
    parser.add_argument("--site-grl-lambda", type=float, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-main-scans", type=int, default=None)
    parser.add_argument("--max-travel-scans", type=int, default=None)
    parser.add_argument("--replay-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = FusionJEPAConfig()
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
    if args.model_family is not None:
        cfg.model_family = args.model_family
    if args.temporal_variant is not None:
        cfg.temporal_variant = args.temporal_variant

    resolve_architecture(cfg)

    if args.mask_profile is not None:
        cfg.mask_profile = args.mask_profile
    if args.pos_embed_mode is not None:
        cfg.pos_embed_mode = args.pos_embed_mode
    if args.use_conv_stem:
        cfg.use_conv_stem = True
    if args.no_conv_stem:
        cfg.use_conv_stem = False
    if args.conv_kernel_size is not None:
        cfg.conv_kernel_size = args.conv_kernel_size
    if args.use_network_embeddings:
        cfg.use_network_embeddings = True
    if args.no_network_embeddings:
        cfg.use_network_embeddings = False
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
    if args.st_time_patch is not None:
        cfg.st_time_patch = args.st_time_patch
    if args.st_roi_group_size is not None:
        cfg.st_roi_group_size = args.st_roi_group_size
    if args.global_loss_weight is not None:
        cfg.global_loss_weight = args.global_loss_weight
    if args.replay_alignment_weight is not None:
        cfg.replay_alignment_weight = args.replay_alignment_weight
        if cfg.replay_alignment_weight > 0.0:
            cfg.replay_pair_sampling = True
    if args.replay_pair_sampling:
        cfg.replay_pair_sampling = True
    if args.infonce_weight is not None:
        cfg.infonce_weight = args.infonce_weight
        if cfg.infonce_weight > 0.0:
            cfg.replay_pair_sampling = True
    if args.infonce_temperature is not None:
        cfg.infonce_temperature = args.infonce_temperature
    if args.adversarial_site_weight is not None:
        cfg.adversarial_site_weight = args.adversarial_site_weight
    if args.site_adversary_hidden_dim is not None:
        cfg.site_adversary_hidden_dim = args.site_adversary_hidden_dim
    if args.site_grl_lambda is not None:
        cfg.site_grl_lambda = args.site_grl_lambda
    if args.steps_per_epoch is not None:
        cfg.steps_per_epoch = args.steps_per_epoch
    if args.max_main_scans is not None:
        cfg.max_main_scans = args.max_main_scans
    if args.max_travel_scans is not None:
        cfg.max_travel_scans = args.max_travel_scans
    if args.replay_fraction is not None:
        cfg.replay_fraction = args.replay_fraction
    if args.seed is not None:
        cfg.seed = args.seed
    if args.deterministic:
        cfg.deterministic = True
    if args.non_deterministic:
        cfg.deterministic = False
    if args.data_parallel:
        cfg.use_data_parallel = True

    if args.quick:
        cfg.pooled_epochs = 1
        cfg.phase1_epochs = 1
        cfg.phase2_epochs = 1
        cfg.batch_size = min(cfg.batch_size, 8)
        cfg.steps_per_epoch = cfg.steps_per_epoch or 2
        cfg.max_main_scans = cfg.max_main_scans or 48
        cfg.max_travel_scans = cfg.max_travel_scans or 32
        if args.save_dir is None:
            cfg.save_dir = f"checkpoints/jepa_fusion/quick_{cfg.model_family}_{cfg.training_mode}"

    setup_logging(args.log_file)
    logger = logging.getLogger("JEPA-Fusion-Train")
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    logger.info("Config: %s", json.dumps(asdict(cfg), indent=2))

    if cfg.training_mode == "pooled":
        train_pooled(cfg, logger)
    elif cfg.training_mode == "curriculum":
        train_curriculum(cfg, logger)
    else:
        raise ValueError(f"Unsupported training_mode: {cfg.training_mode}")


if __name__ == "__main__":
    main()
