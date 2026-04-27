import argparse
import copy
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import (
    compute_sliding_windows,
    load_scan_records,
    make_subject_id,
    resample_time_window,
)
from .evaluate_subject_vote import (
    TASKS,
    add_site_std_to_summary,
    build_model_from_checkpoint,
    compute_binary_metrics,
    compute_site_level_metrics,
)
from .plot_tsne_fc_vs_raw import run_grouped_probe
from .repro import make_torch_generator, make_worker_init_fn, seed_everything


@dataclass
class SubjectPhase3Config:
    checkpoint_path: str = "checkpoints/jepa_clean_curriculum_vicreg_temporal_allsubjects/phase2_final.pt"
    data_dirs: list[str] = field(
        default_factory=lambda: [
            "preprocessed_data_fir/full/roi_timeseries_1600_full",
            "preprocessed_data_fir/full/roi_timeseries_abide_full",
        ]
    )
    tasks: list[str] = field(default_factory=lambda: ["hc_all"])
    held_out_sites: list[str] | None = None
    held_out_datasets: list[str] | None = None
    mixed_sites_only_train: bool = False
    mixed_sites_only_test: bool = False

    window_seconds: float = 90.0
    window_stride_seconds: float = 30.0
    target_timepoints: int | None = None

    ft_epochs: int = 20
    ft_batch_size: int = 8
    ft_encoder_lr: float = 1e-5
    ft_head_lr: float = 1e-3
    ft_weight_decay: float = 0.01
    ft_warmup_epochs: int = 3
    early_stop_patience: int = 5
    val_fraction: float = 0.2
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.1

    pooling_mode: str = "mean"
    attention_hidden: int = 64
    calibration_mode: str = "none"
    train_windows_per_subject: int | None = None

    head_hidden: int = 128
    head_dropout: float = 0.3

    max_subjects_per_class: int | None = None
    save_dir: str = "results/jepa_clean_phase3_subject_pooling"
    seed: int = 42
    deterministic: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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


class DiagnosisHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


class AttentionPool(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, window_embeddings: torch.Tensor, counts: list[int]):
        subject_embeddings = []
        offset = 0
        for count in counts:
            window_chunk = window_embeddings[offset:offset + count]
            offset += count

            if count == 1:
                subject_embeddings.append(window_chunk[0])
                continue

            attn_logits = self.scorer(window_chunk).squeeze(-1)
            attn_weights = torch.softmax(attn_logits, dim=0)
            pooled = (attn_weights[:, None] * window_chunk).sum(dim=0)
            subject_embeddings.append(pooled)

        return torch.stack(subject_embeddings, dim=0)


class SubjectPoolModel(nn.Module):
    def __init__(
        self,
        encoder_model,
        embed_dim: int,
        head_hidden: int,
        head_dropout: float,
        pooling_mode: str = "mean",
        attention_hidden: int = 64,
    ):
        super().__init__()
        self.token_embed = encoder_model.token_embed
        self.context_encoder = encoder_model.context_encoder
        self.pooling_mode = pooling_mode
        self.attention_pool = None
        if self.pooling_mode == "attention":
            self.attention_pool = AttentionPool(embed_dim=embed_dim, hidden_dim=attention_hidden)
        self.head = DiagnosisHead(embed_dim, hidden_dim=head_hidden, dropout=head_dropout)

    def encode_windows(self, windows: torch.Tensor):
        seq_len = windows.size(1)
        indices = torch.arange(seq_len, device=windows.device)
        embedded = self.token_embed(windows, indices)
        rep = self.context_encoder(embedded)
        return rep.mean(dim=1)

    def aggregate_subjects(self, window_embeddings: torch.Tensor, counts: list[int]):
        if self.pooling_mode == "attention":
            return self.attention_pool(window_embeddings, counts)

        subject_embeddings = []
        offset = 0
        for count in counts:
            subject_embeddings.append(window_embeddings[offset:offset + count].mean(dim=0))
            offset += count
        return torch.stack(subject_embeddings, dim=0)

    def forward(self, windows: torch.Tensor, counts: list[int]):
        window_embeddings = self.encode_windows(windows)
        subject_embeddings = self.aggregate_subjects(window_embeddings, counts)
        logits = self.head(subject_embeddings)
        return logits, subject_embeddings


def get_lr(epoch: int, warmup_epochs: int, total_epochs: int, base_lr: float):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
):
    if embeddings.size(0) < 2:
        return embeddings.new_zeros(())

    z = F.normalize(embeddings, dim=-1)
    logits = torch.matmul(z, z.T) / max(temperature, 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.T).float()
    identity = torch.eye(embeddings.size(0), device=embeddings.device, dtype=positive_mask.dtype)
    positive_mask = positive_mask - identity

    logits_mask = 1.0 - identity
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return embeddings.new_zeros(())

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)
    loss = -mean_log_prob_pos[valid].mean()
    return loss


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in np.linspace(0.1, 0.9, num=81):
        preds = (y_prob >= threshold).astype(int)
        score = compute_binary_metrics(y_true, preds, y_prob)["bacc"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _sigmoid(x: np.ndarray):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


class IdentityCalibrator:
    name = "none"

    def fit(self, margins: np.ndarray, y_true: np.ndarray):
        return self

    def predict_proba(self, margins: np.ndarray):
        return _sigmoid(margins)

    def describe(self):
        return "identity"


class TemperatureCalibrator:
    name = "temperature"

    def __init__(self):
        self.temperature = 1.0

    def fit(self, margins: np.ndarray, y_true: np.ndarray):
        if len(margins) == 0 or len(np.unique(y_true)) < 2:
            self.temperature = 1.0
            return self

        logits = torch.tensor(margins, dtype=torch.float32)
        targets = torch.tensor(y_true, dtype=torch.float32)
        log_temp = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            temp = torch.exp(log_temp).clamp(min=1e-3, max=100.0)
            loss = F.binary_cross_entropy_with_logits(logits / temp, targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temp).clamp(min=1e-3, max=100.0).item())
        return self

    def predict_proba(self, margins: np.ndarray):
        return _sigmoid(margins / max(self.temperature, 1e-6))

    def describe(self):
        return f"temperature={self.temperature:.3f}"


def build_calibrator(mode: str):
    if mode == "none":
        return IdentityCalibrator()
    if mode == "temperature":
        return TemperatureCalibrator()
    raise ValueError(f"Unsupported calibration_mode: {mode}")


def build_subject_entries(
    data_dirs,
    window_seconds: float,
    stride_seconds: float,
):
    records = load_scan_records(data_dirs)
    by_subject = {}

    for record in records:
        subject_id = make_subject_id(record)
        windows = compute_sliding_windows(
            n_timepoints=record.n_timepoints,
            tr=record.tr,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        )
        entry = by_subject.setdefault(
            subject_id,
            {
                "subject_id": subject_id,
                "subject": record.subject,
                "dataset": record.dataset,
                "site": record.site,
                "diag": record.diag,
                "diag_name": record.diag_name,
                "records": [],
            },
        )
        if entry["site"] != record.site:
            raise ValueError(f"Subject {subject_id} appears at multiple sites in main data")
        if entry["diag"] != record.diag:
            raise ValueError(f"Subject {subject_id} appears with inconsistent diagnosis labels")
        entry["records"].append((record, windows))

    entries = list(by_subject.values())
    entries.sort(key=lambda item: (item["site"], item["subject_id"]))
    return entries


def filter_subject_entries(entries, task_key: str):
    task_name, hc_diag, disorder_diag = TASKS[task_key]
    filtered = []
    for entry in entries:
        diag = int(entry["diag"])
        if disorder_diag == "ALL":
            y = int(diag != hc_diag)
        elif diag in {hc_diag, disorder_diag}:
            y = int(diag == disorder_diag)
        else:
            continue

        copied = dict(entry)
        copied["y"] = y
        filtered.append(copied)

    return task_name, filtered


def maybe_subsample_subject_entries(entries, max_subjects_per_class: int | None, seed: int):
    if max_subjects_per_class is None:
        return entries

    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for entry in entries:
        by_class[int(entry["y"])].append(entry)

    selected = []
    for cls in sorted(by_class):
        cls_entries = by_class[cls]
        if len(cls_entries) > max_subjects_per_class:
            idx = rng.choice(len(cls_entries), size=max_subjects_per_class, replace=False)
            cls_entries = [cls_entries[i] for i in idx]
        selected.extend(cls_entries)

    selected.sort(key=lambda item: (item["site"], item["subject_id"]))
    return selected


def filter_entries_to_mixed_sites(entries):
    site_labels = defaultdict(set)
    for entry in entries:
        site_labels[(entry["dataset"], entry["site"])].add(int(entry["y"]))

    keep_keys = {key for key, labels in site_labels.items() if labels == {0, 1}}
    filtered = [entry for entry in entries if (entry["dataset"], entry["site"]) in keep_keys]
    removed = [entry for entry in entries if (entry["dataset"], entry["site"]) not in keep_keys]
    return filtered, removed, keep_keys


class SubjectBagDataset(Dataset):
    def __init__(
        self,
        subject_entries,
        target_timepoints: int,
        max_windows_per_subject: int | None = None,
        random_sample_windows: bool = False,
    ):
        self.subject_entries = list(subject_entries)
        self.target_timepoints = int(target_timepoints)
        self.max_windows_per_subject = max_windows_per_subject
        self.random_sample_windows = random_sample_windows

    def __len__(self):
        return len(self.subject_entries)

    def __getitem__(self, idx):
        entry = self.subject_entries[idx]
        candidates = []
        for record, spans in entry["records"]:
            for start, end in spans:
                candidates.append((record, start, end))

        if self.max_windows_per_subject is not None and len(candidates) > 0:
            if len(candidates) >= self.max_windows_per_subject:
                if self.random_sample_windows:
                    selected_idx = np.random.choice(
                        len(candidates),
                        size=self.max_windows_per_subject,
                        replace=False,
                    )
                else:
                    selected_idx = np.linspace(
                        0,
                        len(candidates) - 1,
                        num=self.max_windows_per_subject,
                        dtype=int,
                    )
            else:
                if self.random_sample_windows:
                    extra_idx = np.random.choice(
                        len(candidates),
                        size=self.max_windows_per_subject - len(candidates),
                        replace=True,
                    )
                    selected_idx = np.concatenate([np.arange(len(candidates)), extra_idx])
                else:
                    selected_idx = np.linspace(
                        0,
                        len(candidates) - 1,
                        num=self.max_windows_per_subject,
                        dtype=int,
                    )
            selected = [candidates[i] for i in selected_idx.tolist()]
        else:
            selected = candidates

        grouped = defaultdict(list)
        for record, start, end in selected:
            grouped[record.ts_path].append((start, end))

        windows = []
        for ts_path, spans in grouped.items():
            ts = np.load(ts_path).astype(np.float32)
            for start, end in spans:
                window = resample_time_window(ts[start:end], self.target_timepoints)
                windows.append(torch.from_numpy(window))

        bag = torch.stack(windows, dim=0)
        labels = {
            "subject_id": entry["subject_id"],
            "site": entry["site"],
            "dataset": entry["dataset"],
            "diag": entry["diag"],
            "diag_name": entry["diag_name"],
            "y": int(entry["y"]),
            "n_windows": int(bag.size(0)),
        }
        return bag, labels


def collate_subject_bags(batch):
    bags, labels = zip(*batch)
    counts = [bag.size(0) for bag in bags]
    flat_windows = torch.cat(bags, dim=0)
    y = torch.tensor([label["y"] for label in labels], dtype=torch.long)
    return flat_windows, counts, y, list(labels)


def make_class_balanced_sampler(entries, seed: int):
    labels = [int(entry["y"]) for entry in entries]
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=make_torch_generator(seed),
    )


def split_train_val_entries(train_entries, val_fraction: float, seed: int):
    labels = np.array([int(entry["y"]) for entry in train_entries], dtype=int)
    if len(train_entries) < 8 or len(np.unique(labels)) < 2 or labels.min() == labels.max():
        return train_entries, []

    class_counts = Counter(labels.tolist())
    if min(class_counts.values()) < 2:
        return train_entries, []

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(train_entries)), labels))
    inner_train = [train_entries[i] for i in train_idx]
    inner_val = [train_entries[i] for i in val_idx]
    return inner_train, inner_val


def build_subject_model(cfg: SubjectPhase3Config, checkpoint_cfg, device: str):
    encoder_model, _ = build_model_from_checkpoint(
        cfg.checkpoint_path,
        override_target_timepoints=cfg.target_timepoints,
    )
    encoder_model = encoder_model.to(device)
    model = SubjectPoolModel(
        encoder_model=encoder_model,
        embed_dim=checkpoint_cfg.get("embed_dim", 128),
        head_hidden=cfg.head_hidden,
        head_dropout=cfg.head_dropout,
        pooling_mode=cfg.pooling_mode,
        attention_hidden=cfg.attention_hidden,
    ).to(device)
    return model


def collect_subject_outputs(model, loader, device: str):
    model.eval()
    all_margins = []
    all_true = []
    all_labels = []

    with torch.no_grad():
        for flat_windows, counts, y, labels in loader:
            flat_windows = flat_windows.to(device)
            logits, _ = model(flat_windows, counts)
            margins = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            y_np = y.numpy()

            all_margins.extend(margins.tolist())
            all_true.extend(y_np.tolist())
            all_labels.extend(labels)

    return (
        np.array(all_margins, dtype=float),
        np.array(all_true, dtype=int),
        all_labels,
    )


def collect_subject_embeddings(model, loader, device: str):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for flat_windows, counts, _, labels in loader:
            flat_windows = flat_windows.to(device)
            _, subject_embeddings = model(flat_windows, counts)
            all_embeddings.append(subject_embeddings.detach().cpu().numpy())
            all_labels.extend(labels)

    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32), []
    return np.concatenate(all_embeddings, axis=0), all_labels


def run_post_finetune_site_probe(model, loader, device: str, seed: int):
    embeddings, labels = collect_subject_embeddings(model, loader, device=device)
    if len(labels) == 0:
        return None

    sites = [label["site"] for label in labels]
    groups = [label["subject_id"] for label in labels]
    result = run_grouped_probe(
        embeddings,
        sites,
        groups=groups,
        seed=seed,
        folds=5,
        task_name="site",
    )
    if result is None:
        return None
    result["majority_accuracy"] = float("nan")
    result["majority_balanced_accuracy"] = float("nan")
    result["n_sites"] = int(result.get("n_classes", len(set(sites))))
    result["n_subjects"] = int(len(labels))
    return result


def evaluate_subject_loader(model, loader, device: str, threshold: float = 0.5, calibrator=None):
    calibrator = calibrator or IdentityCalibrator()
    margins, y_true, labels = collect_subject_outputs(model, loader, device=device)

    if len(labels) == 0:
        return {
            "bacc": float("nan"),
            "auc": float("nan"),
            "f1_macro": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "n_subjects": 0,
            "mean_windows_per_subject": float("nan"),
        }, []

    probs = calibrator.predict_proba(margins)
    preds = (probs >= threshold).astype(int)

    all_rows = []
    for label, prob, pred, truth, margin in zip(labels, probs, preds, y_true, margins):
        all_rows.append(
            {
                "subject_id": label["subject_id"],
                "site": label["site"],
                "dataset": label["dataset"],
                "y": int(truth),
                "pred": int(pred),
                "prob": float(prob),
                "margin": float(margin),
                "n_windows": int(label["n_windows"]),
            }
        )

    metrics = compute_binary_metrics(y_true, preds, probs)
    metrics["n_subjects"] = len(all_rows)
    metrics["mean_windows_per_subject"] = float(np.mean([row["n_windows"] for row in all_rows]))
    return metrics, all_rows


def evaluate_subject_loader_with_loss(model, loader, device: str, criterion, threshold: float = 0.5, calibrator=None):
    model.eval()
    calibrator = calibrator or IdentityCalibrator()

    batch_losses = []
    all_rows = []
    all_margins = []
    all_y = []

    with torch.no_grad():
        for flat_windows, counts, y, labels in loader:
            flat_windows = flat_windows.to(device)
            y = y.to(device)

            logits, _ = model(flat_windows, counts)
            loss = criterion(logits, y)
            batch_losses.append(float(loss.item()))

            margins = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
            probs = calibrator.predict_proba(margins)
            preds = (probs >= threshold).astype(int)
            y_np = y.detach().cpu().numpy().astype(int)
            all_margins.extend(margins.tolist())
            all_y.extend(y_np.tolist())

            for label, prob, pred, truth, margin in zip(labels, probs, preds, y_np, margins):
                all_rows.append(
                    {
                        "subject_id": label["subject_id"],
                        "site": label["site"],
                        "dataset": label["dataset"],
                        "y": int(truth),
                        "pred": int(pred),
                        "prob": float(prob),
                        "margin": float(margin),
                        "n_windows": int(label["n_windows"]),
                    }
                )

    if not all_rows:
        return {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "bacc": float("nan"),
            "auc": float("nan"),
            "f1_macro": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "n_subjects": 0,
            "mean_windows_per_subject": float("nan"),
        }, []

    y_true = np.array([row["y"] for row in all_rows], dtype=int)
    y_pred = np.array([row["pred"] for row in all_rows], dtype=int)
    y_prob = np.array([row["prob"] for row in all_rows], dtype=float)
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = float(np.mean(batch_losses)) if batch_losses else float("nan")
    metrics["accuracy"] = float(np.mean(y_pred == y_true))
    metrics["n_subjects"] = len(all_rows)
    metrics["mean_windows_per_subject"] = float(np.mean([row["n_windows"] for row in all_rows]))
    return metrics, all_rows


def train_one_fold(cfg: SubjectPhase3Config, checkpoint_cfg, train_entries, val_entries, device: str, logger=None):
    model = build_subject_model(cfg, checkpoint_cfg, device)

    encoder_params = list(model.token_embed.parameters()) + list(model.context_encoder.parameters())
    head_params = list(model.head.parameters())
    if model.attention_pool is not None:
        head_params += list(model.attention_pool.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.ft_encoder_lr},
            {"params": head_params, "lr": cfg.ft_head_lr},
        ],
        weight_decay=cfg.ft_weight_decay,
    )

    train_dataset = SubjectBagDataset(
        train_entries,
        target_timepoints=cfg.target_timepoints,
        max_windows_per_subject=cfg.train_windows_per_subject,
        random_sample_windows=True,
    )
    sampler = make_class_balanced_sampler(train_entries, seed=cfg.seed + 101)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.ft_batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_subject_bags,
        worker_init_fn=make_worker_init_fn(cfg.seed + 102),
    )

    if val_entries:
        val_dataset = SubjectBagDataset(
            val_entries,
            target_timepoints=cfg.target_timepoints,
            max_windows_per_subject=None,
            random_sample_windows=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.ft_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=collate_subject_bags,
            worker_init_fn=make_worker_init_fn(cfg.seed + 103),
        )
    else:
        val_loader = None

    class_counts = Counter(int(entry["y"]) for entry in train_entries)
    weights = torch.tensor(
        [1.0 / max(class_counts.get(0, 1), 1), 1.0 / max(class_counts.get(1, 1), 1)],
        dtype=torch.float32,
        device=device,
    )
    weights = weights / weights.sum() * 2.0
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    best_threshold = 0.5
    patience_left = cfg.early_stop_patience

    for epoch in range(cfg.ft_epochs):
        model.train()
        lr_encoder = get_lr(epoch, cfg.ft_warmup_epochs, cfg.ft_epochs, cfg.ft_encoder_lr)
        lr_head = get_lr(epoch, cfg.ft_warmup_epochs, cfg.ft_epochs, cfg.ft_head_lr)
        optimizer.param_groups[0]["lr"] = lr_encoder
        optimizer.param_groups[1]["lr"] = lr_head

        epoch_loss_values = []
        epoch_cls_values = []
        epoch_contrastive_values = []

        for flat_windows, counts, y, _ in train_loader:
            flat_windows = flat_windows.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, subject_embeddings = model(flat_windows, counts)
            cls_loss = criterion(logits, y)
            contrastive_loss = logits.new_zeros(())
            loss = cls_loss
            if cfg.contrastive_weight > 0.0:
                contrastive_loss = supervised_contrastive_loss(
                    subject_embeddings,
                    y,
                    temperature=cfg.contrastive_temperature,
                )
                loss = loss + cfg.contrastive_weight * contrastive_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss_values.append(float(loss.item()))
            epoch_cls_values.append(float(cls_loss.item()))
            epoch_contrastive_values.append(float(contrastive_loss.item()) if cfg.contrastive_weight > 0.0 else 0.0)

        train_metrics, _ = evaluate_subject_loader_with_loss(
            model,
            train_loader,
            device=device,
            criterion=criterion,
            threshold=0.5,
            calibrator=IdentityCalibrator(),
        )

        if val_loader is None:
            best_state = copy.deepcopy(model.state_dict())
            if logger is not None:
                logger.info(
                    "    Epoch %02d | train_loss=%.4f cls=%.4f contrast=%.4f train_acc=%.1f%% train_bAcc=%.1f%% | no validation",
                    epoch + 1,
                    float(np.mean(epoch_loss_values)) if epoch_loss_values else float("nan"),
                    float(np.mean(epoch_cls_values)) if epoch_cls_values else float("nan"),
                    float(np.mean(epoch_contrastive_values)) if epoch_contrastive_values else 0.0,
                    100.0 * train_metrics["accuracy"],
                    100.0 * train_metrics["bacc"],
                )
            continue

        val_metrics, val_rows = evaluate_subject_loader(model, val_loader, device=device, threshold=0.5)
        val_threshold = find_optimal_threshold(
            np.array([row["y"] for row in val_rows], dtype=int),
            np.array([row["prob"] for row in val_rows], dtype=float),
        )
        val_metrics_thr, _ = evaluate_subject_loader(model, val_loader, device=device, threshold=val_threshold)
        val_loss_metrics, _ = evaluate_subject_loader_with_loss(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            threshold=val_threshold,
            calibrator=IdentityCalibrator(),
        )
        score = val_metrics_thr["bacc"]
        improved = score > best_score

        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = val_threshold
            patience_left = cfg.early_stop_patience
        else:
            patience_left -= 1
        if logger is not None:
            logger.info(
                "    Epoch %02d | train_loss=%.4f cls=%.4f contrast=%.4f train_acc=%.1f%% train_bAcc=%.1f%% | val_loss=%.4f val_bAcc@0.5=%.1f%% val_bAcc*=%.1f%% val_auc=%.3f thr=%.2f best=%.1f%% patience=%d %s",
                epoch + 1,
                float(np.mean(epoch_loss_values)) if epoch_loss_values else float("nan"),
                float(np.mean(epoch_cls_values)) if epoch_cls_values else float("nan"),
                float(np.mean(epoch_contrastive_values)) if epoch_contrastive_values else 0.0,
                100.0 * train_metrics["accuracy"],
                100.0 * train_metrics["bacc"],
                val_loss_metrics["loss"],
                100.0 * val_metrics["bacc"],
                100.0 * val_metrics_thr["bacc"],
                val_metrics_thr["auc"],
                val_threshold,
                100.0 * best_score,
                patience_left,
                "[best]" if improved else "",
            )
        if patience_left <= 0:
            break

    model.load_state_dict(best_state)

    calibrator = build_calibrator(cfg.calibration_mode)
    if val_loader is not None:
        val_margins, val_y, _ = collect_subject_outputs(model, val_loader, device=device)
        calibrator.fit(val_margins, val_y)
        val_probs = calibrator.predict_proba(val_margins)
        best_threshold = find_optimal_threshold(val_y, val_probs)

    return model, calibrator, best_threshold


def run_task_loso(cfg: SubjectPhase3Config, task_key: str, subject_entries, checkpoint_cfg, logger):
    task_name, task_entries = filter_subject_entries(subject_entries, task_key)
    if cfg.mixed_sites_only_train and cfg.mixed_sites_only_test:
        task_entries, removed_entries, kept_keys = filter_entries_to_mixed_sites(task_entries)
        logger.info(
            "Mixed-site filter (all entries) for %s | kept=%d removed=%d mixed_dataset_sites=%d",
            task_name,
            len(task_entries),
            len(removed_entries),
            len(kept_keys),
        )
    task_entries = maybe_subsample_subject_entries(task_entries, cfg.max_subjects_per_class, cfg.seed)

    subject_count = len(task_entries)
    class_counts = Counter(int(entry["y"]) for entry in task_entries)
    logger.info(
        "Task %s | subjects=%d | class_counts=%s",
        task_name,
        subject_count,
        dict(sorted(class_counts.items())),
    )

    aggregated_rows = []
    site_probe_rows = []
    valid_units = []
    holdout_key = "site"
    holdout_units = sorted({entry["site"] for entry in task_entries})

    if cfg.held_out_datasets is not None:
        holdout_key = "dataset"
        held_out_filter = set(cfg.held_out_datasets)
        holdout_units = sorted({entry["dataset"] for entry in task_entries if entry["dataset"] in held_out_filter})
        logger.info(
            "Restricting held-out datasets for %s to %s",
            task_name,
            holdout_units,
        )
    elif cfg.held_out_sites is not None:
        held_out_filter = set(cfg.held_out_sites)
        holdout_units = [site for site in holdout_units if site in held_out_filter]
        logger.info(
            "Restricting held-out sites for %s to %s",
            task_name,
            holdout_units,
        )

    for held_out_unit in holdout_units:
        train_entries = [entry for entry in task_entries if entry[holdout_key] != held_out_unit]
        test_entries = [entry for entry in task_entries if entry[holdout_key] == held_out_unit]

        if cfg.mixed_sites_only_train and not (cfg.mixed_sites_only_train and cfg.mixed_sites_only_test):
            orig_train = len(train_entries)
            train_entries, removed_train, kept_train_keys = filter_entries_to_mixed_sites(train_entries)
            logger.info(
                "  %s %s | mixed-site train filter | kept=%d removed=%d mixed_dataset_sites=%d",
                holdout_key.capitalize(),
                held_out_unit,
                len(train_entries),
                orig_train - len(train_entries),
                len(kept_train_keys),
            )

        if cfg.mixed_sites_only_test and not (cfg.mixed_sites_only_train and cfg.mixed_sites_only_test):
            orig_test = len(test_entries)
            test_entries, removed_test, kept_test_keys = filter_entries_to_mixed_sites(test_entries)
            logger.info(
                "  %s %s | mixed-site test filter | kept=%d removed=%d mixed_dataset_sites=%d",
                holdout_key.capitalize(),
                held_out_unit,
                len(test_entries),
                orig_test - len(test_entries),
                len(kept_test_keys),
            )

        if len(test_entries) == 0:
            continue

        train_labels = {int(entry["y"]) for entry in train_entries}
        test_labels = {int(entry["y"]) for entry in test_entries}
        if len(train_labels) < 2 or len(test_labels) < 2:
            continue

        train_inner, val_entries = split_train_val_entries(train_entries, cfg.val_fraction, cfg.seed)
        model, calibrator, threshold = train_one_fold(
            cfg=cfg,
            checkpoint_cfg=checkpoint_cfg,
            train_entries=train_inner,
            val_entries=val_entries,
            device=cfg.device,
            logger=logger,
        )

        test_dataset = SubjectBagDataset(test_entries, target_timepoints=cfg.target_timepoints)
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.ft_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=collate_subject_bags,
            worker_init_fn=make_worker_init_fn(cfg.seed + 104),
        )
        test_metrics, test_rows = evaluate_subject_loader(
            model,
            test_loader,
            device=cfg.device,
            threshold=threshold,
            calibrator=calibrator,
        )
        logger.info(
            "  %s %s | subjects=%d | bAcc=%.1f%% AUC=%.3f threshold=%.2f calibrator=%s",
            holdout_key.capitalize(),
            held_out_unit,
            test_metrics["n_subjects"],
            100.0 * test_metrics["bacc"],
            test_metrics["auc"],
            threshold,
            calibrator.describe(),
        )
        if holdout_key == "dataset":
            site_probe = run_post_finetune_site_probe(
                model,
                test_loader,
                device=cfg.device,
                seed=cfg.seed,
            )
            if site_probe is not None:
                site_probe_row = {
                    "held_out_unit": held_out_unit,
                    **site_probe,
                }
                site_probe_rows.append(site_probe_row)
                logger.info(
                    "  %s %s | post-ft site probe | bAcc=%.1f%% acc=%.1f%% majority_acc=%.1f%% n_sites=%d",
                    holdout_key.capitalize(),
                    held_out_unit,
                    100.0 * site_probe["balanced_accuracy"],
                    100.0 * site_probe["accuracy"],
                    100.0 * site_probe["majority_accuracy"],
                    site_probe["n_sites"],
                )
        aggregated_rows.extend(test_rows)
        valid_units.append(held_out_unit)

    if not aggregated_rows:
        summary = {
            "task": task_name,
            "task_key": task_key,
            "bacc": float("nan"),
            "auc": float("nan"),
            "f1_macro": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "n_valid_sites": 0,
            "held_out_sites": [],
            "n_valid_datasets": 0,
            "held_out_datasets": [],
            "n_subjects": 0,
            "mean_windows_per_subject": float("nan"),
            "site_probe_accuracy": float("nan"),
            "site_probe_balanced_accuracy": float("nan"),
            "site_probe_majority_accuracy": float("nan"),
            "site_probe_majority_balanced_accuracy": float("nan"),
            "site_probe_n_evals": 0,
        }
        add_site_std_to_summary(summary, [])
        return summary, [], []

    y_true = np.array([row["y"] for row in aggregated_rows], dtype=int)
    y_pred = np.array([row["pred"] for row in aggregated_rows], dtype=int)
    y_prob = np.array([row["prob"] for row in aggregated_rows], dtype=float)
    summary = compute_binary_metrics(y_true, y_pred, y_prob)
    summary["task"] = task_name
    summary["task_key"] = task_key
    summary["holdout_key"] = holdout_key
    summary["n_valid_sites"] = len(valid_units) if holdout_key == "site" else 0
    summary["held_out_sites"] = valid_units if holdout_key == "site" else []
    summary["n_valid_datasets"] = len(valid_units) if holdout_key == "dataset" else 0
    summary["held_out_datasets"] = valid_units if holdout_key == "dataset" else []
    summary["n_subjects"] = len(aggregated_rows)
    summary["mean_windows_per_subject"] = float(np.mean([row["n_windows"] for row in aggregated_rows]))
    summary["pooling_mode"] = cfg.pooling_mode
    summary["calibration_mode"] = cfg.calibration_mode
    summary["train_windows_per_subject"] = cfg.train_windows_per_subject
    summary["contrastive_weight"] = cfg.contrastive_weight
    summary["contrastive_temperature"] = cfg.contrastive_temperature
    summary["site_probe_accuracy"] = float("nan")
    summary["site_probe_balanced_accuracy"] = float("nan")
    summary["site_probe_majority_accuracy"] = float("nan")
    summary["site_probe_majority_balanced_accuracy"] = float("nan")
    summary["site_probe_n_evals"] = len(site_probe_rows)
    summary["site_probe_n_sites"] = int(site_probe_rows[0]["n_sites"]) if site_probe_rows else 0
    if site_probe_rows:
        summary["site_probe_accuracy"] = float(np.mean([row["accuracy"] for row in site_probe_rows]))
        summary["site_probe_balanced_accuracy"] = float(np.mean([row["balanced_accuracy"] for row in site_probe_rows]))
        majority_acc = np.array([row["majority_accuracy"] for row in site_probe_rows], dtype=float)
        majority_bacc = np.array([row["majority_balanced_accuracy"] for row in site_probe_rows], dtype=float)
        if np.any(~np.isnan(majority_acc)):
            summary["site_probe_majority_accuracy"] = float(np.nanmean(majority_acc))
        if np.any(~np.isnan(majority_bacc)):
            summary["site_probe_majority_balanced_accuracy"] = float(np.nanmean(majority_bacc))
    site_metrics = compute_site_level_metrics(aggregated_rows)
    add_site_std_to_summary(summary, site_metrics)
    return summary, aggregated_rows, site_probe_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Phase 3 subject-level fine-tuning with pooled sliding-window embeddings")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dirs", nargs="+", default=None)
    parser.add_argument("--task", nargs="+", default=None, choices=sorted(TASKS))
    parser.add_argument("--held-out-sites", nargs="+", default=None)
    parser.add_argument("--held-out-datasets", nargs="+", default=None)
    parser.add_argument("--mixed-sites-only-train", action="store_true")
    parser.add_argument("--mixed-sites-only-test", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=None)
    parser.add_argument("--target-timepoints", type=int, default=None)
    parser.add_argument("--ft-epochs", type=int, default=None)
    parser.add_argument("--ft-batch-size", type=int, default=None)
    parser.add_argument("--ft-encoder-lr", type=float, default=None)
    parser.add_argument("--ft-head-lr", type=float, default=None)
    parser.add_argument("--contrastive-weight", type=float, default=None)
    parser.add_argument("--contrastive-temperature", type=float, default=None)
    parser.add_argument("--pooling", choices=["mean", "attention"], default=None)
    parser.add_argument("--attention-hidden", type=int, default=None)
    parser.add_argument("--calibration-mode", choices=["none", "temperature"], default=None)
    parser.add_argument("--train-windows-per-subject", type=int, default=None)
    parser.add_argument("--head-hidden", type=int, default=None)
    parser.add_argument("--head-dropout", type=float, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--max-subjects-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = SubjectPhase3Config()

    if args.checkpoint is not None:
        cfg.checkpoint_path = args.checkpoint
    if args.data_dirs is not None:
        cfg.data_dirs = args.data_dirs
    if args.task is not None:
        cfg.tasks = args.task
    if args.held_out_sites is not None:
        cfg.held_out_sites = args.held_out_sites
    if args.held_out_datasets is not None:
        cfg.held_out_datasets = args.held_out_datasets
    if args.mixed_sites_only_train:
        cfg.mixed_sites_only_train = True
    if args.mixed_sites_only_test:
        cfg.mixed_sites_only_test = True
    if args.device is not None:
        cfg.device = args.device
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.window_seconds is not None:
        cfg.window_seconds = args.window_seconds
    if args.window_stride_seconds is not None:
        cfg.window_stride_seconds = args.window_stride_seconds
    if args.target_timepoints is not None:
        cfg.target_timepoints = args.target_timepoints
    if args.ft_epochs is not None:
        cfg.ft_epochs = args.ft_epochs
    if args.ft_batch_size is not None:
        cfg.ft_batch_size = args.ft_batch_size
    if args.ft_encoder_lr is not None:
        cfg.ft_encoder_lr = args.ft_encoder_lr
    if args.ft_head_lr is not None:
        cfg.ft_head_lr = args.ft_head_lr
    if args.contrastive_weight is not None:
        cfg.contrastive_weight = args.contrastive_weight
    if args.contrastive_temperature is not None:
        cfg.contrastive_temperature = args.contrastive_temperature
    if args.pooling is not None:
        cfg.pooling_mode = args.pooling
    if args.attention_hidden is not None:
        cfg.attention_hidden = args.attention_hidden
    if args.calibration_mode is not None:
        cfg.calibration_mode = args.calibration_mode
    if args.train_windows_per_subject is not None:
        cfg.train_windows_per_subject = args.train_windows_per_subject
    if args.head_hidden is not None:
        cfg.head_hidden = args.head_hidden
    if args.head_dropout is not None:
        cfg.head_dropout = args.head_dropout
    if args.val_fraction is not None:
        cfg.val_fraction = args.val_fraction
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    if args.max_subjects_per_class is not None:
        cfg.max_subjects_per_class = args.max_subjects_per_class
    if args.seed is not None:
        cfg.seed = args.seed
    if args.deterministic:
        cfg.deterministic = True
    if args.non_deterministic:
        cfg.deterministic = False
    if cfg.held_out_sites is not None and cfg.held_out_datasets is not None:
        raise ValueError("Specify only one of --held-out-sites or --held-out-datasets.")

    if args.quick:
        cfg.tasks = cfg.tasks[:1]
        cfg.ft_epochs = 1
        cfg.ft_batch_size = min(cfg.ft_batch_size, 4)
        cfg.max_subjects_per_class = cfg.max_subjects_per_class or 8
        if args.save_dir is None:
            cfg.save_dir = "results/jepa_clean_phase3_subject_pooling_quick"

    checkpoint_cfg = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False).get("config", {})
    if cfg.target_timepoints is None:
        cfg.target_timepoints = checkpoint_cfg.get("target_timepoints", 45)

    setup_logging(args.log_file)
    logger = logging.getLogger("JEPA-Clean-Phase3-Subject")
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    logger.info("Config: %s", json.dumps(asdict(cfg), indent=2))

    subject_entries = build_subject_entries(
        data_dirs=cfg.data_dirs,
        window_seconds=cfg.window_seconds,
        stride_seconds=cfg.window_stride_seconds,
    )
    logger.info(
        "Loaded %d subject bags from %d datasets",
        len(subject_entries),
        len(cfg.data_dirs),
    )

    all_results = []
    for task_key in cfg.tasks:
        summary, rows, site_probe_rows = run_task_loso(
            cfg=cfg,
            task_key=task_key,
            subject_entries=subject_entries,
            checkpoint_cfg=checkpoint_cfg,
            logger=logger,
        )
        all_results.append({"summary": summary, "subjects": rows, "site_probe": site_probe_rows})
        logger.info(
            "%s | bAcc=%.1f%% AUC=%.3f F1=%.3f precision=%.3f recall=%.3f | holdout=%s valid_units=%d subjects=%d mean_windows=%.2f",
            summary["task"],
            100.0 * summary["bacc"],
            summary["auc"],
            summary["f1_macro"],
            summary["precision"],
            summary["recall"],
            summary.get("holdout_key", "site"),
            summary["n_valid_sites"] if summary.get("holdout_key") == "site" else summary.get("n_valid_datasets", 0),
            summary["n_subjects"],
            summary["mean_windows_per_subject"],
        )
        if summary["site_probe_n_evals"] > 0:
            logger.info(
                "%s | post-ft site probe | bAcc=%.1f%% acc=%.1f%% majority_acc=%.1f%%",
                summary["task"],
                100.0 * summary["site_probe_balanced_accuracy"],
                100.0 * summary["site_probe_accuracy"],
                100.0 * summary["site_probe_majority_accuracy"],
            )

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "subject_pooling_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved results: %s", out_path)


if __name__ == "__main__":
    main()
