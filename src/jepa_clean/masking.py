from dataclasses import dataclass
from functools import lru_cache
import random

import numpy as np
import torch


@lru_cache(maxsize=1)
def get_schaefer400_network_labels():
    from nilearn.datasets import fetch_atlas_schaefer_2018

    atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    labels = [
        lbl.decode("utf-8") if isinstance(lbl, bytes) else str(lbl)
        for lbl in atlas["labels"]
    ]
    network_names = [lbl.split("_")[2] for lbl in labels]
    unique_networks = list(dict.fromkeys(network_names))
    name_to_id = {name: i + 1 for i, name in enumerate(unique_networks)}
    return np.array([name_to_id[name] for name in network_names], dtype=np.int64)


@dataclass
class SpatioTemporalMask:
    target_indices: torch.LongTensor
    masked_roi_indices: torch.LongTensor
    masked_network_ids: tuple
    time_ranges: tuple


def _sample_time_ranges(seq_len: int, num_blocks: int, block_size_range: tuple):
    masked = set()
    ranges = []
    for _ in range(num_blocks):
        block_size = random.randint(*block_size_range)
        block_size = min(block_size, seq_len)
        start = random.randint(0, max(0, seq_len - block_size))
        end = start + block_size
        ranges.append((start, end))
        masked.update(range(start, end))

    if not masked:
        masked.add(random.randrange(seq_len))
        ranges.append((min(masked), max(masked) + 1))

    return tuple(sorted(ranges)), torch.tensor(sorted(masked), dtype=torch.long)


def sample_temporal_mask(
    seq_len: int,
    num_time_blocks: int = 2,
    time_block_size: tuple = (6, 10),
):
    time_ranges, target_indices = _sample_time_ranges(
        seq_len=seq_len,
        num_blocks=num_time_blocks,
        block_size_range=time_block_size,
    )
    return {
        "target_indices": target_indices,
        "time_ranges": time_ranges,
    }


def sample_spatiotemporal_mask(
    seq_len: int,
    num_networks_range: tuple = (1, 2),
    num_time_blocks: int = 2,
    time_block_size: tuple = (6, 10),
):
    network_labels = get_schaefer400_network_labels()
    n_networks = random.randint(num_networks_range[0], num_networks_range[1])
    network_ids = tuple(sorted(random.sample(list(range(1, 8)), n_networks)))
    roi_mask = np.isin(network_labels, network_ids)
    roi_indices = torch.tensor(np.where(roi_mask)[0], dtype=torch.long)
    time_ranges, target_indices = _sample_time_ranges(
        seq_len=seq_len,
        num_blocks=num_time_blocks,
        block_size_range=time_block_size,
    )
    return SpatioTemporalMask(
        target_indices=target_indices,
        masked_roi_indices=roi_indices,
        masked_network_ids=network_ids,
        time_ranges=time_ranges,
    )


def apply_temporal_mask(x: torch.Tensor, target_indices: torch.LongTensor, mask_value: float = 0.0):
    masked_x = x.clone()
    if target_indices.numel() == 0:
        return masked_x
    time_idx = target_indices.to(masked_x.device)
    masked_x[:, time_idx, :] = mask_value
    return masked_x


def apply_spatiotemporal_mask(x: torch.Tensor, mask: SpatioTemporalMask, mask_value: float = 0.0):
    masked_x = x.clone()
    if mask.target_indices.numel() == 0 or mask.masked_roi_indices.numel() == 0:
        return masked_x

    time_idx = mask.target_indices.to(masked_x.device)
    roi_idx = mask.masked_roi_indices.to(masked_x.device)
    mask_matrix = torch.zeros(
        masked_x.size(1),
        masked_x.size(2),
        dtype=torch.bool,
        device=masked_x.device,
    )
    mask_matrix[time_idx[:, None], roi_idx[None, :]] = True
    masked_x.masked_fill_(mask_matrix.unsqueeze(0), mask_value)
    return masked_x
