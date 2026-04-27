from __future__ import annotations

from dataclasses import dataclass
import random

import torch


@dataclass(frozen=True)
class TemporalMask:
    target_indices: torch.LongTensor
    time_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RectangularPatchMask:
    target_indices: torch.LongTensor
    time_patch_indices: torch.LongTensor
    roi_group_indices: torch.LongTensor
    time_patch_ranges: tuple[tuple[int, int], ...]
    roi_group_ranges: tuple[tuple[int, int], ...]


def _sample_block_union(length: int, num_blocks: int, block_size: int | tuple[int, int]):
    masked = set()
    ranges: list[tuple[int, int]] = []

    if isinstance(block_size, int):
        size_range = (int(block_size), int(block_size))
    else:
        size_range = (int(block_size[0]), int(block_size[1]))

    for _ in range(max(0, int(num_blocks))):
        size = random.randint(*size_range)
        size = min(max(1, size), int(length))
        start = random.randint(0, max(0, int(length) - size))
        end = start + size
        ranges.append((start, end))
        masked.update(range(start, end))

    if not masked:
        fallback = random.randrange(int(length))
        masked.add(fallback)
        ranges.append((fallback, fallback + 1))

    return tuple(sorted(ranges)), torch.tensor(sorted(masked), dtype=torch.long)


def complement_indices(seq_len: int, target_indices: torch.LongTensor, device: torch.device):
    keep_mask = torch.ones(int(seq_len), dtype=torch.bool, device=device)
    keep_mask[target_indices.to(device)] = False
    return torch.arange(int(seq_len), device=device)[keep_mask]


def sample_temporal_mask(
    seq_len: int,
    num_time_blocks: int = 2,
    time_block_size: tuple[int, int] = (6, 10),
):
    time_ranges, target_indices = _sample_block_union(
        length=seq_len,
        num_blocks=num_time_blocks,
        block_size=time_block_size,
    )
    return TemporalMask(
        target_indices=target_indices,
        time_ranges=time_ranges,
    )


def sample_rectangular_patch_mask(
    num_time_patches: int,
    num_roi_groups: int,
    num_time_blocks: int = 2,
    time_block_size: int = 2,
    num_roi_blocks: int = 2,
    roi_block_size: int = 3,
):
    time_ranges, time_patch_indices = _sample_block_union(
        length=num_time_patches,
        num_blocks=num_time_blocks,
        block_size=time_block_size,
    )
    roi_ranges, roi_group_indices = _sample_block_union(
        length=num_roi_groups,
        num_blocks=num_roi_blocks,
        block_size=roi_block_size,
    )

    target_indices = []
    for time_idx in time_patch_indices.tolist():
        row_offset = int(time_idx) * int(num_roi_groups)
        for roi_idx in roi_group_indices.tolist():
            target_indices.append(row_offset + int(roi_idx))
    target_indices = sorted(set(target_indices))
    if not target_indices:
        target_indices = [0]

    return RectangularPatchMask(
        target_indices=torch.tensor(target_indices, dtype=torch.long),
        time_patch_indices=time_patch_indices,
        roi_group_indices=roi_group_indices,
        time_patch_ranges=time_ranges,
        roi_group_ranges=roi_ranges,
    )
