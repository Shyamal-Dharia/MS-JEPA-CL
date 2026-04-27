from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from ..jepa_clean.data import WindowedFMRIDataset
    from .atlas import get_schaefer400_roi_centroids
    from .masking import complement_indices, sample_rectangular_patch_mask, sample_temporal_mask
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_clean.data import WindowedFMRIDataset
    from src.jepa_fusion.atlas import get_schaefer400_roi_centroids
    from src.jepa_fusion.masking import complement_indices, sample_rectangular_patch_mask, sample_temporal_mask


MAIN_DATASET_DIRS = {
    "srpbs": "preprocessed_data_fir/full/roi_timeseries_1600_full",
    "abide1": "preprocessed_data_fir/full/roi_timeseries_abide_full",
    "abide2": "preprocessed_data_fir/full/roi_timeseries_abide2_full",
}


class FourierEmb4D(nn.Module):
    """
    Continuous Fourier positional embedding for `(x, y, z, t)` coordinates.

    Input shape: `[B, N, 4]`
    Output shape: `[B, N, dimension]`
    """

    def __init__(self, dimension: int, freqs: int, increment_time: float = 0.1, margin: float = 0.4):
        super().__init__()
        self.dimension = int(dimension)
        self.freqs = int(freqs)
        self.increment_time = float(increment_time)
        self.margin = float(margin)

    def forward(self, positions: torch.Tensor):
        if positions.ndim != 3 or positions.size(-1) != 4:
            raise ValueError(f"Expected positions with shape [B, N, 4], got {tuple(positions.shape)}")

        coords = positions.clone()
        coords[:, :, -1] *= self.increment_time
        batch_size, num_tokens, _ = coords.shape

        freqs_w = torch.arange(self.freqs, device=coords.device, dtype=coords.dtype)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]

        width = 1.0 + 2.0 * self.margin
        coords = coords + self.margin

        p_x = 2.0 * math.pi * freqs_x / width
        p_y = 2.0 * math.pi * freqs_y / width
        p_z = 2.0 * math.pi * freqs_z / width
        p_w = 2.0 * math.pi * freqs_w / width

        coords = coords[..., None, None, None, None, :]
        loc = (
            coords[..., 0] * p_x
            + coords[..., 1] * p_y
            + coords[..., 2] * p_z
            + coords[..., 3] * p_w
        ).view(batch_size, num_tokens, -1)

        half_dim = self.dimension // 2
        if loc.size(-1) < half_dim:
            raise ValueError(
                f"Fourier basis too small for dimension={self.dimension}. "
                f"Increase freqs; current half-dim basis={loc.size(-1)}."
            )
        loc = loc[:, :, :half_dim]
        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        if emb.size(-1) != self.dimension:
            raise RuntimeError(
                f"Unexpected Fourier embedding width: expected {self.dimension}, got {emb.size(-1)}"
            )
        return emb


class ScalarValueTokenizer(nn.Module):
    """
    Tokenize each scalar `(roi, time)` value with:
    1. a learned value projection from scalar -> embed_dim
    2. a Fourier positional encoding from `(x, y, z, t)` -> embed_dim

    This is the minimal content-preserving version of a Fourier tokenization.
    Pure positional encoding alone would drop the fMRI signal entirely.
    """

    def __init__(self, embed_dim: int, fourier_freqs: int, increment_time: float = 0.1):
        super().__init__()
        self.value_proj = nn.Linear(1, embed_dim)
        self.fourier = FourierEmb4D(
            dimension=embed_dim,
            freqs=fourier_freqs,
            increment_time=increment_time,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, values: torch.Tensor, positions: torch.Tensor):
        value_tokens = self.value_proj(values)
        pe = self.fourier(positions)
        tokens = value_tokens + pe
        return self.norm(tokens), value_tokens, pe


def collate_fn(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), list(labels)


def expand_positions_with_time(roi_positions: torch.Tensor, num_timepoints: int, batch_size: int):
    """
    Convert ROI coordinates `[R, 3]` into space-time coordinates `[B, T*R, 4]`
    using time-major flattening: `(t0,r0), (t0,r1), ..., (t1,r0), ...`.
    """
    num_rois = roi_positions.size(0)
    pos_repeated = roi_positions.unsqueeze(0).unsqueeze(0).expand(batch_size, num_timepoints, num_rois, 3)
    time_values = torch.arange(num_timepoints, device=roi_positions.device, dtype=roi_positions.dtype)
    time_values = time_values.view(1, num_timepoints, 1, 1).expand(batch_size, num_timepoints, num_rois, 1)
    pos_with_time = torch.cat([pos_repeated, time_values], dim=-1)
    return pos_with_time.reshape(batch_size, num_timepoints * num_rois, 4)


def flatten_scalar_values(x: torch.Tensor):
    """
    Input: `[B, T, R]`
    Output: `[B, T*R, 1]` with time-major flattening.
    """
    batch_size, num_timepoints, num_rois = x.shape
    return x.reshape(batch_size, num_timepoints * num_rois, 1)


def sample_temporal_spacetime_mask(
    num_timepoints: int,
    num_rois: int,
    device: torch.device,
    num_time_blocks: int = 2,
    time_block_size: tuple[int, int] = (6, 10),
):
    """
    Mask complete time slices in a flattened `(time, roi)` token grid.
    If time index `t` is masked, all `R` ROI tokens for that time become targets.
    """
    time_mask = sample_temporal_mask(
        seq_len=num_timepoints,
        num_time_blocks=num_time_blocks,
        time_block_size=time_block_size,
    )
    target_indices = []
    for time_idx in time_mask.target_indices.tolist():
        start = int(time_idx) * int(num_rois)
        target_indices.extend(range(start, start + int(num_rois)))
    target_indices = torch.tensor(target_indices, dtype=torch.long, device=device)
    context_indices = complement_indices(num_timepoints * num_rois, target_indices, device=device)
    return time_mask, context_indices, target_indices


def sample_rectangular_spacetime_mask(
    num_timepoints: int,
    num_rois: int,
    device: torch.device,
    num_time_blocks: int = 2,
    time_block_size: int = 4,
    num_roi_blocks: int = 2,
    roi_block_size: int = 32,
):
    """
    Mask rectangular regions in the `(time, roi)` grid, then flatten to token indices.
    """
    patch_mask = sample_rectangular_patch_mask(
        num_time_patches=num_timepoints,
        num_roi_groups=num_rois,
        num_time_blocks=num_time_blocks,
        time_block_size=time_block_size,
        num_roi_blocks=num_roi_blocks,
        roi_block_size=roi_block_size,
    )
    target_indices = patch_mask.target_indices.to(device)
    context_indices = complement_indices(num_timepoints * num_rois, target_indices, device=device)
    return patch_mask, context_indices, target_indices


def load_batch(args, device: torch.device):
    if args.synthetic:
        x = torch.randn(args.batch_size, args.target_timepoints, args.input_dim, device=device)
        labels = [{"source": "synthetic"} for _ in range(args.batch_size)]
        return x, labels

    data_dir = MAIN_DATASET_DIRS[args.dataset]
    dataset = WindowedFMRIDataset(
        data_dirs=data_dir,
        window_seconds=args.window_seconds,
        target_timepoints=args.target_timepoints,
        sampling=args.sampling,
        stride_seconds=args.window_stride_seconds,
        tokenization_mode="temporal",
        max_scans=args.max_scans,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    x, labels = next(iter(loader))
    return x.to(device), labels


def parse_args():
    parser = argparse.ArgumentParser(description="Prototype Fourier space-time tokenization and masking.")
    parser.add_argument("--dataset", choices=sorted(MAIN_DATASET_DIRS), default="srpbs")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--target-timepoints", type=int, default=45)
    parser.add_argument("--input-dim", type=int, default=400)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--fourier-freqs", type=int, default=4)
    parser.add_argument("--increment-time", type=float, default=0.1)
    parser.add_argument("--window-seconds", type=float, default=90.0)
    parser.add_argument("--window-stride-seconds", type=float, default=30.0)
    parser.add_argument("--sampling", choices=["random", "center", "sliding"], default="random")
    parser.add_argument("--max-scans", type=int, default=8)
    parser.add_argument("--mask-mode", choices=["temporal", "rectangular"], default="temporal")
    parser.add_argument("--mask-num-time-blocks", type=int, default=2)
    parser.add_argument("--mask-time-block-min", type=int, default=6)
    parser.add_argument("--mask-time-block-max", type=int, default=10)
    parser.add_argument("--st-mask-time-block-size", type=int, default=4)
    parser.add_argument("--st-mask-num-roi-blocks", type=int, default=2)
    parser.add_argument("--st-mask-roi-block-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    x, labels = load_batch(args, device)

    batch_size, num_timepoints, num_rois = x.shape
    if num_rois != 400:
        raise ValueError(
            f"The current atlas-coordinate helper expects Schaefer-400 ordering, got num_rois={num_rois}"
        )
    roi_positions_mni = get_schaefer400_roi_centroids(space="mni", as_tensor=True).to(device)
    roi_positions = get_schaefer400_roi_centroids(space="normalized", as_tensor=True).to(device)
    spacetime_positions = expand_positions_with_time(roi_positions, num_timepoints, batch_size)
    scalar_values = flatten_scalar_values(x)

    tokenizer = ScalarValueTokenizer(
        embed_dim=args.embed_dim,
        fourier_freqs=args.fourier_freqs,
        increment_time=args.increment_time,
    ).to(device)
    tokens, value_tokens, pe_tokens = tokenizer(scalar_values, spacetime_positions)

    if args.mask_mode == "temporal":
        mask_info, context_indices, target_indices = sample_temporal_spacetime_mask(
            num_timepoints=num_timepoints,
            num_rois=num_rois,
            device=device,
            num_time_blocks=args.mask_num_time_blocks,
            time_block_size=(args.mask_time_block_min, args.mask_time_block_max),
        )
    else:
        mask_info, context_indices, target_indices = sample_rectangular_spacetime_mask(
            num_timepoints=num_timepoints,
            num_rois=num_rois,
            device=device,
            num_time_blocks=args.mask_num_time_blocks,
            time_block_size=args.st_mask_time_block_size,
            num_roi_blocks=args.st_mask_num_roi_blocks,
            roi_block_size=args.st_mask_roi_block_size,
        )

    context_tokens = tokens[:, context_indices, :]
    target_tokens = tokens[:, target_indices, :]
    context_pos = spacetime_positions[:, context_indices, :]
    target_pos = spacetime_positions[:, target_indices, :]

    print("=== Fourier Space-Time Prototype ===")
    print(f"source batch label example: {labels[0] if labels else {}}")
    print(f"input x:                 {tuple(x.shape)}  [B, T, R]")
    print(f"roi_positions_mni:       {tuple(roi_positions_mni.shape)}  [R, 3]")
    print(f"roi_positions:           {tuple(roi_positions.shape)}  [R, 3]")
    print(
        "roi_positions_mni range: "
        f"min={roi_positions_mni.amin(dim=0).tolist()} "
        f"max={roi_positions_mni.amax(dim=0).tolist()}"
    )
    print(
        "roi_positions_norm range:"
        f" min={roi_positions.amin(dim=0).tolist()} "
        f"max={roi_positions.amax(dim=0).tolist()}"
    )
    print(f"spacetime_positions:     {tuple(spacetime_positions.shape)}  [B, T*R, 4]")
    print(f"scalar_values:           {tuple(scalar_values.shape)}  [B, T*R, 1]")
    print(f"value_tokens:            {tuple(value_tokens.shape)}  [B, T*R, D]")
    print(f"fourier_pe:              {tuple(pe_tokens.shape)}  [B, T*R, D]")
    print(f"combined tokens:         {tuple(tokens.shape)}  [B, T*R, D]")
    print(f"context_indices:         {tuple(context_indices.shape)}")
    print(f"target_indices:          {tuple(target_indices.shape)}")
    print(f"context_tokens:          {tuple(context_tokens.shape)}  [B, N_ctx, D]")
    print(f"target_tokens:           {tuple(target_tokens.shape)}  [B, N_tgt, D]")
    print(f"context_positions:       {tuple(context_pos.shape)}  [B, N_ctx, 4]")
    print(f"target_positions:        {tuple(target_pos.shape)}  [B, N_tgt, 4]")

    if args.mask_mode == "temporal":
        print(f"masked time ranges:      {mask_info.time_ranges}")
        print(f"masked time indices:     {mask_info.target_indices.tolist()}")
        print(
            "mask interpretation:     each masked time index expands to all ROI tokens for that time,"
            f" so N_tgt = (#masked_times) * R = {target_tokens.size(1)}"
        )
    else:
        print(f"masked time ranges:      {mask_info.time_patch_ranges}")
        print(f"masked roi ranges:       {mask_info.roi_group_ranges}")
        print(
            "mask interpretation:     target tokens are rectangles in the flattened (time, roi) grid,"
            f" so N_tgt = (#masked_time_cells) * (#masked_roi_cells) after block union = {target_tokens.size(1)}"
        )


if __name__ == "__main__":
    main()
