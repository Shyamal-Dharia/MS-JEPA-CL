from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn

from .atlas import get_schaefer400_roi_centroids
from .model import Predictor, TransformerEncoder, build_sincos_pos_embed, gradient_reverse


class FourierEmb4D(nn.Module):
    """
    Fourier positional embedding for 4D positions `(x, y, z, t)`.

    Expected input shape: `[B, N, 4]`
    Output shape: `[B, N, dimension]`
    """

    def __init__(self, dimension: int, freqs: int, increment_time: float = 0.1, margin: float = 0.4):
        super().__init__()
        self.dimension = int(dimension)
        self.freqs = int(freqs)
        self.increment_time = float(increment_time)
        self.margin = float(margin)

    def forward(self, positions_: torch.Tensor):
        positions = positions_.clone()
        positions[:, :, -1] *= self.increment_time
        *leading_dims, _ = positions.shape

        freqs_w = torch.arange(self.freqs, device=positions.device, dtype=positions.dtype)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]

        width = 1.0 + 2.0 * self.margin
        positions = positions + self.margin
        p_x = 2.0 * math.pi * freqs_x / width
        p_y = 2.0 * math.pi * freqs_y / width
        p_z = 2.0 * math.pi * freqs_z / width
        p_w = 2.0 * math.pi * freqs_w / width

        positions = positions[..., None, None, None, None, :]
        loc = (
            positions[..., 0] * p_x
            + positions[..., 1] * p_y
            + positions[..., 2] * p_z
            + positions[..., 3] * p_w
        ).view(*leading_dims, -1)

        half_dim = self.dimension // 2
        if loc.size(-1) < half_dim:
            raise ValueError(
                f"Fourier basis too small for dimension={self.dimension}. "
                f"Increase freqs; current basis width is {loc.size(-1)}."
            )
        loc = loc[:, :, :half_dim]
        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        if emb.size(-1) != self.dimension:
            raise RuntimeError(f"Expected embedding width {self.dimension}, got {emb.size(-1)}")
        return emb


class TemporalPatchFourierTokenizer(nn.Module):
    """
    Build one token per temporal patch.

    Input:  `[B, T, R]`
    Patches: `[B, P, time_patch, R]`
    Output: `[B, P, D]`

    Patch tokens are formed independently per patch to avoid pre-mask leakage:
    - content token: linear projection of the flattened raw patch `[time_patch * R] -> D`
    - position token: mean Fourier 4D embedding across all `(roi, time)` cells inside the patch
    """

    def __init__(
        self,
        *,
        input_dim: int,
        target_timepoints: int,
        time_patch: int,
        embed_dim: int,
        fourier_freqs: int,
        fourier_increment_time: float,
        fourier_margin: float = 0.4,
        pos_embed_mode: str = "sincos",
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim != 400:
            raise ValueError(f"TemporalPatchFourierTokenizer expects input_dim=400, got {input_dim}")
        if target_timepoints % time_patch != 0:
            raise ValueError(
                f"target_timepoints={target_timepoints} must be divisible by time_patch={time_patch}"
            )
        self.input_dim = int(input_dim)
        self.target_timepoints = int(target_timepoints)
        self.time_patch = int(time_patch)
        self.embed_dim = int(embed_dim)
        self.num_tokens = self.target_timepoints // self.time_patch
        self.patch_dim = self.time_patch * self.input_dim
        self.pos_embed_mode = str(pos_embed_mode)

        self.patch_proj = nn.Linear(self.patch_dim, self.embed_dim)
        self.fourier = FourierEmb4D(
            dimension=self.embed_dim,
            freqs=fourier_freqs,
            increment_time=fourier_increment_time,
            margin=fourier_margin,
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        roi_xyz = get_schaefer400_roi_centroids(space="normalized", as_tensor=True)
        self.register_buffer("roi_xyz", roi_xyz, persistent=True)
        patch_positions = self._build_patch_positions()
        self.register_buffer("patch_positions", patch_positions, persistent=True)

        if self.pos_embed_mode == "rotary":
            self.pos_embed = None
        elif self.pos_embed_mode == "sincos":
            self.register_buffer(
                "pos_embed",
                build_sincos_pos_embed(self.num_tokens, self.embed_dim),
                persistent=True,
            )
        else:
            raise ValueError(f"Unsupported pos_embed_mode: {self.pos_embed_mode}")

    def _build_patch_positions(self):
        """
        Build fixed cell-wise `(x, y, z, t)` positions for every patch.

        Output shape: `[P, time_patch * R, 4]`
        """
        num_rois = self.roi_xyz.size(0)
        if self.target_timepoints <= 1:
            global_times = torch.zeros(self.target_timepoints, dtype=self.roi_xyz.dtype)
        else:
            global_times = torch.linspace(0.0, 1.0, steps=self.target_timepoints, dtype=self.roi_xyz.dtype)

        patch_rows = []
        for patch_idx in range(self.num_tokens):
            start = patch_idx * self.time_patch
            end = start + self.time_patch
            patch_times = global_times[start:end]
            roi_pos = self.roi_xyz.unsqueeze(0).expand(self.time_patch, num_rois, 3)
            time_pos = patch_times.view(self.time_patch, 1, 1).expand(self.time_patch, num_rois, 1)
            patch_pos = torch.cat([roi_pos, time_pos], dim=-1)
            patch_rows.append(patch_pos.reshape(self.patch_dim, 4))
        return torch.stack(patch_rows, dim=0)

    def patchify(self, x: torch.Tensor):
        batch_size, num_timepoints, num_rois = x.shape
        if num_timepoints != self.target_timepoints or num_rois != self.input_dim:
            raise ValueError(
                f"Expected input shape [B, {self.target_timepoints}, {self.input_dim}], "
                f"got {tuple(x.shape)}"
            )
        return x.view(batch_size, self.num_tokens, self.time_patch, self.input_dim)

    def encode_tokens(self, x: torch.Tensor):
        patches = self.patchify(x)
        batch_size = patches.size(0)

        content = self.patch_proj(patches.reshape(batch_size, self.num_tokens, self.patch_dim))

        cell_positions = self.patch_positions.unsqueeze(0).expand(batch_size, -1, -1, -1)
        cell_positions = cell_positions.reshape(batch_size * self.num_tokens, self.patch_dim, 4)
        fourier = self.fourier(cell_positions)
        fourier = fourier.mean(dim=1).view(batch_size, self.num_tokens, self.embed_dim)

        tokens = self.norm(content + fourier)
        return self.dropout(tokens)

    def add_positions(self, tokens: torch.Tensor, indices: torch.Tensor | None = None):
        if self.pos_embed is None:
            return tokens
        if indices is None:
            pos = self.pos_embed[:, : tokens.size(1), :]
        else:
            pos = self.pos_embed[:, indices, :]
        return tokens + pos


class FourierPatchJEPA(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 400,
        target_timepoints: int = 45,
        embed_dim: int = 400,
        encoder_depth: int = 8,
        predictor_depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        ema_decay: float = 0.996,
        pos_embed_mode: str = "sincos",
        time_patch: int = 5,
        fourier_freqs: int = 4,
        fourier_increment_time: float = 0.1,
        fourier_margin: float = 0.4,
        num_site_classes: int = 0,
        site_adversary_hidden_dim: int = 256,
        site_grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.ema_decay = float(ema_decay)
        self.num_site_classes = int(num_site_classes)
        self.site_grl_lambda = float(site_grl_lambda)

        self.token_embed = TemporalPatchFourierTokenizer(
            input_dim=input_dim,
            target_timepoints=target_timepoints,
            time_patch=time_patch,
            embed_dim=embed_dim,
            fourier_freqs=fourier_freqs,
            fourier_increment_time=fourier_increment_time,
            fourier_margin=fourier_margin,
            pos_embed_mode=pos_embed_mode,
            dropout=dropout,
        )
        self.target_token_embed = TemporalPatchFourierTokenizer(
            input_dim=input_dim,
            target_timepoints=target_timepoints,
            time_patch=time_patch,
            embed_dim=embed_dim,
            fourier_freqs=fourier_freqs,
            fourier_increment_time=fourier_increment_time,
            fourier_margin=fourier_margin,
            pos_embed_mode=pos_embed_mode,
            dropout=dropout,
        )
        self.num_tokens = int(self.token_embed.num_tokens)
        self.pos_embed_mode = str(pos_embed_mode)

        self.context_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=encoder_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pos_embed_mode=pos_embed_mode,
        )
        self.target_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=encoder_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pos_embed_mode=pos_embed_mode,
        )
        predictor_pos = (
            None if pos_embed_mode == "rotary" else build_sincos_pos_embed(self.num_tokens, embed_dim)
        )
        self.predictor = Predictor(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=predictor_depth,
            num_tokens=self.num_tokens,
            pos_embed_mode=pos_embed_mode,
            pos_embed=predictor_pos,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.site_adversary = None
        if self.num_site_classes > 1:
            hidden_dim = int(site_adversary_hidden_dim)
            self.site_adversary = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_site_classes),
            )

        self.target_token_embed.load_state_dict(self.token_embed.state_dict())
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_token_embed.parameters():
            param.requires_grad = False
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        tau = self.ema_decay
        for target_param, context_param in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target_param.data.mul_(tau).add_(context_param.data, alpha=1 - tau)
        for target_param, context_param in zip(self.target_token_embed.parameters(), self.token_embed.parameters()):
            target_param.data.mul_(tau).add_(context_param.data, alpha=1 - tau)

    def _prepare_indices(self, indices: torch.Tensor | None, seq_len: int, device: torch.device):
        if indices is None:
            return torch.arange(seq_len, device=device, dtype=torch.long)
        if indices.ndim == 2:
            indices = indices[0]
        return indices.to(device=device, dtype=torch.long)

    def _encode_subset(
        self,
        tokenizer: TemporalPatchFourierTokenizer,
        encoder: TransformerEncoder,
        x: torch.Tensor,
        indices: torch.Tensor | None = None,
    ):
        full_tokens = tokenizer.encode_tokens(x)
        if indices is None:
            indices = torch.arange(full_tokens.size(1), device=x.device, dtype=torch.long)
            token_subset = full_tokens
        else:
            indices = self._prepare_indices(indices, full_tokens.size(1), x.device)
            token_subset = full_tokens[:, indices, :]
        token_subset = tokenizer.add_positions(
            token_subset,
            indices if tokenizer.pos_embed_mode != "rotary" else None,
        )
        positions = indices
        return encoder(token_subset, positions=positions)

    def encode_sequence(self, x: torch.Tensor):
        return self._encode_subset(self.token_embed, self.context_encoder, x, indices=None)

    def encode_pooled(self, x: torch.Tensor):
        return self.encode_sequence(x).mean(dim=1)

    @torch.no_grad()
    def encode_target_sequence(self, x: torch.Tensor):
        return self._encode_subset(self.target_token_embed, self.target_encoder, x, indices=None)

    def predict_site_logits(self, pooled: torch.Tensor, grl_lambda: float | None = None):
        if self.site_adversary is None:
            return None
        lambda_ = self.site_grl_lambda if grl_lambda is None else float(grl_lambda)
        return self.site_adversary(gradient_reverse(pooled, lambda_))

    def forward(
        self,
        x: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
        return_site_logits: bool = False,
        context_x: torch.Tensor | None = None,
        target_x: torch.Tensor | None = None,
    ):
        if context_x is None:
            context_x = x
        if target_x is None:
            target_x = x

        context_indices = self._prepare_indices(context_indices, self.num_tokens, x.device)
        target_indices = self._prepare_indices(target_indices, self.num_tokens, x.device)

        # JEPA-standard masking:
        # - context encoder only sees visible patches
        # - target encoder sees the full unmasked patch sequence
        context_repr = self._encode_subset(
            self.token_embed,
            self.context_encoder,
            context_x,
            indices=context_indices,
        )
        with torch.no_grad():
            full_target_repr = self.encode_target_sequence(target_x)
            target_repr = full_target_repr[:, target_indices, :]

        predicted = self.predictor(context_repr, context_indices, target_indices)
        pooled = self.encode_pooled(context_x)
        site_logits = self.predict_site_logits(pooled) if return_site_logits else None
        if return_site_logits:
            return predicted, target_repr, pooled, site_logits
        return predicted, target_repr, pooled


def build_model_from_config(config: dict[str, Any]):
    return FourierPatchJEPA(
        input_dim=int(config.get("input_dim", 400)),
        target_timepoints=int(config.get("target_timepoints", 45)),
        embed_dim=int(config.get("embed_dim", 400)),
        encoder_depth=int(config.get("encoder_depth", 8)),
        predictor_depth=int(config.get("predictor_depth", 2)),
        num_heads=int(config.get("num_heads", 8)),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        dropout=float(config.get("dropout", 0.1)),
        ema_decay=float(config.get("ema_decay", 0.996)),
        pos_embed_mode=config.get("pos_embed_mode", "sincos"),
        time_patch=int(config.get("time_patch", 5)),
        fourier_freqs=int(config.get("fourier_freqs", 4)),
        fourier_increment_time=float(config.get("fourier_increment_time", 0.1)),
        fourier_margin=float(config.get("fourier_margin", 0.4)),
        num_site_classes=int(config.get("num_site_classes", len(config.get("site_class_names", [])))),
        site_adversary_hidden_dim=int(config.get("site_adversary_hidden_dim", 256)),
        site_grl_lambda=float(config.get("site_grl_lambda", 1.0)),
    )
