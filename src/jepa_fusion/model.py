from __future__ import annotations

import math
import re
from typing import Any

from nilearn.datasets import fetch_atlas_schaefer_2018
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F

from ..jepa_clean.masking import get_schaefer400_network_labels


def build_sincos_pos_embed(max_seq_len: int, embed_dim: int):
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
    )
    pos_embed = torch.zeros(1, max_seq_len, embed_dim, dtype=torch.float32)
    pos_embed[0, :, 0::2] = torch.sin(positions * div_term)
    if embed_dim > 1:
        pos_embed[0, :, 1::2] = torch.cos(
            positions * div_term[: pos_embed[0, :, 1::2].shape[-1]]
        )
    return pos_embed


def _build_axis_sincos(size: int, embed_dim: int):
    if embed_dim % 2 != 0:
        raise ValueError(f"Axis sin-cos embedding requires even dim, got {embed_dim}")
    positions = torch.arange(size, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
    )
    pos_embed = torch.zeros(size, embed_dim, dtype=torch.float32)
    pos_embed[:, 0::2] = torch.sin(positions * div_term)
    pos_embed[:, 1::2] = torch.cos(positions * div_term[: pos_embed[:, 1::2].shape[-1]])
    return pos_embed


def build_2d_sincos_pos_embed(height: int, width: int, embed_dim: int):
    if embed_dim % 4 != 0:
        raise ValueError(f"2D sin-cos embedding requires embed_dim divisible by 4, got {embed_dim}")

    axis_dim = embed_dim // 2
    emb_h = _build_axis_sincos(height, axis_dim)
    emb_w = _build_axis_sincos(width, axis_dim)
    rows = []
    for h_idx in range(height):
        for w_idx in range(width):
            rows.append(torch.cat([emb_h[h_idx], emb_w[w_idx]], dim=0))
    return torch.stack(rows, dim=0).unsqueeze(0)


def _parse_schaefer_label(label: str):
    parts = label.split("_")
    hemi = parts[1] if len(parts) > 1 else "UNK"
    network = parts[2] if len(parts) > 2 else "UNK"
    match = re.search(r"_(\d+)$", label)
    parcel_index = int(match.group(1)) if match else 0
    return hemi, network, parcel_index


def get_roi_permutation(mode: str):
    atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    labels = [lbl.decode("utf-8") if isinstance(lbl, bytes) else str(lbl) for lbl in atlas["labels"]]
    indices = list(range(len(labels)))

    if mode == "none":
        return torch.as_tensor(indices, dtype=torch.long)
    if mode == "hemi":
        order = sorted(indices, key=lambda idx: (0 if _parse_schaefer_label(labels[idx])[0] == "LH" else 1, idx))
        return torch.as_tensor(order, dtype=torch.long)
    if mode == "hemi_network":
        order = sorted(
            indices,
            key=lambda idx: (
                0 if _parse_schaefer_label(labels[idx])[0] == "LH" else 1,
                _parse_schaefer_label(labels[idx])[1],
                _parse_schaefer_label(labels[idx])[2],
            ),
        )
        return torch.as_tensor(order, dtype=torch.long)
    raise ValueError(f"Unsupported reorder mode: {mode}")


def apply_rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def gradient_reverse(x: torch.Tensor, lambda_: float = 1.0):
    return _GradientReversal.apply(x, float(lambda_))


class RotaryMultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"Rotary attention requires an even head dimension, got head_dim={self.head_dim}"
            )

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = float(dropout)
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / float(self.head_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, positions: torch.Tensor, device: torch.device, dtype: torch.dtype):
        pos = positions.to(device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq.to(device))
        cos = freqs.cos()[None, None, :, :].to(dtype=dtype)
        sin = freqs.sin()[None, None, :, :].to(dtype=dtype)
        return cos, sin

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None):
        batch_size, seq_len, _ = x.shape
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)

        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self._cos_sin(positions=positions, device=x.device, dtype=q.dtype)
        q = apply_rotary_embedding(q, cos, sin)
        k = apply_rotary_embedding(k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn)


class RotaryTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        mlp_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = RotaryMultiheadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.ff_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None):
        x = x + self.dropout1(self.self_attn(self.norm1(x), positions=positions))
        ff = self.linear2(self.ff_dropout(self.activation(self.linear1(self.norm2(x)))))
        x = x + self.dropout2(ff)
        return x


class TemporalConvStem(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"Temporal conv stem expects an odd kernel size >= 1, got {kernel_size}")
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        residual = x
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.dropout(self.activation(x))
        return self.norm(residual + x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pos_embed_mode: str = "sincos",
    ):
        super().__init__()
        self.pos_embed_mode = pos_embed_mode
        if pos_embed_mode == "rotary":
            self.layers = nn.ModuleList(
                [
                    RotaryTransformerEncoderLayer(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                    for _ in range(depth)
                ]
            )
        else:
            mlp_dim = int(embed_dim * mlp_ratio)
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.layers = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None):
        if self.pos_embed_mode == "rotary":
            for layer in self.layers:
                x = layer(x, positions=positions)
            return self.norm(x)
        return self.norm(self.layers(x))


class Predictor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        num_tokens: int,
        *,
        pos_embed_mode: str = "sincos",
        pos_embed: torch.Tensor | None = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_mode = pos_embed_mode
        if pos_embed_mode == "rotary":
            self.pos_embed = None
        else:
            if pos_embed is None:
                pos_embed = build_sincos_pos_embed(num_tokens, embed_dim)
            self.register_buffer("pos_embed", pos_embed, persistent=True)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        if pos_embed_mode == "rotary":
            self.layers = nn.ModuleList(
                [
                    RotaryTransformerEncoderLayer(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                    for _ in range(depth)
                ]
            )
        else:
            mlp_dim = int(embed_dim * mlp_ratio)
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.layers = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context_repr: torch.Tensor, context_indices: torch.Tensor, target_indices: torch.Tensor):
        batch_size = context_repr.size(0)
        num_target = len(target_indices)
        mask_tokens = self.mask_token.expand(batch_size, num_target, -1)

        if self.pos_embed_mode == "rotary":
            full_seq = torch.cat([context_repr, mask_tokens], dim=1)
            positions = torch.cat([context_indices, target_indices]).to(context_repr.device)
            for layer in self.layers:
                full_seq = layer(full_seq, positions=positions)
            full_seq = self.norm(full_seq)
            return full_seq[:, -num_target:, :]

        context_with_pos = context_repr + self.pos_embed[:, context_indices, :]
        mask_with_pos = mask_tokens + self.pos_embed[:, target_indices, :]
        full_seq = torch.cat([context_with_pos, mask_with_pos], dim=1)
        full_seq = self.norm(self.layers(full_seq))
        return full_seq[:, -num_target:, :]


class TemporalTokenizer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        embed_dim: int,
        max_seq_len: int,
        pos_embed_mode: str,
        use_conv_stem: bool,
        conv_kernel_size: int,
        dropout: float,
        use_network_embeddings: bool,
    ):
        super().__init__()
        if use_network_embeddings and input_dim != 400:
            raise ValueError(
                f"Network-aware embeddings currently expect input_dim=400, got input_dim={input_dim}"
            )
        self.input_dim = int(input_dim)
        self.num_tokens = int(max_seq_len)
        self.embed_dim = int(embed_dim)
        self.pos_embed_mode = pos_embed_mode
        self.proj = nn.Linear(input_dim, embed_dim)
        self.conv_stem = (
            TemporalConvStem(embed_dim=embed_dim, kernel_size=conv_kernel_size, dropout=dropout)
            if use_conv_stem
            else nn.Identity()
        )
        if use_network_embeddings:
            network_ids = torch.as_tensor(get_schaefer400_network_labels(), dtype=torch.long) - 1
            self.register_buffer("roi_network_ids", network_ids, persistent=True)
            self.network_embed = nn.Parameter(torch.zeros(int(network_ids.max().item()) + 1, embed_dim))
            nn.init.trunc_normal_(self.network_embed, std=0.02)
        else:
            self.roi_network_ids = None
            self.network_embed = None

        if pos_embed_mode == "rotary":
            self.pos_embed = None
        elif pos_embed_mode == "sincos":
            self.register_buffer(
                "pos_embed",
                build_sincos_pos_embed(max_seq_len, embed_dim),
                persistent=True,
            )
        else:
            raise ValueError(f"Unsupported temporal pos_embed_mode: {pos_embed_mode}")

    def encode_tokens(self, x: torch.Tensor):
        raw_x = x
        tokens = self.proj(raw_x)
        if self.network_embed is not None:
            network_basis = self.network_embed[self.roi_network_ids]
            tokens = tokens + torch.einsum("bti,id->btd", raw_x, network_basis)
        return self.conv_stem(tokens)

    def add_positions(self, tokens: torch.Tensor, indices: torch.Tensor | None = None):
        if self.pos_embed is None:
            return tokens
        if indices is None:
            pos = self.pos_embed[:, : tokens.size(1), :]
        else:
            pos = self.pos_embed[:, indices, :]
        return tokens + pos


class SpatioTemporalPatchTokenizer(nn.Module):
    def __init__(
        self,
        *,
        target_timepoints: int,
        num_rois: int,
        embed_dim: int,
        time_patch: int,
        roi_group_size: int,
        reorder_mode: str = "hemi_network",
    ):
        super().__init__()
        if target_timepoints % time_patch != 0:
            raise ValueError(
                f"target_timepoints={target_timepoints} must be divisible by time_patch={time_patch}"
            )
        if num_rois % roi_group_size != 0:
            raise ValueError(
                f"num_rois={num_rois} must be divisible by roi_group_size={roi_group_size}"
            )
        self.target_timepoints = int(target_timepoints)
        self.num_rois = int(num_rois)
        self.time_patch = int(time_patch)
        self.roi_group_size = int(roi_group_size)
        self.num_time_patches = self.target_timepoints // self.time_patch
        self.num_roi_groups = self.num_rois // self.roi_group_size
        self.num_tokens = self.num_time_patches * self.num_roi_groups
        self.patch_dim = self.time_patch * self.roi_group_size
        self.embed_dim = int(embed_dim)
        self.pos_embed_mode = "sincos"
        self.reorder_mode = str(reorder_mode)
        self.proj = nn.Linear(self.patch_dim, embed_dim)

        self.register_buffer(
            "roi_permutation",
            get_roi_permutation(reorder_mode),
            persistent=True,
        )
        self.register_buffer(
            "pos_embed",
            build_2d_sincos_pos_embed(self.num_time_patches, self.num_roi_groups, embed_dim),
            persistent=True,
        )

    def patchify(self, x: torch.Tensor):
        x = x[:, :, self.roi_permutation]
        batch_size = x.size(0)
        x = x.view(
            batch_size,
            self.num_time_patches,
            self.time_patch,
            self.num_roi_groups,
            self.roi_group_size,
        )
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        return x.view(batch_size, self.num_tokens, self.patch_dim)

    def encode_tokens(self, x: torch.Tensor):
        patches = self.patchify(x)
        return self.proj(patches)

    def add_positions(self, tokens: torch.Tensor, indices: torch.Tensor | None = None):
        if indices is None:
            pos = self.pos_embed[:, : tokens.size(1), :]
        else:
            pos = self.pos_embed[:, indices, :]
        return tokens + pos


class FusionJEPA(nn.Module):
    def __init__(
        self,
        *,
        model_family: str = "temporal",
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
        use_conv_stem: bool = False,
        conv_kernel_size: int = 3,
        use_network_embeddings: bool = False,
        st_time_patch: int = 5,
        st_roi_group_size: int = 40,
        st_reorder_mode: str = "hemi_network",
        num_site_classes: int = 0,
        site_adversary_hidden_dim: int = 256,
        site_grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.model_family = str(model_family)
        self.ema_decay = float(ema_decay)
        self.num_site_classes = int(num_site_classes)
        self.site_grl_lambda = float(site_grl_lambda)

        if self.model_family == "temporal":
            self.token_embed = TemporalTokenizer(
                input_dim=input_dim,
                embed_dim=embed_dim,
                max_seq_len=target_timepoints,
                pos_embed_mode=pos_embed_mode,
                use_conv_stem=use_conv_stem,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                use_network_embeddings=use_network_embeddings,
            )
            predictor_pos = (
                None
                if pos_embed_mode == "rotary"
                else build_sincos_pos_embed(target_timepoints, embed_dim)
            )
            predictor_pos_mode = pos_embed_mode
        elif self.model_family == "st_roi_patch":
            self.token_embed = SpatioTemporalPatchTokenizer(
                target_timepoints=target_timepoints,
                num_rois=input_dim,
                embed_dim=embed_dim,
                time_patch=st_time_patch,
                roi_group_size=st_roi_group_size,
                reorder_mode=st_reorder_mode,
            )
            predictor_pos = build_2d_sincos_pos_embed(
                self.token_embed.num_time_patches,
                self.token_embed.num_roi_groups,
                embed_dim,
            )
            predictor_pos_mode = "sincos"
            pos_embed_mode = "sincos"
        else:
            raise ValueError(f"Unsupported model_family: {model_family}")

        self.num_tokens = int(self.token_embed.num_tokens)
        self.pos_embed_mode = pos_embed_mode
        self.context_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=encoder_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pos_embed_mode=pos_embed_mode,
        )
        self.target_token_embed = type(self.token_embed)(**self._tokenizer_kwargs())  # type: ignore[misc]
        self.target_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=encoder_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pos_embed_mode=pos_embed_mode,
        )
        self.predictor = Predictor(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=predictor_depth,
            num_tokens=self.num_tokens,
            pos_embed_mode=predictor_pos_mode,
            pos_embed=predictor_pos,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        mlp_dim_global = int(embed_dim * mlp_ratio)
        self.global_predictor = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim_global),
            nn.BatchNorm1d(mlp_dim_global),
            nn.GELU(),
            nn.Linear(mlp_dim_global, embed_dim),
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

    def _tokenizer_kwargs(self):
        if isinstance(self.token_embed, TemporalTokenizer):
            return {
                "input_dim": self.token_embed.input_dim,
                "embed_dim": self.token_embed.embed_dim,
                "max_seq_len": self.token_embed.num_tokens,
                "pos_embed_mode": self.token_embed.pos_embed_mode,
                "use_conv_stem": not isinstance(self.token_embed.conv_stem, nn.Identity),
                "conv_kernel_size": (
                    self.token_embed.conv_stem.conv.kernel_size[0]
                    if isinstance(self.token_embed.conv_stem, TemporalConvStem)
                    else 3
                ),
                "dropout": (
                    self.token_embed.conv_stem.dropout.p
                    if isinstance(self.token_embed.conv_stem, TemporalConvStem)
                    else 0.1
                ),
                "use_network_embeddings": self.token_embed.network_embed is not None,
            }
        if isinstance(self.token_embed, SpatioTemporalPatchTokenizer):
            return {
                "target_timepoints": self.token_embed.target_timepoints,
                "num_rois": self.token_embed.num_rois,
                "embed_dim": self.token_embed.embed_dim,
                "time_patch": self.token_embed.time_patch,
                "roi_group_size": self.token_embed.roi_group_size,
                "reorder_mode": self.token_embed.reorder_mode,
            }
        raise TypeError(f"Unsupported tokenizer type: {type(self.token_embed)}")

    @torch.no_grad()
    def update_target_encoder(self):
        tau = self.ema_decay
        for target_param, context_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            target_param.data.mul_(tau).add_(context_param.data, alpha=1 - tau)
        for target_param, context_param in zip(
            self.target_token_embed.parameters(),
            self.token_embed.parameters(),
        ):
            target_param.data.mul_(tau).add_(context_param.data, alpha=1 - tau)

    def _prepare_indices(self, indices: torch.Tensor | None, seq_len: int, device: torch.device):
        if indices is None:
            return torch.arange(seq_len, device=device, dtype=torch.long)
        if indices.ndim == 2:
            indices = indices[0]
        return indices.to(device=device, dtype=torch.long)

    def _encode_branch_tokens(
        self,
        tokenizer: TemporalTokenizer | SpatioTemporalPatchTokenizer,
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
        token_subset = tokenizer.add_positions(token_subset, indices if tokenizer.pos_embed_mode != "rotary" else None)
        positions = indices if tokenizer.pos_embed_mode == "rotary" else indices
        return encoder(token_subset, positions=positions)

    def encode_sequence(self, x: torch.Tensor):
        return self._encode_branch_tokens(self.token_embed, self.context_encoder, x)

    def encode_pooled(self, x: torch.Tensor):
        return self.encode_sequence(x).mean(dim=1)

    @torch.no_grad()
    def encode_target_sequence(self, x: torch.Tensor):
        return self._encode_branch_tokens(self.target_token_embed, self.target_encoder, x)

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
        return_global: bool = False,
        return_site_logits: bool = False,
    ):
        context_indices = self._prepare_indices(context_indices, self.num_tokens, x.device)
        target_indices = self._prepare_indices(target_indices, self.num_tokens, x.device)

        context_repr = self._encode_branch_tokens(
            self.token_embed,
            self.context_encoder,
            x,
            indices=context_indices,
        )
        with torch.no_grad():
            target_repr = self._encode_branch_tokens(
                self.target_token_embed,
                self.target_encoder,
                x,
                indices=target_indices,
            )
        predicted = self.predictor(context_repr, context_indices, target_indices)
        pooled = self.encode_pooled(x)
        site_logits = self.predict_site_logits(pooled) if return_site_logits else None
        if return_global:
            context_global = context_repr.mean(dim=1)
            predicted_global = self.global_predictor(context_global)
            with torch.no_grad():
                full_target_encoded = self.encode_target_sequence(x)
                target_global = full_target_encoded.mean(dim=1)
            if return_site_logits:
                return predicted, target_repr, pooled, predicted_global, target_global, site_logits
            return predicted, target_repr, pooled, predicted_global, target_global
        if return_site_logits:
            return predicted, target_repr, pooled, site_logits
        return predicted, target_repr, pooled


def build_model_from_config(config: dict[str, Any]):
    return FusionJEPA(
        model_family=config.get("model_family", "temporal"),
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
        use_conv_stem=bool(config.get("use_conv_stem", False)),
        conv_kernel_size=int(config.get("conv_kernel_size", 3)),
        use_network_embeddings=bool(config.get("use_network_embeddings", False)),
        st_time_patch=int(config.get("st_time_patch", 5)),
        st_roi_group_size=int(config.get("st_roi_group_size", 40)),
        st_reorder_mode=config.get("st_reorder_mode", "hemi_network"),
        num_site_classes=int(config.get("num_site_classes", len(config.get("site_class_names", [])))),
        site_adversary_hidden_dim=int(config.get("site_adversary_hidden_dim", 256)),
        site_grl_lambda=float(config.get("site_grl_lambda", 1.0)),
    )
