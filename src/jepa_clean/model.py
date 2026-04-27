import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masking import get_schaefer400_network_labels


def build_sincos_pos_embed(max_seq_len: int, embed_dim: int):
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
    )
    pos_embed = torch.zeros(1, max_seq_len, embed_dim, dtype=torch.float32)
    pos_embed[0, :, 0::2] = torch.sin(positions * div_term)
    if embed_dim > 1:
        pos_embed[0, :, 1::2] = torch.cos(positions * div_term[: pos_embed[0, :, 1::2].shape[-1]])
    return pos_embed


def apply_rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)


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


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        max_seq_len: int,
        pos_embed_mode: str = "learned",
        use_conv_stem: bool = False,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
        use_roi_embeddings: bool = False,
        use_network_embeddings: bool = False,
    ):
        super().__init__()
        if use_network_embeddings and input_dim != 400:
            raise ValueError(
                f"Network-aware ROI embeddings currently expect input_dim=400, got input_dim={input_dim}"
            )
        self.proj = nn.Linear(input_dim, embed_dim)
        self.pos_embed_mode = pos_embed_mode
        self.use_conv_stem = use_conv_stem
        self.use_roi_embeddings = use_roi_embeddings
        self.use_network_embeddings = use_network_embeddings
        self.conv_stem = (
            TemporalConvStem(embed_dim=embed_dim, kernel_size=conv_kernel_size, dropout=dropout)
            if use_conv_stem
            else nn.Identity()
        )
        if use_roi_embeddings:
            self.roi_embed = nn.Parameter(torch.zeros(input_dim, embed_dim))
            nn.init.trunc_normal_(self.roi_embed, std=0.02)
        else:
            self.roi_embed = None
        if use_network_embeddings:
            network_ids = torch.as_tensor(get_schaefer400_network_labels(), dtype=torch.long) - 1
            self.register_buffer("roi_network_ids", network_ids, persistent=True)
            self.network_embed = nn.Parameter(torch.zeros(int(network_ids.max().item()) + 1, embed_dim))
            nn.init.trunc_normal_(self.network_embed, std=0.02)
        else:
            self.roi_network_ids = None
            self.network_embed = None
        if pos_embed_mode == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_mode == "sincos":
            self.register_buffer("pos_embed", build_sincos_pos_embed(max_seq_len, embed_dim), persistent=True)
        elif pos_embed_mode == "rotary":
            self.pos_embed = None
        else:
            raise ValueError(f"Unsupported pos_embed_mode: {pos_embed_mode}")

    def forward(self, x, indices=None):
        raw_x = x
        x = self.proj(raw_x)
        if self.use_roi_embeddings or self.use_network_embeddings:
            roi_basis = None
            if self.roi_embed is not None:
                roi_basis = self.roi_embed
            if self.network_embed is not None:
                network_basis = self.network_embed[self.roi_network_ids]
                roi_basis = network_basis if roi_basis is None else roi_basis + network_basis
            x = x + torch.einsum("bti,id->btd", raw_x, roi_basis)
        x = self.conv_stem(x)
        if self.pos_embed_mode == "rotary":
            return x
        if indices is None:
            pos = self.pos_embed[:, : x.size(1), :]
        else:
            pos = self.pos_embed[:, indices, :]
        return x + pos


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pos_embed_mode: str = "learned",
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

    def forward(self, x, positions: torch.Tensor | None = None):
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
        max_seq_len: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pos_embed_mode: str = "learned",
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_mode = pos_embed_mode
        if pos_embed_mode == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_mode == "sincos":
            self.register_buffer("pos_embed", build_sincos_pos_embed(max_seq_len, embed_dim), persistent=True)
        elif pos_embed_mode == "rotary":
            self.pos_embed = None
        else:
            raise ValueError(f"Unsupported pos_embed_mode: {pos_embed_mode}")
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

    def forward(self, context_repr, context_indices, target_indices):
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


class JEPA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        max_seq_len: int,
        embed_dim: int = 128,
        encoder_depth: int = 3,
        predictor_depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        ema_decay: float = 0.996,
        pos_embed_mode: str = "learned",
        use_conv_stem: bool = False,
        conv_kernel_size: int = 3,
        use_roi_embeddings: bool = False,
        use_network_embeddings: bool = False,
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.num_tokens = max_seq_len
        self.pos_embed_mode = pos_embed_mode
        self.token_embed = TokenEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            pos_embed_mode=pos_embed_mode,
            use_conv_stem=use_conv_stem,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            use_roi_embeddings=use_roi_embeddings,
            use_network_embeddings=use_network_embeddings,
        )
        self.context_encoder = TransformerEncoder(
            embed_dim, num_heads, encoder_depth, mlp_ratio, dropout, pos_embed_mode=pos_embed_mode
        )
        self.target_token_embed = TokenEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            pos_embed_mode=pos_embed_mode,
            use_conv_stem=use_conv_stem,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            use_roi_embeddings=use_roi_embeddings,
            use_network_embeddings=use_network_embeddings,
        )
        self.target_encoder = TransformerEncoder(
            embed_dim, num_heads, encoder_depth, mlp_ratio, dropout, pos_embed_mode=pos_embed_mode
        )
        self.predictor = Predictor(
            embed_dim,
            num_heads,
            predictor_depth,
            self.num_tokens,
            mlp_ratio,
            dropout,
            pos_embed_mode=pos_embed_mode,
        )

        mlp_dim_global = int(embed_dim * mlp_ratio)
        self.global_predictor = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim_global),
            nn.BatchNorm1d(mlp_dim_global),
            nn.GELU(),
            nn.Linear(mlp_dim_global, embed_dim)
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

    def encode_sequence(self, x):
        embedded = self.token_embed(x)
        positions = torch.arange(x.size(1), device=x.device)
        return self.context_encoder(embedded, positions=positions)

    def encode_pooled(self, x):
        return self.encode_sequence(x).mean(dim=1)

    @torch.no_grad()
    def encode_target_sequence(self, x):
        embedded = self.target_token_embed(x)
        positions = torch.arange(x.size(1), device=x.device)
        return self.target_encoder(embedded, positions=positions)

    def forward_with_target(self, context_x, target_x, context_indices, target_indices):
        context_tokens = context_x[:, context_indices, :]
        target_tokens = target_x[:, target_indices, :]

        context_embedded = self.token_embed(context_tokens, context_indices)
        context_repr = self.context_encoder(context_embedded, positions=context_indices)

        with torch.no_grad():
            target_embedded = self.target_token_embed(target_tokens, target_indices)
            target_repr = self.target_encoder(target_embedded, positions=target_indices)

        predicted = self.predictor(context_repr, context_indices, target_indices)
        return predicted, target_repr

    def forward(self, x, context_indices, target_indices, return_global=False):
        if context_indices.ndim == 2:
            context_indices = context_indices[0]
        if target_indices.ndim == 2:
            target_indices = target_indices[0]

        context_tokens = x[:, context_indices, :]

        predicted, target_repr = self.forward_with_target(x, x, context_indices, target_indices)
        pooled = self.encode_pooled(x)

        if return_global:
            context_embedded = self.token_embed(context_tokens, context_indices)
            context_repr = self.context_encoder(context_embedded, positions=context_indices)

            context_global = context_repr.mean(dim=1)
            predicted_global = self.global_predictor(context_global)

            with torch.no_grad():
                full_target_encoded = self.encode_target_sequence(x)
                target_global = full_target_encoded.mean(dim=1)

            return predicted, target_repr, pooled, predicted_global, target_global

        return predicted, target_repr, pooled
