# src/jepa/model.py

import copy
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Projects raw input tokens into the transformer's embedding space.
    
    Unlike NLP where tokens are discrete words (requiring nn.Embedding),
    our tokens are continuous vectors (400-dim for temporal, 93-dim for spatial).
    So we use a simple linear projection.
    
    Also adds learnable positional embeddings so the transformer knows
    WHERE each token is in the sequence (timepoint 5 vs timepoint 50,
    or parcel 10 vs parcel 300).
    """
    
    def __init__(self, input_dim: int, embed_dim: int, max_seq_len: int):
        super().__init__()
        
        # Linear projection: input_dim → embed_dim
        self.proj = nn.Linear(input_dim, embed_dim)
        
        # Learnable position embeddings: one vector per possible position
        # Shape: (1, max_seq_len, embed_dim)
        # The "1" is for broadcasting across the batch dimension
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # Initialize position embeddings with small random values
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x, indices=None):
        """
        Parameters
        ----------
        x : Tensor of shape (batch, num_tokens, input_dim)
            The raw token values (could be fewer than max_seq_len if masked).
        indices : LongTensor of shape (num_tokens,), optional
            The ORIGINAL position indices of these tokens in the full sequence.
            Needed because after masking, token 0 in the input might actually
            be position 5 in the original sequence.
            If None, assumes tokens are at positions 0, 1, 2, ...
        """
        # Project to embedding space
        x = self.proj(x)                    # (batch, num_tokens, embed_dim)
        
        # Add positional information
        if indices is not None:
            # Select only the position embeddings for the given indices
            pos = self.pos_embed[:, indices, :]   # (1, num_tokens, embed_dim)
        else:
            pos = self.pos_embed[:, :x.size(1), :]
        
        return x + pos


class TransformerEncoder(nn.Module):
    """
    Standard transformer encoder stack.
    
    This is the core of both the Context Encoder and Target Encoder.
    Uses PyTorch's built-in TransformerEncoderLayer for the heavy lifting.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, depth: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        """
        Parameters
        ----------
        embed_dim : int
            Dimension of token embeddings (e.g., 256).
        num_heads : int
            Number of attention heads. Must divide embed_dim evenly.
            Each head attends to embed_dim // num_heads dimensions.
        depth : int
            Number of transformer layers stacked.
        mlp_ratio : float
            The hidden layer in each transformer block's feedforward network
            is embed_dim * mlp_ratio wide. Default 4.0 is standard.
        dropout : float
            Dropout rate for regularization.
        """
        super().__init__()
        
        mlp_dim = int(embed_dim * mlp_ratio)
        
        # Each layer: Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',       # GELU is standard in modern transformers
            batch_first=True,         # so input is (batch, seq, dim) not (seq, batch, dim)
            norm_first=True,          # Pre-norm (more stable training than post-norm)
        )
        
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)    # Final normalization
    
    def forward(self, x):
        """
        x : (batch, num_tokens, embed_dim) → (batch, num_tokens, embed_dim)
        """
        x = self.layers(x)
        x = self.norm(x)
        return x


class Predictor(nn.Module):
    """
    A smaller transformer that predicts target representations from context.
    
    It receives:
      - Context representations (from the Context Encoder)
      - Learnable mask tokens at the positions that were masked
    And outputs predicted representations for the masked positions.
    
    It's intentionally smaller (fewer layers, possibly fewer heads) than
    the encoder — this forces the encoder to do the heavy representational
    work rather than the predictor memorizing everything.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, depth: int,
                 max_seq_len: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        # Learnable mask token — a single vector that stands in for
        # "I don't know what's here, please predict it"
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Predictor has its own positional embeddings
        # (it needs to know WHERE the masked tokens are)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Smaller transformer stack
        mlp_dim = int(embed_dim * mlp_ratio)
        pred_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(pred_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, context_repr, context_indices, target_indices):
        """
        Parameters
        ----------
        context_repr : (batch, num_context, embed_dim)
            Output of the Context Encoder for visible tokens.
        context_indices : LongTensor (num_context,)
            Original positions of context tokens.
        target_indices : LongTensor (num_target,)
            Original positions of masked tokens we need to predict.
        
        Returns
        -------
        predicted : (batch, num_target, embed_dim)
            Predicted representations for the masked positions.
        """
        batch_size = context_repr.size(0)
        num_target = len(target_indices)
        
        # Create mask tokens for each target position
        # Expand the single learnable mask_token to (batch, num_target, embed_dim)
        mask_tokens = self.mask_token.expand(batch_size, num_target, -1)
        
        # Add positional embeddings to both context repr and mask tokens
        context_with_pos = context_repr + self.pos_embed[:, context_indices, :]
        mask_with_pos    = mask_tokens  + self.pos_embed[:, target_indices,  :]
        
        # Concatenate: [context representations | mask tokens]
        # The transformer will attend over ALL of these
        full_seq = torch.cat([context_with_pos, mask_with_pos], dim=1)
        
        # Run through predictor transformer
        full_seq = self.layers(full_seq)
        full_seq = self.norm(full_seq)
        
        # Extract only the mask token outputs (the predictions)
        # They are the last num_target tokens in the sequence
        predicted = full_seq[:, -num_target:, :]
        
        return predicted


class JEPA(nn.Module):
    """
    Complete JEPA model combining all three components.
    
    Parameters
    ----------
    input_dim : int
        Dimension of each input token.
        Model A (temporal): 400  (each time step = 400 parcel values)
        Model B (spatial):  93   (each parcel = 93 time step values)
    max_seq_len : int
        Maximum sequence length.
        Model A (temporal): 93
        Model B (spatial):  400
    embed_dim : int
        Internal transformer dimension.
    encoder_depth : int
        Number of transformer layers in the encoder.
    predictor_depth : int
        Number of transformer layers in the predictor (should be smaller).
    num_heads : int
        Attention heads.
    ema_decay : float
        EMA decay rate for the Target Encoder. 0.996 is standard.
    """
    
    def __init__(self, input_dim: int, max_seq_len: int,
                 embed_dim: int = 256, encoder_depth: int = 6,
                 predictor_depth: int = 2, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 ema_decay: float = 0.996):
        super().__init__()
        
        self.ema_decay = ema_decay
        
        # ── Token Embedding (shared structure, but context & target are separate) ──
        self.token_embed = TokenEmbedding(input_dim, embed_dim, max_seq_len)
        
        # ── Context Encoder (trainable) ──
        self.context_encoder = TransformerEncoder(
            embed_dim, num_heads, encoder_depth, mlp_ratio, dropout
        )
        
        # ── Target Encoder (EMA copy — not trained by gradient) ──
        self.target_encoder = TransformerEncoder(
            embed_dim, num_heads, encoder_depth, mlp_ratio, dropout
        )
        # Initialize target encoder as exact copy of context encoder
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        # Freeze: no gradients ever flow through target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Target encoder needs its own token embedding too
        self.target_token_embed = TokenEmbedding(input_dim, embed_dim, max_seq_len)
        self.target_token_embed.load_state_dict(self.token_embed.state_dict())
        for param in self.target_token_embed.parameters():
            param.requires_grad = False
        
        # ── Predictor (trainable, smaller) ──
        self.predictor = Predictor(
            embed_dim, num_heads, predictor_depth, max_seq_len, mlp_ratio, dropout
        )
    
    @torch.no_grad()
    def update_target_encoder(self):
        """
        EMA update: slowly move target encoder toward context encoder.
        
        Called ONCE after each optimizer step.
        θ_target = τ * θ_target + (1 - τ) * θ_context
        
        With τ=0.996, the target encoder is ~99.6% its old self and
        ~0.4% the new context encoder. This creates a slowly-moving
        target that stabilizes training.
        """
        tau = self.ema_decay
        
        # Update encoder weights
        for target_param, context_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters()
        ):
            target_param.data.mul_(tau).add_(context_param.data, alpha=1 - tau)
        
        # Update token embedding weights
        for target_param, context_param in zip(
            self.target_token_embed.parameters(),
            self.token_embed.parameters()
        ):
            target_param.data.mul_(tau).add_(context_param.data, alpha=1 - tau)
    
    def forward(self, x, context_indices, target_indices):
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)
            Full input (before masking — we split inside here).
        context_indices : LongTensor (num_context,)
        target_indices  : LongTensor (num_target,)
        
        Returns
        -------
        predicted : (batch, num_target, embed_dim) — from predictor
        target_repr : (batch, num_target, embed_dim) — from target encoder
        """
        # ── Split input using mask indices ──
        context_tokens = x[:, context_indices, :]     # (batch, num_context, input_dim)
        target_tokens  = x[:, target_indices,  :]     # (batch, num_target,  input_dim)
        
        # ── Context path (trainable) ──
        context_embedded = self.token_embed(context_tokens, context_indices)
        context_repr = self.context_encoder(context_embedded)
        
        # ── Target path (no gradients) ──
        with torch.no_grad():
            target_embedded = self.target_token_embed(target_tokens, target_indices)
            target_repr = self.target_encoder(target_embedded)
        
        # ── Predictor: context repr → predicted target repr ──
        predicted = self.predictor(context_repr, context_indices, target_indices)
        
        return predicted, target_repr
    

