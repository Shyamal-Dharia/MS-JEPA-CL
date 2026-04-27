import torch


def _build_covariance_projection(
    input_dim: int,
    output_dim: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
):
    output_dim = int(min(max(1, output_dim), input_dim))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_matrix = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
    projection, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return projection.to(device=device, dtype=dtype)


def pool_sequence_features(
    sequence_embeddings: torch.Tensor,
    mode: str = "mean",
    covariance_dim: int = 32,
    covariance_seed: int = 0,
):
    if sequence_embeddings.ndim != 3:
        raise ValueError(
            f"Expected sequence embeddings of shape (batch, time, dim), got {tuple(sequence_embeddings.shape)}"
        )

    if mode == "mean":
        return sequence_embeddings.mean(dim=1)

    mean = sequence_embeddings.mean(dim=1)
    centered = sequence_embeddings - mean.unsqueeze(1)

    if mode == "mean_std":
        std = centered.pow(2).mean(dim=1).clamp_min(1e-6).sqrt()
        return torch.cat([mean, std], dim=-1)

    if mode == "mean_cov":
        batch_size, seq_len, embed_dim = sequence_embeddings.shape
        proj = _build_covariance_projection(
            input_dim=embed_dim,
            output_dim=covariance_dim,
            seed=covariance_seed,
            device=sequence_embeddings.device,
            dtype=sequence_embeddings.dtype,
        )
        projected = centered @ proj
        cov = projected.transpose(1, 2) @ projected
        cov = cov / max(1, seq_len - 1)
        tri = torch.triu_indices(cov.size(1), cov.size(2), device=cov.device)
        cov_upper = cov[:, tri[0], tri[1]]
        return torch.cat([mean, cov_upper], dim=-1)

    raise ValueError(f"Unsupported pooling mode: {mode}")
