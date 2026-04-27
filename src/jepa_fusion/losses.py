from __future__ import annotations

import torch
import torch.nn.functional as F


def conditional_site_invariance_loss(
    core_loss: torch.Tensor,
    z: torch.Tensor,
    site: torch.Tensor,
    age: torch.Tensor,
    sex: torch.Tensor,
    *,
    lambda_cond: float = 0.1,
    lambda_rep: float = 0.05,
    sigma_age: float = 5.0,
    eta: float = 0.5,
    margin: float = 1.0,
    eps: float = 1e-8,
):
    """Combine a core JEPA loss with conditional cross-site attraction and repulsion.

    The regularizer attracts embeddings from different sites when they are
    biologically similar in age/sex, and repels them when they are not.

    Args:
        core_loss: Scalar JEPA loss tensor.
        z: Subject embeddings of shape ``[B, D]``.
        site: Site ids of shape ``[B]``.
        age: Ages of shape ``[B]``.
        sex: Binary sex labels of shape ``[B]`` with values 0/1.

    Returns:
        ``(loss_total, loss_cond, loss_rep)``

    Example:
        >>> import torch
        >>> from src.jepa_fusion.losses import conditional_site_invariance_loss
        >>> core_loss = torch.tensor(0.25, requires_grad=True)
        >>> z = torch.randn(8, 128, requires_grad=True)
        >>> site = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        >>> age = torch.tensor([22., 24., 23., 25., 31., 30., 29., 28.])
        >>> sex = torch.tensor([0, 1, 0, 1, 1, 1, 0, 0])
        >>> loss_total, loss_cond, loss_rep = conditional_site_invariance_loss(
        ...     core_loss, z, site, age, sex
        ... )
        >>> loss_total.backward()
    """
    if z.ndim != 2:
        raise ValueError(f"Expected z to have shape [B, D], got {tuple(z.shape)}")
    batch_size = z.shape[0]
    if batch_size < 2:
        zero = z.new_zeros(())
        return core_loss, zero, zero

    if site.ndim != 1 or age.ndim != 1 or sex.ndim != 1:
        raise ValueError("site, age, and sex must all have shape [B]")
    if not (site.shape[0] == age.shape[0] == sex.shape[0] == batch_size):
        raise ValueError(
            "site, age, and sex must all have the same batch dimension as z. "
            f"Got z={tuple(z.shape)}, site={tuple(site.shape)}, age={tuple(age.shape)}, sex={tuple(sex.shape)}"
        )

    dtype = z.dtype
    device = z.device

    # L2-normalize embeddings before pairwise comparisons.
    z_hat = F.normalize(z, p=2, dim=-1, eps=eps)

    # Pairwise masks. The diagonal is excluded explicitly.
    off_diag = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    cross_site = (site[:, None] != site[None, :]) & off_diag
    cross_site_f = cross_site.to(dtype)

    # Biological similarity weight w_ij.
    age = age.to(device=device, dtype=dtype)
    sex = sex.to(device=device)
    age_diff = age[:, None] - age[None, :]
    age_weight = torch.exp(-(age_diff.square()) / (2.0 * float(sigma_age) ** 2))
    sex_same = (sex[:, None] == sex[None, :]).to(dtype)
    sex_weight = float(eta) + (1.0 - float(eta)) * sex_same
    weight = age_weight * sex_weight

    # Pairwise distances on the unit sphere:
    # ||u - v||^2 = 2 - 2 <u, v>, which is more stable than explicit broadcasting.
    cosine = z_hat @ z_hat.T
    sq_dist = (2.0 - 2.0 * cosine).clamp_min(0.0)
    dist = torch.sqrt(sq_dist.clamp_min(eps))

    # Attraction across sites for biologically similar pairs.
    cond_weight = cross_site_f * weight
    cond_num = (cond_weight * sq_dist).sum()
    cond_den = cond_weight.sum() + float(eps)
    loss_cond = cond_num / cond_den

    # Repulsion across sites for biologically dissimilar pairs.
    rep_weight = cross_site_f * (1.0 - weight)
    rep_term = F.relu(float(margin) - dist).square()
    rep_num = (rep_weight * rep_term).sum()
    rep_den = rep_weight.sum() + float(eps)
    loss_rep = rep_num / rep_den

    loss_total = core_loss + (float(lambda_cond) * loss_cond) + (float(lambda_rep) * loss_rep)
    return loss_total, loss_cond, loss_rep


__all__ = ["conditional_site_invariance_loss"]
