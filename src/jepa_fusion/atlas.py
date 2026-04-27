from __future__ import annotations

from functools import lru_cache

import nibabel as nib
from nilearn.datasets import fetch_atlas_schaefer_2018
import numpy as np
import torch


@lru_cache(maxsize=1)
def _load_schaefer400_centroids_and_labels():
    atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    labels = tuple(
        lbl.decode("utf-8") if isinstance(lbl, bytes) else str(lbl)
        for lbl in atlas["labels"]
    )

    img = nib.load(atlas["maps"])
    data = np.asarray(img.get_fdata(), dtype=np.int32)
    affine = img.affine

    centroids = []
    for label_idx in range(1, len(labels) + 1):
        voxels_ijk = np.argwhere(data == label_idx)
        if voxels_ijk.size == 0:
            raise RuntimeError(f"Schaefer atlas label {label_idx} has no voxels.")
        voxels_xyz = nib.affines.apply_affine(affine, voxels_ijk)
        centroids.append(voxels_xyz.mean(axis=0))

    centroids = np.asarray(centroids, dtype=np.float32)
    return centroids, labels


def get_schaefer400_roi_labels():
    _, labels = _load_schaefer400_centroids_and_labels()
    return list(labels)


def get_schaefer400_roi_centroids(space: str = "mni", as_tensor: bool = True):
    """
    Return Schaefer-400 ROI centroids in atlas order.

    `space="mni"` returns raw MNI coordinates in millimeters.
    `space="normalized"` rescales each axis to `[0, 1]`, preserving atlas order.
    """
    centroids, _ = _load_schaefer400_centroids_and_labels()

    if space == "mni":
        coords = centroids.copy()
    elif space == "normalized":
        mins = centroids.min(axis=0, keepdims=True)
        maxs = centroids.max(axis=0, keepdims=True)
        denom = np.clip(maxs - mins, a_min=1e-6, a_max=None)
        coords = (centroids - mins) / denom
    else:
        raise ValueError(f"Unsupported space: {space!r}. Expected 'mni' or 'normalized'.")

    if as_tensor:
        return torch.as_tensor(coords, dtype=torch.float32)
    return coords
