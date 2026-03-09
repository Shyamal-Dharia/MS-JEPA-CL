# src/jepa/masking.py

import random
import numpy as np
import torch


def generate_temporal_mask(seq_len: int, num_blocks: int = 2,
                           block_size_range: tuple = (10, 15)):
    """
    Masks contiguous blocks of time steps.

    Returns
    -------
    context_indices : LongTensor  — what the Context Encoder sees
    target_indices  : LongTensor  — what the Predictor must predict
    """
    all_indices = set(range(seq_len))
    masked = set()

    for _ in range(num_blocks):
        block_size = random.randint(*block_size_range)
        max_start = seq_len - block_size
        if max_start < 0:
            continue
        start = random.randint(0, max_start)
        masked |= set(range(start, start + block_size))

    context_indices = sorted(all_indices - masked)
    target_indices  = sorted(masked)

    return (torch.tensor(context_indices, dtype=torch.long),
            torch.tensor(target_indices,  dtype=torch.long))


def get_schaefer400_network_labels():
    """
    Derives network labels directly from the Schaefer 2018 atlas.
    Returns array of length 400: entry i → Yeo network ID (1-7).

    Instead of hardcoding parcel counts (which can be wrong due to
    asymmetric hemisphere sizes), we parse the network name from
    each parcel label string provided by nilearn.
    """
    from nilearn.datasets import fetch_atlas_schaefer_2018

    atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    labels = [lbl.decode('utf-8') if isinstance(lbl, bytes) else str(lbl)
              for lbl in atlas['labels']]

    # Each label looks like: "7Networks_LH_Vis_1" or "7Networks_RH_Default_pCunPCC_5"
    # The network name is always the 3rd underscore-separated field (index 2)
    network_names = [lbl.split('_')[2] for lbl in labels]

    # Map to integer IDs in order of first appearance (Vis=1, SomMot=2, ... Default=7)
    unique_networks = list(dict.fromkeys(network_names))
    name_to_id = {name: i + 1 for i, name in enumerate(unique_networks)}

    return np.array([name_to_id[n] for n in network_names])


def generate_spatial_mask(seq_len: int = 400, num_networks_to_mask: int = 2):
    """
    Masks entire Yeo networks from the Schaefer 400 atlas.

    Returns
    -------
    context_indices : LongTensor
    target_indices  : LongTensor
    """
    network_labels = get_schaefer400_network_labels()

    all_network_ids = list(range(1, 8))
    networks_to_mask = random.sample(all_network_ids, num_networks_to_mask)

    masked = set()
    for net_id in networks_to_mask:
        parcel_indices = np.where(network_labels == net_id)[0]
        masked |= set(parcel_indices.tolist())

    all_indices = set(range(seq_len))
    context_indices = sorted(all_indices - masked)
    target_indices  = sorted(masked)

    return (torch.tensor(context_indices, dtype=torch.long),
            torch.tensor(target_indices,  dtype=torch.long))