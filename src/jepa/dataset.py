# src/jepa/dataset.py

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class FMRIDataset(Dataset):
    """
    Loads preprocessed fMRI ROI timeseries from a directory.

    Each sample consists of:
      - A tensor of shape (seq_len, input_dim) ready for a transformer
      - A dictionary of metadata labels (site, diagnosis, subject ID)

    Parameters
    ----------
    data_dir : str
        Path to the folder containing *_roi_ts.npy and *_meta.json files.
    tokenization_mode : str
        'temporal'  →  shape (93,  400)  — time is the sequence dimension
        'spatial'   →  shape (400,  93)  — parcels are the sequence dimension
    """

    def __init__(self, data_dir: str, tokenization_mode: str = 'temporal',
                 augment: bool = False, num_views: int = 1):

        assert tokenization_mode in ('temporal', 'spatial'), \
            "tokenization_mode must be 'temporal' or 'spatial'"

        self.mode = tokenization_mode
        self.augment = augment
        self.num_views = num_views

        # Collect every .npy file; the matching .json has the same stem
        self.ts_files = sorted(glob.glob(os.path.join(data_dir, '*_roi_ts.npy')))

        if len(self.ts_files) == 0:
            raise FileNotFoundError(f"No *_roi_ts.npy files found in {data_dir}")

    def __len__(self):
        return len(self.ts_files)

    def __getitem__(self, idx):
        ts_path = self.ts_files[idx]

        # ── 1. Load the timeseries ──
        ts = np.load(ts_path).astype(np.float32)

        # ── 2. Convert to tensor ──
        ts_tensor = torch.from_numpy(ts)            # (400, 93)

        # ── 3. Reshape for the chosen mode ──
        if self.mode == 'temporal':
            # Raw is (93, 400) — already (seq_len=93, input_dim=400)
            # No transpose needed
            pass

        else:  # spatial
            # We need (seq_len=400, input_dim=93)
            # Raw is (93, 400) so transpose: .T gives (400, 93)
            ts_tensor = ts_tensor.T

        # ── 4. Load metadata ──
        meta_path = ts_path.replace('_roi_ts.npy', '_meta.json')
        with open(meta_path) as f:
            meta = json.load(f)

        labels = {
            'subject': meta['subject'],
            'site':    meta['site'],
            'diag':    meta.get('diag', -1),
            'diag_name': meta.get('diag_name', 'unknown'),
        }

        if self.num_views > 1:
            views = []
            for _ in range(self.num_views):
                v = augment_fmri(ts_tensor, self.mode) if self.augment else ts_tensor.clone()
                views.append(v)
            return views, labels
        else:
            if self.augment:
                ts_tensor = augment_fmri(ts_tensor, self.mode)
            return ts_tensor, labels

def augment_fmri(x_tensor, mode='temporal'):
    """
    Applies fMRI-specific data augmentations to a single tensor.
    x_tensor: shape (seq_len, input_dim) e.g. (93, 400) for temporal.
    """
    x = x_tensor.clone()
    seq_len, input_dim = x.shape
    
    # 1. Temporal Jitter (shift along seq_len for temporal, along input_dim for spatial)
    shift = torch.randint(-3, 4, (1,)).item()
    if shift != 0:
        if mode == 'temporal':
            if shift > 0: # shift right, pad left with first time step
                pad = x[0:1, :].expand(shift, input_dim)
                x = torch.cat([pad, x[:-shift, :]], dim=0)
            else: # shift left, pad right with last time step
                shift = -shift
                pad = x[-1:, :].expand(shift, input_dim)
                x = torch.cat([x[shift:, :], pad], dim=0)
        else: # spatial
            if shift > 0: # shift right, pad left
                pad = x[:, 0:1].expand(seq_len, shift)
                x = torch.cat([pad, x[:, :-shift]], dim=1)
            else:
                shift = -shift
                pad = x[:, -1:].expand(seq_len, shift)
                x = torch.cat([x[:, shift:], pad], dim=1)
                
    # 2. Gaussian Noise
    x = x + torch.randn_like(x) * 0.01
    
    # 3. Parcel Dropout
    if mode == 'temporal':
        # Drop 5% of parcels (input_dim=400)
        mask = torch.rand(input_dim) > 0.05
        mask = mask.float().unsqueeze(0) # (1, 400)
        x = x * mask
    else:
        # spatial: parcels are seq_len=400
        mask = torch.rand(seq_len) > 0.05
        mask = mask.float().unsqueeze(1) # (400, 1)
        x = x * mask
        
    return x