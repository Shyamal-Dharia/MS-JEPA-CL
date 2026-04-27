import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from .data import WindowedFMRIDataset
from .repro import seed_everything, make_worker_init_fn, make_torch_generator
from .evaluate_subject_vote import build_model_from_checkpoint, aggregate_subject_predictions, compute_binary_metrics, threshold_scores, collate_fn

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

class FinetunedJEPA(nn.Module):
    def __init__(self, jepa_model):
        super().__init__()
        self.backbone = jepa_model
        # Unfreeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = True

        embed_dim = self.backbone.predictor.predictor.embed_dim
        # A simple linear head mapping the JEPA pooled state to a single logit (ASD vs HC)
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        # Forward completely through the JEPA context encoder
        pooled = self.backbone.encode_pooled(x)
        logits = self.head(pooled)
        return logits.squeeze(-1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dirs", nargs="+", default=["preprocessed_data_fir/full/roi_timeseries_abide_full"])
    parser.add_argument("--held-out-site", default="NYU")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("End2End-Finetune")
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading Base JEPA model...")
    base_model, cfg = build_model_from_checkpoint(args.checkpoint)
    model = FinetunedJEPA(base_model).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    logger.info("Loading dataset and splitting...")
    dataset = WindowedFMRIDataset(
        args.data_dirs,
        window_seconds=cfg.get("window_seconds", 90.0),
        stride_seconds=30.0,
        target_timepoints=cfg.get("target_timepoints", 45),
        sampling="sliding", # Must be sliding to capture all windows
        diag_filter=[0, 1]  # Only HC vs ASD
    )

    # Manually split WindowRecords by site
    train_records = []
    val_records = []

    for wr in dataset.window_records:
        if wr.record.site == args.held_out_site:
            val_records.append(wr)
        else:
            train_records.append(wr)

    # Swap out records dynamically for DataLoaders
    # Since Dataset uses self.window_records in __getitem__ we can literally create subsets
    class SubsetDataset(torch.utils.data.Dataset):
        def __init__(self, parent_dataset, subset_records):
            self.parent = parent_dataset
            self.window_records = subset_records

        def __len__(self):
            return len(self.window_records)

        def __getitem__(self, idx):
            old_records = self.parent.window_records
            self.parent.window_records = self.window_records
            item = self.parent[idx]
            self.parent.window_records = old_records
            return item

    train_ds = SubsetDataset(dataset, train_records)
    val_ds = SubsetDataset(dataset, val_records)

    logger.info(f"Train windows: {len(train_ds)} (Exclude {args.held_out_site})")
    logger.info(f"Validation windows: {len(val_ds)} (Strictly {args.held_out_site})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for x, meta_batch in train_loader:
            x = x.to(device)
            # labels 0=HC, 1=ASD
            y = torch.tensor([m["diag"] for m in meta_batch], dtype=torch.float32, device=device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation Pass (Zero-Shot Held Out)
        model.eval()
        val_losses = []
        val_metadata = []
        val_probs = []

        with torch.no_grad():
            for x, meta_batch in val_loader:
                x = x.to(device)
                y = torch.tensor([m["diag"] for m in meta_batch], dtype=torch.float32, device=device)

                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits).cpu().numpy()
                for m in meta_batch:
                    m["y"] = m["diag"] # Expected by aggregate_subject_predictions
                val_metadata.extend(meta_batch)
                val_probs.extend(probs.tolist())

        # Aggregate logic
        subject_rows = aggregate_subject_predictions(val_metadata, val_probs, classifier="logreg", threshold=0.5)
        if len(subject_rows) > 0:
            y_true = np.array([r["y"] for r in subject_rows], dtype=int)
            y_pred = np.array([r["pred"] for r in subject_rows], dtype=int)
            y_prob = np.array([r["prob"] for r in subject_rows], dtype=float)
            metrics = compute_binary_metrics(y_true, y_pred, y_prob)

            logger.info(
                f"Epoch {epoch+1:02d} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f} | "
                f"Val bACC: {metrics['bacc']*100:.1f}% | AUC: {metrics['auc']:.3f}"
            )
        else:
            logger.info(f"Epoch {epoch+1:02d} | No validation data found!")

if __name__ == "__main__":
    main()
