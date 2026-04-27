import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from .pooling import pool_sequence_features


def collate_fn(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), list(labels)


@torch.no_grad()
def extract_embeddings(
    model,
    dataset,
    device,
    batch_size: int = 32,
    window_feature: str = "mean",
    covariance_dim: int = 32,
    covariance_seed: int = 0,
):
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    embeddings = []
    sites = []
    subjects = []

    for x, labels in loader:
        x = x.to(device)
        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        sequence = raw_model.encode_sequence(x)
        pooled = pool_sequence_features(
            sequence,
            mode=window_feature,
            covariance_dim=covariance_dim,
            covariance_seed=covariance_seed,
        )
        embeddings.append(pooled.cpu().numpy())
        sites.extend(label["site"] for label in labels)
        subjects.extend(label["subject"] for label in labels)

    return np.concatenate(embeddings, axis=0), np.array(sites), np.array(subjects)


def _balanced_accuracy_present(y_true, y_pred):
    recalls = []
    for cls in np.unique(y_true):
        cls_mask = y_true == cls
        recalls.append(float((y_pred[cls_mask] == cls).mean()))
    return float(np.mean(recalls))


def run_site_probe(
    model,
    dataset,
    device,
    batch_size: int = 32,
    seed: int = 42,
    window_feature: str = "mean",
    covariance_dim: int = 32,
    covariance_seed: int = 0,
):
    x, sites, subjects = extract_embeddings(
        model,
        dataset,
        device,
        batch_size=batch_size,
        window_feature=window_feature,
        covariance_dim=covariance_dim,
        covariance_seed=covariance_seed,
    )
    site_encoder = LabelEncoder()
    y = site_encoder.fit_transform(sites)

    # Use 5-Fold Subject-Grouped CV (as per paper protocol) instead of 1100-fold LOSO
    # This prevents the script from hanging for hours if probe_dataset is huge.
    n_groups = len(np.unique(subjects))
    cv = GroupKFold(n_splits=min(5, n_groups)) if n_groups >= 2 else LeaveOneGroupOut()

    fold_acc = []
    fold_bacc = []
    majority_acc = []
    majority_bacc = []

    for train_idx, test_idx in cv.split(x, y, groups=subjects):
        y_train = y[train_idx]
        y_test = y[test_idx]
        # Only skip if the TRAIN set has less than 2 classes (LogisticRegression requires >=2 to fit).
        # We can still evaluate if y_test is a single class (recall is just 0 or 1).
        if len(np.unique(y_train)) < 2:
            continue

        n_comp = min(50, len(y_train))
        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_comp, random_state=seed),
            LogisticRegression(
                C=0.1,
                max_iter=2000,
                random_state=seed,
                class_weight="balanced",
            ),
        )

        clf.fit(x[train_idx], y_train)
        pred = clf.predict(x[test_idx])
        fold_acc.append(accuracy_score(y_test, pred))
        fold_bacc.append(_balanced_accuracy_present(y_test, pred))

        majority_class = np.bincount(y_train).argmax()
        majority_pred = np.full_like(y_test, fill_value=majority_class)
        majority_acc.append(accuracy_score(y_test, majority_pred))
        majority_bacc.append(_balanced_accuracy_present(y_test, majority_pred))

    if not fold_acc:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "majority_accuracy": float("nan"),
            "majority_balanced_accuracy": float("nan"),
            "n_sites": len(site_encoder.classes_),
            "n_subjects": len(np.unique(subjects)),
            "cv": "grouped_5_fold",
        }

    return {
        "accuracy": float(np.mean(fold_acc)),
        "balanced_accuracy": float(np.mean(fold_bacc)),
        "majority_accuracy": float(np.mean(majority_acc)),
        "majority_balanced_accuracy": float(np.mean(majority_bacc)),
        "n_sites": len(site_encoder.classes_),
        "n_subjects": len(np.unique(subjects)),
        "cv": "grouped_5_fold",
    }
