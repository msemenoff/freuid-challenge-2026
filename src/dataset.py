"""FREUIDDataset – image loading, metadata, and stratified K-Fold splits."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

# Decode JPEGs via libjpeg-turbo (OpenCV) instead of system libjpeg (PIL): the
# latter segfaults (signal 11) on certain malformed JPEGs during long training
# runs, a C-level fault that LOAD_TRUNCATED_IMAGES cannot prevent.
cv2.setNumThreads(0)
# Tolerate truncated/malformed JPEGs in the PIL fallback path instead of segfaulting.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _encode_doc_type(doc_type: object) -> int:
    text = "" if pd.isna(doc_type) else str(doc_type)
    if not text or text == "UNKNOWN":
        return 0
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class FREUIDDataset(Dataset):
    """PyTorch Dataset for identity document images.

    Args:
        df:         DataFrame with at least columns [id, image_path].
                    Optional: label, is_digital, type.
        root:       Base directory prepended to image_path.
        transform:  Callable applied to PIL Image, returns tensor.
        has_labels: If True, expects a 'label' column in df.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root: str | Path,
        transform=None,
        has_labels: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.has_labels = has_labels

        # Pre-compute document-type integer codes for metadata features
        if "type" in df.columns:
            self.type_codes = np.array(
                [_encode_doc_type(doc_type) for doc_type in df["type"]],
                dtype=np.int64,
            )
        else:
            self.type_codes = np.zeros(len(df), dtype=np.int64)

        if "is_digital" in df.columns:
            self.is_digital = df["is_digital"].map(
                {True: 1, False: 0, 1: 1, 0: 0, "True": 1, "False": 0}
            ).fillna(-1).astype(np.int64).values
        else:
            self.is_digital = np.full(len(df), -1, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img_path = self.root / row["image_path"]
        # Robust decode via libjpeg-turbo (OpenCV); returns None on bad data
        # rather than segfaulting like system libjpeg. Fall back to PIL only if
        # OpenCV cannot read the file (e.g. non-JPEG formats).
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is not None:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        if self.transform is not None:
            image = self.transform(image)

        item = {
            "image": image,
            "id": str(row["id"]),
            "is_digital": int(self.is_digital[idx]),
            "doc_type_code": int(self.type_codes[idx]),
        }
        if self.has_labels:
            item["label"] = torch.tensor(float(row["label"]), dtype=torch.float32)
        return item

    # ------------------------------------------------------------------
    # Utility: return all unique document types seen in this split
    # ------------------------------------------------------------------
    def unique_types(self) -> list[str]:
        if "type" in self.df.columns:
            return sorted(self.df["type"].unique().tolist())
        return []


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def make_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    label_col: str = "label",
    strategy: str = "stratified",
    group_col: str = "type",
) -> pd.DataFrame:
    """Add a 'fold' column to df using the configured validation strategy."""
    df = df.copy()
    df["fold"] = -1

    if strategy == "stratified":
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(df, df[label_col])
    elif strategy == "ood_type":
        if group_col not in df.columns:
            raise KeyError(f"group_col '{group_col}' not found in dataframe")
        unique_groups = df[group_col].nunique(dropna=False)
        if unique_groups < n_folds:
            raise ValueError(
                f"Need at least {n_folds} unique groups in '{group_col}' for strategy='{strategy}', "
                f"found {unique_groups}."
            )
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(df, df[label_col], groups=df[group_col].astype(str))
    else:
        raise ValueError(f"Unsupported fold strategy: {strategy}")

    for fold_idx, (_, val_idx) in enumerate(split_iter):
        df.loc[val_idx, "fold"] = fold_idx

    return df


def get_train_val_datasets(
    df: pd.DataFrame,
    fold: int,
    root: str | Path,
    train_transform=None,
    val_transform=None,
    n_folds: int = 5,
    seed: int = 42,
    label_col: str = "label",
    strategy: str = "stratified",
    group_col: str = "type",
) -> tuple[FREUIDDataset, FREUIDDataset]:
    """Build train/val FREUIDDatasets for a given fold."""
    if "fold" not in df.columns:
        df = make_folds(
            df,
            n_folds=n_folds,
            seed=seed,
            label_col=label_col,
            strategy=strategy,
            group_col=group_col,
        )
    train_df = df[df["fold"] != fold]
    val_df = df[df["fold"] == fold]
    train_ds = FREUIDDataset(train_df, root=root, transform=train_transform, has_labels=True)
    val_ds = FREUIDDataset(val_df, root=root, transform=val_transform, has_labels=True)
    return train_ds, val_ds
