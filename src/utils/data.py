"""Competition dataframe helpers for released train/public-test layouts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _stringify_relpath(path: Path) -> str:
    return path.as_posix()


def _normalize_image_path(root: Path, image_path: str) -> str:
    rel_path = Path(image_path)
    if (root / rel_path).exists():
        return _stringify_relpath(rel_path)

    if rel_path.parts:
        nested_rel_path = Path(rel_path.parts[0]) / rel_path
        if (root / nested_rel_path).exists():
            return _stringify_relpath(nested_rel_path)

    return _stringify_relpath(rel_path)


def _normalize_image_paths(df: pd.DataFrame, root: Path, image_col: str) -> pd.DataFrame:
    if image_col in df.columns:
        df = df.copy()
        df[image_col] = [
            _normalize_image_path(root, image_path)
            for image_path in df[image_col].astype(str)
        ]
    return df


def filter_existing_images(
    df: pd.DataFrame,
    root: str | Path,
    image_col: str = "image_path",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    exists_mask = df[image_col].astype(str).map(lambda image_path: (root / image_path).exists())
    existing_df = df.loc[exists_mask].reset_index(drop=True)
    missing_df = df.loc[~exists_mask].reset_index(drop=True)
    return existing_df, missing_df


def _resolve_train_args(root: str | Path | object, train_csv: str | Path | None) -> tuple[Path, str | Path]:
    if train_csv is None and hasattr(root, "data"):
        cfg = root
        return Path(cfg.data.root), cfg.data.train_csv
    if train_csv is None:
        raise TypeError("train_csv is required when a config object is not provided")
    return Path(root), train_csv


def _resolve_test_args(
    root: str | Path | object,
    test_csv: str | Path | None,
    test_image_dir: str | Path | None,
    image_ext: str,
    id_col: str,
    image_col: str,
) -> tuple[Path, str | Path, str | Path, str, str, str]:
    if test_csv is None and hasattr(root, "data"):
        cfg = root
        return (
            Path(cfg.data.root),
            cfg.data.test_csv,
            cfg.data.test_image_dir,
            getattr(cfg.data, "test_image_ext", image_ext),
            getattr(cfg.data, "id_col", id_col),
            getattr(cfg.data, "image_col", image_col),
        )

    if test_csv is None or test_image_dir is None:
        raise TypeError(
            "test_csv and test_image_dir are required when a config object is not provided"
        )

    return Path(root), test_csv, test_image_dir, image_ext, id_col, image_col


def load_train_dataframe(
    root: str | Path | object,
    train_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Load the released training annotations as-is."""
    root, train_csv = _resolve_train_args(root, train_csv)
    df = pd.read_csv(root / train_csv)
    return _normalize_image_paths(df, root, image_col="image_path")


def load_public_test_dataframe(
    root: str | Path | object,
    test_csv: str | Path | None = None,
    test_image_dir: str | Path | None = None,
    image_ext: str = ".jpeg",
    id_col: str = "id",
    image_col: str = "image_path",
) -> pd.DataFrame:
    """Load the public-test ids and infer image paths and placeholder metadata.

    Kaggle's released public test CSV contains only `id,label`. The pipeline needs
    `image_path`, and metadata fields are unavailable at test time, so we fill them
    with neutral placeholders.
    """
    root, test_csv, test_image_dir, image_ext, id_col, image_col = _resolve_test_args(
        root,
        test_csv,
        test_image_dir,
        image_ext,
        id_col,
        image_col,
    )
    df = pd.read_csv(root / test_csv)

    if image_col not in df.columns:
        image_dir = Path(test_image_dir)
        df[image_col] = [
            _stringify_relpath(image_dir / f"{sample_id}{image_ext}")
            for sample_id in df[id_col].astype(str)
        ]

    if "is_digital" not in df.columns:
        df["is_digital"] = -1
    if "type" not in df.columns:
        df["type"] = "UNKNOWN"

    return df