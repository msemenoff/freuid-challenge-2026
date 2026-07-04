"""Batch feature extraction pipeline.

Runs all three branches (ConvNeXtV2, DINOv2, physical) over a dataset split
and saves the resulting feature arrays and metadata to features/.

Usage:
    python src/extract_features.py --split train --fold 0
    python src/extract_features.py --split test
    python src/extract_features.py --split all   # train+val together (no fold split)

Output files in features/<split>/:
    global_feats.npy         (N, 1536)
    dinov2_feats.npy         (N, dino_dim)
    physical_feats.npy       (N, phys_dim)
    metadata.npy             (N, 2)  [is_digital, doc_type_code]
    labels.npy               (N,)    only for train splits
    ids.npy                  (N,)    string IDs
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.data import filter_existing_images, load_public_test_dataframe, load_train_dataframe
from src.utils.seed import seed_everything
from src.dataset import FREUIDDataset, make_folds
from src.transforms import get_val_transform
from src.models.global_backbone import build_global_backbone, build_dinov2
from src.models.physical_features import extract_physical_features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Optional override for extraction resize; defaults to cfg.augmentation.image_size.",
    )
    p.add_argument(
        "--branches",
        nargs="+",
        choices=["global", "dinov2", "physical"],
        default=["global", "dinov2", "physical"],
        help="Feature branches to extract. Defaults to all branches.",
    )
    return p.parse_args()


@torch.inference_mode()
def extract_nn_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> np.ndarray:
    model.eval()
    model.to(device)
    all_feats = []
    for batch in tqdm(dataloader, desc=f"  {model.__class__.__name__}"):
        imgs = batch["image"].to(device, non_blocking=True)
        feats = model(imgs)
        all_feats.append(feats.cpu().float().numpy())
    return np.concatenate(all_feats, axis=0)


def extract_physical_batch(df: pd.DataFrame, root: Path) -> np.ndarray:
    feats = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Physical"):
        img_path = root / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        f = extract_physical_features(img)
        feats.append(f)
    return np.stack(feats, axis=0)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.data.seed)
    root = Path(cfg.data.root)
    out_dir = Path(cfg.paths.features_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load DataFrames ----
    is_test = args.split == "test"
    if is_test:
        df = load_public_test_dataframe(
            root=root,
            test_csv=cfg.data.test_csv,
            test_image_dir=cfg.data.test_image_dir,
            image_ext=cfg.data.test_image_ext,
            id_col=cfg.data.id_col,
            image_col=cfg.data.image_col,
        )
        df, missing_df = filter_existing_images(df, root=root, image_col=cfg.data.image_col)
        if len(missing_df) > 0:
            print(f"Skipping {len(missing_df)} test rows with missing local image files.")
        has_labels = False
    else:
        df = load_train_dataframe(root=root, train_csv=cfg.data.train_csv)
        has_labels = True
        if args.split in ("train", "val"):
            df = make_folds(
                df,
                n_folds=cfg.data.n_folds,
                seed=cfg.data.seed,
                label_col=cfg.data.label_col,
                strategy=getattr(cfg.data, "fold_strategy", "stratified"),
                group_col=getattr(cfg.data, "fold_group_col", "type"),
            )
            if args.split == "train":
                df = df[df["fold"] != args.fold].reset_index(drop=True)
            else:
                df = df[df["fold"] == args.fold].reset_index(drop=True)

    print(f"Extracting features for {len(df)} samples → {out_dir}")

    selected_branches = set(args.branches)
    dataset = None
    loader = None
    if selected_branches & {"global", "dinov2"}:
        # ---- Build dataset/loader only when a neural branch is requested ----
        image_size = args.image_size or cfg.augmentation.image_size
        transform = get_val_transform(image_size=image_size)
        dataset = FREUIDDataset(df, root=root, transform=transform, has_labels=has_labels)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        dataset = FREUIDDataset(df, root=root, transform=None, has_labels=has_labels)

    device = args.device
    print(f"Device: {device}")

    # ---- Branch 1: ConvNeXtV2 ----
    if "global" in selected_branches:
        print("Branch 1: ConvNeXtV2...")
        global_model = build_global_backbone(cfg)
        global_feats = extract_nn_features(global_model, loader, device)
        np.save(out_dir / "global_feats.npy", global_feats)
        print(f"  → {global_feats.shape}")
        del global_model

    # ---- Branch 2: DINOv2 ----
    if "dinov2" in selected_branches:
        print("Branch 2: DINOv2...")
        dino_model = build_dinov2(cfg)
        dinov2_feats = extract_nn_features(dino_model, loader, device)
        np.save(out_dir / "dinov2_feats.npy", dinov2_feats)
        print(f"  → {dinov2_feats.shape}")
        del dino_model

    # ---- Branch 3: Physical ----
    if "physical" in selected_branches:
        print("Branch 3: Physical artifacts...")
        physical_feats = extract_physical_batch(df, root)
        np.save(out_dir / "physical_feats.npy", physical_feats)
        print(f"  → {physical_feats.shape}")

    # ---- Metadata ----
    meta = np.stack([
        dataset.is_digital.astype(np.float32),
        dataset.type_codes.astype(np.float32),
    ], axis=1)
    np.save(out_dir / "metadata.npy", meta)

    # ---- IDs + labels ----
    ids = df[cfg.data.id_col].values.astype(str)
    np.save(out_dir / "ids.npy", ids)
    if has_labels:
        labels = df[cfg.data.label_col].values.astype(np.float32)
        np.save(out_dir / "labels.npy", labels)

    print(f"\nDone. Feature files saved to {out_dir}/")


if __name__ == "__main__":
    main()
