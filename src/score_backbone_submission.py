"""Score a submission CSV directly from a trained backbone checkpoint."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.dataset import FREUIDDataset
from src.models.global_backbone import ConvNeXtV2Backbone, DINOv2Backbone
from src.train_backbone import BackboneClassifier
from src.transforms import get_tta_transforms, get_val_transform
from src.utils.config import load_config
from src.utils.data import filter_existing_images, load_public_test_dataframe, load_train_dataframe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="convnext", choices=["convnext", "dinov2"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tta", action="store_true",
                        help="Average predictions over light test-time augmentation")
    parser.add_argument("--rect_aspect", type=float, default=None,
                        help="Aspect-preserving rectangular resize (width/height) instead of square pad")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def build_model(cfg, backbone_name: str, model_args: dict | None = None) -> BackboneClassifier:
    model_args = model_args or {}
    head_dropout = float(model_args.get("head_dropout", 0.3))
    if backbone_name == "convnext":
        backbone = ConvNeXtV2Backbone(
            model_name=model_args.get("model_name") or cfg.global_backbone.name,
            pretrained=False,
            frozen=False,
        )
        embed_dim = backbone.embed_dim
    else:
        backbone = DINOv2Backbone(
            model_name=cfg.dinov2.model_name,
            frozen=False,
            use_cls=cfg.dinov2.use_cls,
            use_patch_mean=cfg.dinov2.use_patch_mean,
            use_patch_stats=cfg.dinov2.use_patch_stats,
        )
        embed_dim = backbone.embed_dim
    return BackboneClassifier(backbone, embed_dim, dropout=head_dropout)


@torch.inference_mode()
def predict_scores(model, loader, device: str):
    model.eval()
    scores = []
    ids = []
    for batch in tqdm(loader, desc="  infer"):
        imgs = batch["image"].to(device, non_blocking=True)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().float().numpy()
        scores.extend(probs.tolist())
        ids.extend(batch["id"])
    return ids, scores


def build_loader(df, root: Path, transform, batch_size: int, num_workers: int) -> DataLoader:
    dataset = FREUIDDataset(df, root=root, transform=transform, has_labels=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def predict_scores_with_tta(model, df, root: Path, image_size: int, batch_size: int, num_workers: int, device: str, rect_aspect: float | None = None):
    transforms = get_tta_transforms(image_size, rect_aspect=rect_aspect)
    all_scores = []
    ids = None
    for transform in transforms:
        loader = build_loader(df, root, transform, batch_size, num_workers)
        current_ids, current_scores = predict_scores(model, loader, device)
        if ids is None:
            ids = current_ids
        all_scores.append(np.asarray(current_scores, dtype=np.float32))
    mean_scores = np.mean(np.stack(all_scores, axis=0), axis=0)
    return ids, mean_scores.tolist()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    root = Path(cfg.data.root)

    test_df = load_public_test_dataframe(cfg)
    available_df, _missing_df = filter_existing_images(test_df, root=root, image_col=cfg.data.image_col)
    train_df = load_train_dataframe(cfg)
    fallback_score = float(train_df[cfg.data.label_col].mean())

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(cfg, args.backbone, checkpoint.get("model_args"))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device)

    if args.tta:
        ids, scores = predict_scores_with_tta(
            model,
            available_df,
            root,
            args.image_size,
            args.batch_size,
            args.num_workers,
            args.device,
            args.rect_aspect,
        )
    else:
        transform = get_val_transform(args.image_size, rect_aspect=args.rect_aspect)
        loader = build_loader(available_df, root, transform, args.batch_size, args.num_workers)
        ids, scores = predict_scores(model, loader, args.device)
    scored_df = pd.DataFrame({cfg.data.id_col: ids, cfg.data.label_col: scores})

    submission = test_df[[cfg.data.id_col]].copy()
    submission = submission.merge(scored_df, on=cfg.data.id_col, how="left")
    fallback_rows = int(submission[cfg.data.label_col].isna().sum())
    submission[cfg.data.label_col] = submission[cfg.data.label_col].fillna(fallback_score)

    output_path = Path(args.output or Path(cfg.paths.submissions_dir) / f"submission_{args.backbone}_backbone.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"scored_rows={len(scored_df)}")
    print(f"fallback_rows={fallback_rows}")
    print(f"fallback_score={fallback_score:.10f}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()