"""End-to-end backbone fine-tuning script.

Trains ConvNeXtV2 or DINOv2 end-to-end with focal loss + cosine LR schedule.
Intended for Phase 3 (when full data is available).  Saves best checkpoint
by FREUID Score on the validation fold.

Usage:
    python src/train_backbone.py --backbone convnext --fold 0
    python src/train_backbone.py --backbone dinov2   --fold 0
    python src/train_backbone.py --backbone convnext --fit_all
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import amp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.data import load_train_dataframe
from src.utils.seed import seed_everything
from src.utils.metrics import freuid_score
from src.dataset import FREUIDDataset, get_train_val_datasets
from src.transforms import get_train_transform, get_val_transform
from src.models.global_backbone import ConvNeXtV2Backbone, DINOv2Backbone

# Disable OpenCV internal threading in the MAIN process too (num_workers=0 does
# all augmentation here). Multi-threaded OpenCV on large images can segfault.
try:
    import cv2
    cv2.setNumThreads(0)
except Exception:
    pass


def _worker_init_fn(worker_id: int):
    # Prevent OpenCV/albumentations internal threading from conflicting with
    # forked DataLoader workers (a common cause of worker segfaults on long runs).
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce
        return loss.mean() if self.reduction == "mean" else loss


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------

class BackboneClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats).squeeze(-1)    # (B,) logits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, scaler, device, grad_clip, accum, use_amp,
    checkpoint_callback=None,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    for step, batch in enumerate(tqdm(loader, desc="  train")):
        imgs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device)
        with amp.autocast(device_type=device_type, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels) / accum
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer_step_ran = True
            if use_amp:
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_ran = scaler.get_scale() >= old_scale
            else:
                optimizer.step()
            if scheduler is not None and optimizer_step_ran:
                scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accum
        if checkpoint_callback is not None:
            checkpoint_callback(step + 1, total_loss / (step + 1))
    return total_loss / len(loader)


@torch.inference_mode()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores, all_labels = [], []
    for batch in tqdm(loader, desc="  val"):
        imgs = batch["image"].to(device, non_blocking=True)
        logits = model(imgs)
        scores = torch.sigmoid(logits).cpu().float().numpy()
        all_scores.append(scores)
        all_labels.append(batch["label"].numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="convnext", choices=["convnext", "dinov2"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--fit_all", action="store_true",
                   help="Train on all labeled data without a validation split")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--strong_capture_aug", action="store_true",
                   help="Use stronger capture-style augmentations during training")
    p.add_argument("--rect_aspect", type=float, default=None,
                   help="Aspect-preserving rectangular resize (width/height) instead of square pad")
    p.add_argument("--accum", type=int, default=None,
                   help="Gradient accumulation steps override (effective batch = batch_size * accum)")
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader workers override (default from config)")
    p.add_argument("--model_name", default=None,
                   help="Override timm backbone model name (convnext path only)")
    p.add_argument("--head_dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr_backbone", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="Override random seed (for multi-seed ensembling)")
    p.add_argument("--capture_repeat_factor", type=int, default=1,
                   help="Repeat real captured training rows this many times (fit_all only)")
    p.add_argument("--save_every_steps", type=int, default=0,
                   help="Save partial checkpoints every N train batches (0 disables)")
    p.add_argument("--out", default=None, help="Override checkpoint output path")
    return p.parse_args()


def _repeat_captured_rows(df, repeat_factor: int):
    if repeat_factor <= 1:
        return df
    if "is_digital" not in df.columns:
        raise KeyError("--capture_repeat_factor requires an is_digital column")

    is_digital = df["is_digital"].map({True: 1, False: 0, 1: 1, 0: 0, "True": 1, "False": 0})
    captured = df[is_digital == 0]
    if captured.empty:
        raise ValueError("--capture_repeat_factor found no captured rows (is_digital == False)")

    repeats = [df, *[captured.copy() for _ in range(repeat_factor - 1)]]
    return pd.concat(repeats, ignore_index=True)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.data.seed)
    seed_everything(seed)
    root = Path(cfg.data.root)
    ckpt_dir = Path(cfg.paths.checkpoints_dir)
    ckpt_dir.mkdir(exist_ok=True)
    device = args.device
    use_amp = bool(cfg.training.amp and str(device).startswith("cuda"))
    epochs = args.epochs or cfg.training.epochs
    lr = float(args.lr if args.lr is not None else cfg.training.lr)
    lr_backbone = float(args.lr_backbone if args.lr_backbone is not None else cfg.training.lr_backbone)
    weight_decay = float(args.weight_decay if args.weight_decay is not None else cfg.training.weight_decay)
    grad_clip = float(cfg.training.grad_clip)
    accum = int(args.accum if args.accum is not None else cfg.training.accumulate_grad)
    focal_gamma = float(cfg.training.focal_gamma)
    focal_alpha = float(cfg.training.focal_alpha)
    head_dropout = float(args.head_dropout)
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.training.batch_size)

    df = load_train_dataframe(cfg)
    capture_repeat_factor = int(args.capture_repeat_factor)
    if capture_repeat_factor > 1:
        if not args.fit_all:
            raise ValueError("--capture_repeat_factor is only supported with --fit_all")
        before = len(df)
        df = _repeat_captured_rows(df, capture_repeat_factor)
        print(
            f"Repeated captured rows by factor {capture_repeat_factor}: "
            f"{before} -> {len(df)} train rows"
        )
    img_size = args.image_size or cfg.augmentation.image_size
    if args.fit_all:
        train_ds = FREUIDDataset(
            df,
            root=root,
            transform=get_train_transform(img_size, strong_capture_aug=args.strong_capture_aug,
                                          rect_aspect=args.rect_aspect),
            has_labels=True,
        )
        val_ds = None
        print(f"Train: {len(train_ds)}  Val: none (fit_all)")
    else:
        train_ds, val_ds = get_train_val_datasets(
            df, fold=args.fold, root=root,
            train_transform=get_train_transform(img_size, strong_capture_aug=args.strong_capture_aug,
                                                rect_aspect=args.rect_aspect),
            val_transform=get_val_transform(img_size, rect_aspect=args.rect_aspect),
            n_folds=cfg.data.n_folds, seed=cfg.data.seed,
            label_col=cfg.data.label_col,
            strategy=getattr(cfg.data, "fold_strategy", "stratified"),
            group_col=getattr(cfg.data, "fold_group_col", "type"),
        )
        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    num_workers = int(args.num_workers if args.num_workers is not None else cfg.training.num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=not args.fit_all,
                              worker_init_fn=_worker_init_fn,
                              persistent_workers=(num_workers > 0))
    val_loader = None if val_ds is None else DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, worker_init_fn=_worker_init_fn,
        persistent_workers=(num_workers > 0),
    )

    # ---- Build model ----
    if args.backbone == "convnext":
        backbone = ConvNeXtV2Backbone(
            model_name=args.model_name or cfg.global_backbone.name,
            pretrained=cfg.global_backbone.pretrained,
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

    model = BackboneClassifier(backbone, embed_dim, dropout=head_dropout).to(device)

    # ---- Optimizer: separate LR for backbone vs. head ----
    head_params = list(model.head.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr},
    ], weight_decay=weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / max(1, accum))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch
    pct_start = min(0.99, warmup_steps / total_steps) if total_steps > 0 else 0.1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[lr_backbone, lr],
        total_steps=total_steps, pct_start=pct_start,
    )
    criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    scaler = amp.GradScaler(device="cuda", enabled=use_amp)

    best_freuid = float("inf")
    default_name = f"{args.backbone}_full.pth" if args.fit_all else f"{args.backbone}_fold{args.fold}_best.pth"
    ckpt_path = Path(args.out) if args.out else ckpt_dir / default_name
    checkpoint_meta = {
        "backbone": args.backbone,
        "model_name": args.model_name or cfg.global_backbone.name,
        "image_size": int(img_size),
        "head_dropout": head_dropout,
        "strong_capture_aug": bool(args.strong_capture_aug),
        "lr": lr,
        "lr_backbone": lr_backbone,
        "weight_decay": weight_decay,
        "fit_all": bool(args.fit_all),
        "fold": None if args.fit_all else int(args.fold),
        "capture_repeat_factor": capture_repeat_factor,
    }
    save_every_steps = int(args.save_every_steps)

    def save_partial_checkpoint(step: int, running_loss: float) -> None:
        if save_every_steps <= 0 or step % save_every_steps != 0:
            return
        partial_path = ckpt_path.with_name(f"{ckpt_path.stem}_step{step:06d}{ckpt_path.suffix}")
        torch.save({
            "epoch": None,
            "step": int(step),
            "state_dict": model.state_dict(),
            "metrics": {"running_loss": float(running_loss)},
            "model_args": {
                **checkpoint_meta,
                "partial_epoch": True,
                "partial_step": int(step),
            },
        }, partial_path)
        tqdm.write(f"  ✓ Saved partial checkpoint @ step {step} → {partial_path}")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler,
            device, grad_clip, accum, use_amp, save_partial_checkpoint,
        )
        if val_loader is None:
            print(f"  loss={train_loss:.4f}")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "metrics": None,
                "model_args": checkpoint_meta,
            }, ckpt_path)
            print(f"  ✓ Saved checkpoint → {ckpt_path}")
        else:
            scores, labels = evaluate(model, val_loader, device)
            metrics = freuid_score(labels, scores, cfg.evaluation.bpcer_target)
            print(
                f"  loss={train_loss:.4f}  "
                f"AuDET={metrics['audet']:.4f}  "
                f"APCER@1%={metrics['apcer_bpcer1']:.4f}  "
                f"FREUID={metrics['freuid']:.4f}"
            )
            if metrics["freuid"] < best_freuid:
                best_freuid = metrics["freuid"]
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "metrics": metrics,
                    "model_args": checkpoint_meta,
                }, ckpt_path)
                print(f"  ✓ Saved best checkpoint → {ckpt_path}")

    if val_loader is not None:
        print(f"\nBest FREUID Score: {best_freuid:.4f}")


if __name__ == "__main__":
    main()
