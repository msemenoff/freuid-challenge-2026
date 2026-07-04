"""Albumentations-based augmentation pipelines for FREUID.

Returns callables that accept a PIL Image and return a float32 torch.Tensor
of shape (3, H, W) normalised to ImageNet stats.
"""
from __future__ import annotations
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# ImageNet statistics (used by all pretrained backbones)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.uint8)


class AlbumentationsWrapper:
    """Wraps an albumentations Compose so it accepts PIL or numpy images."""
    def __init__(self, pipeline: A.Compose):
        self.pipeline = pipeline

    def __call__(self, img) -> torch.Tensor:
        if isinstance(img, Image.Image):
            img = _pil_to_np(img)
        result = self.pipeline(image=img)
        return result["image"]


def get_train_transform(image_size: int = 512, strong_capture_aug: bool = False,
                        rect_aspect: float | None = None) -> AlbumentationsWrapper:
    # Resolution stage: either aspect-preserving rectangle (no padding waste) or
    # longest-side resize + square pad.
    if rect_aspect is not None:
        target_w = int(round(image_size * rect_aspect / 32) * 32)
        resize_ops = [A.Resize(height=image_size, width=target_w)]
    else:
        resize_ops = [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=0, fill=0),
        ]
    capture_style_ops = []
    if strong_capture_aug:
        capture_style_ops = [
            A.Perspective(scale=(0.01, 0.03), keep_size=True, fit_output=False,
                          border_mode=0, fill=0, p=0.15),
            A.OneOf([
                A.Downscale(scale_range=(0.7, 0.9), p=1.0),
                A.ImageCompression(compression_type="jpeg", quality_range=(55, 90), p=1.0),
            ], p=0.35),
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.06), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.08, 0.25), p=1.0),
            ], p=0.25),
            A.OneOf([
                A.RandomGamma(gamma_limit=(90, 120), p=1.0),
                A.CLAHE(clip_limit=(1.0, 2.5), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.18,
                                           contrast_limit=0.18, p=1.0),
            ], p=0.25),
        ]

    pipeline = A.Compose([
        *resize_ops,
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1),
                 rotate=(-5, 5), p=0.5),
        *capture_style_ops,
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
        ], p=0.3),
        A.GaussNoise(std_range=(0.04, 0.12), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10,
                             sat_shift_limit=15,
                             val_shift_limit=10, p=0.4),
        # Simulate JPEG compression artifacts (relevant for physical artifacts)
        A.ImageCompression(compression_type="jpeg", quality_range=(70, 100), p=0.5),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32),
                        hole_width_range=(16, 32), fill=0, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return AlbumentationsWrapper(pipeline)


def get_val_transform(image_size: int = 512, rect_aspect: float | None = None) -> AlbumentationsWrapper:
    if rect_aspect is not None:
        target_w = int(round(image_size * rect_aspect / 32) * 32)
        resize_ops = [A.Resize(height=image_size, width=target_w)]
    else:
        resize_ops = [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=0, fill=0),
        ]
    pipeline = A.Compose([
        *resize_ops,
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return AlbumentationsWrapper(pipeline)


def get_tta_transforms(image_size: int = 512, rect_aspect: float | None = None) -> list[AlbumentationsWrapper]:
    """Return a list of TTA augmentation transforms."""
    if rect_aspect is not None:
        target_w = int(round(image_size * rect_aspect / 32) * 32)
        base_resize = [A.Resize(height=image_size, width=target_w)]
    else:
        base_resize = [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=0, fill=0),
        ]
    tta_list = []
    for hflip in [False, True]:
        extra = [A.HorizontalFlip(p=1.0)] if hflip else []
        full_base = list(base_resize)
        pipe = A.Compose(full_base + extra + [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        tta_list.append(AlbumentationsWrapper(pipe))
    return tta_list
