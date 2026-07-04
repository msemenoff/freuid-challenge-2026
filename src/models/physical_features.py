"""Physical artifact feature extractor.

Computes handcrafted features that capture:
  1. SRM (Steganalysis Rich Model) high-pass noise residuals  – expose
     GenAI / re-sampling / compression traces.
  2. Wavelet energy statistics (Daubechies-4) per channel – detect
     resampling and print-and-scan artifacts.
  3. Per-channel color histogram statistics – catch hue/saturation
     inconsistencies between genuine and tampered regions.

All computations run on CPU (numpy/scipy). Input can be a PIL Image,
a numpy uint8 array (H,W,3), or a torch tensor (3,H,W) float [0,1].

Output: 1-D numpy float32 feature vector.
"""
from __future__ import annotations
import numpy as np
import pywt
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# SRM Filters  (30 of the 30-filter SRM bank, truncated to 5×5 kernels)
# ---------------------------------------------------------------------------

# We use a compact 3×3 high-pass bank (approximation of full SRM).
_SRM_KERNELS_3x3 = np.array([
    # Horizontal / vertical first-order
    [[ 0, 0, 0], [-1, 2,-1], [ 0, 0, 0]],
    [[ 0,-1, 0], [ 0, 2, 0], [ 0,-1, 0]],
    # Diagonal
    [[-1, 0, 0], [ 0, 2, 0], [ 0, 0,-1]],
    [[ 0, 0,-1], [ 0, 2, 0], [-1, 0, 0]],
    # Second-order horizontal
    [[ 0, 0, 0], [ 1,-2, 1], [ 0, 0, 0]],
    [[ 0, 1, 0], [ 0,-2, 0], [ 0, 1, 0]],
    # Laplacian
    [[ 0,-1, 0], [-1, 4,-1], [ 0,-1, 0]],
    # Cross
    [[-1,-1,-1], [-1, 8,-1], [-1,-1,-1]],
], dtype=np.float32)  # (8, 3, 3)


def _srm_residuals(img_gray: np.ndarray) -> np.ndarray:
    """Apply SRM high-pass filters to a gray uint8 image.

    Returns (n_filters,) vector of per-filter residual statistics.
    """
    img_f = img_gray.astype(np.float32)
    feats = []
    for k in _SRM_KERNELS_3x3:
        residual = cv2.filter2D(img_f, -1, k)
        feats.extend([
            float(residual.mean()),
            float(residual.std()),
            float(np.abs(residual).mean()),
        ])
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Wavelet features
# ---------------------------------------------------------------------------

def _wavelet_energy(img_gray: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """Compute per-subband energy (H,V,D) at each decomposition level."""
    img_f = img_gray.astype(np.float64)
    coeffs = pywt.wavedec2(img_f, wavelet, level=level)
    feats = []
    for detail_tuple in coeffs[1:]:          # skip approximation
        for subband in detail_tuple:         # (cH, cV, cD)
            energy = float(np.mean(subband ** 2))
            feats.append(energy)
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Color channel statistics
# ---------------------------------------------------------------------------

def _color_channel_stats(img_bgr: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Per-channel histogram + mean/std/skew for RGB and HSV images."""
    feats = []
    # RGB stats
    for c in range(3):
        ch = img_bgr[:, :, c].astype(np.float32) / 255.0
        hist, _ = np.histogram(ch, bins=n_bins, range=(0.0, 1.0), density=True)
        feats.extend(hist.tolist())
        feats.extend([float(ch.mean()), float(ch.std())])

    # HSV stats – capture hue/saturation inconsistencies
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    for c in range(3):
        ch = img_hsv[:, :, c]
        feats.extend([float(ch.mean()), float(ch.std())])

    # Color inconsistency: std of per-patch mean (4×4 grid)
    h, w = img_bgr.shape[:2]
    ph, pw = h // 4, w // 4
    patch_means = []
    for i in range(4):
        for j in range(4):
            patch = img_bgr[i*ph:(i+1)*ph, j*pw:(j+1)*pw].astype(np.float32)
            patch_means.append(patch.mean(axis=(0,1)))     # (3,)
    patch_means = np.array(patch_means, dtype=np.float32)  # (16, 3)
    feats.extend(patch_means.std(axis=0).tolist())          # (3,)

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_physical_features(
    img,
    wavelet: str = "db4",
    wavelet_level: int = 3,
    color_bins: int = 32,
    resize_to: int = 256,
) -> np.ndarray:
    """Extract all physical artifact features from a single image.

    Args:
        img: PIL Image, numpy uint8 (H,W,3), or torch.Tensor (3,H,W) float [0,1].
        wavelet: PyWavelets wavelet name.
        wavelet_level: Decomposition levels.
        color_bins: Number of histogram bins per channel.
        resize_to: Resize longest dimension to this for speed.

    Returns:
        1-D float32 numpy array.
    """
    # --- normalise input ---
    if isinstance(img, Image.Image):
        img_np = np.array(img.convert("RGB"), dtype=np.uint8)
    elif hasattr(img, "numpy"):
        arr = img.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        img_np = (arr * 255).clip(0, 255).astype(np.uint8)
    else:
        img_np = np.asarray(img, dtype=np.uint8)

    # --- resize for speed ---
    h, w = img_np.shape[:2]
    if max(h, w) > resize_to:
        scale = resize_to / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    srm_feats      = _srm_residuals(img_gray)
    wavelet_feats  = _wavelet_energy(img_gray, wavelet=wavelet, level=wavelet_level)
    color_feats    = _color_channel_stats(img_bgr, n_bins=color_bins)

    return np.concatenate([srm_feats, wavelet_feats, color_feats])
