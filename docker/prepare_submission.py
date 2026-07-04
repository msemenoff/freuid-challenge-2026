#!/usr/bin/env python3
"""
FREUID Challenge 2026 — reproducibility inference entrypoint.

Reproduces our selected public-leaderboard submission
(``submission_convnext_rect832_capaug_e1_s736_notta_cv2.csv``, public LB
score **0.00465**) against the organizer's no-network Docker sandbox contract.

Organizers mount:
  /data/           read-only   test images only (flat directory, no CSV)
  /submissions/    read-write  must contain submission.csv after exit

Image filenames define row ids: ``{id}.jpeg`` (``.jpg`` / ``.png`` / ``.webp`` /
``.bmp`` / ``.tif`` / ``.tiff`` also accepted). The document id is the filename
stem.

Output schema: ``id,label`` where ``label`` is a real-valued fraud score
(higher = more confident the document is fraudulent) — same semantics as the
Kaggle leaderboard.

Model: ConvNeXtV2-large (timm ``convnextv2_large``), full end-to-end fine-tune,
fit on the entire labeled training set with capture-style augmentation. Input
resolution 736 (aspect-preserving rectangle, width = round(736*1.585/32)*32),
no test-time augmentation. See ../models/README.md for the checkpoint SHA-256
and ../report/freuid_technical_report.pdf for the full method description.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# `/app` (this file's directory) is added to sys.path by Python automatically
# when invoked as `python /app/prepare_submission.py`; the explicit insert
# below is defensive in case the entrypoint is ever invoked differently.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import FREUIDDataset          # noqa: E402
from src.models.global_backbone import ConvNeXtV2Backbone, DINOv2Backbone  # noqa: E402
from src.train_backbone import BackboneClassifier  # noqa: E402
from src.transforms import get_val_transform   # noqa: E402
from src.utils.config import load_config       # noqa: E402

DATA_DIR = Path(os.environ.get("FREUID_DATA_DIR", "/data"))
OUTPUT_DIR = Path(os.environ.get("FREUID_OUTPUT_DIR", "/submissions"))
SUBMISSION_PATH = Path(os.environ.get("FREUID_SUBMISSION_PATH", str(OUTPUT_DIR / "submission.csv")))

MODEL_PATH = Path(os.environ.get("FREUID_MODEL_PATH", "/app/models/convnext_full_drop01_rect832_capaug_e1.pth"))
CONFIG_PATH = os.environ.get("FREUID_CONFIG_PATH", "/app/config.yaml")
IMAGE_SIZE = int(os.environ.get("FREUID_IMAGE_SIZE", "736"))
RECT_ASPECT = float(os.environ.get("FREUID_RECT_ASPECT", "1.585"))
BATCH_SIZE = int(os.environ.get("FREUID_BATCH_SIZE", "1"))
NUM_WORKERS = int(os.environ.get("FREUID_NUM_WORKERS", "0"))
DEVICE = os.environ.get("FREUID_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Empirical mean label rate of the released training set (69,352 rows). Used
# only as a last-resort fallback if a single image cannot be decoded at all
# (by both OpenCV and the PIL fallback inside FREUIDDataset), so that every
# requested id still gets a finite score and the run never aborts outright.
FALLBACK_SCORE = 0.4231601107


def discover_images(data_dir: Path) -> list[tuple[str, Path]]:
    """Return (id, path) pairs for every image file directly under ``data_dir``."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pairs: list[tuple[str, Path]] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        row_id = path.stem
        if not row_id:
            raise ValueError(f"Cannot derive id from filename: {path.name}")
        pairs.append((row_id, path))

    if not pairs:
        raise FileNotFoundError(
            f"No images found in {data_dir}. Expected flat files like '{{id}}.jpeg'."
        )
    return pairs


def build_model(cfg, model_args: dict | None):
    """Mirror of src/score_backbone_submission.py::build_model (convnext path)."""
    model_args = model_args or {}
    head_dropout = float(model_args.get("head_dropout", 0.3))
    backbone_kind = model_args.get("backbone", "convnext")
    if backbone_kind == "convnext":
        backbone = ConvNeXtV2Backbone(
            model_name=model_args.get("model_name") or cfg.global_backbone.name,
            pretrained=False,   # inference-only: weights come entirely from our checkpoint
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


def run_inference(
    dataset: FREUIDDataset,
    ids_in_order: list[str],
    model: torch.nn.Module,
    device: str,
    batch_size: int,
    num_workers: int,
) -> list[float]:
    scores: list[float | None] = [None] * len(ids_in_order)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
    )
    offset = 0
    loader_iter = iter(loader)
    while True:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        except Exception as exc:
            # A whole batch failed to load/decode. We can only safely attribute
            # the failure to a single id when batch_size == 1; otherwise we
            # cannot tell *which* image in the batch was the cause.
            if batch_size != 1:
                raise RuntimeError(
                    "A batch failed to load and FREUID_BATCH_SIZE != 1, so the "
                    "failing image cannot be identified safely. Re-run with "
                    "FREUID_BATCH_SIZE=1 (the default), or remove/replace the "
                    "offending image."
                ) from exc
            failing_id = ids_in_order[offset]
            print(
                f"WARNING: failed to decode image id={failing_id}: {exc}; "
                f"using fallback score {FALLBACK_SCORE}",
                file=sys.stderr,
            )
            scores[offset] = FALLBACK_SCORE
            offset += 1
            continue

        n = batch["image"].shape[0]
        imgs = batch["image"].to(device, non_blocking=True)
        with torch.inference_mode():
            logits = model(imgs)
            probs = torch.sigmoid(logits).float().cpu().numpy()
        for j in range(n):
            scores[offset + j] = float(probs[j])
        offset += n

    if any(s is None for s in scores):
        missing = [ids_in_order[i] for i, s in enumerate(scores) if s is None]
        raise RuntimeError(f"Inference did not produce scores for {len(missing)} id(s): {missing[:5]}")
    return scores  # type: ignore[return-value]


def predict_labels(image_rows: list[tuple[str, Path]]) -> pd.DataFrame:
    """Run the champion ConvNeXtV2 model on every discovered test image."""
    cfg = load_config(CONFIG_PATH)

    df = pd.DataFrame(
        {"id": [row_id for row_id, _ in image_rows],
         "image_path": [path.name for _, path in image_rows]}
    )
    dataset = FREUIDDataset(df, root=DATA_DIR, transform=get_val_transform(IMAGE_SIZE, rect_aspect=RECT_ASPECT),
                             has_labels=False)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = build_model(cfg, checkpoint.get("model_args"))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"Running inference on {len(dataset)} images "
          f"(device={DEVICE}, image_size={IMAGE_SIZE}, rect_aspect={RECT_ASPECT}, "
          f"batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}, tta=False)", file=sys.stderr)

    ids_in_order = list(df["id"].astype(str))
    scores = run_inference(dataset, ids_in_order, model, DEVICE, BATCH_SIZE, NUM_WORKERS)

    out = pd.DataFrame({"id": ids_in_order, "label": scores})
    if not np.isfinite(out["label"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite labels produced.")
    return out


def validate_submission(submission: pd.DataFrame, expected_ids: set[str]) -> None:
    if list(submission.columns) != ["id", "label"]:
        raise ValueError(
            f"submission.csv must have columns ['id', 'label']; got {list(submission.columns)}"
        )

    got = set(submission["id"].astype(str))
    missing = expected_ids - got
    extra = got - expected_ids
    if missing:
        raise ValueError(f"submission.csv missing {len(missing)} id(s), e.g. {sorted(missing)[:3]}")
    if extra:
        raise ValueError(f"submission.csv has {len(extra)} unexpected id(s), e.g. {sorted(extra)[:3]}")


def main() -> int:
    data_dir = DATA_DIR.resolve()
    output_path = SUBMISSION_PATH.resolve()

    image_rows = discover_images(data_dir)
    expected_ids = {row_id for row_id, _ in image_rows}
    submission = predict_labels(image_rows)
    validate_submission(submission, expected_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} rows to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
