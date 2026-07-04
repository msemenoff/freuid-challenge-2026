# Model weights

## Champion checkpoint

| Field | Value |
| --- | --- |
| File | `convnext_full_drop01_rect832_capaug_e1.pth` |
| Architecture | `convnextv2_large` (timm), full end-to-end fine-tune (no frozen layers) |
| Head | `Dropout(p=0.1)` → `Linear(1536, 1)` (binary logit), see `BackboneClassifier` in `src/train_backbone.py` |
| Training resolution | 832×1312 (aspect-preserving rectangle, `rect_aspect=1.585`, no padding) |
| Training data | Full labeled train set, 69,352 images, `--fit_all` (no held-out fold) |
| Augmentation | `--strong_capture_aug` (perspective jitter, downscale/JPEG re-encode, sensor noise, gamma/CLAHE/brightness — simulates print-and-capture) + standard flip/affine/blur/noise/color/JPEG/dropout |
| Epochs | 1 (full-data) |
| Loss | Focal loss (`gamma=2.0`, `alpha=0.75`) |
| Optimizer | AdamW, separate LR for backbone (`1e-5`) and head (`1e-4`), `OneCycleLR` |
| Best public LB (selected submission) | `submission_convnext_rect832_capaug_e1_s736_notta_cv2.csv` = **0.00465** |
| SHA-256 | `baeedfaa009a20caa85014807480a38a42fa5c8542347415615dc5a471edcb4c` |
| Size | 750 MB |

Verify integrity after cloning / pulling via Git LFS:

```bash
sha256sum models/convnext_full_drop01_rect832_capaug_e1.pth
# expect: baeedfaa009a20caa85014807480a38a42fa5c8542347415615dc5a471edcb4c
```

## Checkpoint format

Standard PyTorch checkpoint dict:

```python
{
  "epoch": int,
  "state_dict": <BackboneClassifier state_dict>,
  "metrics": None,          # fit_all runs have no held-out validation fold
  "model_args": {
     "backbone": "convnext",
     "image_size": 832,
     "head_dropout": 0.1,
     "strong_capture_aug": True,
     "lr": 0.0001,
     "lr_backbone": 1e-05,
     "weight_decay": 0.0001,
     "fit_all": True,
     "fold": None,
  },
}
```

`model_args` does not include an explicit `model_name` key for this checkpoint (an
older field), so the loader falls back to `config.yaml`'s `global_backbone.name:
convnextv2_large`, which is exactly what this checkpoint was trained with. See
`docker/prepare_submission.py` / `src/score_backbone_submission.py::build_model`.

## Retrieval via Git LFS

This file is tracked with [Git LFS](https://git-lfs.com/) (see `.gitattributes`).
After cloning:

```bash
git lfs install
git lfs pull
```

If your environment cannot use Git LFS, the file can alternatively be fetched at
Docker **build** time (network is only forbidden at `docker run`, not `docker
build`) from a release asset you host — see `docker/README.md`.
