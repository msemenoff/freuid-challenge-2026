# Docker reproducibility contract — this repo's implementation

This folder implements the organizer's [no-network sandbox
contract](https://freuid2026.microblink.com/reproducibility.html#docker-sandbox-contract)
for **The FREUID Challenge 2026**. It runs our champion model exactly as used
to produce the selected public-leaderboard submission
(`submission_convnext_rect832_capaug_e1_s736_notta_cv2.csv`, **public LB
0.00465**).

## Layout

| File | Purpose |
| ---- | ------- |
| `Dockerfile` | Builds the fully offline inference image (build context = **repo root**) |
| `prepare_submission.py` | Real entrypoint — loads the champion checkpoint and writes `/submissions/submission.csv` |
| `requirements.txt` | Python deps for inference |

The Dockerfile also copies in `../src/`, `../config.yaml`, and `../models/` from
the repo root, so it must be built with the **repo root as build context**, not
from inside `docker/`.

## Build

```bash
# from the repository root
docker build -f docker/Dockerfile -t freuid-repro:local .
```

Network is required at **build** time only (`pip install`). No weights are
downloaded — `../models/convnext_full_drop01_rect832_capaug_e1.pth` is copied
into the image directly (see `../models/README.md` for its SHA-256).

## Local test (no network at run time)

```bash
docker run --rm \
  --network none \
  -v /path/to/flat/test_images:/data:ro \
  -v "$(pwd)/_local_out:/submissions" \
  freuid-repro:local
cat _local_out/submission.csv
```

`/path/to/flat/test_images` must be a **flat** directory of image files only
(no CSV/manifest/subfolders) — see the [Input layout
contract](https://freuid2026.microblink.com/reproducibility.html#docker-sandbox-contract).
`train_sample/train_sample/` in this repo (13 real images) can be used for a
quick smoke test.

This exact script (outside a container, same code path) was validated against
those 13 sample images: 12/13 predictions land on the correct side of 0.5
relative to their known training labels, consistent with the champion model's
near-perfect fit on training-distribution images.

## Runtime configuration

All inference parameters are environment variables with defaults matching the
validated 0.00465 recipe — **do not change these for the graded run** unless
you are intentionally reproducing a different submission:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FREUID_DATA_DIR` | `/data` | Input image directory (read-only mount) |
| `FREUID_OUTPUT_DIR` | `/submissions` | Output directory (read-write mount) |
| `FREUID_SUBMISSION_PATH` | `/submissions/submission.csv` | Output CSV path |
| `FREUID_MODEL_PATH` | `/app/models/convnext_full_drop01_rect832_capaug_e1.pth` | Checkpoint path |
| `FREUID_CONFIG_PATH` | `/app/config.yaml` | Model/config file |
| `FREUID_IMAGE_SIZE` | `736` | Inference height (rect resize) |
| `FREUID_RECT_ASPECT` | `1.585` | Aspect-preserving resize ratio (width = round(height·ratio/32)·32) |
| `FREUID_BATCH_SIZE` | `1` | Inference batch size (validated recipe uses 1; see note below) |
| `FREUID_NUM_WORKERS` | `0` | DataLoader workers |
| `FREUID_DEVICE` | auto (`cuda` if available, else `cpu`) | Force `cpu` or `cuda` |

Note on `FREUID_BATCH_SIZE`: the model has no BatchNorm (ConvNeXtV2 uses
LayerNorm), so batching does not change numerics in eval mode — increasing it
purely speeds up inference. The default of `1` matches exactly what produced
the graded submission and additionally allows the script to attribute a
decode failure to a single image (see "Corrupt-image handling" below); at
`FREUID_BATCH_SIZE != 1`, a whole failing batch raises instead of guessing.

## Corrupt-image handling

If an image cannot be decoded (by OpenCV, then by the Pillow fallback baked
into `src/dataset.py::FREUIDDataset`), the script logs a warning to stderr and
assigns a fallback score of `0.4231601107` (the empirical mean label rate of
the released training set) for that id only, rather than aborting the whole
run. Every id present in `/data` still gets exactly one row in the output.

## Requirements for verification (per the organizer contract)

- [x] Container starts with **no network** (`docker run --network none`) — no
      runtime downloads; all weights are `COPY`'d into the image.
- [x] No writes outside `/submissions/`.
- [x] Model weights bundled in the image (`COPY models/ /app/models/`).
- [x] GPU vs CPU documented in the technical report (`../report/`); the
      script auto-detects CUDA and falls back to CPU.
- [x] One output row per image file in `/data`; no missing or extra ids
      (enforced by `validate_submission()`).
- [ ] Pin the base image digest before publishing a pre-built image (left to
      the team at final freeze — see outstanding items in the top-level
      `README.md`).
