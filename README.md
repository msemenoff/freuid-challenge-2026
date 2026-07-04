# The FREUID Challenge 2026 — Solution & Reproducibility Package

Identity document fraud detection (physical manipulation, GenAI-generated
forgeries, print-and-capture attacks) for [The FREUID Challenge 2026
(IJCAI-ECAI)](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai).
Lower FREUID Score is better.

**Selected final submission:** `submission_convnext_rect832_capaug_e1_s736_notta_cv2.csv`
— **public leaderboard score 0.00465**.

This repository contains the training/inference code, the frozen champion
checkpoint, and the reproducibility package (Docker sandbox + technical
report) required for prize eligibility under the [competition's
reproducibility
requirements](https://freuid2026.microblink.com/reproducibility.html).

## TL;DR — reproduce the selected submission

```bash
# from the repository root
docker build -f docker/Dockerfile -t freuid-repro:local .

docker run --rm \
  --network none \
  -v /path/to/flat/test_images:/data:ro \
  -v "$(pwd)/_local_out:/submissions" \
  freuid-repro:local
```

See [`docker/README.md`](docker/README.md) for the full sandbox contract, and
[`models/README.md`](models/README.md) for the checkpoint and its SHA-256.

## Method summary

- **Architecture:** ConvNeXtV2-large (`timm convnextv2_large`), full
  end-to-end fine-tune (no frozen layers), `Dropout(0.1)` → `Linear(1536, 1)`
  binary head.
- **Training:** full labeled training set (69,352 images, no held-out fold —
  `--fit_all`), 1 epoch, focal loss, AdamW with separate backbone/head
  learning rates, `OneCycleLR`.
- **Resolution:** aspect-preserving rectangular resize, no padding
  (`rect_aspect=1.585`), trained at 832×1312. This was the single largest
  lever discovered during development — see `discussion.txt` for the full
  experimental log (square-pad @384 → 0.198 FREUID → rect @832 → 0.00495 →
  0.00465 after a no-TTA / decode fix).
- **Augmentation:** standard flip/affine/blur/noise/color/JPEG/dropout, plus
  `--strong_capture_aug` (perspective jitter, downscale/re-JPEG, sensor noise,
  gamma/CLAHE/brightness) to simulate the print-and-capture domain gap between
  training data (69,332 digital / only 20 captured) and the test distribution.
- **Inference:** scale 736 (0.88× training resolution — empirically the
  sweet spot; see `discussion.txt`), **no test-time augmentation**, OpenCV
  (libjpeg-turbo) image decode with a Pillow fallback.
- **What did *not* work** (kept for the technical report / ablations):
  ensembling a diverse but weaker architecture (NFNet) always hurt; multi-seed
  averaging of two independently-trained same-recipe checkpoints also hurt
  once validated on the public LB (0.00685 vs 0.00465), despite looking
  promising on internal agreement metrics; oversampling the 20 real captured
  training images 200× did not clearly beat the champion (LB pending at
  session end — check `discussion.txt` / repo memory for the final verdict
  before citing it in the report).

## Repository layout

```
config.yaml              Master config (data paths, model, training hyperparameters)
src/                      Training + inference code
  dataset.py              FREUIDDataset (cv2/PIL image loading, fold splitting)
  transforms.py           Train/val/TTA augmentation pipelines (Albumentations)
  train_backbone.py       ConvNeXtV2 / DINOv2 end-to-end fine-tuning entrypoint
  score_backbone_submission.py   Reference scoring script (Kaggle CSV workflow)
  models/global_backbone.py     ConvNeXtV2Backbone / DINOv2Backbone
  utils/                  config loader, dataframe helpers, metrics, seeding
models/
  convnext_full_drop01_rect832_capaug_e1.pth   Champion checkpoint (Git LFS)
  README.md               Checkpoint provenance, SHA-256, format
docker/                   No-network reproducibility sandbox (see docker/README.md)
report/                   Technical report (LaTeX) required for prize eligibility
REPLY_TEMPLATE.txt         Draft of the one required Kaggle discussion reply
```

Large released datasets (`train/`, `public_test/`, `train_labels.csv`,
`sample_submission.csv`) and local artifacts (`checkpoints/`, `features/`,
`logs/`, `submissions/`) are intentionally **not** committed — see
`.gitignore`. `train_sample/` (13 real images) is kept for quick smoke-testing
the Docker image without downloading the full dataset.

## Environment (development / training)

Training and experimentation were run in a conda env named `neural-debris`
(not itself part of this repo) with:

| Package | Version |
| --- | --- |
| Python | 3.11 |
| torch | 2.11.0 (+cu126) |
| timm | 1.0.27 |
| opencv-python (cv2) | 4.13.0 |
| albumentations | 2.0.8 |
| pandas | 2.2.3 |
| numpy | 2.2.6 |
| PyYAML | 6.0.3 |
| tqdm | 4.67.3 |
| Pillow | 12.2.0 |

`docker/requirements.txt` pins broader-but-compatible ranges for portability;
see that file's header comment if you need to reproduce the exact dev
environment bit-for-bit.

**Hardware used for training:** single 16 GB GPU (16,376 MiB), 54 GB RAM + 64
GB swap, WSL2/Linux. Full-data 1-epoch fine-tuning at 832×1312 with
`batch_size=1, accumulate_grad=8` took roughly 6–7 hours (~3 it/s). Inference
at scale 736 on the same GPU processes the ~7,821 publicly-scored rows in
about 15–18 minutes (no TTA). CPU-only inference is supported but
substantially slower (untimed here — document actual wall-clock once
measured, before finalizing the technical report).

## Training the champion recipe from scratch

```bash
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8 OMP_NUM_THREADS=2 \
  python src/train_backbone.py \
    --backbone convnext --fit_all \
    --image_size 832 --rect_aspect 1.585 \
    --batch_size 1 --accum 8 --num_workers 0 --epochs 1 \
    --head_dropout 0.1 --strong_capture_aug \
    --save_every_steps 10000 \
    --out checkpoints/convnext_full_drop01_rect832_capaug_e1.pth
```

`--save_every_steps` writes periodic partial checkpoints during the ~6–7 hour
run so a silent crash near the end does not lose the entire run.

## Scoring / reproducing the exact champion submission (CSV workflow)

```bash
PYTHONPATH=. python src/score_backbone_submission.py \
  --backbone convnext \
  --checkpoint checkpoints/convnext_full_drop01_rect832_capaug_e1.pth \
  --image_size 736 --rect_aspect 1.585 \
  --batch_size 1 --num_workers 0 \
  --output submissions/submission_convnext_rect832_capaug_e1_s736_notta_cv2.csv
```

(No `--tta` flag — the champion submission does **not** use test-time
augmentation.) This is the Kaggle-CSV-based workflow used during development,
against `sample_submission.csv` + `public_test/`. The Docker entrypoint in
`docker/prepare_submission.py` reimplements the same model/transform/decode
logic against the organizer's flat-directory sandbox contract instead of a
CSV, for the final private-test evaluation.

## Reproducibility package (prize eligibility)

Per the [official
requirements](https://freuid2026.microblink.com/reproducibility.html),
by **July 15, 2026, 23:59 AoE**:

1. This repository — public, OSI-licensed (MIT, see [`LICENSE`](LICENSE)), at
   a frozen commit SHA.
2. A technical report PDF — skeleton in [`report/`](report/), **still needs
   final content and a compiled PDF** (see Outstanding items below).
3. A runnable Docker artifact — [`docker/`](docker/), validated locally
   (dry-run, not yet inside an actual container — see Outstanding items).
4. Exactly one reply on the pinned Kaggle discussion thread — draft in
   [`REPLY_TEMPLATE.txt`](REPLY_TEMPLATE.txt).

**Code freeze:** private test images are released **July 13, 2026**. After
that date, no changes to model weights, architecture, or training code are
allowed — only inference, documentation, and Docker packaging updates.

## Outstanding items (see the task summary for the full list with owners)

This package is prepared as completely as possible ahead of the private test
release. Items that require the user's action, credentials, or the private
data itself are listed in detail in the accompanying task summary/response —
short version:

- Push this repo to a public OSI-licensed remote (e.g. GitHub) — needs your
  account/credentials.
- Install Docker + Git LFS locally (not available in this dev sandbox) and
  run an actual `docker build` / `docker run --network none` end-to-end test.
- Fill in team name, authors, and affiliations in `LICENSE`, `report/`, and
  `REPLY_TEMPLATE.txt`.
- Compile `report/freuid_technical_report.tex` to PDF (needs a LaTeX
  toolchain, e.g. Overleaf or `latexmk`, not available in this dev sandbox).
- After July 13: run inference on the released private images (Docker or CSV
  workflow), update the private-row Kaggle submission, freeze the commit SHA,
  and submit the final reply by July 15.

## License

[MIT](LICENSE) — replace the placeholder copyright holder name before
publishing.
