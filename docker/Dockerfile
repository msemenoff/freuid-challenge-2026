# FREUID Challenge 2026 — reproducibility Dockerfile
#
# Builds a fully offline inference image for the champion ConvNeXtV2-large
# checkpoint (public LB 0.00465). Organizers run:
#   docker build -t freuid-repro:local .
#   docker run --rm --network none -v /path/to/images:/data:ro \
#              -v "$(pwd)/out:/submissions" freuid-repro:local
#
# Network is required at BUILD time (pip install) but NOT at run time — all
# model weights are baked into the image below.

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Default paths inside the sandbox (must match the organizer contract)
    FREUID_DATA_DIR=/data \
    FREUID_OUTPUT_DIR=/submissions \
    FREUID_SUBMISSION_PATH=/submissions/submission.csv \
    # Inference recipe that produced our selected 0.00465 public submission.
    FREUID_MODEL_PATH=/app/models/convnext_full_drop01_rect832_capaug_e1.pth \
    FREUID_CONFIG_PATH=/app/config.yaml \
    FREUID_IMAGE_SIZE=736 \
    FREUID_RECT_ASPECT=1.585 \
    FREUID_BATCH_SIZE=1 \
    FREUID_NUM_WORKERS=0 \
    # Albumentations phones home for a version check at import time by
    # default; this is harmless under --network none (fails silently with a
    # warning) but disabling it keeps logs clean and avoids the DNS lookup.
    NO_ALBUMENTATIONS_UPDATE=1

WORKDIR /app

# System libraries required by OpenCV (image decode) and Pillow fallback.
# build-essential/cmake are a fallback in case a transitive dependency (e.g.
# stringzilla, pulled in by albumentations>=2.0 -> albucore) ever needs to be
# built from source instead of installed from a prebuilt wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Inference code (vendored, same modules used to produce the champion submission).
COPY src/ /app/src/
COPY config.yaml /app/config.yaml

# Model weights baked into the image — no runtime download required.
# (Tracked via Git LFS in this repo; see models/README.md for the SHA-256.)
COPY models/ /app/models/

COPY docker/prepare_submission.py /app/prepare_submission.py

RUN useradd --create-home --uid 1000 runner \
    && chown -R runner:runner /app
USER runner

ENTRYPOINT ["python", "/app/prepare_submission.py"]
CMD []
