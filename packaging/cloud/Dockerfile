# OASIS headless runner — the pipeline and the validation suite on a CUDA GPU.
#
# WHAT THIS IS FOR. LoFTR is the slow part of certification and MPS is a measured dead end
# for it (its attention ops fall back to the CPU anyway), so on Apple Silicon the matcher runs
# on CPU no matter what. CUDA is the only accelerator that actually helps, and
# `loftr_matcher._device()` picks it up with no configuration. Together with the device fix in
# oasis/common/device.py, both halves of the pipeline — LoFTR and InstanSeg — land on the GPU.
#
# WHAT THIS IS NOT FOR. It does not put the desktop UI in the cloud, and it deliberately does
# not offload interactive certification over the network. Round-tripping crops to a remote GPU
# adds latency to a path that is already content-cached (565x on re-run), and it would mean
# shipping tissue images off the operator's machine for every click. Batch work is where a GPU
# pays: validation runs and cohort pipelines, started once and collected later.
#
# PATIENT DATA. Everything here reads from a directory YOU mount. Nothing is uploaded anywhere
# by this image. But a VM is still someone else's computer: histology is patient data, and WSI
# metadata routinely carries identifiers. Run public sets (ANHIR, ACROBAT, DeepLIIF, CODEX) in
# the cloud freely; for a real cohort, either keep it local or satisfy yourself it is
# de-identified and that your institution permits it. The image prints this warning on start.
#
# Base is a plain slim Python rather than a CUDA base image: torch's Linux wheels already
# bundle CUDA, and pinning torch==2.13.0 exactly matters more here than base-image tidiness —
# requirements.txt explains why every pin is fixed (library drift moves null envelopes and can
# change a significance verdict). The host needs an NVIDIA driver + the container toolkit;
# GCP's Deep Learning VM images ship both.
FROM python:3.12-slim

# libgl1/libglib2.0-0: opencv-python (the pinned, non-headless wheel) links against them.
# Switching to opencv-python-headless would avoid this, but it is a DIFFERENT package and this
# codebase pins opencv precisely because its version can move Otsu output.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /oasis

# One source of truth for dependencies. pywebview is the only GUI dependency and neither
# run_pipeline.py nor validation/run.py imports it, so it is filtered here rather than
# duplicated into a second requirements file that would silently drift out of sync.
COPY requirements.txt /tmp/requirements.txt
RUN grep -viE '^\s*pywebview' /tmp/requirements.txt > /tmp/req-headless.txt \
    && pip install --no-cache-dir -r /tmp/req-headless.txt

COPY . /oasis

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OASIS_HEADLESS=1

# Default: say what hardware was actually found. The failure this image exists to avoid is
# silent — a run that works, costs GPU-hours, and quietly used the CPU for half the work.
CMD ["python", "packaging/cloud/gpu_check.py"]
