---
title: OASIS
emoji: 🔬
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.22.0
app_file: hf_space/app.py
python_version: "3.12.12"
pinned: false
license: mit
short_description: IHC quantification and serial-section spatial association
---

# OASIS

Nuclear and membranous IHC quantification, and cross-type spatial association between
serial sections — the same application that runs on the desktop, in a browser.

This is the OASIS desktop app unmodified: the same `oasis/webui/index.html`, the same
`oasis/webui/api.py`, the same `run_pipeline.py`. Everything specific to running it as a
Space lives in `hf_space/` and is additive. It shows the same four tabs a lab's install
shows — **Quant**, **Spatial**, **Classifier**, **Settings**.

## Using it

The **Browse** buttons open your browser's file picker and upload what you choose; the
path box then fills in with where it landed, exactly as a desktop file dialog would.
Folders upload whole, for batch runs.

No sample data is shipped, and the app opens with its path boxes empty, exactly as the
desktop build does. Bring your own images.

You must enter the pixel size yourself. OASIS never assumes one, because every density,
radius and area it reports is computed from it — the Quant tab will refuse to start until
it has one, either typed in or measured from a scale bar.

## What is different here, and why

**Results may not match the desktop build digit for digit.** ZeroGPU supports PyTorch
2.8.0–2.11.0; the desktop app is pinned to 2.13.0, which is the version every validated
number in this project was produced under. The Space runs 2.11.0 because it has to. The
segmenter's output feeds every downstream count, so this is a real difference rather than
a formality. **Treat this Space as a demonstration of the tool, not as the instrument.**
For work you intend to publish, run the desktop build.

**GPU time is rationed.** ZeroGPU gives each visitor a daily allowance — 2 minutes
unauthenticated, 5 minutes signed in, more for PRO accounts — and a GPU is held only for
the duration of each segmentation or matching call. Modest crops are the point — a couple
of megapixels segments in seconds. A whole-slide image will exhaust a free allowance.

**One run at a time.** Every visitor shares one container and one GPU allocation, so runs
queue rather than overlap. A waiting run says so in its own log.

**Stop means "stop showing me this".** On the desktop, Stop kills the worker process. Here
the work runs in the server process, which cannot be killed from outside without taking
the Space down, so the run is abandoned rather than halted.

**Files persist only if a Storage Bucket is attached.** A Space's own disk is wiped on
restart, rebuild and sleep. With a bucket mounted, uploads, results, saved calibrations
and trained classifiers survive; without one they do not.

## Running it locally

    pip install gradio==6.22.0 spaces python-multipart
    python hf_space/app.py            # then open http://127.0.0.1:7860

Everything works except the GPU: `spaces` is inert off-platform, so segmentation runs on
whatever device `oasis/common/device.py` resolves — CUDA, MPS or CPU. Files go to
`hf_space/.local-data/` unless `OASIS_SPACE_DATA` says otherwise. `/__health` reports which
GPU entry points were patched.

## Deploying it

1. Create a Gradio Space and set its hardware to **ZeroGPU** in Space settings. ZeroGPU is
   only available on the Gradio SDK — on any other SDK the `@spaces.GPU` decorator does
   nothing and the whole pipeline runs on CPU while looking healthy.
2. Attach a **Storage Bucket** as a volume, mounted at `/data` (or set `OASIS_SPACE_DATA`
   to wherever you mount it).
3. Run `hf_space/deploy.sh <owner>/<space-name>` from a checkout to assemble and push.

Confirm it really is on the GPU after deploying: run any small image and open
`Full settings for this run` — or read the run's `*_summary.json`, where
`segmenter_device` must say `cuda`. If it says `cpu`, ZeroGPU is not being reached and
the run is merely slow rather than wrong. `/__health` reports the same thing.

## Citing

OASIS's segmentation depends on InstanSeg, which should be cited alongside it. The
licence, provenance and citations are in `models/NOTICE.md`.
