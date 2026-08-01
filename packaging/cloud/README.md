# Running OASIS on a cloud GPU

## What a GPU actually buys, and what it does not

LoFTR dominates certification, and MPS is a **measured** dead end for it — its attention ops
fall back to the CPU, so on Apple Silicon the matcher runs on CPU whatever you set. CUDA is
the only accelerator that helps. `loftr_matcher._device()` picks CUDA up with no
configuration, and since `oasis/common/device.py` landed, InstanSeg does too.

What a GPU does **not** fix:

- **A re-run.** LoFTR passes are content-cached (565× on a repeat), so the second pass over
  the same crops is already near-free. The pain is first-run cost.
- **Interactive certification.** Shipping crops to a remote GPU per click adds network latency
  to a cached path, and sends tissue images off the operator's machine every time. This image
  deliberately does not do that.

Where a GPU pays is **batch**: the validation suite and cohort pipeline runs — start once,
collect later.

## Patient data

Nothing here uploads anything; the container reads only what you mount. But a VM is someone
else's computer. Histology is patient data and WSI metadata routinely carries identifiers.

- **Public sets** (ANHIR, ACROBAT, DeepLIIF, CODEX) — run them in the cloud freely.
- **A real cohort** — keep it local, or satisfy yourself it is de-identified and that your
  institution permits it.

Most of the value below is on public data, so you can get it with no exposure at all.

## Build and check

```bash
docker build -f packaging/cloud/Dockerfile -t oasis-headless .
docker run --rm --gpus all oasis-headless
```

That last command is the important one. It prints what **each half** of the pipeline resolved
to and times one LoFTR pass. The failure this guards against is silent: a run that works,
costs GPU-hours, and quietly did half the work on the CPU. It exits non-zero if either half
landed on CPU.

## Run something

```bash
# validation suite (public data), reports written to the mounted volume
docker run --rm --gpus all \
  -v ~/oasis_validation_datasets:/data:ro \
  -v $PWD/validation_reports:/oasis/validation_reports \
  oasis-headless python -m validation.run all --tier long

# one validation
docker run --rm --gpus all -v ~/oasis_validation_datasets:/data:ro \
  oasis-headless python validation/validate_scale_filter_anhir.py --limit 400

# a cohort pipeline run
docker run --rm --gpus all -v /path/to/images:/in -v /path/to/out:/out \
  oasis-headless python run_pipeline.py --config /in/config.yaml
```

## On GCP, concretely

**Check GPU quota before planning anything.** New and trial projects frequently have
`GPUS_ALL_REGIONS` quota of **0**, and the increase request is a separate approval that can be
refused on a trial account. Find out first — everything else is wasted effort if this is
blocked.

```bash
gcloud compute regions describe us-central1 --format="table(quotas.metric,quotas.limit)" \
  | grep -i gpu
```

Then a Deep Learning VM, which already has the NVIDIA driver and container toolkit:

```bash
gcloud compute instances create oasis-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=common-cu123-debian-11 --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT          # ~70% cheaper; these jobs are restartable
```

Set a budget alert before you start, and **delete the instance when done** — an idle GPU VM
bills at the same rate as a busy one, and that is the usual way free credit disappears.

```bash
gcloud compute instances delete oasis-gpu --zone=us-central1-a
```

## Rough cost

A T4 plus `n1-standard-4` is roughly **$0.50/hr** on demand, well under $0.20/hr on spot.
The entire validation backlog below is tens of GPU-hours, not hundreds — see the sizing note
in the repo discussion. Budget in the tens of dollars, not the hundreds.
