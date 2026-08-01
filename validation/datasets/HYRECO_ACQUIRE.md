# Getting HyReCo

## Why this one

Every registration number OASIS has comes from one of two places, and neither is the thing we
actually ship against:

- **A synthetic warp** (`validate_loftr_fle_groundtruth.py`) — ground truth is exact, but the
  two images are the same section resampled, so there is no biological difference to absorb.
  It measures the matcher's floor and says so.
- **ANHIR** — real expert landmarks, but lung/mammary/kidney and largely cross-stain. It is
  the control, not the target.

HyReCo is **CD8 on consecutive sections with expert landmarks**. Nine cases, five stains each
(H&E, CD8, CD45, Ki67, PHH3), 11–19 landmarks per section, 690 total, placed by hand and
verified by two researchers. That is the calibration `research/registration.md` § 11.6 names
as missing, on the target modality.

It settles, on real serial sections rather than by proxy:

- the true FLE between two *different* physical sections (the term the synthetic harness
  structurally cannot see);
- whether the 5 µm gate and its confidence-bound stacking (§ 4) are calibrated on H-DAB;
- the window-size question — at what diameter does a real serial pair certify;
- the unvalidated `_RADIUS_FLOOR_FACTOR = 3.0`.

## Download it on a VM, not on your laptop

`HyReCo.zip` is **233.36 GB**. The two `HyReCo-Additional-*` archives (273.73 GB and
232.58 GB) are H&E + PHH3 only — **do not download them**, they have no CD8 and nothing we
need.

This is the case where cloud credit genuinely earns its keep, and the reason is bandwidth and
disk, not GPU:

- A home connection at 100 Mbps takes ~5.5 hours; a GCP VM ingresses at multi-Gbps and
  **ingress is free**.
- You need ~500 GB of disk (233 GB archive + extracted), which is ~$20/month of standard
  persistent disk — a few dollars for the week you actually need it.
- You can then run the validations in the same place the data already is, and delete the disk.

```bash
gcloud compute instances create oasis-hyreco \
  --zone=us-central1-a --machine-type=n2-standard-8 \
  --boot-disk-size=600GB --boot-disk-type=pd-balanced \
  --image-family=common-cu123-debian-11 --image-project=deeplearning-platform-release
```

Add `--accelerator=type=nvidia-tesla-t4,count=1 --maintenance-policy=TERMINATE` if you want to
run certification on the same box. The download itself needs no GPU.

## The login

Access is **"Open Access dataset files are accessible to all logged in users"** and the page
states **"IEEE Membership is not required"** — though it also mentions a DataPort
subscription, so confirm on your own account which applies before planning around it.

**You have to do this part.** Signing in and downloading is yours to run — never paste an IEEE
password or session token into this repo, a script, or a chat. The practical route is:

1. Sign in on the dataset page and start `HyReCo.zip` in the browser.
2. Copy the resulting signed S3/CloudFront URL from your browser's download manager (it is
   time-limited and already carries the authorisation).
3. On the VM, fetch it with that URL:

```bash
curl -L -C - -o /data/HyReCo.zip 'PASTE_THE_SIGNED_URL'
unzip -q /data/HyReCo.zip -d /data/HyReCo/inputs
```

`-C -` resumes a partial transfer, which matters at this size. If the signed URL expires
mid-download, re-copy a fresh one and re-run the same command — it picks up where it stopped.

Then point OASIS at it (`validation_data_dir` in Settings, or the `paths.yaml` entry), so it
lands at `<validation_data_dir>/HyReCo/inputs/`.

## Before anything is trusted

Landmarks are **millimetre world coordinates**, not pixels. Converting them needs each
slide's microns-per-pixel and origin out of the BigTIFF metadata, and a sign or origin error
there produces TRE numbers that look plausible and are wrong — the same class of mistake as
the transform-direction trap in `registration.md` § 9, which cost a run before it was caught.

So the first thing to do with real files, before any loader is written against them:

```bash
ls -R /data/HyReCo/inputs | head -40
head -3 /data/HyReCo/inputs/*/*.csv | head -20
python -c "import openslide,sys; s=openslide.OpenSlide(sys.argv[1]); \
print(s.dimensions, s.level_count, dict(list(s.properties.items())[:25]))" \
  /data/HyReCo/inputs/<case>/<one>.tif
```

Send that output back and the loader can be written against the real format instead of
against an assumption about it.
