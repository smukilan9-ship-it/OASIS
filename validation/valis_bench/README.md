# VALIS vs OASIS — serial-section registration benchmark (ANHIR/CIMA)

Independent, **non-circular** comparison of OASIS's registration (LoFTR→similarity, and
the structural automated path) + our certification **gate** against **VALIS** (rigid, and
rigid+non-rigid) on ANHIR/CIMA expert landmarks. Built to answer: *does VALIS register
serial sections better than we do, and is our gate honest?* — without touching the main
pipeline.

## Why it's non-circular (the whole point)

1. **Landmarks are never given to any registration.** Both methods register from image
   pixels only; the expert landmarks are used *solely* to score rTRE. (VALIS is benchmarked
   the same way in its Nature Comms paper — held-out ANHIR landmarks, rTRE.)
2. **One shared scorer.** `common.py` (numpy+PIL only) is imported *identically* by both
   the main `.venv` and the isolated VALIS venv, so the rTRE metric cannot drift between
   the two methods.
3. **Identity baseline always reported**, so neither method is credited for pre-alignment.
4. **Gate calibration is judged by landmarks the gate never saw.** We feed the gate the
   LoFTR correspondences, record its verdict, then check the *independent* expert-landmark
   rTRE per verdict bucket. A trustworthy gate ⇒ low rTRE for pass verdicts.
5. **VALIS-rigid (distance-preserving) is separated from VALIS-nonrigid.** Only rigid is
   apples-to-apples with our similarity and cross-K-safe; non-rigid is reported *only* as an
   accuracy upper bound — it is the warp OASIS forbids for spatial-association stats
   (`serial_registration.assert_distance_preserving`).

## rTRE

Relative target registration error = ‖T(moving_landmark) − fixed_landmark‖ / fixed-image
diagonal (ANHIR convention). Relative ⇒ invariant to working scale. Per-pair median, then
median-over-pairs = **MMrTRE**.

## Data layout expected

```
~/oasis_validation_datasets/CIMA_ANHIR/inputs/
  dataset/<set>/scale-<N>pc/<stem>.jpg           # the images to register
  annotations/<set>/user-<XX>_scale-<M>pc/<stem>.csv   # expert landmarks ( ,X,Y )
```
`common.enumerate_pairs` auto-matches stems, rescales landmarks from the annotation scale
(M) to the image scale (N), and forms every usable directed pair (both stains have a CSV
with equal point counts AND an image on disk).

**Current data gap:** only `lung-lesion_3` has images locally, at 5pc (~895 px) — too small
for LoFTR (starves to <6 matches on cross-stain pairs). A rigorous run needs the higher-res
CIMA images (~25–50pc) for all 9 sets (lung-lesion 1–3, lung-lobes 1–4, mammary-gland 1–2).

## Run

```bash
# 1) OASIS side — main project venv (torch/kornia/SimpleITK):
.venv/bin/python -m validation.valis_bench.run_ours          # -> ours_results.json

# 2) VALIS side — isolated venv (needs libvips on DYLD path):
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
  ~/valis_runtime/venv/bin/python -m validation.valis_bench.run_valis   # -> valis_results.json

# 3) Merge + report (main venv):
.venv/bin/python -m validation.valis_bench.compare          # -> REPORT.md
```

## Environments

- Main `.venv`: unchanged; provides LoFTR (kornia/torch), SimpleITK, the OASIS gate.
- `~/valis_runtime/venv` (uv, py3.11): `valis-wsi` 1.2.0 + native libvips (brew `vips`),
  openslide, jpype/bioformats. Isolated exactly like the DeepLIIF benchmark so the 2021-era
  stacks never touch the project. rebuild:
  `uv venv --python 3.11 ~/valis_runtime/venv && uv pip install --python ~/valis_runtime/venv/bin/python valis-wsi`
