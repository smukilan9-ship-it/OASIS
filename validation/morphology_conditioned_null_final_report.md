# Morphology-Conditioned Dense Null - Final Focused Report

This report summarizes the dense-tissue null work after the smaller-bandwidth
reweighted null failed.

## Methods Tested

Three dense-tissue null ideas were tested:

1. Smaller reweighted primary null, h = 35-45 um.
2. Square-tile compartment-conditioned null.
3. External morphology-conditioned null.

Only the third approach showed serious promise.

## Why Smaller Reweighted Bandwidth Failed

Focused calibration rejected h=35-45 um reweighted null presets. They had good
power, but they stayed anti-conservative under dense shared architecture.

Representative results:

| Candidate | Dense shared H0 p05 | Banded H0 p05 | Short power | Mid power | Decision |
|---|---:|---:|---:|---:|---|
| h45, 10-30 | 0.250 | 0.497 | 0.963 | 0.907 | reject |
| h35, 10-30 | 0.147 | 0.113 | 0.897 | 0.763 | reject |
| h40, 5-20 | 0.150 | 0.200 | 0.983 | 0.843 | reject |

Conclusion:

```text
Smaller h alone is not enough.
```

## Why Square-Tile Conditioning Failed

A serial-section-safe compartment null was tested:

```text
keep A fixed
preserve observed B count per square tile
redraw B uniformly within each tile
```

Pilot result:

| Best tile candidate | Uniform H0 | Dense shared H0 | Intermediate H0 | Banded H0 | Power 5 um | Power 12 um | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| tile 20 um, band 5-20 | 0.050 | 0.225 | 0.075 | 0.050 | 1.000 | 0.875 | reject |

Larger tiles were much worse, with dense shared H0 often near 1.0.

Conclusion:

```text
Tile-count conditioning is too crude. It preserves coarse local abundance but
does not preserve the continuous dense architecture inside the tiles.
```

## External Morphology-Conditioned Null

The promising method uses a marker-independent architecture field:

```text
lambda_M(x) = morphology / tissue-architecture intensity field
```

Then:

```text
keep A fixed
sample B* from lambda_M(x)
compare observed cross-K to B* null curves
```

This is serial-section-safe because it does not permute A/B labels across
different physical sections. It only randomizes B under a marker-independent
architecture model.

In the validation harness, `lambda_M(x)` is the oracle field used to generate the
synthetic tissue. This is a best-case proof of concept. Real OASIS would still
need to estimate `lambda_M(x)` from hematoxylin, total nuclei, tissue structure,
or certified morphology channels.

## Focused Calibration: Morphology Null, Band 5-20 um

Command:

```bash
.venv/bin/python validation/validate_morphology_conditioned_null.py \
  --sims 300 --nperm 199 \
  --morph-smooth-um 0 \
  --bands 5-20 \
  --planted-fraction 1.0
```

Results:

| Regime | P(p <= 0.05) | Verdict |
|---|---:|---|
| Uniform CSR H0 | 0.047 | pass |
| Dense shared-architecture H0 | 0.043 | pass |
| Intermediate shared-architecture H0 | 0.067 | pass |
| Banded shared-compartment H0 | 0.080 | borderline |

Power:

| Positive control | Power |
|---|---:|
| 5 um attraction | 1.000 |
| 12 um attraction | 0.967 |

Decision:

```text
BORDERLINE_NEEDS_MORE_SIM
```

Interpretation:

This is the first dense-tissue method that controls the main dense shared-field
failure mode and retains strong power. The only issue is the banded-compartment
H0 at 0.080, slightly above the strict 0.07 target.

## Focused Calibration: Morphology Null, Band 10-30 um

Command:

```bash
.venv/bin/python validation/validate_morphology_conditioned_null.py \
  --sims 300 --nperm 199 \
  --morph-smooth-um 0 \
  --bands 10-30 \
  --planted-fraction 1.0
```

Results:

| Regime | P(p <= 0.05) | Verdict |
|---|---:|---|
| Uniform CSR H0 | 0.057 | pass |
| Dense shared-architecture H0 | 0.047 | pass |
| Intermediate shared-architecture H0 | 0.040 | pass |
| Banded shared-compartment H0 | 0.073 | borderline |

Power:

| Positive control | Power |
|---|---:|
| 5 um attraction | 0.993 |
| 12 um attraction | 0.943 |

Decision:

```text
BORDERLINE_NEEDS_MORE_SIM
```

Interpretation:

The 10-30 um band is slightly cleaner than 5-20 um on the banded H0 and still has
excellent power. It is the better candidate if OASIS wants a dense-neighborhood
claim closer to the existing 10-50 um spatial framework.

## Effect Of Morphology Smoothing

Pilot sweep:

| Morphology smoothing | Outcome |
|---:|---|
| 0 um | controls dense shared H0; strong power |
| 10 um | borderline / partial leakage |
| 20 um | anti-conservative |
| 30 um | strongly anti-conservative |

This is critical:

```text
The morphology field must preserve dense architecture. Oversmoothing lambda_M
brings back the original false-positive problem.
```

## Current Verdict

The external morphology-conditioned null is biologically and statistically
promising, but not yet fully production-validated.

Best current candidate:

```text
external morphology-conditioned null
DCLF band: 10-30 um
morphology field smoothing: as little as possible
status: borderline, needs more simulation and real-image morphology validation
```

This is much better than the 35-45 um reweighted null, because it directly
conditions on marker-independent architecture rather than trying to infer
architecture from the marker-positive cells themselves.

But it is not yet a paper-ready production claim because:

1. The test used oracle morphology, not an H-derived field.
2. The banded H0 is slightly above target at 0.073.
3. Real serial-section morphology extraction must be validated.

## What To Do Next

### Step 1: Public Real-Architecture Calibration

Completed with `validation/validate_public_codex_dense_null.py` on the Schürch CRC
CODEX single-cell table. This uses real dense CRC cell-coordinate fields as
marker-independent architecture templates, then simulates known-truth null and
planted-positive marker populations on top. It does not assume any biological
marker pair is a true null.

```bash
.venv/bin/python validation/validate_public_codex_dense_null.py \
  --spot-cap 60 --sims-per-spot 5 --nperm 199 \
  --generator-sigmas-um 2,5,10 \
  --null-sigmas-um 2 \
  --power-jitters-um 5,12 \
  --planted-fraction 0.75 \
  --include-homogeneous
```

Focused result:

| Method | Band | H0 generator | p<=0.05 | Verdict | Power 5 um | Power 12 um |
|---|---:|---|---:|---|---:|---:|
| homogeneous CSR | 10-30 | 2 um | 0.223 | anti-conservative | 1.000 | 1.000 |
| homogeneous CSR | 10-30 | 5 um | 0.250 | anti-conservative | 1.000 | 1.000 |
| homogeneous CSR | 10-30 | 10 um | 0.167 | anti-conservative | 1.000 | 1.000 |
| morphology-conditioned, 2 um support jitter | 10-30 | 2 um | 0.067 | pass | 1.000 | 1.000 |
| morphology-conditioned, 2 um support jitter | 10-30 | 5 um | 0.037 | pass | 1.000 | 1.000 |
| morphology-conditioned, 2 um support jitter | 10-30 | 10 um | 0.057 | pass | 1.000 | 1.000 |
| morphology-conditioned, 2 um support jitter | 5-20 | worst H0 | 0.080 | borderline / reject | 1.000 | 1.000 |

Decision:

```text
candidate promoted for further validation:
  morphology-conditioned total-cell support field
  support jitter: 2 um
  DCLF band: 10-30 um

production status:
  do not ship yet
```

The public coordinate-level calibration is stronger than the oracle-only synthetic
result because it uses real CRC tissue architecture. It still does not validate
the image-derived `lambda_M(x)` OASIS would need from H-DAB/hematoxylin images.

### Step 2: Rendered H-DAB-Like Image-Derived Morphology

Completed with `validation/validate_dense_null_image_derived_morphology.py`.
The harness renders real CODEX all-cell coordinates into hematoxylin-like pixels,
extracts hematoxylin with the OASIS deconvolution helper, detects nuclei from the
rendered morphology image, and uses those detected nuclei as the marker-independent
`lambda_M(x)` support.

Focused command:

```bash
.venv/bin/python validation/validate_dense_null_image_derived_morphology.py \
  --spot-cap 60 --sims-per-spot 5 --nperm 199 \
  --generator-sigmas-um 2,5,10 \
  --null-sigmas-um 2 \
  --power-jitters-um 5,12 \
  --planted-fraction 0.75 \
  --no-homogeneous
```

Morphology recovery:

| Metric | Result |
|---|---:|
| field correlation, median | 0.9389 |
| detected/true nuclei ratio, median | 0.8436 |
| median nearest detected nucleus distance | 0.2335 um |

Focused calibration:

| Method | Band | Worst H0 p<=0.05 | Min power | Decision |
|---|---:|---:|---:|---|
| oracle coordinate morphology | 10-30 | 0.067 | 1.000 | passes screen |
| image-derived nuclei morphology | 10-30 | 0.063 | 1.000 | passes screen |
| oracle coordinate morphology | 5-20 | 0.077 | 1.000 | reject / mildly anti-conservative |
| image-derived nuclei morphology | 5-20 | 0.073 | 1.000 | reject / mildly anti-conservative |

Decision:

```text
rendered-pixel bridge passed for:
  image-derived nuclei morphology
  support jitter: 2 um
  DCLF band: 10-30 um

production status:
  still do not ship; next required check was real LL477 H-DAB serial-section ROIs
```

### Step 3: Real LL477 H-DAB Serial-Section Demonstration

Completed with `validation/validate_dense_null_real_ll477.py`. This reads the
completed OASIS LL477 spatial bundles, uses certified transforms and A∩B windows,
builds marker-independent morphology support from all reference-section OASIS
detections, and runs the `10-30 um / 2 um` support-jitter candidate. This is a
real-use demonstration, not calibration, because LL477 has no known null truth.

Focused command:

```bash
.venv/bin/python validation/validate_dense_null_real_ll477.py \
  --results-root /Users/mukilan/Desktop/ihc_spatial_results \
  --asset-roots /Users/mukilan/Desktop/assets \
  --nperm 999
```

Result:

| Pair | Status | A+ in window | B+ in window | Candidate 10-30 p | Direction | Existing OASIS |
|---|---|---:|---:|---:|---|---|
| LL477_CD8_x10_1 | tested | 236 | 52 | 0.007 | association | robust |
| LL477_CD8_x10_2 | skipped | 72 | 10 | - | insufficient TIM-3 events | csr_only |
| LL477_CD8_x10_3 | tested | 59 | 75 | 0.024 | association | csr_only |

Decision:

```text
real-use demonstration passed:
  usable certified LL477 pairs run under the dense candidate
  sparse pair is excluded fail-closed

production status:
  still do not ship until the candidate is wired into production with provenance,
  ROI handling, sparsity gates, and reviewer-facing wording
```

### Step 4: Real H-DAB Morphology Field

Candidate sources:

- hematoxylin structural density,
- total nuclei density,
- tissue mask distance / local tissue density,
- gland/lumen/stroma/tumor compartment maps,
- local H texture features.

Requirements:

- marker-independent,
- same coordinate system as the reference image,
- restricted to certified analysis ROI,
- not derived from CD8/TIM-3 positivity,
- not oversmoothed beyond dense architecture.

### Step 5: Validate Real H-DAB Field Against The Rendered/Oracle Bridge

For real images or rendered synthetic morphology:

1. Generate or estimate true architecture field.
2. Extract H-derived `lambda_M`.
3. Compare calibration:
   - oracle morphology null,
   - H-derived morphology null,
   - current reweighted null.

### Step 6: Production Gate

Only add a dense mode if:

```text
uniform H0 p05 in [0.03, 0.07]
dense shared H0 p05 in [0.03, 0.07]
banded H0 p05 acceptably controlled
power at 5 and 12 um >= 0.80
real H-derived morphology behaves like oracle morphology
```

## Paper Framing If This Passes

If the H-derived version passes, the dense-tissue method can be framed as:

> A morphology-conditioned serial-section null that preserves marker-independent
> dense tissue architecture while testing whether marker-positive populations are
> locally associated within certified serial-section ROIs.

This is stronger and more biologically honest for dense tissue than a global
smoothed reweighted K, because it separates:

```text
architecture: encoded by morphology lambda_M
marker relationship: tested as residual local proximity beyond lambda_M
```

## Current Production Recommendation

Do not ship dense mode yet.

But unlike the 35-45 um reweighted null, this method is worth pursuing.
