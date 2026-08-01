# registration.md — the FLE problem

**Status: RESOLVED. The FLE was correct, the attribution was correct, and the gate was right
to refuse. Read § 11.4a and § 11.4b; everything before them is the route to that answer.**

The residual is a real, spatially continuous deformation field — 2 % of its variance is
random, Moran's I is +0.60/+0.39 at p ≤ 0.001, and three independent methods put FLE at
0.2–0.4 µm. What remains is engineering (§ 11.5): propose a certifiable window size from the
field's spatial range instead of refusing whole fields, and fix Phase C's reporting.

Companion to `research/ihc.md` § 3.5 (the gate), § 7.1 (VALIS/ANHIR benchmark) and § 12.4
(where this was found). Read § 3.5 first — the gate's design is not restated here.

**One-line summary — as first written, and now known to be wrong.** *"The certification gate
attributes matcher localisation error to tissue deformation, because FLE is estimated by a
method that measures precision rather than accuracy."* Phase A measured LoFTR's localisation
error against a warp we chose and got **≈0.2 µm** — essentially the value the app already
ships. Then the variogram of the real pair's residuals put **98 % of the variance in a
spatially structured field**, confirming the deformation the gate had booked. The FLE was not
the defect and neither was the attribution.

**The standing claim that "every cell-error number on H-DAB is an upper bound of unknown
tightness" is withdrawn.** The tightness is now known: FLE 0.2–0.4 µm against a 4 µm
deformation, so the cell-error numbers are dominated by a term that has been independently
verified to be real.

**Read §§ 11.4a–11.4b before §§ 5–7.** Sections 5–7 are kept verbatim because they are the
reasoning that led to the experiments, and the experiments are only interpretable next to
them — but their conclusions are superseded.

---

## 1. What was observed

Driving the shipped app on `LL477_CD8_x10_1 ↔ LL477_Tim3_x10_1`, one region was certified
twice with two different answers:

| path | verdict | cell error p90 | FLE shown |
|---|---|---|---|
| `Certify (auto: whole field → regions)` | LOCALLY_CERTIFIED | **3.2 µm** | `—` |
| `Certify all regions` (drawn) | RADIUS_LIMITED | **14.5 µm** | 0.17 µm |

Identical polygon (810,000 µm²), identical 82 correspondences. Reproduced under controlled
conditions on a synthetic ROI of the same pair: **3.1 µm vs 12.8 µm**, 59 correspondences both
ways, with `fit_residual_um = 4.144` and `landmark_noise_um = 4.073` **identical in both runs**.

## 2. Mechanism — established, not inferred

The two paths call `loftr_matcher.certify_local_roi` with different `fle_fast`:

| call site | path | `fle_fast` | `work_max_dim` |
|---|---|---|---|
| `api.py:1749` (`auto_certify_regions`) | the **Certify** button, FINAL answer | `True` | 800 |
| `api.py:1436` (`certify_local_roi_multi`) | **Certify drawn regions** | `False` | 800 |
| `api.py:1869` (`certify_spatial_auto`) | whole-field attempt | `False` | 1000 |
| `api.py:1631` | auto probe | `True` | 800 |

`fle_fast=True` **declares** `fle_um = 0.7`; `fle_fast=False` **measures** it via `loftr_fle`
(0.199 µm here). That single constant changes `n_good` — the count of landmarks whose residual
is "consistent with localisation noise", a threshold proportional to FLE — from **1 to 14**.
Fourteen is enough to trigger the sub-ROI rescue in `serial_registration.py` (~line 1525),
which discards the drawn window, rebuilds a smaller one around those landmarks (**32 % of the
field**) and certifies *that* at 3.1 µm.

So the verdict difference is not FLE arithmetic in the variance decomposition — that term
moves by <1 µm² — it is a **discrete branch flip** into a different, smaller analysis window.

`loftr_fle` is called with `n_trials=2`, commented "lower bound; 2 is enough".

## 3. What the residuals actually are

```
n = 59   median 4.14 µm   p75 5.32   p90 7.57   max 14.67
p90/median = 1.83          isotropic Gaussian (Rayleigh) = 1.90
core RMS (≤p75) = 3.58 µm   >10 µm: 8.5 % of matches
```

**The distribution is Rayleigh.** That is the signature of isotropic random localisation
error. It is *not* an outlier-contaminated distribution and *not* a smooth deformation field.

This retires two earlier explanations of the same symptom recorded in `ihc.md` § 3.5:
- "gross errors set the residual p90" — there is no gross-error tail here;
- the local-smoothness filter added after *84 real CD8/TIM-3 field pairs certified zero
  regions* — it drops 9.7 % here (`local_drop_frac`) and the survivors are still Rayleigh.

Both treated a calibration problem as an outlier problem.

## 4. Why the gate then fails a state-of-the-art registration

ANHIR's best methods report **median rTRE 0.19–0.38 % of image diagonal**. This image's
diagonal is 2400 px × 0.7519 = 1804 µm, so SOTA on histology is **3.4–6.9 µm median TRE**.
Our median is **4.14 µm** — inside that range.

The gate compares `≤5 µm` against an **upper confidence bound on a p90**:

| quantity | value | factor |
|---|---|---|
| median residual | 4.14 µm | — |
| p90 (inherent for Rayleigh) | 7.57 µm | ×1.83 |
| upper bound on the p90 (`deformation p90_ub`) | 12.815 µm | ×1.69 |
| threshold | 5 µm | |

Passing therefore requires a **median residual ≤ 5/(1.83×1.69) ≈ 1.6 µm ≈ 2 px** — about
2–4× better than the best published automatic histology registration. Two independent
conservatisms (a quantile, and a confidence bound on that quantile) multiply silently against
a threshold that reads like it applies to a point estimate.

`tre_pred_p90_um` was **0.091 µm** at n=59, i.e. the Fitzpatrick–West term is negligible and
the entire budget is the deformation term.

## 5. Root cause — REFUTED BY MEASUREMENT (§ 11)

> **This section is wrong and is kept deliberately.** It is the hypothesis Phase A was built
> to test, and it failed the test: the predicted FLE of ≈2.9 µm is ~15× the measured value.
> Everything below is the reasoning as it stood before the measurement existed.

```
σ_fit² = 2·FLE² + model²          model² = deformation
```

With `σ_fit = 4.073` and `FLE = 0.199`, `model = 4.06` — **all** of the scatter is booked as
deformation. But § 3 shows the scatter is localisation noise.

`loftr_fle` re-localises matches under added image noise. That measures **repeatability
(precision)**, not **distance from truth (accuracy)**. A dense matcher on repetitive tissue
texture can be highly repeatable while systematically mis-localising, so repeatability
underestimates true FLE. `ihc.md` § 3.5 already calls it *"a conservative lower bound"*.

A lower bound on FLE is mechanically an **upper bound on deformation**, hence an upper bound
on cell error, hence a gate that fails good work.

If LoFTR's true FLE on H-DAB is ≈2.9 µm (≈3.9 px), then `2·FLE² ≈ σ_fit²` and the deformation
term goes to ≈ 0: this pair has no measurable deformation, only matcher noise.

**Neither shipped mode is correct.** 0.199 µm (0.26 px) and 0.7 µm (0.93 px) both under-state
FLE. Fast mode is merely less wrong, which is why it reproduces § 3.5's recorded result
(`LOCALLY_CERTIFIED, 67 % of field, p90 2.85 µm`).

The FW bound was calibrated on lung + mammary (`validate_fw_anhir_calibration.py`, predicted/
realized p90 ratio 0.96/1.03/1.10) and, as § 3.5 states, **never on H-DAB**.

## 6. What must NOT be done

- **Do not loosen the ≤5 µm gate.** It is the right target for a 10–20 µm contact claim.
  The gate is not the defect; its input is.
- **Do not "fix" this by making the drawn path use `fle_fast=True`.** That hides the
  disagreement rather than resolving it, and locks in an unmeasured constant.
- **Do not treat it as outliers again.** That has been tried twice (§ 3).

## 7. Plan — measure FLE against ground truth

> **Phase A is DONE (`c83cfc0`); its result is § 11 and it refutes the premise of Phase B as
> written below. Phases B–D are kept as planned so the change of direction is legible.**

### Phase A — build the ground-truth harness *(the unblocking step)* — DONE

`validation/validate_loftr_fle_groundtruth.py`.

Apply a **known** warp to a real H-DAB section and measure how far LoFTR's matches land from
the answer that warp defines. Because the displacement field is known analytically at every
pixel, the per-match error is true FLE — no annotator, no circularity.

- **Warps:** pure translation (2, 5, 10, 30 µm), rotation (1°, 3°, 10°), isotropic scale
  (±2 %, ±5 %), and a smooth B-spline field of known magnitude to separate "matcher error"
  from "matcher error under deformation". Include the ~30 µm shift regime, where
  `phase-a-registration-052526` records patch-flow aliasing.
- **Report:** per-match error distribution (median, p90, Rayleigh ratio), and whether error
  scales with warp magnitude (systematic) or not (noise).
- **Confounder to control:** resample the warped image once with a fixed interpolator, and
  verify the harness returns ≈0 error on an identity warp before trusting anything else.

**Success criterion:** a defensible FLE(µm) for H-DAB at 0.7519 µm/px, with its dependence on
tissue texture characterised.

### Phase B — re-derive the gate with the measured FLE

- Re-run `validate_fw_anhir_calibration.py` with the Phase-A FLE. Check predicted/realized p90
  ratio and 95 % coverage still hold (they were 0.96–1.10 and 89–93 % on lung + mammary).
- Decide the gate statistic explicitly: p90 **point estimate** vs upper bound. If the bound is
  kept, state its confidence level and set the threshold *for a bound*, not for a point
  estimate. Record the decision and why.
- Re-run the § 3.5a neighbourhood sweep and the 84-pair CD8/TIM-3 set. **If the calibration is
  right, a material fraction of those 84 pairs should now certify.** Zero certifying was the
  original symptom.

### Phase C — make the two paths agree

- `auto_certify_regions` must re-certify surviving regions with the measured FLE before
  returning — which is what `loftr_matcher.py`'s own comment instructs ("Re-certify a chosen
  region without fast mode for the principled measured FLE"). Fast mode is for the sweep only.
- Run the **FLE-consistency audit** on the auto path. `ihc.md` § 3.5 says it exists to reject
  "a declared FLE that is larger than the residuals can support" — exactly `fle_fast`'s 0.7.
- **Reporting bug, independent of all the above:** when the sub-ROI rescue fires, the UI shows
  the *drawn* area (810,000 µm²) beside the *rescued window's* error (3.2 µm). Those describe
  different shapes, and the analysis window is the smaller one. Report the certified window's
  area and draw it.
- A blank FLE column on a certified region (as the auto path currently shows) must be
  impossible.

### Phase D — re-run the record

Everything in `ihc.md` § 12.6, at the corrected pixel size (0.7519) **and** the corrected
certification. Then, and only then, revisit § 8's cohort-FDR bullet (`ihc.md` § 12.5).

## 8. Datasets

Ordered by what each one settles. Phase A needs only the first.

**1. LL477 (local, in hand) — the FLE measurement itself.**
`~/Desktop/cd8_input`, `~/Desktop/tim3 input`. Six 10× H-DAB fields at 0.7519 µm/px with
burned-in bars. This is the target tissue and the cohort every disputed number comes from, and
synthetic warps need no second section — the ground truth is the warp. **Phase A is fully
unblocked today.**

**2. ANHIR — external calibration with expert landmarks.**
<https://anhir.grand-challenge.org/> — 355 images, 481 pairs, 18 stains, manual landmarks.
Already used for `validate_fw_anhir_calibration.py` and the VALIS benchmark (§ 7.1), so the
loader exists. Gives real inter-annotator FLE and lets Phase B check whether the recalibrated
gate still holds where it was originally calibrated. **Note the limitation:** ANHIR is
lung/mammary/kidney, largely cross-stain — it is the control, not the target.

**3. ACROBAT — breast H&E ↔ IHC WSI, already downloaded.**
22 GB in Expansion, openslide-readable at 0.907 µm/px. Closest public analogue to the target
modality (brightfield IHC, real serial sections). Landmarks are submission-only via
grand-challenge, so it validates *registration behaviour*, not TRE directly — useful for
checking FLE's dependence on tissue texture across a cohort we did not tune on.

**4. HyReCo — the ideal set, still blocked.**
Serial H-DAB with expert landmarks, which is exactly the missing calibration. 233 GB and login
(`public-data-certification-062126`). Only worth the effort if Phase A/B leave the H-DAB FLE
genuinely ambiguous.

**Not needed here:** CODEX, Keren, DeepLIIF, HNSCC — they validate nulls, dense scaffolds and
classification, none of which touch FLE.

### On the cloud GPU

Phase A is the right use of it: many warps × many fields × repeated LoFTR passes, embarrassingly
parallel, and **public or already-local data only — no patient identifiers leave the machine**.
LoFTR auto-selects CUDA (`loftr_matcher._device`), so a T4 needs no code change; at ~$0.35/hr
the $300 credits are ~850 GPU-hours, far more than the whole plan needs.

**Two things to fix before renting anything:**
- ~~`run_pipeline.py` does `cfg.setdefault("device", "mps")`, and `_torch_device("mps")` on a
  CUDA box finds MPS unavailable and returns **"cpu"**.~~ **FIXED** (`a8a8a75`). Policy now
  lives in `oasis/common/device.py`: default `auto` → cuda → mps → cpu, a named backend is a
  preference that falls through to the next accelerator rather than to the CPU, and `cpu`
  stays an absolute opt-out. The run banner and the Settings tab both state the device chosen.
- Containerise the headless path first (`run_pipeline.py` imports no pywebview). Needed for the
  cloud run anyway, and it is the reproducibility artefact JOSS/JPI will ask for.

**But note what § 11 does to the case for renting at all.** Phase A ran to completion on a
laptop CPU in minutes. The experiments it now points to — a `tol_um` sweep on one real pair,
a spatial-autocorrelation test on 59 residuals — are smaller still. The GPU becomes worth
paying for at Phase B/D scale (the 84-pair CD8/TIM-3 set, full ANHIR), not before.

## 9. Reproducing the observation

```
LL477_CD8_x10_1.tif ↔ LL477_Tim3_x10_1.tif, pixel size 0.7519 µm/px
thumbnail max_side=1920 (→ scale 1.0), provisional = register_similarity
ROI = centred axis-aligned box, half-width 0.45·W, half-height 0.45·H  (span 1299 µm)
certify_local_roi(..., work_max_dim=800, fle_fast=True | False)

fle_fast=True   LOCALLY_CERTIFIED  cell_err 3.1    FLE 0.7     n_good 14  floor  9.3
fle_fast=False  RADIUS_LIMITED     cell_err 12.816 FLE 0.1992  n_good  1  floor 38.4
both:           fit_residual 4.144  landmark_noise 4.073  n 59  tre_pred_p90 0.091
```

Residuals: map `corr_mov` through `local_matrix` (**moving → reference** — the codebase
convention; the other direction gives ~148 µm nonsense) and compare to `corr_ref`.

## 10. Open questions — as first written

1. ~~What **is** LoFTR's FLE on H-DAB? Everything else follows. (Phase A)~~ **ANSWERED: § 11.**
2. Is the correct gate statistic the p90 point estimate or a bound on it — and at what
   confidence? **Still open, and now the main question.**
3. With a correct FLE, does the sub-ROI rescue still fire, and should it? It is a
   residual-based selection of the analysis window, which § 3.5 already flags as a circularity
   risk. **Still open.**
4. Does the 84-pair CD8/TIM-3 set certify once calibrated? If it still does not, the problem is
   not FLE. **The premise is gone — the FLE was already right, so recalibration will not move
   those 84 pairs. Whatever refused them is what § 11.4 has to identify.**
5. Does `deformation_detectable=True` / `deformation_is_validated=False` mean anything useful
   here, or is it another artefact of the same mis-attribution? **Still open, but it is no
   longer "the same mis-attribution" — see § 11.4.**

---

## 11. Phase A result — the FLE was right all along

`validation/validate_loftr_fle_groundtruth.py`, commit `c83cfc0`. Registry id
`loftr_fle_groundtruth`. Full output in `validation/loftr_fle_groundtruth_results.json`.

### 11.1 What was measured

One real H-DAB field (`LL477_CD8_x10_1`, 0.7519 µm/px), densest-tissue crop, warped by a
transform of our choosing and matched against itself with `loftr_correspondences` at its
shipped defaults and at certification's own working resolution (`work_max_dim = 800`).
Because the moving image is *constructed* as `moving(W(X)) = ref(X)`, the partner of every
reference point is exactly `W(X)` — so for a returned correspondence `(p, q)`, `q − W(p)` is
its true localisation error. No annotator, no second section, no circularity.

Controls, all passing, and each one guards a failure that would look plausible:

| control | why it exists | result |
|---|---|---|
| `cv2.warpAffine` moves content forward by `M` | a direction error inverts every truth while leaving every magnitude believable | mean\|diff\| = 0.0000 grey levels |
| identity warp returns ≈0 | catches an off-by-one crop, a swapped x/y, a `to_work`/`to_full` slip | n=2213, **median 0.095 µm**, max 0.49 |
| dense-field inversion converges | if it has not, the B-spline image does not correspond to the warp called truth | round-trip residual 7.1e-3 px |

### 11.2 The numbers

Thirteen known warps at `work_max_dim = 800` — what `certify_local_roi` actually uses. Crop
1083 µm (this field is 1920×1440 px, too small for the disputed ROI's 1300 µm), tissue 42 %:

| warp | n | median | p90 | p90/med | bias | **FLE (per point)** | gross |
|---|---|---|---|---|---|---|---|
| translation 2 µm | 2213 | 0.258 | 0.562 | 2.18 | 0.15 | **0.172** | 0 |
| translation 5 µm | 2213 | 0.251 | 0.521 | 2.07 | 0.09 | **0.164** | 0 |
| translation 10 µm | 2185 | 0.302 | 0.504 | 1.67 | 0.26 | **0.171** | 0 |
| translation 30 µm | 2166 | 0.335 | 0.591 | 1.77 | 0.27 | **0.195** | 0 |
| rotation 1° | 2041 | 0.388 | 0.692 | 1.78 | 0.03 | **0.230** | 0 |
| rotation 3° | 2029 | 0.365 | 0.680 | 1.86 | 0.02 | **0.224** | 0 |
| rotation 10° | 1820 | 0.430 | 0.807 | 1.88 | 0.07 | **0.270** | 0 |
| scale +2 % | 2013 | 0.381 | 0.691 | 1.81 | 0.02 | **0.229** | 0 |
| scale −2 % | 1939 | 0.393 | 0.700 | 1.78 | 0.06 | **0.234** | 0 |
| scale +5 % | 1900 | 0.353 | 0.672 | 1.91 | 0.02 | **0.219** | 0 |
| scale −5 % | 1928 | 0.382 | 0.702 | 1.84 | 0.08 | **0.231** | 0 |
| B-spline p90 5 µm | 2213 | 0.346 | 0.617 | 1.78 | 0.04 | **0.205** | 0 |
| B-spline p90 15 µm | 2092 | 0.380 | 0.679 | 1.79 | 0.02 | **0.228** | 0 |

**Measured FLE: 0.224 µm** (median over the eleven rigid warps; range 0.164–0.270). The
shipped `loftr_fle` reported **0.199 µm** on the disputed ROI. It was accurate.

Three things the sweep settles beyond the headline:

- **Error does not grow with the warp.** Pearson r between warp magnitude and median error is
  **−0.07** — flat. That includes the 30 µm translation, where `phase-a-registration-052526`
  records patch-flow *aliasing*; LoFTR shows none (FLE 0.195 µm, 2166 matches). The error is
  the matcher's fixed localisation noise, not a function of how hard the transform is.
- **p90/median = 1.839** against a Rayleigh 1.823, and **zero** gross errors (>25 µm) in any
  of the thirteen. Pure isotropic noise, no blunder tail.
- **FLE is closer to a matcher-grid property than a tissue property.** Across a 2× change in
  working scale the error's coefficient of variation is **0.09 in working pixels vs 0.41 in
  µm** — i.e. roughly constant at **0.12–0.20 working px**, so its value in µm scales with
  `px_work`:

  | warp | work_max_dim | px_work | n | median | **FLE µm** | FLE px |
  |---|---|---|---|---|---|---|
  | translation 5 µm | 400 | 2.71 µm/px | 444 | 0.797 | 0.456 | 0.168 |
  | translation 5 µm | 800 | 1.35 µm/px | 2213 | 0.251 | 0.164 | 0.121 |
  | rotation 3° | 400 | 2.71 µm/px | 374 | 0.877 | 0.527 | 0.195 |
  | rotation 3° | 800 | 1.35 µm/px | 2029 | 0.365 | 0.224 | 0.166 |

  Only two scales, so read it as a direction rather than a law — but it is the direction that
  matters, because **the disputed ROI ran at `px_work` = 1.82 µm/px** (a ~1940 px crop into
  `work_max_dim` 800, r ≈ 0.41), coarser than either row above. At ~0.15 working px that is
  **FLE ≈ 0.27 µm** there. Reaching 2.88 µm would need `px_work` ≈ 19 µm/px.

  Note also the collapse in **n** at the coarser scale: 2213 → 444. Certification at
  `work_max_dim` 800 on a large ROI is buying its speed with an order of magnitude of
  correspondences, which is worth knowing independently of FLE.

Appearance mismatch — moving image restained, defocused, illumination-graded and noised, so
the two images stop looking like the same slide — barely moves it:

| appearance | n | median | **FLE** |
|---|---|---|---|
| matched | 2029 | 0.365 µm | 0.224 µm |
| DAB density ×0.45 | 2028 | 0.365 µm | 0.224 µm |
| DAB density ×1.9 | 2025 | 0.366 µm | 0.224 µm |
| counterstain ×0.6 | 2026 | 0.367 µm | 0.225 µm |
| defocus σ 1.5 px | 2019 | 0.362 µm | 0.225 µm |
| illumination ±15 % | 2027 | 0.368 µm | 0.225 µm |
| sensor noise σ 6 | 2025 | 0.372 µm | 0.226 µm |
| all of the above | 2022 | 0.394 µm | **0.247 µm** (×1.10, keeping 100 % of matches) |

### 11.3 What that does to the disputed ROI

```
σ_fit                      4.073 µm     (measured on the real pair, both runs)
FLE needed to explain it   2.88 µm      = σ_fit / √2
FLE actually measured      0.224 µm     (0.164–0.270 across 11 rigid warps)
  … at that ROI's coarser px_work        ≈ 0.27 µm  (extrapolated at ~0.15 working px)
  … under the worst appearance mismatch  0.247 µm
```

The gap is **~11×**, and the strongest appearance perturbation this harness can produce closes
under 1 % of it. Substituting the measured FLE changes the deformation term from 4.0633 µm to
4.0606 µm: **the FLE explains 1 % of the residual variance.**

So § 5 is dead. `loftr_fle` measuring precision rather than accuracy is a true statement about
the method that turns out not to matter, because on this tissue LoFTR's precision and its
accuracy are both sub-micron.

### 11.4 What this leaves — three live hypotheses, in order of cheapness

The 4.07 µm is *real*. Two sections, after the best local similarity fit, genuinely disagree
by that much per correspondence. Why is now the whole question, and § 3's Rayleigh finding no
longer implies what it seemed to: a Rayleigh residual is the signature of *isotropic random*
error, and having ruled the matcher out, the randomness has to come from somewhere else.

1. **The filter tolerance sets the scale.** `tol_um = 4.0` is the admission tolerance of all
   three correspondence filters, and the observed median residual is **4.14 µm**. That
   coincidence is close enough to be worth one experiment: sweep `tol_um` ∈ {1, 2, 4, 8} on
   the real pair and see whether the residual median tracks it. If it does, 4.14 µm is an
   artefact of what the filters admit, not a property of the tissue — and the whole dispute
   dissolves. *Cheapest decisive test available; run it first.* Note the confound: tightening
   `tol_um` also selects, so read the change in **n** alongside the change in residual.
2. **Serial-section correspondence ambiguity.** The sections are ~4 µm apart in the block, so
   the "same" vessel wall is a different cross-section and the true partner of a point is
   genuinely undefined by roughly a nuclear diameter. This *is* fiducial localisation error in
   the sense the budget means, but it belongs to the specimen, not the matcher, and no
   single-section harness can produce it — which is exactly the limitation § 11.1's method
   carries.
3. **Real short-wavelength deformation**, as the identity currently assumes.

**(2) and (3) are distinguishable, cheaply.** Deformation is a smooth field, so its residuals
are spatially *correlated*; correspondence ambiguity is spatially *white*. Compute the spatial
autocorrelation (or a semivariogram) of the 59 residual vectors on the disputed ROI. Correlated
at some length scale → deformation, and the gate is right to refuse. White → ambiguity, and the
budget is double-counting: `transform_prediction_error` was **0.091 µm** at n=59, so a
similarity fitted from 59 noisy correspondences is accurate even when each one is noisy, and
charging the full per-match scatter to a systematic cell displacement is wrong.

### 11.4a RESOLVED — the residuals are a real deformation field

`validation/validate_residual_origin.py`. Both experiments of § 11.4 ran on the disputed ROI,
after reproducing it exactly (n = 59, `fit_residual` 4.144, `landmark_noise` 4.073 — the
recorded values to three decimals, so this is the thing under dispute and not a lookalike).

**Hypothesis (1), filter truncation — refuted.** Sweeping `tol_um`:

| `tol_um` | n | median residual | σ_fit | med/tol | raw → cycle → scale → local |
|---|---|---|---|---|---|
| 1 | — | *fewer than 8 matches* | | | |
| 2 | 15 | 2.468 µm | 2.352 | 1.23 | 495 → 194 → 17 → 16 |
| 3 | 43 | 4.205 µm | 4.191 | 1.40 | 495 → 301 → 49 → 46 |
| **4 (shipped)** | **59** | **4.144 µm** | **4.073** | 1.04 | 495 → 365 → 72 → 65 |
| 6 | 102 | 3.915 µm | 4.268 | 0.65 | 495 → 403 → 121 → 111 |
| 8 | 124 | 4.165 µm | 4.627 | 0.52 | 495 → 418 → 154 → 136 |
| 12 | 153 | 5.732 µm | 5.406 | 0.48 | 495 → 424 → 190 → 169 |

The median is **flat at ~4 µm across tol 3→8 while n triples** (43 → 124). Under truncation
`med/tol` would be roughly constant; instead it falls by 2.7×. The 4.14 µm is a property of
the pair. (`tol_um = 2` *does* buy `LOCALLY_CERTIFIED` at σ_fit 2.35 — by keeping 15 of 153
matches, i.e. by selecting the region's most-agreeing subset. That is the circularity this
module exists to avoid, and it is why the tolerance must not be tuned on the verdict.)

**Hypotheses (2) vs (3) — decisively (3).** Semivariogram of the 59 residual vectors:

| lag | 100 | 325 | 461 | 572 | 704 | 801 | 896 | 1075 µm |
|---|---|---|---|---|---|---|---|---|
| γ | 11.6 | 31.8 | 48.6 | 49.7 | 41.7 | 31.6 | 17.3 | 26.4 |

γ rises steeply with separation — the hallmark of a spatially continuous field. (The fall
beyond ~570 µm is the ordinary detrending artefact: these are *post*-fit residuals, so the
similarity has already absorbed the global linear component. That biases the test **against**
finding structure, and structure was found anyway.)

```
sill                    31.889          (√(sill/2) = 3.99 µm, i.e. σ_fit as expected)
nugget (γ → 0)           0.668
nugget / sill            0.021           only 2 % of the variance is spatially random
Moran's I  dx  +0.5959   dy  +0.3907     null −0.0172, permutation p ≤ 0.001 both
```

**Variogram decomposition of σ_fit — the split the budget needs, read off the real pair:**

```
random (nugget)      -> FLE  0.409 µm   (combined 0.578)
structured           -> deformation  3.951 µm
```

**The 0.409 µm is the important number.** It is an entirely independent estimate of FLE —
from the spatial statistics of the real CD8↔TIM-3 pair, with no synthetic warp anywhere — and
it lands next to Phase A's 0.224 µm and the shipped 0.199 µm. Three methods, one answer:
**FLE on this material is a few tenths of a micron.** Hypothesis (2), serial-section
correspondence ambiguity inflating FLE to ~2.9 µm, is dead; had it been true the nugget would
have carried most of the variance and it carries 2 %.

### 11.4b So the gate is right — and this is what closes the file

`σ_fit² = 2·FLE² + deformation²` with FLE ≈ 0.2–0.4 µm gives deformation ≈ 3.95–4.06 µm. That
is exactly what the gate computed. **The attribution was correct all along.** This tissue
genuinely deforms by ~4 µm across a 1299 µm window, and no similarity transform can align it
better — so a ≤5 µm cell-error claim over that whole window is not available, and refusing it
is the right answer, not a calibration failure.

Note this stands **without** § 4's stacking argument: the p90 residual *point estimate* is
7.57 µm, already past the 5 µm threshold before any confidence bound is applied. § 4's
observation (that the ×1.83 quantile and the ×1.69 bound multiply against a threshold that
reads like a point estimate) remains a fair criticism of how strict the gate is *in general*,
but it is not what refused this ROI.

**And it retro-justifies the sub-ROI rescue.** A spatially continuous deformation field means
a smaller window contains less of it — precisely what `certify_local_roi`'s docstring claims
and what the ANHIR mammary 335 → 117 µm measurement showed. Certifying a 32 % sub-window at
3.1 µm is therefore scientifically sound rather than a residual-chasing artefact. What is
*not* sound is choosing that window by which landmarks had small residuals, and reporting the
drawn area beside the rescued window's error (§ 11.5).

One thing this does **not** license: § 3 read the Rayleigh p90/median 1.83 as proof of
localisation noise. That reading was wrong. A smooth field sampled at scattered points
produces a residual magnitude distribution that can look Rayleigh while being strongly
autocorrelated — the marginal distribution and the spatial structure are different questions,
and only the second one distinguishes noise from deformation.

### 11.5 What Phase B should now be

Not "re-derive the gate with the measured FLE" — that changes nothing (§ 11.3). Items 1 and 2
below are **done** (§ 11.4a); what remains is real engineering, not calibration.

1. ~~The `tol_um` sweep.~~ Done — refuted.
2. ~~The residual-autocorrelation test.~~ Done — the field is structured; the gate is right.
3. § 4's surviving criticism, now the only calibration question left: the gate compares 5 µm
   against an upper confidence bound on a p90 (×1.83 then ×1.69), which demands a median
   residual ≤1.6 µm ≈ 2 px — 2–4× better than the best published automatic histology
   registration. It did **not** refuse the disputed ROI (the point estimate already fails),
   but it will refuse marginal ones. Decide the statistic explicitly, for a *bound* rather
   than a point estimate, and record why.
4. **New, and the one with real value for users: derive the certifiable window size from the
   variogram.** The deformation field is spatially continuous, so there is some window
   diameter at which any given pair certifies. `roi_certification_neighbourhood.py` already
   sweeps window size; pair it with the variogram range so the app can *propose* a window
   instead of shrinking one by residual-chasing. This turns "RADIUS_LIMITED, sorry" into
   "certifiable below ~N µm", which is an answer a biologist can use.
5. Re-run the 84-pair CD8/TIM-3 set with (4). The original symptom — zero of 84 certifying —
   is now expected rather than mysterious: they were all being asked to certify whole fields
   carrying ~4 µm of real deformation.

Phase C's reporting bugs (the drawn area shown beside the rescued window's error; a blank FLE
on a certified region; `auto_certify_regions` never re-certifying with a measured FLE) are
untouched by any of this and are still real — and § 11.4b makes the first one worse, not
better: now that the rescue is known to be scientifically justified, showing the wrong area
next to its error is the only thing standing between it and being defensible in a paper.

### 11.6 Limits of this result — stated plainly

- It measures the matcher's **floor**. A section matched against a resampled copy of itself
  has no biological difference to absorb; the real FLE cannot be *smaller* than this, and this
  harness gives no upper bound on it.
- One field, one tissue, one magnification. `--image` takes any other.
- The appearance perturbations are synthetic. A real CD8 → TIM-3 pair differs in ways a
  restain cannot imitate.
- The **selection** is the shipped one (cycle + scale + local smoothness). On self-matched
  images essentially everything survives (2213 of 2213); on the real pair 59 of ~750 did.
  A heavily-filtered population should be *better* localised, not worse — but this harness has
  not verified that on a population filtered that hard, which is another reason § 11.4(1)
  matters.
- **The harness cannot see a truncation effect, by construction.** All three filters admit
  disagreement up to `tol_um = 4.0 µm`; the errors measured here are ~0.25 µm, so the
  tolerance never binds. That is exactly why § 11.4(1) needs the real pair — where the median
  residual (4.14 µm) sits *at* the tolerance, which is what a truncated distribution looks
  like. Note the mechanism is plausible and not merely numerological: the filters cap the
  *pairwise* displacement difference between neighbours, so a population whose true scatter
  exceeded `tol_um` would be culled back to roughly `tol_um` and its survivors' median would
  land there regardless of the tissue.

---

## 12. HyReCo — the external validation, and what it says about our slides

Downloaded 2026-08-02 from IEEE DataPort. Only 12 % of the 233 GB archive was needed: all 45
expert landmark CSVs (15 KB) plus CD8/H&E/CD45 for cases 611 and 679 (27.9 GB), pulled member
by member over HTTP range requests (`validation/datasets/remote_zip_probe.py`).

**Format verified, not assumed.** Slides are 95,601 × 218,145 px at 0.2430 µm/px, 17 pyramid
levels. Landmarks are millimetre world coordinates: `px = mm × 1000 / 0.2430`, origin (0,0),
no offset. Proof rather than arithmetic — every landmark was converted, a patch read at that
location, and **14/14 and 11/11 landed on tissue**. A sign or origin error would have put them
on glass. Counts match across all stains per case, so they are index-corresponded.

### 12.1 Predicted vs realized error at expert landmarks — section scale

First external test of the gate on serial H-DAB. LoFTR never sees a landmark, so the expert
set is fully held out.

| case | pair | verdict | predicted p90 | provisional TRE | certified TRE |
|---|---|---|---|---|---|
| 611 | CD8→HE | DEFORMED | 504.9 µm | 52.2 | 90.0 |
| 611 | CD8→CD45 | DEFORMED | 117.8 µm | 35.8 | 31.2 |
| 611 | HE→CD45 | NO_MATCHES | — | **10,672** | — |
| 679 | all three | NO_MATCHES | — | 25–33 | — |

**The gate never under-predicted (2/2), by a factor of ~3.** That is the safe direction.

But read the rest honestly: **4 of 6 pairs returned NO_MATCHES**, and `register_similarity`
failed catastrophically once (10,672 µm). At the ~13 µm/px a whole 23 × 53 mm section renders
to, `certify_local_roi` downscales to `work_max_dim` and LoFTR has almost nothing to work
with. **OASIS does not register whole sections**, and this is the measurement that says so.
It is a field-scale tool and these numbers are outside its regime.

A first version of this test reported that 10,672 µm as a *certified* TRE, because it fell
back to the provisional matrix when LoFTR found nothing. The two transforms are now measured
separately — a failed registration must not be reported under the certification's name.

### 12.2 The blunder rate is our slides, not the matcher

`validate_hyreco_field_blunders.py`, 1920 × 1440 fields at exactly 0.7519 µm/px — LL477's own
frame — centred on each expert landmark:

Final, all 50 fields over both cases:

| stratum | fields | n (median) | residual median | gross median | gross mean |
|---|---|---|---|---|---|
| ALL pooled | 50 | 1,029 | 4.80 µm | 3.2 % | **14.3 %** |
| **CD8 ↔ CD45** (IHC↔IHC) | 25 | **2,686** | 3.06 µm | **0.3 %** | **1.1 %** |
| CD8 ↔ H&E (IHC↔H&E) | 25 | 365 | 8.47 µm | 21.9 % | 27.5 % |
| **LL477 CD8 ↔ TIM-3** (IHC↔IHC) | 3 | 77–632 | 3.5–8.3 µm | — | **13.6 %** |

**POOLING IS A TRAP HERE, AND THE SCRIPT'S OWN HEADLINE FALLS INTO IT.** Pooled, HyReCo's mean
gross rate is 14.3 % against LL477's 13.6 % — indistinguishable, and a naive reading would
conclude the slides are fine. That number mixes a genuine cross-MODALITY problem (DAB-on-
haematoxylin against H&E, a different colour space entirely) with the like-for-like one. The
`validate_hyreco_field_blunders.py` summary prints the pooled median, which happens to favour
the conclusion; the stratified table is the honest analysis and it is what the argument rests
on.

Stratified, CD8↔CD45 — the correct analogue for CD8↔TIM-3, both DAB on haematoxylin — gives
**2,686 correspondences at 1.1 % gross**, consistently across both cases (611: 1.7 %, 679:
0.2 %).

CD8↔CD45 is the correct analogue for CD8↔TIM-3 — both are DAB-on-haematoxylin IHC. On
published, expert-annotated slides that pairing yields **4,754 correspondences at 0.7 % gross**.
LL477's equivalent yields **77–632 at 13.6 %**: an order of magnitude fewer correspondences and
twenty times the blunder rate, on the same class of stain pairing at the same magnification and
pixel size.

**LL477's CD8↔TIM-3 behaves like a hard cross-modality pair rather than the IHC↔IHC pair it
actually is.** Combined with § 12 of `ihc.md` — TIM-3 thresholded below its own background,
4–58 callable cells reported as 11,584 — the conclusion is that the slides are the limiting
factor, and no model work fixes a slide.

### 12.3 Correspondence density predicts blunders

Across all 30 fields, log(n) against gross fraction gives **r = −0.66**:

```
fields with n <  600 :  9 fields, mean gross 21.0 %
fields with n >= 600 : 21 fields, mean gross  3.6 %
```

This is the mechanism behind everything chased this session. Sparse correspondences do not
merely weaken a fit — they are *contaminated*, because the local-smoothness filter has no
dense neighbourhood to appeal to and `reject_local_residual_outliers` has no local consensus.
Density is not a nice-to-have; below roughly 600 correspondences per field the blunder rate
rises sixfold.

### 12.4 Getting more correspondences — do not fine-tune first

Fine-tuning LoFTR was rejected earlier on similarity-ceiling grounds, and § 11.3 strengthens
that: deformation dominates the cell error and better correspondences cannot reduce
deformation. But § 12.3 shows density has a second, independent effect through the blunder
rate, so the question was reopened.

**It is probably already solved upstream.** *MatchAnything* (arXiv 2501.07556, zju3dv — the
LoFTR authors) large-scale pre-trains detector-free matchers for cross-modality matching and
reports, on ANHIR: *"our trained ELoFTR model achieves a 55.3 % relative improvement"* in
Average-Average rTRE, and 33.2 % for ROMA. Weights are stated to be on HuggingFace and the
training code is "available later".

So the ranking for getting better correspondences on this modality:

1. **Try MatchAnything's pre-trained ELoFTR weights.** No training, no dataset, no GPU. If the
   ANHIR improvement transfers, it is a weight swap. **Verify the licence before adopting** —
   this repo is heading for JOSS and neither the code nor the weight licence was visible from
   the project page or README.
2. **Then re-measure with `validate_matchers_on_cohort.py`**, which already compares five
   matchers on LL477 both ways.
3. **Only then consider fine-tuning**, and if so the corpus is **ACROBAT** (750 training cases,
   3,406 WSIs, H&E ↔ ER/Ki67/PGR/HER2 — by far the largest) with **HyReCo subset B** (54
   re-stained H&E↔PHH3 pairs: *same physical section*, so geometric truth is near-exact while
   the stain difference is real) for stain-invariance specifically. ANHIR's 481 pairs are for
   validation, not training — its landmarks are far too sparse.

### 12.5 What is still open

- Only 2 of 9 cases downloaded (27.9 GB of 259.5 GB). The signed URL expired; more needs a
  fresh one.
- No mid-scale test yet. The landmarks are ~3 mm apart, so a 1.4 mm field holds at most one and
  the section is 13 µm/px — neither is OASIS's 450 µm certification regime. A ~5 mm window at
  ~1 µm/px would hold 3–5 landmarks and *is* close to it. That is the test that would validate
  the ROI gate externally, and it has not been run.
