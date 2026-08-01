# registration.md — the FLE problem

**Status: OPEN. This blocks every certification claim OASIS makes on H-DAB.**

Companion to `research/ihc.md` § 3.5 (the gate), § 7.1 (VALIS/ANHIR benchmark) and § 12.4
(where this was found). Read § 3.5 first — the gate's design is not restated here.

**One-line summary.** The certification gate attributes matcher localisation error to tissue
deformation, because FLE is estimated by a method that measures precision rather than
accuracy. The consequence is that well-registered regions fail, and every cell-error number
the tool has produced on H-DAB is an upper bound of unknown tightness.

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

## 5. Root cause

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

### Phase A — build the ground-truth harness *(the unblocking step)*

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
- `run_pipeline.py` does `cfg.setdefault("device", "mps")`, and `_torch_device("mps")` on a
  CUDA box finds MPS unavailable and returns **"cpu"**. LoFTR would use the GPU while InstanSeg
  silently would not. Default should be auto → cuda → mps → cpu.
- Containerise the headless path first (`run_pipeline.py` imports no pywebview). Needed for the
  cloud run anyway, and it is the reproducibility artefact JOSS/JPI will ask for.

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

## 10. Open questions for the next session

1. What **is** LoFTR's FLE on H-DAB? Everything else follows. (Phase A)
2. Is the correct gate statistic the p90 point estimate or a bound on it — and at what
   confidence?
3. With a correct FLE, does the sub-ROI rescue still fire, and should it? It is a
   residual-based selection of the analysis window, which § 3.5 already flags as a circularity
   risk.
4. Does the 84-pair CD8/TIM-3 set certify once calibrated? If it still does not, the problem is
   not FLE.
5. Does `deformation_detectable=True` / `deformation_is_validated=False` mean anything useful
   here, or is it another artefact of the same mis-attribution?
