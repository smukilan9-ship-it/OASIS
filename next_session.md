# Next session — registration certification: state, the training question, and the plan

Written 2026-07-10. Read this end to end before touching registration code. It is the full
handoff: what we built, what it means, whether to train a model, and exactly what to do next.

---

## 0. TL;DR

- The old certification gate was measuring the **wrong thing** (landmark-set tidiness, not
  registration accuracy). It false-rejected good hand-clicked pairs and false-accepted
  badly-deformed machine-proposed ones. This is a known result: **fiducial registration
  error and target registration error are uncorrelated** (Fitzpatrick 1998/2009).
- We built a replacement — the **Fitzpatrick–West cell-error gate** — and a **model-free
  correspondence source (LoFTR)**, and we **externally calibrated** the gate against a
  second human annotator on ANHIR (predicted/realized p90 ratio 0.96–1.10, coverage 89–93%).
- **Your LL477 CD8↔TIM-3 pair = `LOCALLY_CERTIFIED`** (67% of field, cell-error p90 2.85 µm),
  with honest caveats below. It is *not* misaligned; it is under-certified over the full field.
- **Nothing is wired into the app.** The app still runs the old gate.
- **On "download all of ANHIR and train the model": no — not now, and not in that framing.**
  Full reasoning in § 6. Short version: there is no model of ours to train; the only learned
  part is pretrained LoFTR; the bottleneck is calibration and a mismatch tail, not model
  capacity; there is no CUDA on this machine; and we have zero ground-truth CD8/TIM-3
  correspondences to train on. The next step is **calibration**, which needs a *small*
  targeted download, not 2 TB.

---

## 1. The scientific arc (why any of this happened)

The user started from a shaky-registration complaint and a specific symptom: **hand-placed
landmarks fail to certify, but auto-proposed and guided landmarks certify — which is
backwards.** Chasing that produced the following chain, each step forced by a measurement:

1. **The old gate (median leave-one-out landmark TRE ≤ 5 µm) is the wrong statistic.** LOO
   measures how self-consistent a landmark *set* is. Hand clicks are noisy → self-inconsistent
   → fail, however good the alignment. RANSAC-proposed points are self-consistent *by
   construction* → pass, however bad the alignment. Verdict is anticorrelated with truth.
2. **The fix is Fitzpatrick–West:** don't drive residuals to zero. Measure FLE (fiducial
   localisation error) independently; predict target error at the *cells*; recover deformation
   by variance decomposition. Built and unit-validated.
3. **Your pair still wouldn't globally certify — because the correspondences were wrong.**
   Lumen centroids (the only landmarks the auto-proposer knows) **cannot be matched by
   appearance** across CD8/TIM-3: no patch descriptor beats AUC 0.64, SIFT returns 0 matches.
   The RANSAC step in `propose_landmarks` was not filtering matches, it was *creating* them —
   which is exactly why they can't test the transform.
4. **LoFTR breaks the deadlock.** Detector-free, whole-image attention → ~750 matches where
   lumens gave 8, model-free, and its deformation estimate tracks injected ground truth
   (0/18.7/37.3 µm → measured 4/20/33 µm). It has the global context SIFT/NCC/lumens lack.
5. **The gate then failed its first *external* calibration** (against a 2nd annotator) and had
   to be fixed twice — robust FLE and a quantile deformation bound. After the fixes it is
   calibrated. This is the single most important result of the session: a predictive safety
   bound that has been checked against reality, not just against tests we wrote.

The through-line, and the thing to protect at all costs: **never let a correspondence set be
selected by, or a bound be tuned on, the transform it is meant to certify.** Every bug this
session was a version of that circularity.

---

## 2. What was built (files + functions)

### `serial_registration.py` (edited)
- `fle_from_repeat(pass1, pass2, px, robust=True)` — FLE from two annotation passes.
  Robust median; centres out bias; reports `discordant_frac` and a `concordant` mask (drop
  landmarks where two experts marked different structures). Uses only annotations, never a
  transform.
- `fle_by_relocalization(...)` — FLE of an *automatic* detector via re-localisation under
  image noise. Lower bound (image-noise only).
- `deformation_from_landmarks(ref, mov, M, px, fle, method="robust")` — variance
  decomposition `σ_fit² = 2·FLE² + model²`. Robust (leverage-standardised median +
  bootstrap) **and** classical branches. Reports both an RMS bound and a **p90 field
  quantile** bound, an FLE-consistency test (`fle_consistent`, catches overstated FLE), and
  `tail_ratio` (RMS/median — a bad-match detector).
- `cell_error_budget(...)` — `sqrt(TRE_pred² + deformation²)` at p90 over the analysis
  window; gates on `max(quantile_ub, rms_ub)`.
- `residual_field_assay(ref, mov, M, px, fle)` — **the adjudicator.** Moran's I on residual
  *vectors*, permutation-tested. Smooth field ⇒ `REAL_DEFORMATION`; random ⇒
  `CORRESPONDENCES_BAD`; small ⇒ `CORRESPONDENCES_GOOD_NO_DEFORMATION`. Controls verified
  (smooth I=0.33 p=0.001; random I=-0.01 p=0.15). **The one test that separates "tissue is
  bent" from "matcher is wrong" with no ground truth.**
- `correspondences_for_certification(...)` — lumen-based model-free matcher. **Kept as a
  documented NEGATIVE result** — it does not work on CD8/TIM-3 (proves lumens aren't
  appearance-matchable).
- `landmark_register_and_verify(..., fle_um=None, landmarks_are_model_selected=False,
  censor_um=None)` — new FW gate reached when `fle_um` is passed; otherwise the old LOO
  path runs unchanged (back-compat). `_certify_fitzpatrick_west(...)` is the new verdict
  engine (CERTIFIED / LOCALLY_CERTIFIED / RADIUS_LIMITED / DEFORMED / NOT_CERTIFIABLE).

### `loftr_matcher.py` (new — needs torch + kornia)
- `loftr_correspondences(ref, mov, px, weights="outdoor", scales=(0.75,0.5), tol_um=4.0)` —
  **cycle + scale consistency**, no RANSAC, no residual threshold. Displacement-based
  agreement (not position — positions are grid-offset). Returns points + counts + msg.
- `loftr_fle(...)` — re-runs the *whole selected pipeline* under noise to get the FLE of the
  selected population (not the raw matcher). Lower bound.

### Validation (new, both PASS)
- `validation/validate_fw_certification.py` — E1/E2/E3 on LL477 with injected ground truth.
- `validation/validate_fw_anhir_calibration.py` — 2-annotator external calibration.

### Docs
- `ihc.md` § 3.5 (new), § 3.4 note, § 6 + § 7 additions.

### Scratchpad (NOT in repo — under the session scratchpad dir)
- `loftr_test.py`, `loftr_certify.py`, `loftr_sweep.py`, `loftr_final.py` — exploratory LoFTR
  runs. Reproduce the LL477 numbers. Move logic into a committed script next session.

---

## 3. Current validation status

| check | result |
|---|---|
| `pytest tests/` | 43 pass, 1 skip |
| `validate_fw_certification.py` | E1 PASS · E2 PASS · E3 PASS |
| `validate_fw_anhir_calibration.py` | ratio 0.96/1.03/1.10, coverage 89–93% — PASS |
| LL477 under FW + LoFTR | LOCALLY_CERTIFIED (67% field, cell-err p90 2.85 µm) |

Environment note: torch 2.13.0 + kornia 0.8.3 installed in `.venv` (CPU/MPS only). LoFTR
weight download needs `export SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi;print(certifi.where())')`.

---

## 4. The LL477 verdict, honestly

- Under the **old gate**: CERTIFIED at 4.9 µm LOO (but the gate is meaningless — it certified
  a 31 µm-deformed pair too).
- Under the **new gate with a tuned confidence threshold** (last session): CERTIFIED 4.67 µm.
  **Retracted** — the threshold was picked by watching the residual tail, a mild circularity.
- Under the **new gate with a-priori cycle/scale selection** (no tuning): **LOCALLY_CERTIFIED**,
  67% of field, cell-error p90 2.85 µm, field-wide deformation p90 ~18 µm.

Two caveats that are load-bearing and must not be smoothed over:
1. **The LoFTR confidence floor is not yet a-priori-calibrated.** The residual field is only
   weakly smooth (Moran I 0.018, p 0.001) — consistent with a small real deformation buried
   in matcher error. We don't yet know the split.
2. **`LOCALLY_CERTIFIED`'s ROI is itself a residual-based selection** (keep landmarks within
   3σ of the fit). That narrows a window rather than certifies a field, but it *is* the same
   species of circularity we're fighting. Declared in the reason string; not clean evidence.

Bottom line: LL477 is **probably fine over ~two-thirds of the field, not demonstrated over
the whole field**, and the limiting factor is now the matcher's error tail — not the tissue.

---

## 5. Open problems, ranked

1. **LoFTR confidence floor is not a-priori-calibrated.** (Blocks a clean LL477 verdict.)
   Fix: calibrate the floor on ANHIR where expert landmarks give ground truth, then apply the
   fixed floor to LL477. Same trick that made the gate calibration work.
2. **`outdoor` vs `indoor` weights.** `indoor` gives confidently-wrong-but-SMOOTH matches that
   `residual_field_assay` mislabels as REAL_DEFORMATION. Weight choice is empirical (one pair).
   The assay catches *random* error, not *systematic* error — a real gap.
3. **`LOCALLY_CERTIFIED` ROI circularity** (see § 4). Consider replacing residual-based ROI
   with a spatial-block ROI chosen before fitting, or drop local certification for LoFTR sets.
4. **χ²/bootstrap assume independent residuals.** Dense LoFTR matches are spatially correlated
   → effective dof < 2n−4 → interval slightly too tight (anti-conservative). Quantify with a
   block bootstrap.
5. **Calibrated only on lung + mammary H&E↔IHC.** Not on H-DAB, not on CD8/TIM-3. Generalisation
   across tissue/stain is assumed, not shown.
6. **Nothing wired into the app.** `webui/api.py` still calls the LOO path.
7. New validations not yet in `validation/registry.py` (so they don't appear in the Validation
   tab / `python -m validation.run`).

---

## 6. THE TRAINING QUESTION — full analysis and recommendation

**Question asked:** "should we download the entire ANHIR dataset (2 TB drive) and train the
model?"

**Recommendation: No. Do a small targeted download and *calibrate*, don't train.** Reasoning,
because this is an easy place to waste a month:

**a. There is no "our model" to train.** The pipeline's intelligence is statistical — the FW
gate, the assay, FLE, the deformation decomposition. None of it is learned. The only learned
component is **LoFTR, which is already pretrained.** So "train the model" can only mean
"fine-tune LoFTR on histology." Worth naming that precisely before spending disk on it.

**b. The bottleneck is not model capacity.** Pretrained LoFTR already gave 998 correspondences
and tracked injected deformation almost unbiased. What's unresolved is (i) the confidence-floor
calibration and (ii) the gross-mismatch tail. Both are fixable *without training* — the floor
via ANHIR ground truth (§ 7 step 1), the tail via the robust estimator we already built.
Training would be solving a problem we have not shown we have.

**c. No compute.** `torch.cuda.is_available() == False`. This is an Apple-silicon Mac (MPS
only). LoFTR training/fine-tuning is a multi-GPU-day job; MPS won't run it credibly (and some
LoFTR ops fall back to CPU on MPS). Training is blocked on **hardware**, not disk. The 2 TB
drive solves the wrong constraint.

**d. No ground truth for the actual target.** ANHIR is lung/mammary/kidney H&E↔IHC. It is
**not** CD8/TIM-3 DAB. Fine-tuning on ANHIR adapts the matcher to *ANHIR's* stain pairs. And
you have **zero** GT correspondences on CD8/TIM-3 — which is precisely the "cross-stain,
no labels" wall that forced the very first paper you brought (CGNReg, J Pathol Inform 2023) to
build a whole CycleGAN. Training your own matcher walks straight into that wall.

**e. Training makes the circularity *worse*, not better.** If you train a matcher and then
certify registration with its correspondences, you must prove it did not learn to emit
transform-agreeing (self-consistent) matches — the exact failure we spent this session killing.
A trained matcher is *harder* to audit for that than a frozen pretrained one.

**f. What ANHIR IS good for: validation, not training.** A broad ANHIR download (it's ~100–150
GB at full res, not 2 TB — the drive is fine but overkill) is genuinely useful as a
**generalisation corpus**: run the gate + LoFTR across many tissue types and stain pairs to
see where it holds and where it breaks. That is validation. We only currently have the 5pc
(tiny) images locally — too small for LoFTR. We need the medium/50pc scale for the pairs that
have expert landmarks.

**When would training actually be worth revisiting?** Only if calibration (step 1) shows
pretrained LoFTR *fundamentally cannot* match CD8/TIM-3 at any usable confidence — i.e. the
floor that certifies ANHIR gives too few or too-wrong matches on H-DAB. Even then the right
move is **self-supervised domain adaptation on your own unlabeled CD8/TIM-3** (synthetic warps
+ photometric augmentation), on rented CUDA, with the assay as the acceptance test — not
supervised training on ANHIR. Park it; don't start it.

**Concrete download to do instead:** the ANHIR/CIMA **50pc images** for the three
two-annotator pairs we already have landmarks for (lung-lesion_3, mammary-gland_1,
mammary-gland_2) plus a handful of single-annotator pairs across other tissues for
generalisation. Tens of GB, not 2 TB. Source: ANHIR challenge (grand-challenge.org) / CIMA
(Borovec et al.). We currently hold only landmarks + 5pc images (`~/oasis_validation_datasets/
CIMA_ANHIR`, 1.6 MB).

---

## 7. The next-session plan (do these in order)

### Step 1 — Calibrate the LoFTR confidence floor on ANHIR ground truth (HIGHEST VALUE)
The one thing that unblocks a clean LL477 verdict and validates LoFTR beyond one pair.
- Download 50pc images for the 3 two-annotator ANHIR pairs (+ a few single-annotator pairs).
- For each pair: run `loftr_correspondences` at a sweep of internal confidence floors, fit a
  similarity, and compare LoFTR's implied error against the **expert landmark TRE** (ground
  truth). Pick the floor where LoFTR's correspondences agree with expert landmarks — chosen
  with **no knowledge of LL477's residuals**. That floor is now a-priori.
- Also confirm `outdoor` vs `indoor` on real data with ground truth (resolves open problem 2).
- Then re-run LL477 with the fixed floor and report the verdict for real.
- Deliverable: `validation/validate_loftr_calibration.py` (+ registry entry).

### Step 2 — Harden the LoFTR path into a committed module
- Move `loftr_final.py` scratch logic into a committed function
  (`loftr_matcher.certify_pair_loftr(ref, mov, px, image_wh)` returning the full verdict dict).
- Use the a-priori floor from Step 1. Handle the no-torch case gracefully (fall back /
  clear error). Add a fast down-sampling path (LoFTR at 0.5 scale is fine and ~4× faster).

### Step 3 — Fix the two residual-independence / ROI gaps (open problems 3, 4)
- Replace `LOCALLY_CERTIFIED`'s residual-based ROI with a pre-fit spatial-block ROI, OR
  disable local certification for LoFTR sets and report field-wide only.
- Swap the i.i.d. bootstrap for a spatial **block** bootstrap in
  `deformation_from_landmarks` so correlated matches don't tighten the interval.

### Step 4 — Generalisation sweep across ANHIR tissue types
- Run the whole gate on the single-annotator ANHIR pairs (many tissues/stains). Record where
  it CERTIFIES, RADIUS_LIMITS, or DEFORMS, and whether the assay agrees. This is the evidence
  that the method is not LL477-specific. (Uses LOO where only one annotator exists — note that
  limitation.)

### Step 5 — Wire into the app (only after 1–4)
- `webui/api.py`: add the LoFTR certify path; pass `fle_um`; mark proposal-derived sets
  `landmarks_are_model_selected=True`. Add a repeat-annotation FLE step in the UI (two passes
  over ~8 points, once per tissue/marker) OR use `loftr_fle` when the LoFTR path is used.
- Keep the LOO path as an explicit fallback; never silently.

### Step 6 — Register the new validations
- Add `fw_certification`, `fw_anhir_calibration`, `loftr_calibration` to
  `validation/registry.py` so they surface in the Validation tab and `python -m validation.run`.

### (Deferred / only-if-Step-1-fails) — Training / domain adaptation
- Self-supervised LoFTR fine-tuning on unlabeled CD8/TIM-3 (synthetic warps), on rented CUDA,
  assay as acceptance test. Do NOT start unless Step 1 proves pretrained LoFTR can't match H-DAB.

---

## 8. Reproduction commands

```bash
cd "/Users/mukilan/PycharmProjects/ihc-original copy"
export SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi;print(certifi.where())')

# the three validations
.venv/bin/python validation/validate_fw_certification.py       # E1/E2/E3
.venv/bin/python validation/validate_fw_anhir_calibration.py   # 2-annotator calibration
.venv/bin/python -m pytest tests/ -q

# LL477 under the new gate + LoFTR (scratch script)
.venv/bin/python "<scratchpad>/loftr_final.py"                 # move into repo next session
```

Key paths: LL477 images `~/Desktop/assets/{cd8_input,tim3 input}/LL477_*_3.tif`, px 0.7519.
ANHIR landmarks `~/oasis_validation_datasets/CIMA_ANHIR/inputs/annotations/` (PS + JB).

---

## 9. Things not to forget / traps

- **The circularity rule is the whole game.** Any new correspondence source or bound: prove it
  does not depend on the transform under test. Run `residual_field_assay` on any candidate.
- **`fle_um=None` → old LOO gate.** The new gate is opt-in. Don't assume a run used FW.
- **A smaller FLE is the *conservative* direction** (charges more residual to deformation). The
  relocalization FLE is a deliberate lower bound.
- **RMS under-states a smooth field's p90 by ~1.6×** — always gate on the quantile (or the
  max of quantile and RMS). This was the ANHIR calibration failure.
- **Don't tune a threshold on residuals.** If you catch yourself picking a number by watching
  a residual statistic improve, stop — that's the circularity.
- **LoFTR `indoor` weights are a trap** on this data. Use `outdoor` until Step 1 says otherwise.
- The `discordant_frac` from `fle_from_repeat` matters: on ANHIR mammary ~40% of "same"
  landmarks are two experts marking different structures. Drop them (annotation-only decision).
