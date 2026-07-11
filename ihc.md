# OASIS — Technical Reference

Deterministic pipeline for **cross-type spatial association** on serial-section
single-plex H-DAB IHC (e.g. CD8 vs TIM-3), as a low-cost alternative to multiplex
imaging. No AI/LLM inference — classical CV + spatial statistics, fail-closed.
(Chronological decision log preserved in `ihc_legacy_changelog.md`.)

---

## 1. Core principle

Serial sections are **different physical slices**, so a CD8 cell and a TIM-3 cell can
never be the same cell. The pipeline therefore does **not** claim single-cell
co-expression. It measures whether two cell **populations** are spatially associated
relative to spatial randomness, via cross-type Ripley's K. Single-cell co-expression
needs multiplex on one section (out of scope) or restaining (separate tab).

Two distinct questions, two nulls (§4):
- **Co-infiltration** (compartment co-occupancy) — homogeneous-CSR null. Trivially true
  for almost any two immune markers.
- **Cell-scale engagement** (proximity beyond shared compartment) — reweighted
  inhomogeneous null. The real, strong claim.

---

## 2. Architecture

All code lives in one `oasis/` package. Three entry points at the repo root share one core
and produce identical results:
- **CLI**: `run_pipeline.py --config cfg.yaml --mode {quant|spatial}`
- **Desktop UI**: `app.py` (pywebview) → `oasis/webui/api.py` + `oasis/webui/index.html`; the
  API writes a config and shells to `run_pipeline.py`.
- **Browser UI**: `serve.py` — serves the *same* `oasis/webui` over HTTP (thin `fetch` shim +
  long-poll bridge for the `evaluate_js` push channel), so the identical UI runs in a browser
  and can be driven/validated without the desktop window. pywebview is untouched.

| Module | Role |
|---|---|
| `run_pipeline.py` | Orchestrator: QuPath/InstanSeg segmentation, quant, spatial driver |
| `oasis/common/pixel_size_util.py` | µm/px from burned-in scale bar; per-image resolution |
| `oasis/common/registration.py` | Thumbnail loading, hematoxylin deconvolution, SITK helpers |
| `oasis/common/file_matcher.py` | Pair matching by filename stain tokens |
| `oasis/quant/cell_expansion.py` | Membrane markers: cytoplasmic-ring DAB + completeness cutoffs |
| `oasis/spatial/serial_registration.py` | Serial registration, landmark + FW certification, auto-propose |
| `oasis/spatial/spatial_stats.py` | Cross-type K/g/L, three nulls, DCLF test, cohort FDR |
| `oasis/spatial/spatial.py` | Spatial-association driver |
| `oasis/spatial/loftr_matcher.py` | LoFTR correspondences + LoFTR-in-ROI local certification |
| `oasis/reporting/overlay.py` | Segmentation / density / association figures |
| `oasis/reporting/dashboard.py` | Result dashboards |
| `oasis/webui/calibration.py` | Fit per-marker membrane cutoffs from hand-labelled cells |
| `oasis/restained/restained_coexpression.py` | Separate same-section restained tab (not this flow) |

Validation harnesses stay in `validation/` (the registry references them by filename);
quarantined scratch lives in `legacy/`.

Segmentation: QuPath 0.7 + InstanSeg `brightfield_nuclei` (config `qupath_binary`,
`instanseg_model`, `device=mps`).

---

## 3. Pipeline stages (in order)

### 3.1 Pixel size
Detects the solid horizontal scale bar in the bottom strip (longest contiguous dark run,
voted over a threshold sweep to reject the text label). Per-image µm/px with a session
default + overrides. Wrong pixel size mis-scales every distance, so it is resolved up
front and flows to all images.

### 3.2 Segmentation & quantification
InstanSeg nuclear segmentation → Ruifrok–Johnston/Macenko stain deconvolution → DAB OD
per cell. Default class: nuclear DAB > threshold (per-stain, e.g. CD8 0.20, TIM-3 0.10).

**Membrane mode** (`cell_expansion.py`, CD8/TIM-3): DAB measured in the **cytoplasmic
ring** = (expanded cell ∩ Voronoi) − nucleus. Half-plane Voronoi clipping stops an
expanded cell crossing the midline and stealing a neighbour's membrane DAB in dense
infiltrate. Per-image Macenko vectors with a parity fallback to fixed QuPath vectors when
degenerate (a fixed ±π-wrap collapse bug).

**Membrane completeness** (callable statistic for faint membranous markers): positive iff
a sufficient **fraction** of ring pixels exceeds a calibrated OD (`membrane_pix_thr`),
gated by DAB>H — separates a real faint arc from diffuse background, which the ring
**mean** cannot.

### 3.3 Calibration (Calibrate tab, `webui/calibration.py`)
DAB is **not quantitative** — cutoffs don't transfer across antibody/scanner. Per
protocol: segment your slides → hand-label pos/neg cells → fit `membrane_pix_thr` +
`membrane_frac_min`.
- **Multi-image**: pools labelled cells across ≥1 slide (captures staining variability).
- **Honesty metric**: **leave-one-cell-out** F1/AUC (each cell scored by a fit on the
  *others*). Callable gate = held-out AUC ≥ 0.75, not the optimistic in-sample.
- Built-in preset: CRC-ICM TIM-3 (pix_thr 0.30, frac_min 0.14, AUC 0.93).
- Spatial applies **per-marker** cutoffs (`membrane_overrides` keyed by filename); CD8 and
  TIM-3 resolve their own calibration; missing → warns + falls back to ring-mean.

### 3.4 Registration & certification (serial sections)
**Similarity only** (rotation + uniform scale + translation). Never non-rigid — a warp
fabricates the inter-cell distances K consumes. (Key divergence from HALO's elastic
alignment, which is disqualifying for distance stats.) Operates on a **low-frequency
structural hematoxylin channel** (σ≈12 µm) so non-corresponding nuclei blur away and
shared architecture dominates.

- **Auto `register_similarity`** (diagnostic path): multi-init MI + gradient-magnitude
  (edge) candidates; **selects by Normalized Gradient Field** edge alignment (`_ngf_score`)
  + NGF translation refinement — MI selection aliases on quasi-periodic tissue. SITK has no
  NGF/MIND optimiser metric, so NGF is applied at selection, not in the optimiser.
- **Auto-propose landmarks** (`propose_landmarks`): lumen centroids + structural corners →
  grid-seed → RANSAC similarity → consistent matches → local-NCC snap; coverage-first → ROI
  fallback. Pre-fills the canvas; operator verifies. Proposals are consistent *by
  construction* — human confirmation makes them valid; they never certify alone. No
  confidence score is shown: the operator adjudicates every pair, and a machine confidence
  derived from the same RANSAC that selected the point would only launder that circularity.
  Rejecting a proposal places **neither** point — the operator sets both, unaided.
- **Production `landmark_register_and_verify`**: operator landmarks define a **Huber-IRLS
  robust** similarity (a landmark on a fold bends the fit instead of breaking it; breakdown
  is ~2/12 gross outliers, above which the pair degrades to a weaker verdict but never
  certifies on a corrupted transform). Accuracy on **held-out** points
  (independent-annotator set if given, else leave-one-out). Five verdicts:
  - `CERTIFIED` — n ≥ 6, held-out TRE median ≤ 5 µm, fit-residual ≤ 5 µm
  - `LOCALLY_CERTIFIED` — only a subset passes; analyse that ROI (hull ≥ 10 %)
  - `RADIUS_LIMITED` — the landmarks **do** agree on one similarity, but only to within
    TRE > 5 µm. Serial sections deform; this is expected. Such error blurs cross-K toward
    the null — the test stays correctly sized and loses only power — so the pair is
    analysable over `r ≥ 3·TRE` and cannot be read below it. Accepted while ≥ 50 % of the
    0–100 µm range survives that floor.
  - `DEFORMED` — error leaves no interpretable radius band
  - `NOT_CERTIFIABLE` — too few correspondences (NOT evidence sections are unrelated)

  Precedence is deliberate: field-wide `CERTIFIED` > `LOCALLY_CERTIFIED` > `RADIUS_LIMITED`.
  A smaller window that keeps the contact scale (~10–20 µm) says more than the whole field
  with the contact scale removed. Guided certification therefore pursues `CERTIFIED` first
  and falls back only when it cannot be reached. A failed pair is reported, never warped.
  Every transform is asserted **distance-preserving** (`assert_distance_preserving`) before
  cells move, so cross-K radii keep their meaning. LOO is fit-unbiased but
  **single-annotator**, and floors at the landmark localisation noise σ (≈ 4 µm on real
  H-DAB sections) — it over-states a cell's true registration error. `landmark_noise_um`
  and `prediction_error_um` are reported so an operator can see why a well-aligned pair
  reads a large TRE, but they **do not gate** (see § 6). **This LOO gate is the shipped
  path but is now known-flawed — see § 3.5, which supersedes it.**

### 3.5 Fitzpatrick–West certification (validated 2026-07; wired via the LoFTR-in-ROI local path)

The LOO gate above measures the **self-consistency of a landmark set, not the accuracy of
a registration** — and the two are provably uncorrelated (Fitzpatrick, West & Maurer, IEEE
TMI 17(5):694, 1998; Fitzpatrick 2009). Measured consequences (`validation/
validate_fw_certification.py`): on a **perfect** transform with hand-click noise σ = 3 µm,
LOO rejects at ~70 % and does **not** improve with n; on RANSAC-selected proposals LOO
stays ~5 µm as true deformation goes 0 → 55 µm, so `DEFORMED` never fires and a
31 µm-deformed pair certified. Both symptoms are one bug: LOO fails good hand-clicked work
and passes bad model-selected work.

The replacement, in `serial_registration.py` (`landmark_register_and_verify(..., fle_um=)`):
- **FLE** (fiducial localisation error) is **measured, not inferred from residuals** —
  `fle_from_repeat` (two annotation passes; robust median, drops discordant landmarks) or
  `fle_by_relocalization` / `loftr_matcher.loftr_fle` (re-localise under image noise; a
  conservative lower bound). Residual-derived σ confounds FLE with deformation.
- **Cell-error budget** = `sqrt(TRE_pred² + deformation²)`, gated at ≤ 5 µm on the **p90
  over the analysis window (landmark hull)**. `TRE_pred = σ·√(fᵀ(XᵀX)⁻¹f)` falls like
  1/√n, so **more/better landmarks now genuinely buy certification** (the old gate could
  not be satisfied by working harder).
- **Deformation** is recovered by variance decomposition against the measured FLE
  (`deformation_from_landmarks`): `σ_fit² = 2·FLE² + model²`. **Robust by default** (median
  + bootstrap, breakdown-resistant — SSR read 34 µm from a handful of bad matches where the
  robust scale read 5 µm) and reported as a **field p90 quantile**, not an RMS (a smooth
  field's magnitudes are not Rayleigh; RMS under-states the p90 by ~1.6×). The gate takes
  `max()` of the quantile and RMS bounds — they fail in opposite n-regimes.
- **Circularity guards.** `landmarks_are_model_selected=True` fails closed on any RANSAC-
  selected set (its residuals cannot test the model they were selected under). An
  **FLE-consistency audit** (lower χ²/bootstrap tail) rejects a declared FLE that is larger
  than the residuals can support — closing the "overstate FLE → shrink deformation → buy a
  cert" loophole.

**Correspondence source (`loftr_matcher.py`, needs torch + kornia).** Lumen centroids
**cannot be matched by appearance** across CD8/TIM-3 (no patch descriptor separates
correct from wrong pairings, AUC 0.48–0.64; SIFT mutual-NN returns 0 matches) — so
`propose_landmarks`' RANSAC is not a filter, it is what *establishes* the match, which is
why it can't then test the transform. **LoFTR** (detector-free, whole-image attention) is
model-free and yields ~750 raw matches where lumens gave 8. Selected on **cycle + scale
consistency** (no residuals, no tuned threshold), audited by **`residual_field_assay`**
(Moran's I on residual vectors: smooth field ⇒ real deformation; random ⇒ bad matches —
the only test here that separates the two without ground truth). Caveat: LoFTR `indoor`
weights give confidently-wrong-but-smooth matches the assay mislabels — weight choice is
external and unvalidated beyond `outdoor` on one pair.

**Calibration (the only external check).** `validation/validate_fw_anhir_calibration.py`
fits on annotator PS, predicts, and measures realized error at annotator JB (held out).
After the quantile fix: predicted/realized p90 ratio **0.96 / 1.03 / 1.10** across three
ANHIR pairs, 95 %-bound coverage **89–93 %**. The bound is calibrated on lung + mammary;
**not yet on H-DAB / CD8-TIM-3**.

**Status.** The FW gate is now reachable from the app through the **LoFTR-in-ROI local
certification** path — the operator (or auto-finder) picks *where*, the unrelaxed gate
decides *whether*:
- `webui/api.certify_local_roi_multi` (draw one or many regions of any shape — polygon /
  freehand / rectangle) and `auto_certify_regions` (tissue-masked auto-find with an
  auto-selected region size). Each region: crop both sections → **LoFTR correspondences
  inside the ROI** (its coarse-whole-slide weakness vanishes in a small patch) → **local
  robust similarity fit** → **ordinary FW gate windowed to that ROI**, with FLE from
  `loftr_fle` (or a conservative fixed sub-pixel value in the fast auto sweep — charges
  more residual to deformation, never over-certifies).
- **Division of labour:** LoFTR *only* supplies correspondences; the § 3.5 gate alone
  assigns the verdict. The threshold is identical to the landmark path — a drawn region is
  never given an easier gate. `landmarks_are_model_selected=False` here because LoFTR
  matches are not RANSAC-selected against the fitted similarity.
- **Overlapping drawn regions** are split by a shapely planar partition
  (`webui/api._planar_partition`) so an intersection becomes its **own separate region** —
  no cell is counted under two different transforms.
- **Fan-out:** each certified region becomes one analysable pair in
  `run_spatial_association` (Phase-2), with its own local transform + analysis window,
  analysed **separately**; `DEFORMED` / `NOT_CERTIFIABLE` regions are dropped.
- The § 3.4 LOO landmark path is kept **separate and intact** as a second mode; `fle_um`
  still defaults to `None` there (→ LOO fallback).

**LL477 CD8↔TIM-3 under the new gate = `LOCALLY_CERTIFIED`** (67 % of field, cell-error
p90 2.85 µm; a drawn central ROI re-certifies at 1.8–3.5 µm). Honest caveats: the LoFTR
confidence threshold is not yet a-priori-calibrated on H-DAB, a local ROI is itself a
residual-based selection, and the current **pywebview desktop shell is not an HTTP server,
so an automated browser agent cannot drive/validate the UI** (motivates the browser-served
rebuild).

---

## 4. Spatial statistics (`spatial_stats.py`)

`K_ab(r) = (A/NM)·Σ 1[dist(a_i,b_j) ≤ r]`; pair-correlation `g_ab(r)`; association curve
`L_ab(r) − r` (0 independence, + attraction, − segregation). Points in CD8 pixels; window
= A∩B tissue intersection (Otsu on hematoxylin, holes preserved). **No analytic edge
correction** — the Monte-Carlo null shares the same edge bias so it cancels (validated).

**Three nulls** (`cross_k_all_nulls`):
- **Reweighted inhomogeneous cross-K** — PRIMARY. Pairs weighted 1/(λ_A·λ_B) with
  per-simulation intensity re-estimation (Baddeley–Møller–Waagepetersen). Bandwidth
  **75 µm**, deliberately above the 10–50 µm band so λ captures architecture without
  absorbing cell-scale interaction. Tests **engagement**.
- **Homogeneous CSR** — baseline; uniform in the mask. Tests **co-infiltration**; inflated
  for any compartment-sharing pair.
- **Toroidal** — structure-preserving cross-check.

**DCLF global test** over **10–50 µm** (below ~10 µm hard-core exclusion; above ~50 µm
architecture). Emits `global_p_dclf`, direction, significance. KDE bandwidth 50 µm
(Scott's/Silverman's rejected — over-smooths multimodal intensity); 0.5×/1×/2× sweep
reported.

**Dense-tissue status**: the shipped primary remains the 75 µm reweighted null when
the per-image architecture pre-flight says the tissue field is coarser than the
bandwidth. If that pre-flight fails specifically because the tissue is fine/dense,
OASIS now automatically attempts the dense morphology-conditioned primary null: B*
is sampled from marker-independent reference-section all-cell detections inside the
certified analysis window, plus **2 µm** jitter, with a **10–30 µm** DCLF band. It
is fail-closed unless landmark certification, a real analysis window, ≥30 positives
per marker, and ≥500 support cells pass. Sparse/underpowered fields are not eligible
for this switch; they are recorded as not tested. The fallback was calibrated in
`validation/validate_public_codex_dense_null.py`
on public Schürch CRC CODEX architecture templates: homogeneous CSR over-rejected true
nulls (≈10–25%); the total-cell morphology-conditioned candidate controlled H0 at
3.7–6.7% with planted-positive power 1.0. A rendered-pixel bridge
(`validate_dense_null_image_derived_morphology.py`) then recovered morphology from
synthetic H-DAB-like hematoxylin pixels (median field correlation 0.939) and kept H0
at 3.7–6.3% with power 1.0. A real LL477 demonstration then ran the candidate on completed
certified H-DAB bundles using all reference-section OASIS detections as morphology
support: x10_1 p=0.007, x10_3 p=0.024, and sparse x10_2 was skipped (10 TIM-3 positives
inside the window). A Keren TNBC dense-scaffold pilot then stress-tested the circularity
risk on three pseudo-IHC CD8/PanCK fields: replacing OASIS's all-cell support with the
independent Keren mask-derived scaffold preserved the dense verdict in all three fields
(p13/p16/p32), but perturbations showed an important boundary condition. Strong fields
p13 and p16 stayed stable under all 33 scaffold perturbations; borderline p32 was stable
in only 21/33 variants and became non-significant or fail-closed under some scaffold
damage. The Spatial tab/CLI now ships this as an automatic fallback only when the 75 µm
gate fails specifically because architecture is fine/dense and the dense
gates/provenance/ROI requirements pass. Borderline dense results must be interpreted with
scaffold-sensitivity evidence, not treated as universally invariant.

**Robustness verdict** (never a single null's significance):
`robust` (selected primary-null significant → cell-scale engagement) · `csr_only`
(CSR-only → co-infiltration) · `none` · `mixed`.

**Cohort**: Benjamini–Hochberg FDR across per-pair p; only certified pairs contribute;
never quote the bare minimum p.

**Registration QC gate** (fail-closed): bad alignment (identity fallback, residual
≥ 10 µm) → stats marked invalid + greyed. Weaker than landmark certification; disclosed.

---

## 5. Interpretation — two-question framing (UI)

`robust` = "engaged beyond shared compartments" (green). `csr_only` = "same compartments,
no cell-scale engagement" — a **distinct weaker finding, not an artifact** (cyan);
direction-aware ("compartment-scale exclusion" if segregation). Results lead with a legend
explaining both. Decisive in practice: on LL477 CD8/TIM-3, CSR gave *opposite* directions
on the same marker pair across fields (pair 2 segregation, pair 3 association) — proof CSR
reads per-field architecture; the reweighted null kept only pair 1. All three certs are
n=8 provisional, single-annotator.

---

## 6. Design decisions & rejected alternatives

- MNN single-cell matching → **cross-type Ripley's K** (serial cells don't correspond).
- Non-rigid warp → **similarity only** (warping destroys the measured distances).
- MI/NCC/phase-corr selection → **NGF selection** (dense metrics alias on periodic tissue).
- Homogeneous CSR as the test → **reweighted primary** (CSR conflates co-occupancy with
  engagement).
- Scott's/Silverman's bandwidth → **fixed 75 µm** (adaptive over-smooths).
- Analytic edge correction → **none** (bias cancels in the null; validated).
- Ring mean → **ring completeness fraction** (mean dilutes faint arcs).
- Withholding analysis from every pair with TRE > 5 µm → **`RADIUS_LIMITED`**. Registration
  error cannot manufacture a cross-K association; it only attenuates one. Size stays ≈ α at
  every ε tested, so a significant result under error stands and a null one may simply be
  under-powered. The error bounds what may be *claimed* (contact vs neighbourhood scale),
  not whether the pair runs. Clipping the DCLF band up to the floor was tried and **costs**
  power, so the floor is a **reporting boundary, not a gate on the statistic**
  (`validation/validate_radius_floor.py`). Holds only for landmark-driven, cell-blind
  transforms — an intensity-driven non-rigid warp optimises on a signal correlated with cell
  density and *could* manufacture association.
- Certifying on a **cell-level** registration error → **rejected; gate stays leave-one-out
  TRE.** The motivation was real: LOO TRE floors at the landmark picking noise σ, so a
  well-registered pair reads ≈ 6.5 µm when σ ≈ 4 µm, and cells are never clicked. The
  proposed statistic was `sqrt(estimation² + model²)` — prediction SE of the fit at the
  cells, ⊕ image-measured tissue deformation. Its **model term has no working measurement**.
  `measure_deformation` (Hann-windowed phase-correlation patch flow) is **blind**: on a real
  LL477 pair it reports 0.14 µm for the certified transform and 0.22 µm for an *identity*
  transform leaving the sections ~106 µm apart, and 0.18 µm for a known 48.8 µm translation.
  The cause is `structural_channel`'s σ ≈ 12 µm blur — added to suppress non-corresponding
  nuclei, it removes the high-frequency content a displacement estimator needs, so any two
  patches of blurred parenchyma correlate at zero offset. NCC template matching (27 µm
  median on a *correct* transform), gradient-magnitude phase correlation (no admissible
  patches) and `lumen_tre` (censored by its 12 µm inlier tolerance) all fail on the same
  images. With the model term stuck at ≈ 0 the statistic collapses to prediction SE, which
  shrinks like 1/√n — an operator could certify **any** pair, however deformed, by clicking
  more landmarks. That is fail-open, so it was not adopted. σ and prediction SE are reported
  as diagnostics; a supplied `deformation` dict is recorded and ignored, pinned by
  `validation/validate_deformation_estimator.py` and two regression tests. Consequently the
  legacy fully-automatic `certify_pair` (which gates on the same patch flow) is **superseded
  and unreachable from production**. **⚠ Partially overturned 2026-07 (§ 3.5):** the model
  term is not unmeasurable, only unmeasurable *from image patches*. It is recoverable from
  **landmark residuals against an independently-measured FLE** (`deformation_from_landmarks`).
  The 1/√n fail-open worry is answered: the deformation term does **not** shrink with n, and
  a robust + quantile estimator + FLE-consistency audit keep it honest. The cell-error gate
  is now the validated (not-yet-wired) path, calibrated against a second annotator on ANHIR.
- LOO / fiducial-residual gate → **Fitzpatrick–West cell-error budget** (§ 3.5). Fiducial
  registration error and target registration error are uncorrelated; the residual gate both
  false-rejects good hand-clicked pairs and false-accepts model-selected deformed ones.
- RANSAC/lumen-appearance correspondences for **certification** → **LoFTR + cycle/scale
  consistency** (§ 3.5). A set selected for agreeing with a similarity cannot test that
  similarity; lumens are not matchable by appearance across stains.
- Confidence-threshold tuning on the residual tail → **cycle + scale consistency** (a
  residual-free selection); the residual tail is a function of the transform under test.

---

## 7. Validation

**Research-grade validation framework** — every scientific claim is a registered,
reproducible validation runnable from the desktop **Validation** tab or the CLI (same
runner, same reports):
- `validation/registry.py` — one documented record per validation (claim / purpose / why /
  assumptions / limitations / interpretation / datasets / expected / tier / external deps),
  grouped by pipeline stage (statistical → registration → segmentation → quantification →
  spatial association → end-to-end).
- `validation/runner.py` + `validation/run.py` — `python -m validation.run <id|all|--list>`;
  each run writes `validation_reports/<id>/<ts>/report.json` (metrics, status, expected,
  software + git SHA + lib versions, dataset checksums, timing) + `run.log` + plots. Missing
  datasets/tools yield **SKIP-with-reason**, never a false FAIL.
- `validation/datasets/` — `datasets.yaml` registry (source, license, citation, sha256,
  redistributable), `resolve.py` (path resolution via `validation_data_dir`: env →
  `~/.ihc_analyzer/setup.yaml` → default `~/oasis_validation_datasets`), `verify.py`
  (presence + checksum), `acquire.py` (consolidate raw **inputs** apart from generated
  outputs). Datasets are never committed; restricted sets (HNSCC/TCIA) are documented only.
- `pytest` suite in `tests/` (unit / keystone / integration-skip-if-missing / golden).

- **Statistical correctness** — K on known clustered/CSR patterns; DCLF ~5 % false-positive
  + power; cross-validated vs R **spatstat**.
- **Reweighted null** — 3-regime proof. *Caveat*: mildly **anti-conservative** (~10 %
  type-I vs 5 % on synthetic CSR; homogeneous CSR conservative at 0 %) → p near 0.05 needs
  caution.
- **Dense-tissue null fallback** — smaller 35–45 µm reweighted bandwidths and
  square-tile conditioning were rejected. Public Schürch CRC CODEX calibration on real
  dense cell-coordinate architecture promoted one morphology-conditioned candidate
  (`10–30 µm`, total-cell field jitter `2 µm`). Rendered CODEX H-DAB-like pixels then
  showed image-derived nuclei morphology can recover the field and preserve calibration.
  Keren TNBC pseudo-IHC fields add an external-scaffold and perturbation stress test:
  p13/p16 are stable strong calls, while p32 is correctly flagged as scaffold-sensitive.
  The Spatial tab/CLI now uses this candidate automatically only when the 75 µm gate
  fails because architecture is fine/dense and dense gates/provenance/ROI handling pass;
  otherwise it remains fail-closed.
- **Registration** — TRE vs **ANHIR/CIMA expert landmarks**; best certified real pair
  (lung-lesion Cc10↔proSPC) LOCALLY_CERTIFIED at 3.66 µm ROI. HyReCo blocked (233 GB+login).
  No public two-marker same-section DAB set exists.
- **Radius floor** (`radius_floor`) — registration error costs the cross-K test power,
  never validity; size ≈ α at every ε. Evidence behind `RADIUS_LIMITED`.
- **Deformation estimator** (`deformation_estimator`) — *negative result*. Proves the
  patch-flow deformation measurement is blind (reads ≈ 0 for an unregistered pair) and
  guards against it ever gating a verdict again.
- **Fitzpatrick–West gate** (`validate_fw_certification.py`, 2026-07) — three falsification
  experiments on the real LL477 pair with injected ground truth: **E1** the LOO gate
  false-rejects a perfect transform and does not improve with n while the FW gate does; **E2**
  LOO false-accepts model-selected sets across 0→55 µm deformation while FW fails closed; **E3**
  the robust variance decomposition recovers injected deformation (36.3 µm → 35.9 µm) and its
  95 % bound covers truth every run. All PASS.
- **FW calibration vs a second annotator** (`validate_fw_anhir_calibration.py`, 2026-07) — the
  *only external* check. Fit on ANHIR annotator PS, measure realized error at held-out
  annotator JB. Predicted/realized p90 ratio **0.96 / 1.03 / 1.10**, coverage **89–93 %** on
  three lung+mammary pairs. Forced two fixes: robust FLE (drop discordant landmarks) and a
  **quantile** deformation bound (RMS under-states a smooth field's p90 by ~1.6×). Not yet run
  on H-DAB. LoFTR correspondence path (`loftr_matcher.py`) validated on LL477 only.
- **Detection/membrane** — DeepLIIF IF truth (class F1 ≈ 0.81); membranous CD8 on HNSCC
  mIF (held-out F1 ≈ 0.76, AUC 0.89). IF **proxies** — no same-section DAB+IF truth
  possible (DAB unstrippable).
- **Keystone — degradation** (`tests/test_degradation.py`, the End-to-End validation): CODEX
  same-section truth (CD8 vs PD-1) → split to pseudo-serial + inject registration error →
  verdict must not flip. Real truth `csr_only` stable under 1–3° / 3–8 px; engaged and
  independent regimes preserved. The **only** place true cross-marker association ground
  truth exists (CODEX ships as coordinates, not registrable images). The earlier
  **image-based** degradation experiment was removed (tissue-scale data, not cell-scale;
  §10) — to be redesigned on an appropriate dataset.

---

## 8. Limitations & defensible claims

**Defensible**: population-level cross-type association with size-controlled nulls;
distance-preserving registration with held-out TRE certification and fail-closed refusal;
honest compartment-vs-engagement separation.

**Open / not defensible**:
- No cross-marker DAB ground truth for the targets → CD8/TIM-3 biological claim is
  underpowered (3 pairs, one cohort, nothing survives cohort FDR).
- Certs are single-annotator LOO, n=8 provisional; one annotator-independent number only.
- Segmentation recall ~0.75 non-randomly thins dense infiltrate → biases the pattern.
- Reweighted null mildly anti-conservative; 75 µm architecture-scale is now measured
  per image, but dense tissues still fail closed until the morphology-conditioned
  candidate is validated on real H-DAB/hematoxylin morphology fields.
- DAB not quantitative; membrane accuracy on DAB extrapolated from IF proxies.

**Paper framing**: a **methods/tools paper** (pipeline + honest null framework + fail-
closed certification), validated by registration TRE, statistic operating characteristics
+ spatstat, and the degradation keystone. LL477 pairs = explicitly underpowered proof-of-
concept, never a finding. Not a discovery/biology paper.

---

## 9. Configuration & running

```
python run_pipeline.py --config cfg.yaml --mode spatial   # or quant
# --mode coloc = deprecated alias of spatial
```
Key config: `qupath_binary`, `instanseg_model`, `device`, `default_pixel_size`,
`pixel_overrides`, `threshold_overrides`, `stain_thresholds`, `cytoplasm_overrides`,
`membrane_overrides`, `spatial_pairs`, `require_landmark_certification`,
`reweight_bandwidth_um` (75). Spatial outputs per pair: detections GeoJSON/CSV,
segmentation + consolidated-density + association-curve PNGs, `*_spatial_association.json`,
cohort `spatial_cohort_fdr.json`.

---

## 10. End-to-end validation — the bounding suite

A conclusive real-DAB **cell-scale** end-to-end (real chromogenic pixels of two
*different* markers on corresponding sections, with a known cross-marker association)
cannot be assembled — serial sections put the two markers on different physical slices,
so that ground truth does not exist. We therefore **bound** the untestable case from
three sides rather than claim to close it. (An earlier image-level experiment on ~5 %
scale CIMA/IMC tiles was **removed**: at ~20 µm/px the 10–50 µm band spans <2 px, so it
was tissue-scale, not cell-scale — reading it as a cell-scale result would overclaim.)

| # | Validation | Pixels | Ground truth | Pipeline exercised | Status |
|---|---|---|---|---|---|
| Keystone | `tests/test_degradation.py` | none (coords) | real, cross-marker (CODEX CD8/PD-1) | statistic + registration-error tolerance | ✅ |
| **B** | `validate_e2e_knownwarp_deepliif.py` | **real DAB** | trivial (same cells, known warp) | **full** (InstanSeg → registration → cross-K) | ✅ |
| **A** | `validate_e2e_render_codex.py` (planned) | synthetic brightfield | **real, cross-marker** (CODEX) | **full** | ⏳ TODO.md |

**B (shipped).** Warp a real DeepLIIF IHC panel by a known transform, segment both with
the real pipeline, register, and check: reconstruction TRE small (measured ≈1.6 µm
median, ≤5 µm), the registered verdict recovers association, and the verdict **breaks
without registration** (necessity control). Proves real DAB pixels segment + register +
feed the statistic correctly at cell scale. *Limit:* same marker → association is
trivial.

**A (planned, see TODO.md).** Render real CODEX cross-marker cells into cell-scale
brightfield tiles and run the full pipeline against the known CODEX verdict. Proves the
full pixel pipeline on **real cross-marker truth**. *Limit:* pixels are synthetic.

No single row is the real thing; **B gives real pixels + full pipeline, A gives real
cross-marker truth + full pipeline, and the keystone gives real truth for the statistic
— jointly they bound the gap from every side.** The honest residual, stated plainly: the
combination of real chromogenic pixels *and* real non-trivial cross-marker truth cannot
be built for serial DAB; we bound it, we do not close it.
