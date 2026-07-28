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
InstanSeg was chosen over StarDist (det-F1 0.807 vs 0.665 on DeepLIIF) and over DeepLIIF's
own model (det-F1 0.82 vs 0.65 on HNSCC expert masks); see §7 "Segmenter choice".

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
model-free and yields ~750 raw matches where lumens gave 8. Selected on **cycle + scale +
local-smoothness consistency** (no residuals, no tuned threshold), audited by
**`residual_field_assay`** (Moran's I on residual vectors: smooth field ⇒ real deformation;
random ⇒ bad matches — the only test here that separates the two without ground truth).
Caveat: LoFTR `indoor` weights give confidently-wrong-but-smooth matches the assay
mislabels — weight choice is external and unvalidated beyond `outdoor` on one pair.

*Local smoothness was added after 84 real CD8/TIM-3 field pairs certified zero regions.*
Cycle and scale test the matcher against itself, and a match to the wrong instance of a
repeated structure passes both — the reverse pass and the coarser scale err identically.
About 10–20 % of survivors were gross errors (one field: 7 µm median residual, **631 µm
max**), and nothing downstream rejected them, `_fit_similarity_robust` being Huber
(down-weights, never rejects) under an explicit assumption of human-validated landmarks
that the dense-matcher path silently broke. Those few points set the residual p90 that
`cell_error_budget` gates on **and** collapsed the assay's Moran's I, so pairs were failed
*and* misreported as deformed tissue. The filter keeps a match only if its displacement
agrees with its k nearest neighbours' median — local continuity only, never global form, so
a tissue no similarity can describe passes intact and the certification keeps its power to
reject (`test_local_smoothness_does_not_select_for_a_similarity`). Added assumption, stated:
the displacement field is continuous at the neighbour scale — true away from tears; across a
genuine tear it would drop the minority side, so `local_drop_frac` is reported.

**It is a safety net, not an accuracy improvement, and the ANHIR A/B says so**
(`validate_local_smoothness_anhir.py`, 44 training pairs, expert landmarks held out from both
arms). No overall shift — Wilcoxon **p = 0.93**, 34/44 pairs move ≤ 0.5 px — which is the
correct result for a filter that has nothing to remove from an already-clean set: where arm A
was already < 10 px (n = 35) the mean delta is **−0.04 px**, a no-op. The value is entirely in
the contaminated tail, which is the regime this pipeline's own data sits in: where arm A was
≥ 10 px (n = 9), mean delta **−6.97 px**. Among the ten pairs that move at all, 8 improve and
2 worsen, and the magnitudes are asymmetric — best −41.7 px against worst +2.4 px. Mean cull
9.7 %. The worst regression (mice-kidney_1 9_PAS→6_CD31, 6.2 → 8.6 px at a 22 % cull) is the
honest cost and marks the limit: on cross-stain pairs, local continuity can discard correct
matches. Do not cite the aggregate as evidence the filter registers better; cite the split.

*The assay verdicts on effect size, not p.* It previously called `REAL_DEFORMATION` on
p < 0.05 alone. Permutation power grows with n, so at a dense matcher's n it said yes to
structure far too weak to carry a deformation estimate: n = 441 at I = 0.014 — two
thousandths above its own random control — returned p = 0.001. Every pair tested came back
`REAL_DEFORMATION`, i.e. the adjudicator had stopped adjudicating in exactly the regime it
exists for. Moran's I approximates the spatially-structured *fraction* of residual variance
and so does not inflate with n; `_MORAN_EFFECT_FLOOR = 0.10` sits between the function's two
controls (smooth 0.331, random −0.006), with p retained as a secondary guard.

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
- **Auto region size follows the claim boundary, not the cell count.** The size ladder is
  probed at the seeded anchor and the chosen rung is the **largest one whose radius floor
  still resolves the contact scale** (≤ 20 µm, the same test `spatial.py` uses for
  `contact_scale_resolved`) — not simply the largest rung that certifies. The two differ:
  on LL477_Liver_10X_4, R = 450 µm certifies at a 21.2 µm floor while R = 260 µm certifies
  at 19.0 µm, so size-first silently spends the ~10–20 µm band for area. When *no* rung
  keeps the contact scale, the largest certifying rung is taken for statistical power and
  `size_policy` records that cell-scale engagement is **not** claimable for that pair. The
  floor is not monotone in size (450 → 21.2, 350 → 24.7, 260 → 19.0), so smaller rungs are
  probed whenever the largest certifying one loses the band; when it does not, the ladder's
  descending order already makes it the answer and probing stops. Measured on the 33-pair
  10X set the policy chose the same size as size-first on every pair (16/33 pairs, 18
  regions either way) — it is a **claim guard for the cases where they diverge**, not an
  accuracy win, and the early exit keeps it free (1326 s vs 1337 s; the exhaustive form
  cost 2567 s for the identical result).
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
- Per-image adaptive DAB threshold (GMM valley) → **fixed per-stain cutoff, cohort-wide**
  (§ 11). A threshold that moves per image quantifies staining variation *into* the
  biology, which is the one thing the measurement exists to remove.
- Thresholding a percentile-normalised image → **rejected outright** (§ 11.3). Per-channel
  percentile rescaling is a white balance; it destroys the optical-density relation that
  deconvolution depends on and reintroduces counterstain cross-talk.
- Lowering `ashman_min` to make the adaptive cut fire → **rejected** (§ 11.2). Ashman's D
  is a two-mode statistic; at 1–3 % positives there is no second mode to find, so relaxing
  the gate does not fix a mis-specified rule, it only lets it fire.

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
- **Segmenter choice — InstanSeg vs StarDist** (`validation/stardist_vs_instanseg_RESULTS.md`,
  2026-07-14) — both run headless in QuPath under **identical brightfield conditions**
  (`BRIGHTFIELD_H_DAB`, 0.5 µm, full-image annotation, same GeoJSON export, same 15 px
  centroid matcher) over all 598 DeepLIIF images / 41,428 IF-derived GT cells. InstanSeg
  `brightfield_nuclei` **det-F1 0.807** (recall 0.752, prec 0.871) vs StarDist
  `dsb2018_heavy_augment` on the deconvolved hematoxylin channel (thr 0.5) **det-F1 0.665**
  (recall 0.853, prec 0.546 — over-detects 64.8k objects vs ~41k GT). **InstanSeg better on
  580/598 images.** A hematoxylin-intensity post-filter sweep (proxy for a prob-threshold PR
  curve; QuPath doesn't export per-detection probability) caps StarDist at **det-F1 ≈ 0.723**,
  still −0.084 below InstanSeg; area filtering does nothing → the excess are genuine spurious
  nuclei calls, not splitting fragments. StarDist's only edge is raw recall, not worth the
  precision hit here. Framing is in-domain (InstanSeg, built for brightfield) vs repurposed
  fluorescence (StarDist); the RGB H&E StarDist model was **not** run because the data is
  DAB-IHC, not H&E — the deconvolution route already *is* the correct brightfield path.
  **Decision: InstanSeg stays the segmenter.** TensorFlow has no Python-3.14 build, so the
  native `stardist` package cannot run in-repo; QuPath's bundled TF path executed the `.pb`.
- **Segmenter choice — InstanSeg vs DeepLIIF** (`validation/deepliif_vs_instanseg_RESULTS.md`
  + `score_hnscc_deepliif_vs_instanseg.py`, 2026-07-17) — decided on an **independent**
  expert-labelled set (HNSCC mIHC, 268 tiles, hematoxylin-only input, 0.5 µm, **91,173**
  expert nuclei), *not* DeepLIIF's own test distribution (which would be circular home-turf).
  Both run identically (0.5 µm, adaptive OFF, DAB 0.35). InstanSeg **det-F1 0.82** (@15 px;
  0.77–0.82 across 6–15 px, pixel-F1 0.823) vs DeepLIIF `DeepLIIF_Latest_Model` **det-F1 0.65**
  (0.49–0.65, pixel-F1 0.691). DeepLIIF finds a similar *count* (pred/GT 0.97) but **localises
  poorly and over-segments background** (hallucinated nuclei in stroma on off-distribution
  hematoxylin-only input); InstanSeg tracks the expert mask closely. **Caveat:** DeepLIIF is
  trained on full IHC RGB and internally infers hematoxylin→seg, so hematoxylin-only handicaps
  it — but that IS the nuclear signal the pipeline uses, and DeepLIIF is generative
  signal-inference (§1) that additionally cannot do membranous CD8/TIM-3. Native `deepliif`
  can't run on the repo's Python 3.14 (2021-era stack) → isolated py3.11 env, project `.venv`
  untouched. **Decision: InstanSeg stays.**
- **Segmenter runtime — QuPath vs in-process TorchScript** (`oasis/quant/segment.py`,
  `validation/validate_native_segmenter*.py`, 2026-07-25) — the *model* is unchanged
  (InstanSeg `brightfield_nuclei`); only the thing that runs it changed, from the QuPath binary
  + generated Groovy to an in-process `torch.jit` call. Three checks. (a) **Model fidelity:**
  the TorchScript bundle, fed through our own rdf.yaml `scale_range` preprocessing, reproduces
  the model's shipped reference output **exactly** (336/336 labels, foreground IoU 1.0).
  (b) **Deconvolution parity:** our DAB optical density inside QuPath's *own* nucleus polygons
  vs QuPath's exported `DAB: Mean` — r **0.9986**, slope 0.991, MAE **0.0035 OD** (n=1500,
  LL477 CD8). (c) **The gate — 598-image DeepLIIF IF-truth benchmark re-run and diffed through
  the same scorer:** det-recall 0.752→**0.747**, det-precision 0.871→**0.876**, class-only F1
  0.809→**0.812**, class accuracy 0.928→**0.929**, end-to-end F1 0.666→**0.669**; every
  |Δ| ≤ 0.005. Runtime 0.25 s/image on CPU. Two QuPath behaviours had to be reproduced
  explicitly and are the non-obvious part: **resampling to the model's trained 0.5 µm/px**
  (`getPreferredDownsample`, clamped at 1.0 — never upsample; on 0.752 µm/px LL477 upsampling
  inflates counts 4796→6719 because enlarged nuclei get split) and **512-px tiling with 32-px
  context padding plus centroid-in-core object ownership** (without which every seam
  double-counts). Two further findings came out of validating the parts the DeepLIIF gate
  cannot reach — its 512-px panels resample to 256 px, i.e. a *single tile*, so tiling was
  untested by it. (i) **Contested pixels at seams:** neighbouring tiles infer independently over
  the shared padded band and disagree at object edges, so a later tile could overwrite an object
  an earlier one had emitted — measured on LL477/256 px, one object destroyed and one reduced to
  a fragment out of 5397. Contested pixels now go to the first claimant. (ii) **Normalisation
  must be global, not per-tile.** The rdf `scale_range` percentiles computed per tile make the
  count depend on tile size: LL477 gives 5396 objects at 256 px, 5379 at 512 px, 4657 whole — a
  16% spread driven by a performance knob. Computing them once over the image gives
  4611/4632/4651/4657 (**0.7% spread**) and lands closer to QuPath. Post-fix: whole-vs-tiled
  agreement recall **0.989**, precision **0.999**, **0 duplicate** objects, residual 1%
  disagreement correctly concentrated at seams (54% within 8 px vs 12.6% baseline); LL477 count
  ratio vs QuPath **0.966** (was 1.40), det-F1 **0.819**, class agreement **0.999**. Also
  verified: bit-identical repeat runs, Otsu identical to a transliteration of the Groovy,
  degenerate inputs (blank/tiny/1-px/thin-strip) return empty without raising, and the membrane
  path (`measure_cytoplasm_dab`) consumes native GeoJSON with ring-DAB median shift 0.0026 OD
  and membrane-positive rate shift 0.000. **Whole slides** (validated on ACROBAT
  `0_KI67_val.tif`, 48128×16128 = 776 Mpx, 0.907 µm/px) are **streamed**: holding one whole
  would need ~18.6 GB for the float64 optical-density intermediates alone, on a 17.2 GB machine.
  Stripe streaming reproduces the in-memory path **identically** (same object count, DAB MAE
  0.00000) at bounded peak RSS. Two further findings there: (iii) the global normalisation
  percentiles must be **exact, not estimated** — a downsampled pyramid level is off by 0.35
  normalised units (averaging destroys the tails a 0.1/99.9 percentile is made of) and a
  level-0 grid sample by 0.28; segmenting with the sampled range agreed with the exact one on
  only **63% of nuclei** while the counts looked reassuringly close (120 vs 126). Since the data
  is 8-bit, a 256-bin histogram over a counting pass is exact by construction (verified: 0.0
  difference vs `np.percentile`), and the same pass records which stripes hold tissue so blank
  ones are skipped. (iv) a **dynamic-range floor** (`MIN_DYNAMIC_RANGE = 40`) is required:
  normalising a near-blank crop against its own degenerate range amplified sensor noise into
  **3926 spurious "nuclei"** on a 99.95%-background ACROBAT crop. The floor is set from the
  validated corpus — minimum channel span is 98 over all 598 DeepLIIF panels and 144 on LL477,
  versus 10 for the blank crop — so it cannot alter any validated result. Note this also shows
  global normalisation is a *correctness* property, not only a tile-independence one: the same
  blank crop under true whole-slide ranges yields 0 objects. **Full-slide run (completed):** the
  whole 776 Mpx slide streamed in 32 stripes → **27,156 nuclei**, peak RSS **5.37 GB** against a
  naive requirement of 18.6 GB; median nucleus area 35.4 µm² (≈6.7 µm equivalent diameter) and
  ≈8,500 cells/mm² over the tissue, both well inside the pipeline's own plausibility band.
  Runtime 49 min on CPU, ≈17 min on MPS. **Device:** the corpus was scored on CPU but the
  shipped default is `device: mps` (598 panels 148.7 s → 51.8 s; a 2048-px tile 12.1 s → 2.4 s).
  Equivalence measured, not assumed: identical total cell count (35,286) and **597/598 GeoJSON
  files byte-identical**; the single differing file had the same 54 cells, all matched within
  1 px, identical classifications, and a DAB mean differing by **1.5e-5 OD** — four orders of
  magnitude below the 0.2 threshold. Equivalent for every decision the pipeline makes, but not
  bit-identical in every case, so `segmenter_device` is recorded in each summary for provenance.
  **Decision: QuPath is removable — the published figures survive the swap.**
- **Keystone — degradation** (`tests/test_degradation.py`, the End-to-End validation): CODEX
  same-section truth (CD8 vs PD-1) → split to pseudo-serial + inject registration error →
  verdict must not flip. Real truth `csr_only` stable under 1–3° / 3–8 px; engaged and
  independent regimes preserved. The **only** place true cross-marker association ground
  truth exists (CODEX ships as coordinates, not registrable images). The earlier
  **image-based** degradation experiment was removed (tissue-scale data, not cell-scale;
  §10) — to be redesigned on an appropriate dataset.

### 7.1 Registration benchmarked against VALIS — full ANHIR (2026-07-18)

Independent, **non-circular** head-to-head of OASIS registration (LoFTR→similarity, and the
structural `register_similarity` path) + the certification **gate** against **VALIS** (Virtual
Alignment of pathoLogy Image Series, Nat. Commun. 2023 — the open-source SOTA WSI registrar).
Harness: `validation/valis_bench/` (`common.py` shared scorer, `run_ours.py`, `run_valis.py`,
`run_correspondence.py`, `compare.py`, `run_all.sh`, `RESULTS.md`, `README.md`). **The main
pipeline is untouched** — VALIS runs in an isolated env, exactly like the DeepLIIF benchmark.

**Datasets & where to download**
- **ANHIR** (Automatic Non-rigid Histological Image Registration, ISBI 2019). Full public
  training set, **`dataset_medium` = scale-25pc**, **222 scorable training pairs**, **8 tissue
  types** (lung-lesion, lung-lobes, mammary-gland, COAD, gastric, breast, kidney, mice-kidney).
  Download from the CTU/CMP server by **HTTP basic auth**, guest `ANHIR-guest` / `isbi2019`
  (CC-BY-NC-SA): `http://ptak.felk.cvut.cz/Medical/dataset_ANHIR/images/dataset_medium.{csv,zip,
  z01..z05}` (split zip, ~11.8 GB) + `.../landmarks/dataset_medium.zip`. Recombine
  (`zip -s 0 dataset_medium.zip --out combined.zip`) and unzip images + landmarks into the same
  tree → `<SET>/scale-25pc/{STAIN.jpg, STAIN.csv}` (landmarks co-located, ImageJ `,X,Y`). Pairs
  and the target diagonal come from `dataset_medium.csv` (`status` = training/evaluation;
  **test-set landmarks are server-side, so only the 222 training pairs are scorable locally** —
  the same basis as VALIS's reported 230). Stored at `~/oasis_validation_datasets/ANHIR_medium/`.
  Real µm/px per tissue (from the challenge table, at 25pc): lung-lesion 0.70, lung-lobes 5.10,
  mammary 9.18, COAD 1.87, gastric 1.01, breast 1.01, kidney 1.01, mice-kidney 0.91.
- **CIMA** subset (lung + mammary only) is the openly-mirrored ANHIR fragment used for the earlier
  landmark checks (§7 "Registration"); `common.enumerate_pairs` still reads that split layout.
- **VALIS runtime**: isolated `~/valis_runtime/venv` (uv, Python 3.11, `valis-wsi` 1.2.0) + brew
  `vips`; run with `DYLD_LIBRARY_PATH=/opt/homebrew/lib`. Native `valis` cannot run on the repo's
  Python 3.14, hence isolation; the project `.venv` (pinned opencv/numpy) is never touched.

**Methodology (how circularity is avoided)**
1. **Landmarks are never given to any registration.** Both methods register from image pixels
   only; the expert landmarks are used *solely* to score rTRE (relative target registration error
   = ‖T(moving landmark) − fixed landmark‖ / fixed-image diagonal, the ANHIR metric). MMrTRE =
   median over pairs of the per-pair median.
2. **One shared scorer** (`common.rtre`) is imported *identically* by the main `.venv` and the
   isolated VALIS venv, so the metric cannot drift between methods. The identity (no-registration)
   baseline is always reported so neither method is credited for pre-alignment.
3. **VALIS-rigid (distance-preserving) is scored separately from VALIS-nonrigid.** Only rigid is
   apples-to-apples with OASIS's similarity and cross-K-safe; the non-rigid warp (`warp_xy_from_to
   (…, non_rigid=True)`) is the operation OASIS **forbids** for spatial-association stats
   (`serial_registration.assert_distance_preserving`, §6) and is reported only as an accuracy
   upper bound.
4. **Gate calibration is judged by landmarks the gate never saw.** The gate is fed the LoFTR
   correspondences, its verdict recorded, then the *independent* expert-landmark rTRE is tabulated
   per verdict — a genuine, non-circular test of whether "certified" means "actually accurate."
5. **Correspondence quality** (`common.correspondence_quality`) checks each LoFTR match against the
   ground-truth displacement predicted by a **local affine fit to the nearest expert landmarks** —
   isolating LoFTR error from real tissue deformation. Non-circular (landmarks are independent).
6. **Big-image handling** — at 25pc, COAD/breast/gastric reach 16k+ px, too large for whole-image
   LoFTR (OOMs above ~2000 px). Both images are downsampled to a 2000 px working frame for the
   global fit, then FULL-RES landmarks are warped through it (scale-in → fit → scale-out) so rTRE
   is measured at full resolution — mirroring VALIS, which also downsamples for its rigid step.
7. **Scope** — correspondence on **all 222 pairs**; accuracy on a **stratified 44** (7 per tissue,
   `common.stratified_pairs`) because the full ours+VALIS sweep is ~16 h on this 16 GB machine
   (runs must be sequential — LoFTR OOMs if a second heavy job shares RAM).

**Timing (per-pair wall time, same machine)** — this is the surprise and it matters:

| step | median | mean | range |
|---|---:|---:|---:|
| LoFTR only (correspondence, 2000 px) | 49 s | 51 s | 6–143 s |
| **VALIS rigid+non-rigid** | **29 s** | 34 s | 7–87 s |
| OURS LoFTR + structural (full ours path) | 139 s | 153 s | 52–336 s |

On the same 44 pairs **VALIS is ~1.75× faster than our LoFTR pass** (29 s vs 51 s) and ~4.7× faster
than the full LoFTR+structural pipeline — while also succeeding where LoFTR returns 0 matches.

**Results**
- *Accuracy (stratified 44):* identity MMrTRE 0.0522 → **VALIS-rigid 0.0037**, VALIS-nonrigid 0.0015,
  OASIS-LoFTR 0.0052 (**only 23/44 registered** — 0 matches on 21 cross-modal pairs),
  OASIS-structural 0.0052 median but **mean 0.058** (catastrophic on cross-modal / large-displacement
  pairs — several worse-than-identity). **Within OASIS's regime it ties VALIS-rigid** (better on
  14/23 pairs where LoFTR works).
- *LoFTR correspondence (all 222):* usable matches on **125/222 (56%)**, split by stain appearance —
  **lung-lesion 100%, lung-lobes 95%** (IHC↔IHC, similar), **mammary 0/38, breast 0/1, kidney 0/1**
  (cross-modal H&E↔IHC). LoFTR is reliable on similar stains and **fails outright on cross-modal**.
  OASIS's real use case (CD8 vs TIM-3, both brown DAB) is IHC↔IHC → in LoFTR's good regime, so LoFTR
  is validated *for what OASIS uses it for*.
- *Gate calibration:* **fails closed** — every pass verdict (LOCALLY_CERTIFIED 0.0045, RADIUS_LIMITED
  0.0016) has genuinely low error; it never certified a bad registration. Over-conservative (6/44
  certified), partly an artifact of whole-slide downsampling making the 5 µm LOO threshold sub-pixel.

**Conclusion & decision.** On the full diversity of ANHIR, **VALIS is the better *general* registrar**
— robust to cross-modal staining and large displacements that break both OASIS paths, more accurate,
*and faster*. **Within OASIS's regime (similar-stain serial sections) OASIS ties VALIS-rigid.** OASIS
stays a specialized serial-section CD8/TIM-3 tool with a fail-closed gate, not a general histology
registrar. **VALIS-rigid is worth adding as an invariant-safe option** — a drop-in *equivalent
alternative to LoFTR for different/cross-modal stains where LoFTR fails* (it is faster than LoFTR,
distance-preserving, and recovers the 44% of pairs LoFTR cannot). Its **non-rigid** warp remains
forbidden before any cross-K test. Full writeup + reproduce steps: `validation/valis_bench/RESULTS.md`.

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

---

## 11. Positivity thresholds — evidence and policy

The pipeline classifies a cell positive on a **fixed per-stain DAB OD cutoff** applied
across the whole cohort (CD8 0.20, TIM-3 0.10). This section records why, what was
measured on real slides, and what was rejected. Harness:
`legacy/nuclear_adaptive/threshold_audit_ll477.py` → `threshold_audit_ll477_results.json`
(parked alongside the feature it retired).

### 11.1 What the field does

There is **no automated gold standard**. The Digital Pathology Association white paper
(Aeffner *et al.*, *J Pathol Inform* 2019) states the operating rule plainly: *"Final
algorithm thresholds should be approved by a pathologist before data generation."* The
same paper names our failure mode — fixed thresholds across colour-variable datasets
give inconsistent results — and the field uses them anyway, because a threshold that
moves silently is considered worse than one that is wrong in a known direction.

QuPath, the reference implementation, prescribes no universal value: its guidance is a
visual-validation loop over the `Nucleus: DAB OD mean` histogram, with ~0.1 offered as a
permissive start and **0.2 as the conservative "few false positives, may miss true
positives" point**. Four families exist in practice: (a) manual fixed cutoff on
deconvolved OD — the de facto standard; (b) hue/intensity pixel thresholds (Aperio
Positive Pixel Count, ImageJ IHC Profiler, 88.6 % agreement with manual over n=1703);
(c) **control-derived cutoffs** — DAB-quant sets the threshold from negative-control
slides at a stated false-positive tolerance (0.1 % → NRMB 0.265; 1 % → 0.116), the only
approach giving a threshold a statistical rather than visual meaning; (d) trained
classifiers (QuPath random trees; DeepLIIF).

**Cohort-invariance, not fixedness, is the principle.** Comparing CD8 density between
patients requires every patient's number to mean the same thing. A fixed cutoff and a
cohort-wide trained classifier both satisfy this; a per-image adaptive cut does not.
Precedent: Visiopharm's HER2-CONNECT transfers connectivity cutoffs 0.12 / 0.56 across
different staining, a different scanner and different outlining **without re-tuning**,
then validates on the new cohort (κ 0.86–0.87 vs pathologists).

DeepLIIF was evaluated as a threshold-free alternative and **cannot be shipped**: its
licence is Apache 2.0 **with Commons Clause** (GitHub reports `NOASSERTION` — not
OSI-approved), which is incompatible with our MIT licence and with JOSS. It also does not
remove the constant, it relocates it — `PostProcessSegmentationMask.py` classifies on a
hard `seg_thresh=150` applied to a GAN-generated mask, which is *less* defensible to a
pathologist than an optical density. Its legitimate use here is as a **ground-truth
generator** (co-registered IHC + mpIF), which is how § 7 already uses it.

### 11.2 What was measured (8 real slides, 0.7519 µm/px, scale bars excluded)

| Image | Cells | pos @ fixed | Otsu | Ashman D | GMM @ D≥1.25 |
|---|---|---|---|---|---|
| CD8_10X_4 | 4521 | 2.6 % | **0.177** | 1.64 | 0.102 |
| CD8_x10_1 | 12943 | 1.8 % | **0.211** | 1.55 | 0.138 |
| CD8_x10_2 | 8930 | 1.1 % | 0.141 | 1.52 | 0.074 |
| CD8_x10_3 | 4632 | 2.4 % | **0.200** | 1.48 | 0.099 |
| Tim3_Liver_1 | 2917 | 0.0 % | 0.020 | 1.18 | abstain |
| Tim3_10X_3 | 5564 | 2.6 % | 0.069 | 1.33 | 0.051 |
| Tim3_x10_1 | 12224 | 0.5 % | 0.049 | 1.11 | abstain |
| Tim3_x10_2 | 7835 | 0.1 % | 0.016 | 1.27 | 0.002 |

**CD8 0.20 is corroborated** — Otsu independently returns 0.177 / 0.211 / 0.141 / 0.200,
three of four on the trusted value, and it coincides with QuPath's conservative point.
Two independent routes agreeing is as much support as a cutoff of this kind ever gets.

**TIM-3 0.10 is not corroborated on this cohort** (Otsu 0.016–0.069) and is provisionally
read as an inherited convention rather than a derived value. Open — see § 11.4.

**The adaptive GMM was unsafe as configured, and is now removed.** `nuclear_ashman_min`
defaulted to **1.25** in `run_pipeline.py` (the 2.0 in `nuclear_classify.py` was only the
library default). At 1.25 the cut fires on 6 of 8 images at roughly half the trusted value;
on Tim3_x10_2 it returns 0.0022 and calls **28.8 % positive** against 0.1 % at the fixed
cut. Production was unaffected only because `nuclear_adaptive` defaulted to `False` — a
single toggle away from a wrong cohort. The module and its harnesses are parked in
`legacy/nuclear_adaptive/`; nothing in `oasis/` imports them.

**The cause is statistical, not biological.** The membranous-marker hypothesis was tested
and does not hold: ring separability is barely above nuclear (CD8 1.51 vs 1.48; TIM-3 1.50
vs 1.33), and neither approaches 2.0. The real cause is that positives are rare (1–6 %),
so the distribution is unimodal-with-a-tail and Ashman's D — a two-mode statistic — has no
second mode to find. This is textbook: Otsu requires two clear peaks and **biases toward
the class with larger intra-class variance**; the triangle method and Rosin's unimodal
thresholding exist for background-dominated histograms; median + k·MAD is the recommended
rule for "small signals buried in noise" (Bankhead, *Introduction to Bioimage Analysis*).
Where GMMs *are* used for marker positivity, the cutoff is the **tail of the fitted
negative mode** (µ + 3σ, ≈ DAB-quant's false-positive tolerance in parametric form), not
the crossover of two comparable components. Our GMM uses the wrong cutoff rule for a rare
marker — which is why forcing it to fire lands ~2× low.

**Visually verified** (both checks confirm the measurement, not a bug): `Tim3_Liver_1` is
genuinely near-unstained (DAB p99 0.051, 0.2 % of pixels > 0.1), so zero positives is
correct; `CD8_x10_3` shows clean sparse membranous brown (DAB p99 0.326), so ~2 % is real.

### 11.3 Normalisation before deconvolution — rejected

Percentile-normalising the image and then deconvolving with fixed vectors is **materially
worse**: on `Tim3_x10_1` it calls **99.3 % of cells positive** (Otsu 0.815), the classic
counterstain cross-talk failure; `Tim3_Liver_1` goes to 78.7 %. Per-channel percentile
rescaling is effectively a white balance and destroys the optical-density relation that
deconvolution assumes. This is consistent with the literature: normalisation
(Macenko/Reinhard/Vahadane) exists to stabilise *deep-learning input*, and where it has
been tested on H-DAB extraction specifically, plain deconvolution with well-chosen vectors
won. Per-image Macenko (used by `cell_expansion`) does move CD8 toward 0.2 (0.074 → 0.163)
but **declined on 2 of 8 images**; its estimator failure must stay visible, not silent.

### 11.4 What ships

**Tier 1 (default, v1).** One fixed OD cutoff per stain, applied across the whole cohort.

The cutoff is set in a **review step between measurement and output** (`--stage segment`
→ review → `--stage finish`). Segmentation is expensive and the cutoff is cheap, so they
are separated: the run stops once cells are measured, the operator sets the cutoff against
the measured distribution, and **no overlay, dashboard or export is ever produced from an
unreviewed cutoff**. The histogram is shown beside the count deliberately — at 1–3 %
positives the image is a handful of brown cells among thousands, and the eye cannot
distinguish a cutoff on the shoulder of the negative peak from one past it. This is also
where the DPA's "thresholds approved by a pathologist before data generation" happens.
A membranous image is scored on calibrated ring completeness, so it is shown for context
with **no cutoff control and no slider-derived count** — deriving one from the nuclear
cutoff would put a confident wrong number next to the histogram. Retuning membrane cutoffs
belongs in Calibrate, where they are fitted against labelled cells.
(`oasis/quant/reclassify.py`, `tests/test_reclassify.py`.)

**Tier 2 (escape hatch).** A per-image override for a one-off faint or poor stain. It is
**recorded** — the image, the value, and the cohort default it replaced — and surfaced in
the report, because it is a deliberate exception to the cohort rule rather than a setting.
The caveat that goes with it: re-thresholding a faint slide until the positive count looks
right assumes the answer. NordiQC's data is that ~90 % of IHC failures are too-weak or
false-negative staining, so the first correct response to a faint slide is restain or
exclude; override is second. Fixed OD on faint tissue is *safer*, not more accurate — it
fails closed (under-calls) where an adaptive or trained rule fails open, which is exactly
what § 11.2 and § 11.3 measured.

**Tier 3 (planned).** A per-cohort trained classifier — design proposal in
`docs/classifier_tab_proposal.md`, nothing built. Two findings from writing it are worth
carrying here. First, `webui/calibration.py` already does most of the job but holds out one
**cell** at a time; cells within a slide share staining run, illumination and section
thickness, so leave-one-cell-out scores a model whose own slide is still in training and
systematically overstates generalisation — leave-one-**image**-out (with the per-fold
spread, not an average) has to replace it. Second, the membranous path is blocked on a
feature, not on a model: ring separability is barely above nuclear here (§ 11.2), so a
classifier trained on the current ring features would be a well-validated wrapper around
weak ones. Membrane **connectivity** — the longest contiguous positive arc as a fraction of
the ring, which is what HER2-CONNECT scores and cuts at 0.12/0.56 for κ 0.86 — is the
missing measurement, and it must be built and shown to lift separability *before* the
classifier is worth building.

### 11.5 Membrane contiguity — reading the ring as an arc

`membrane_pos_frac` counts *how many* ring pixels are stained. It cannot distinguish one
clean arc from the same number of pixels scattered around the ring, and real membrane
staining is contiguous where debris, a neighbour's membrane and counterstain bleed are not.
`ring_connectivity` (`cell_expansion.py`) bins the ring into angular sectors about the
centroid and returns the longest contiguous stained run, the number of separate runs, and
plain coverage. It reuses the identical pixels, threshold and DAB-dominance gate as
completeness, so the two differ **only** in whether angular order is taken into account.

This is the measurement the HER2 literature settled on rather than a better threshold:
Visiopharm's HER2-CONNECT skeletonises membrane fragments and scores connectivity 0–1,
cutting at 0.12 / 0.56 for κ 0.86–0.87 against pathologists; Aperio's Membrane algorithm
scores completeness alongside intensity.

**Measured discrimination** at equal completeness — one contiguous arc versus the same
pixel count scattered as speckle, 720 ring pixels / 72 sectors
(`tests/test_ring_connectivity.py`):

| coverage | arc | speckle | ratio |
|---|---|---|---|
| 0.10 | 0.097 | 0.000 | ∞ |
| 0.20 | 0.194 | 0.014 | 14× |
| 0.30 | 0.306 | 0.028 | 11× |
| 0.50 | 0.500 | 0.111 | 4.5× |

Discrimination is strongest at the low coverage real membranous staining actually occupies
and narrows toward 50 %, where each sector becomes a coin flip against the 0.5 sector
threshold. Two properties are pinned as tests rather than left to be rediscovered: an arc
crossing the ±π seam is **one** arc, and speckle finer than one sector is **smoothed into
apparent coverage** — a real limitation of sector binning, traded for robustness to
single-pixel noise. Empty sectors are excluded rather than counted as unstained, since a
thin or Voronoi-clipped ring has angular gaps with no pixels and absence of pixels is not
evidence of absent stain.

**Still unvalidated against labels.** The gate proposed in
`docs/classifier_tab_proposal.md` — does connectivity improve *classification* of real
membranous cells — cannot be run: the hand-labelled TIM-3 set is no longer on disk and the
HNSCC membrane set was never redistributable. What is established is that the feature
carries information completeness does not; whether that information improves a real call
is open.

### 11.6 Per-cohort classifier — what it is worth, measured

`oasis/quant/classifier.py`: ridge logistic regression in numpy over ~6–9 engineered
features, fitted to one cohort's own labelled cells. Persisted with its held-out report,
its coefficients and a fingerprint of its decision function.

**Validated against DeepLIIF's IF-derived labels** — 6,663 matched cells across 149 tiles,
18.1 % positive, ground truth from the co-registered immunofluorescence SegMask panel the
pipeline never sees (`validation/validate_cell_classifier.py`).

**Result 1 — on clean data the classifier does not beat the fixed cutoff.**

| rule | F1 | AUC |
|---|---|---|
| fixed 0.20 OD (shipped) | **0.781** | 0.930 |
| best possible single cutoff (0.30, hindsight only) | 0.793 | — |
| classifier, leave-one-image-out | 0.771 | 0.927 |

The shipped cutoff is within 0.012 F1 of the ceiling for *any* single-threshold rule, and
the classifier does not exceed that ceiling. On clean, well-stained, single-protocol
material the labelling burden buys nothing. This is why the fixed cutoff is tier 1.

Per-fold F1 spread is wide (sd 0.30, min 0.00) but that is **fold size, not slide failure**:
the median DeepLIIF tile carries 39 cells and 7 positives, 6 folds contain no positives at
all, and 11 of the 13 zero-F1 folds have ≤ 2 positives. The pooled figure is the reliable
one here. On real whole slides the same statistic would carry the meaning intended for it.

**Result 2 — the classifier earns its place exactly when staining varies.** Simulating
batch-to-batch variation as a per-image additive OD offset:

| per-image drift (OD) | fixed @0.20 | fixed @0.30 | fixed @0.40 | classifier |
|---|---|---|---|---|
| 0.00 | 0.781 | 0.793 | 0.770 | 0.771 |
| 0.05 | 0.777 | 0.785 | 0.763 | 0.766 |
| 0.10 | 0.715 | 0.776 | 0.759 | **0.766** |
| 0.20 | 0.537 | 0.628 | 0.716 | **0.765** |
| 0.30 | 0.484 | 0.525 | 0.571 | **0.765** |

The classifier is flat; every fixed cutoff collapses. **The crossover is at roughly
0.08 OD of per-image drift** — below it the cutoff wins on simplicity, above it the
classifier wins outright. Note the control: no fixed cutoff at *any* value survives, so
this is not the classifier merely sitting at a higher operating point (its effective cutoff
is ≈ 0.386 OD, and a fixed 0.40 still collapses to 0.571 at drift 0.30).

**The mechanism is not the feature we expected.** Ablating `dab_minus_local_bg` changes
almost nothing (0.766 → 0.762 at drift 0.10); ablating `dab_over_h` as well still leaves
0.762. The robustness comes from the *linear model over co-shifting channels*: when
`dab_mean`, `dab_p90` and `hema_mean` all move together, logistic regression can learn a
near-zero-sum combination that cancels the shared offset while keeping the within-cell
contrast. The fitted raw-unit weight sum over those three channels collapses from **3.52**
at no drift to **≈ −0.6** at drift ≥ 0.10 — the model is deliberately differencing the
channels. A one-channel threshold has no such option. Local background correction remains
justified on principle (a gradient *within* one section is not a per-image offset) but it is
not what produced this result, and the docstring should not claim otherwise.

**Leave-one-IMAGE-out, never leave-one-cell-out.** Cells within a slide share a staining
run, illumination and section thickness, so a cell-wise fold scores a model whose own slide
is still in training. `tests/test_classifier.py` demonstrates the inflation on constructed
data with a per-image offset. `webui/calibration.py` still uses leave-one-cell-out for the
membrane cutoff fit and should be migrated.

**Membrane classifier: built, not validated.** The feature contract includes
`membrane_connectivity` and `membrane_arc_count` (§ 11.5), but no labelled membranous set
is on disk — the hand-labelled TIM-3 export is gone and HNSCC was never redistributable.
The machinery is exercised by tests; whether it *classifies* real membranous cells better
than the calibrated completeness cutoff is untested. The tab therefore requires the user's
own leave-one-image-out report before applying anything, which is the correct answer for a
per-cohort tool: its validation is inherently produced per cohort.

### 11.7 What the Classifier tab does

`Classifier` tab → label cells on ≥ 3 slides (reusing the Calibrate canvas) → fit → the
**leave-one-image-out report is produced before anything can be saved**, with the per-fold
spread and the worst slides named. Gates, all enforced server-side: block below 3 labelled
slides (leave-one-image-out is undefined), warn below 5, block below 50 cells per class,
and refuse to apply cohort-wide below held-out AUC 0.75. The tab states up front that below
~20 slides the fixed cutoff plus the review step is the better tool, and states the § 11.6
measurement — that on consistent staining the classifier does not win.

A fitted model writes the **same `classification` property** the cutoff writes, so Quant,
Spatial and batch need no changes. Provenance travels with it: `positivity_method`, the
classifier name, its fingerprint, and how many cells fell near the decision boundary. The
results row names the method rather than assuming the reader was the person who chose it.

**The applicability gate had a real bug worth recording.** It built its band from the
min/max of individual training *cells* and compared an incoming image's *median* against
it. Cell values span far more than image medians do — one slide's `dab_mean` ran −0.005 to
1.384 — so the band admitted anything, and a slide shifted half an optical density into the
floor was accepted as "within training range". A safety gate that cannot fire is worse than
none, because it reads as a check that passed. The band is now built from **per-image
medians** (5th–95th percentile of cells when image identity is unknown), and the regression
is pinned in `tests/test_classifier.py`.

### 11.7b Nuclear vs membrane, measured (2026-07-28)

The membranous path rests on a compartment argument: a surface marker is not in the
nucleus, so measuring the nucleus reads the wrong thing. That argument had never been
tested, and it turns out to overstate the case badly.

**Harness:** `validation/nuclear_vs_membrane_tim3.py`, results in
`validation/nuclear_vs_membrane_tim3_RESULTS.md`. 599 hand-labelled cells (281 positive /
318 negative) across four CRC-ICM TIM-3 fields, held out **by image**. Single-number rules
get their cutoff from the training images and apply it to the held-out one, so every arm is
scored the same way.

| arm | AUC | F1 |
|---|---|---|
| 1. nuclear cutoff — *Membranous OFF* | 0.933 | 0.808 |
| 2. ring-mean cutoff | 0.757 | 0.701 |
| 3. completeness, **auto** pixel threshold | 0.876 | 0.790 |
| 4. nuclear classifier (6 features) | 0.925 | 0.848 |
| **5. membrane classifier (9 features)** | **0.948** | **0.881** |
| 6. completeness, threshold fitted on labels | 0.637 | 0.635 |

Per-image held-out F1 — `92290_IM` is the visibly faint slide:

| arm | 9212046_CT | **92290_IM** | 92625_CT | 92658_IM |
|---|---|---|---|---|
| 1. nuclear cutoff | 0.943 | 0.525 | 0.892 | 0.898 |
| 3. completeness (auto) | 0.920 | 0.458 | 0.904 | 0.889 |
| 5. membrane classifier | 0.960 | 0.667 | 0.941 | 0.900 |

**Four findings, and what each changed.**

1. **A nuclear cutoff on a membranous marker beats every single-number ring rule**, ring
   completeness included (0.933 vs 0.876). Bleed-through is the likely mechanism: strong
   membrane staining spills across the nuclear mask, so nuclear OD tracks membrane
   positivity by side effect. The UI claimed measuring a membrane marker in the nucleus
   "reads mostly background" — false on this data, and removed.

2. **The ring wins only as the fitted nine-feature combination**, and only just: +0.017,
   +0.049 and +0.002 F1 against the nuclear cutoff on the three well-stained slides, and
   +0.033 F1 / +0.023 AUC against the nuclear *classifier* pooled. So **hand-set membranous
   cutoffs were deleted from the UI entirely.** Offering them offered a rule measurably
   worse than the nuclear cutoff it replaced, dressed as the more careful choice. Membranous
   is now classifier-or-nothing; the nuclear cutoff below it is the fallback for refused
   slides. `_apply_cytoplasm_measurement` no longer applies `ring mean > threshold` when it
   has no cutoffs — it measures and leaves the nuclear classification standing.

3. **The self-calibrating pixel threshold is the better estimator, not a compromise.**
   Deriving "stained" per image from that image's own ring pixels (median + 3·MAD) reaches
   AUC 0.876; fitting it on labelled negatives and carrying it to another image — the
   original `tune_membrane_threshold.py` approach — reaches **0.637**. A threshold fitted on
   one slide's staining is a constant applied to a different slide's. Pinned in
   `tests/test_auto_pixel_threshold.py`, which also documents why a high percentile of
   pooled ring pixels is the wrong estimator: the stained pixels are in the distribution, so
   the threshold climbs with the positivity rate and a slide with more signal is called less
   positive (0.078 → 0.438 across 0–80 % positivity, against 0.086 → 0.110 for MAD).

4. **Nothing works on the faint slide.** Best arm F1 0.667; single-number rules 0.38–0.53.
   This is the contrast floor of § 11.5 restated on labels: where the stain does not clear
   the ring background, no rule separates the classes. It is why the classifier's
   applicability gate hands such slides back to the cutoff.

**Caveats.** Four images, one cohort, one scanner; the per-slide spread (F1 0.46–0.96) is
wider than every difference between arms. Labels are one person's calls on DAB morphology,
not an orthogonal truth such as immunofluorescence. The arms are comparable to each other
and not to the earlier `tune_membrane_threshold.py` figure, which used a different decision
rule and fold protocol.

### 11.8 Open — pending decision

Not yet settled, deliberately left unwritten rather than guessed:

1. **TIM-3 cutoff.** 0.10 is unsupported by this cohort's Otsu values. Provenance to be
   established before it is either kept or moved.
2. ~~**Cohort-wide trained classifier** as an opt-in tier above the fixed cutoff.~~
   **Settled** — shipped, gated on a leave-one-image-out report, and measured in § 11.7b.
   For membranous markers it is not an opt-in tier but the only option: § 11.7b removed the
   hand-set membranous cutoffs after measuring them below the nuclear cutoff they replaced.
3. **Replacing GMM valley selection with a noise-tail rule** (median + k·MAD, or
   triangle/Rosin) validated against the existing DeepLIIF-derived labels.
4. **Negative-control-derived thresholds** (DAB-quant pattern). The most defensible option
   for a methods paper; blocked on slides, not on code.
