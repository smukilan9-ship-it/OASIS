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
confidence threshold is not yet a-priori-calibrated on H-DAB, and a local ROI is itself a
residual-based selection. (The desktop shell being undriveable by an agent is fixed:
`serve.py` serves the identical frontend and the identical validated API over HTTP.)

### 3.5a Does a certifying ROI hold up when you move the window? (2026-07-28)

A local certification's fit and its residual come from the same correspondences, so the
claim is not checkable from inside the ROI. It **is** checkable by moving the window:
deformation is smooth, so a genuine local alignment must certify across a patch of
overlapping windows whose independently-refitted transforms agree.

Neither prior sweep could see this, both by construction — the coarse tiling (`batch.py`)
skips any candidate within `1.8R` of a region it already kept, and the exhaustive search
`break`s on the first window that certifies. **Both therefore report "one region per pair"
for reasons that have nothing to do with the tissue**, and earlier readings of those counts
as evidence of isolation were wrong.

Measured (`validation/roi_certification_neighbourhood.py`, full grid, windows at `R/2` so
neighbours share most of their tissue, no early exit; 13 CD8↔TIM-3 pairs):

| group | certifying windows | with a certifying neighbour |
|---|---|---|
| A — found a path only under exhaustive search (8 pairs) | **9 / 289** (3 %) | **0 / 9** |
| B — certified on the first coarse tiling (5 pairs) | **97 / 242** (40 %) | **93 / 97** |

- **The certification test is not broken.** Where tissue genuinely aligns it says so
  broadly, and the contrast between the groups is not explained by how many windows were
  searched.
- **Where windows overlap, the fit reproduces itself.** Evaluated at the midpoint between
  two *overlapping* certifying windows — tissue both of them saw — four of five group-B
  pairs agree to **1.7–7.1 µm median**, at or inside the 5 µm cell-error budget.
- **Disagreement between DISTANT windows is the deformation field, not error.** It rises
  from 1.7–7.1 µm (overlapping) to 6.7–30.9 µm median and 130 µm at slide scale. Two
  locally-fitted transforms a slide apart are *supposed* to differ — that is the premise of
  certifying locally. A first version of this harness evaluated every transform at every
  certifying centre and reported 171–334 µm; those figures were the deformation field
  mislabelled as disagreement and are **retracted**.
- **One group-B pair is genuinely out of family**: `LL480_Junction_10X_2`, 2 of 40 windows,
  non-adjacent, overlapping-window disagreement **46.7 µm** against 1.7–7.1 for its peers.

**Isolation is not evidence of a bad fit, and must not become a gate.** A researcher who
finds the one well-preserved island on a deformed section and draws around it produces
group A's exact signature: one certifying window, failing neighbours — because the
neighbours reach into the deformed tissue they deliberately excluded. Nothing measured here
separates that from an underdetermined fit, so `n_with_a_neighbour` is a **diagnostic to
report, never a veto**. Gating on it would reject precisely the use the ROI workflow exists
for.

**Consequence for what may be claimed.** Certification behaves as designed and needs no new
gate before Spatial. The residual honest caveats are unchanged from § 3.5: the fit and its
residual still come from the same correspondences, and there are no independent landmarks
in this cohort, so `overlap_disagree` shows two windows agreeing — not that they are right.

Cohort bookkeeping: `LL477_x4_2_scale` is the same field as `LL477_Liver_4X_2` with the
scale bar burnt in, and reproduced its numbers exactly — the cohort is **75** independent
pairs, not 76, and the pipeline is confirmed deterministic on identical tissue.

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
- Certification reproduces itself where windows overlap (1.7–7.1 µm, § 3.5a), but the fit
  and its residual still come from the same correspondences and this cohort has no
  independent landmarks — so that number shows two windows *agreeing*, not being right.
  `LL480_Junction_10X_2` is an unexplained outlier at 46.7 µm.
- No cross-marker DAB ground truth for the targets → CD8/TIM-3 biological claim is
  underpowered (3 pairs, one cohort, nothing survives cohort FDR).
- Certs are single-annotator LOO, n=8 provisional; one annotator-independent number only.
- Segmentation recall ~0.75 non-randomly thins dense infiltrate → biases the pattern.
- Reweighted null anti-conservative — measured at size 0.24 on real tissue (§ 15.1), not
  "mild". The 75 µm architecture-scale is measured per image and now routes both
  `dense_tissue` AND `caution` away from it (§ 15.3); the morphology-conditioned null it
  routes to **is** validated on real H-DAB-derived morphology fields (size 0.02, and immune
  to a saturated marker), so that bullet is discharged.
- The two-band decomposition was not two claims until 2026-08-02 (§ 15.1): a contact-only
  truth fired the co-infiltration band at 0.98 and a regional-only truth was found in its own
  band at 0.06. Any two-band claim published from a run before that date is not supportable.
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

---

## 12. Session 2026-08-01 — scale-bar correction, a dead pipeline, and the FLE finding

Everything below was found by driving the shipped app on real data rather than by reading
code. Three of the four items were invisible to the 155-test suite at the time.

### 12.1 Scale bar — every LL477 distance was 10.5 % wrong

The bar detector Otsu-thresholded the bottom strip, accepted anything up to 25 % of the crop
HEIGHT as "line-like" (54 px on a 1440 px image) and took the WIDEST survivor, so a tissue
edge beat the drawn bar. True bar is 133 px in all six LL477 scale images; the detector
returned 133/139/156/165/174 (to **+31 %**) and batch took the median **0.6802 µm/px** as the
run default.

Replaced with a shape test — near-black (`<100`), solid (fill ≥ 0.80), thin in ABSOLUTE terms
(≤ 10 % of crop height), aspect ≥ 6, and the same length on every row (row-length CV ≤ 0.08).
Fails closed: no bar-shaped component returns `(None, None)` and the operator types the value.
All six images now read 133 px → **0.7519 µm/px, spread 0.00 %**.

**Consequence for the record.** 0.6802 → 0.7519 is **+10.5 %** on the multiplier for every
distance the pipeline reports: TRE, cell error, the 10–20 and 20–50 µm bands, the 75 µm
bandwidth, region areas. **Every LL477-derived number in this document predating this session
is quantitatively stale by ~10 %** and must be re-run before it is cited. External-dataset
results (ANHIR, CODEX, DeepLIIF, Keren, HNSCC) carry their own pixel sizes, never went
through this reader, and are unaffected.

Prior results are kept, not overwritten — the pre-fix numbers are the evidence that the fix
mattered.

### 12.2 The spatial and quant pipelines could not run at all

Rewriting the detector kept only `_detect_scale_bar` (returns a tuple) and dropped the public
`extract_pixel_size_from_scale_bar` wrapper. Both callers — `run_pipeline.resolve_pixel_size`
and the scale matcher in `webui.api` — import that name **inside a function body**, so nothing
raised until a run reached the line. Every spatial and quant run died on its first step with
`import failed: cannot import name extract_pixel_size_from_scale_bar`, while the entire test
suite stayed green, because every scale-bar test called the private function directly and no
test drives a pipeline.

Swept the tree afterwards: 128 distinct names across 22 first-party modules, all resolve. That
one wrapper was the only instance. `tests/test_imports_resolve.py` now parses every
`from oasis… import …` in the tree and checks each name against the module (0.8 s).

The local build's smoke test could not have caught it either: `build.sh` ran the frozen binary
with output discarded and `|| true`, then checked file existence. `packaging/smoke_test.py`,
which drives a real pipeline through the frozen binary, existed but was wired only into the
release workflow. Now wired into `build.sh`, where `set -euo pipefail` makes it fail the build.

### 12.3 Confidence ignored what the pipeline already knew

`compute_confidence` read only cell count and positivity %. An image the trained classifier
REFUSED as out of its trained range — which hands the calls back to the fixed cutoff and flags
`staining_quality: low` — was reported **NORMAL**. Observed on `LL477_CD8_x10_1`
(`area_px=33.36 outside [53, 64]`, 12,942 cells, 1.85 % positive). Both signals now force LOW.
A run that never requested a classifier is unaffected.

### 12.4 Certification: the FLE estimator, not the gate, is what fails good registrations

Reported separately in **`research/registration.md`** — the full investigation, the measurements
and the validation plan. The headline for this document:

On `LL477_CD8_x10_1 ↔ Tim3_x10_1`, an 810,000 µm² region certifies **LOCALLY_CERTIFIED at
3.1 µm** through the auto path and **RADIUS_LIMITED at 12.8 µm** through the drawn-region path
— identical polygon, identical 59 correspondences, identical fit (`fit_residual 4.144 µm`,
`landmark_noise 4.073 µm`). The only difference is `fle_fast`, which declares FLE = 0.7 µm
instead of measuring it (0.199 µm here). That changes `n_good` — the count of landmarks whose
residual is "consistent with localisation noise" — from **1 to 14**, which is enough to trigger
the sub-ROI rescue and certify a 32 % sub-window at 3.1 µm.

Measuring the residuals directly settles what the scatter actually is:

```
n = 59   median 4.14 µm   p75 5.32   p90 7.57   max 14.67
p90/median = 1.83        (isotropic Gaussian localisation noise = 1.90)
```

**The distribution is Rayleigh.** It is localisation noise, not deformation, and not an
outlier tail — which retires two earlier explanations for the same symptom (§ 3.5's "gross
errors set the residual p90", and the local-smoothness filter added after *84 real CD8/TIM-3
pairs certified zero regions*).

A 4.14 µm median is state of the art: ANHIR's best methods report median rTRE 0.19–0.38 % of
image diagonal, which on this 1804 µm diagonal is **3.4–6.9 µm**. The gate nevertheless fails
it, because it compares a **≤5 µm threshold against an upper confidence bound on a p90**:
4.14 → 7.57 (×1.83, inherent to Rayleigh) → 12.815 (×1.69, the bound). Passing therefore needs
a median residual ≤ **1.6 µm ≈ 2 px**, roughly 2–4× better than the best published automatic
histology registration.

**Root cause — proposed here, and REFUTED the same session; see § 12.4a.**
`σ_fit² = 2·FLE² + model²` assigns everything not explained by FLE to
deformation. `loftr_fle` re-localises under image noise, which measures *precision*, not
*accuracy* — this document already labels it "a conservative lower bound". A lower bound on
FLE is mechanically an upper bound on deformation. If LoFTR's true FLE on H-DAB is ≈2.9 µm
(≈3.9 px, entirely plausible for a dense matcher on tissue), then `2·FLE² ≈ σ_fit²` and the
deformation term is ≈ 0.

**Standing consequence: every certification number this tool has produced on H-DAB is an upper
bound of unknown tightness.** Neither `fle_fast=True` (0.7 µm) nor the measured path (0.199 µm)
is right; both under-state FLE and the fast one is merely less wrong, which is why it
reproduces § 3.5's recorded `LOCALLY_CERTIFIED, 67 % of field, p90 2.85 µm`. The FW bound was
calibrated on lung + mammary and, as § 3.5 already notes, **never on H-DAB**.

Nothing in the gate's design is loosened in response. The ≤5 µm target is correct for a
10–20 µm contact claim. What must change is measuring FLE against ground truth.

### 12.4a The FLE was measured, and the § 12.4 root cause is wrong

`validation/validate_loftr_fle_groundtruth.py` (`c83cfc0`) warps a real H-DAB field by a
transform we choose and matches it against itself, so the partner of every reference point is
known analytically and `q − W(p)` is the true localisation error. Controls pass: identity warp
median **0.095 µm**, resample direction verified against the pixels, dense-field inversion
converged to 7e-3 px.

**Measured FLE: 0.224 µm** — the median over the eleven rigid warps of thirteen, range
0.164–0.270 — at certification's
own working resolution, stable under restaining, defocus, illumination gradient and noise
(worst case ×1.10, no matches lost). The shipped `loftr_fle` reported **0.199 µm**. It was
accurate. Error does not grow with warp magnitude (Pearson r = −0.07, including the 30 µm
shift that aliases patch flow), there are no gross errors in any warp, and the residual is
Rayleigh (p90/median 1.84 against 1.82).

Explaining the 4.073 µm σ_fit would need FLE = **2.88 µm**, about **11×** the measured value —
and the sweep across working scales rules out the obvious escape. FLE is roughly constant at
0.12–0.20 *working pixels* (CV 0.09 in px vs 0.41 in µm), so it scales with `px_work`; at the
disputed ROI's coarser `px_work` of 1.82 µm/px that is ≈0.27 µm, not 2.88. Substituting the
measured value moves the deformation term from 4.0633 to 4.0606 µm — it explains **1 %** of the
residual variance. So `loftr_fle` measuring precision rather than accuracy is a true statement
about the method that does not matter here: on this tissue LoFTR's precision and its accuracy
are both sub-micron.

### 12.4b The residual is a real deformation field, and the gate was right

`validation/validate_residual_origin.py` reproduced the disputed certification exactly (n = 59,
`fit_residual` 4.144, `landmark_noise` 4.073) and then asked what the residual is.

**It is not the filter tolerance.** Sweeping `tol_um` 2 → 12, the median residual is flat at
~4 µm across 3 → 8 while n triples (43 → 124); `med/tol` falls 2.7×, where truncation would
hold it constant. (`tol_um = 2` does certify — by keeping 15 of 153 matches, i.e. by selecting
the most-agreeing subset. That is the circularity the matcher module exists to avoid, and it
is why the tolerance must never be tuned against the verdict.)

**It is a spatially continuous field.** The semivariogram of the 59 residual vectors rises from
γ = 11.6 at 100 µm separation to ~49 at 570 µm:

```
sill 31.889   nugget 0.668   nugget/sill 0.021     only 2 % of the variance is random
Moran's I   dx +0.5959   dy +0.3907   null −0.0172, permutation p ≤ 0.001
```

Both tests are biased *against* this conclusion — the residuals are post-fit, so the similarity
has already absorbed the global linear component — and it holds anyway.

**Variogram decomposition of σ_fit, on the real pair, with no synthetic warp anywhere:**

```
random (nugget)   -> FLE 0.409 µm      structured -> deformation 3.951 µm
```

0.409 µm is a third independent estimate of FLE, next to Phase A's 0.224 and the shipped 0.199.
`σ_fit² = 2·FLE² + deformation²` then gives deformation ≈ 3.95–4.06 µm — **exactly what the
gate computed.** The tissue genuinely deforms ~4 µm across a 1299 µm window; no similarity can
align it better; refusing a ≤5 µm cell-error claim over that window is correct. Note this holds
without § 12.4's stacking argument: the p90 residual *point estimate* is 7.57 µm, already past
5 µm before any confidence bound.

**§ 12.4's "standing consequence" paragraph is withdrawn.** The certification numbers are not
upper bounds of unknown tightness; the tightness is known, and the dominant term has been
independently verified to be real. What survives of § 12.4 is the narrower point that the
threshold is applied to an upper bound on a p90 and so is stricter than it reads — true, but
not what refused this ROI.

**Two consequences.** (i) The sub-ROI rescue is retro-justified: a spatially continuous field
means a smaller window carries less of it, so certifying a 32 % sub-window at 3.1 µm is sound
— though choosing that window by which landmarks had small residuals is not, and the UI must
stop reporting the drawn area beside the rescued window's error. (ii) § 3's reading of the
Rayleigh p90/median as proof of localisation noise was **wrong**: a smooth field sampled at
scattered points gives a Rayleigh-looking magnitude distribution while being strongly
autocorrelated. The marginal distribution and the spatial structure are different questions,
and only the second separates noise from deformation.

The useful next step is no longer calibration but a feature: derive the certifiable window size
from the field's spatial range, so the app answers "certifiable below ~N µm" instead of
"RADIUS_LIMITED". `registration.md` § 11.5.

### 12.5 § 8 contradicts the shipped behaviour — RESOLVED 2026-08-02

§ 8 states the CD8/TIM-3 claim is *"underpowered (3 pairs, one cohort, nothing survives
cohort FDR)"*, while this section recorded a full run reporting **2 of 2 significant after
Benjamini–Hochberg**. Both statements were true — of **different runs**, and the artifacts
settle which one counts:

| artifact | date | pixel size | cohort FDR |
|---|---|---|---|
| `~/Desktop/assets/new` | 2026-07-09 | **0.641 µm** | 2 tested, q = 0.026 / 0.038, **2 significant** |
| `~/Desktop/ihc_spatial_results` | 2026-07-20 | **0.7519 µm** | 1 tested, q = 0.088, **0 significant** |

The run behind "2 of 2" used the **pre-scale-bar pixel size** (§ 12.1). 0.641 against the
corrected 0.7519 is a 17 % linear error, so every radius in it is physically misplaced — the
10–50 µm DCLF band actually covered 11.7–58.6 µm of tissue. Its `interaction` block is also
`None`: the two-band decomposition did not exist yet, so its claim rests entirely on the
global test.

**§ 8 stands and this section's counter-claim is withdrawn.** The current, correctly-scaled
artifact is one ROI-restricted pair at q = 0.088, not significant. The "2 of 2" figure must
not be quoted again.

§ 8's other stale bullet — dense tissues *"still fail closed until the morphology-conditioned
candidate is validated on real H-DAB/hematoxylin morphology fields"* — **is now discharged**:
§ 15 validates that candidate on exactly such fields (OASIS's own LL477 detections plus Keren
TNBC), measuring size 0.02 and immunity to a saturated marker. § 8 has been updated.

### 12.6 Re-validation required

Must re-run (LL477-derived, at the old pixel size and the old certification):
registration/certification numbers, the § 3.5a neighbourhood sweep, the LL477 dense-null
demonstration, and the three-pair spatial results.

Unaffected: ANHIR (§ 7.1), CODEX, DeepLIIF, Keren, HNSCC.

Version rather than overwrite — `validation_reports/<name>/pre-scalebar-fix/`.

---

## 13. Session 2026-08-02 — the radius floor is over-strict, and the bands are not independent

Two findings, from one question: the Spatial Association tab exists to claim **cell-scale
engagement** — that two populations are close enough to plausibly overlap — and the constant
deciding whether it may make that claim had never been derived.

`_RADIUS_FLOOR_FACTOR = 3.0` is the tab's only statement of a required registration accuracy.
Contact scale is claimable iff `factor × TRE ≤ _COLOC_RMAX_UM` (20 µm), so 3.0 encodes a spec
of **TRE ≤ 6.67 µm**. Every matcher comparison and certification threshold in this repo has
been aimed at a target that constant defines, and § 12.4's measured LL477 residual
(median 4.14 µm, p90 7.57 µm) straddles it: the claim is *permitted at the median and withheld
at the p90*, on the constant rather than on the data.

`validate_radius_floor.py` had measured **size** and **power** on the single 10–50 µm band.
Neither addresses **localisation**, which is the only thing a floor is for.

### 13.1 The floor is not doing the job it was introduced for

`validation/validate_radius_floor_localisation.py`. Two annulus truths on a shared-architecture
substrate — CONTACT-ONLY (0–12 µm) and REGIONAL-ONLY (25–45 µm) — displaced under two models,
read through the production reweighted primary with the floor disabled.

The displacement model matters and the earlier harness had it wrong. registration.md § 11.4b
measured the real residual as **98 % spatially structured** (nugget/sill 0.021), so B is
displaced by a smooth field, not iid noise. Verified on the harness: neighbouring B points move
coherently (mean cos 0.96, 2.1 µm relative slip at 12 µm absolute) where iid gives cos ≈ 0 and
15.9 µm slip. A smooth field translates a neighbourhood of B *as a block* relative to A, which
is the mechanism that could relocate an excess between bands; iid merely blurs amplitude.

**REGIONAL-ONLY truth → rate of a false contact-scale engagement claim** (60 reps × 199 perms):

| model | ε=0 | ε=3 | ε=5 | ε=8 | ε=12 | ε=20 |
|---|---|---|---|---|---|---|
| smooth field | 0.00 | 0.02 | 0.02 | 0.07 | 0.08 | 0.07 |
| iid noise | 0.00 | 0.02 | 0.02 | 0.02 | 0.07 | 0.05 |

Never above the 0.10 tolerance, at any ε, under either model. The ε=0 control is 0.00 while the
co-infiltration band correctly detects the truth at 0.55, so the construction discriminates.

**Registration error does not manufacture a cell-scale claim.** Within the tested range 3.0 is
over-strict by at least 3×. It is an upper bound, not a point estimate: ε\* is **censored at the
sweep maximum of 20 µm**, and where leakage begins beyond that is unmeasured.

Consistent with the other direction — the CONTACT-ONLY truth is still found in the contact band
at 0.98 at ε=12 and 0.77 at ε=20, so large error does not blind the band either. Measured on a
strong truth (30 % recruitment); a weak one would degrade sooner.

### 13.2 The two-band decomposition is not two independent claims

The same sweep exposed a defect with nothing to do with registration. At **ε = 0**, with the
truth confined to 0–12 µm, the **co-infiltration band (20–50 µm) claims attraction at 0.97**.

Both bands are DCLF tests on the *same* L−r curve, and L derives from K, which is **cumulative**
— K(r) counts every pair closer than r, so an excess at 6 µm raises K(r) at every larger r.
Writing the surplus as a constant c, `L(r) − r = √(r² + c/π) − r ≈ c/(2πr)`: it decays but never
returns to zero. Measured on a contact-only truth at ε = 0:

| r (µm) | 6 | 10 | 16 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|
| obs L−r | 10.82 | 17.53 | 19.96 | 18.58 | 17.59 | 16.58 | 16.04 |
| null upper | 6.13 | 7.91 | 9.21 | 10.65 | 12.62 | 13.96 | 15.84 |
| outside? | YES | YES | YES | YES | YES | YES | YES |
| obs g(r) | 7.16 | 7.83 | 1.57 | 1.59 | 1.28 | 1.46 | 1.33 |

`g(r)`, the derivative of K, carries no such memory and collapses from 7.8 to ~1.3 beyond the
contact band — the density of the excess really is localised. L−r does not follow it.

So the decomposition is asymmetric in the wrong direction. A regional truth correctly stays out
of the contact band (§ 13.1, 0.00 at ε=0), but **a contact-scale truth is reported as regional
co-infiltration as well**, essentially always. § 15.6's promise — *"a pair can co-localize at
short range without regional co-infiltration"* — does not hold as implemented.

**Diagnosed, not fixed.** The indicated repair is to decompose on `g(r)` rather than L−r
(`_pcf_from_k` already computes it), but the DCLF machinery, its null envelope and its
calibration are all built on L−r, so that is a change to the primary statistic and needs its
own calibration first. Nothing was changed in `spatial_stats.py`.

### 13.3 What this means for registration and correspondences

Correspondences matter **only** through cell-error p90, and cell-error p90 matters **only**
relative to `_COLOC_RMAX_UM / factor`. That ratio is the registration specification, and it had
never been written down.

§ 12.4 already establishes that LL477's 4.14 µm median residual is at the published ceiling —
ANHIR's best methods give 3.4–6.9 µm on this image diagonal, and the old ≤5 µm gate needed
≤1.6 µm, "2–4× better than the best published automatic histology registration". So the matcher
work is pushing against a wall that the state of the art also hits, while § 13.1 says the floor
is withholding claims that the error does not actually corrupt. **The lever is the constant,
not the correspondence engine.**

Not yet done, and required before the factor is changed: the sweep is synthetic, ε\* is censored
at 20 µm, and § 13.2 means the co-infiltration verdict is unreliable regardless of the floor.



---

## 14. Session 2026-08-02b — the moving certification window is now the operator's

### 14.1 What the registration workflow actually is

Settled this session, and it narrows the tab rather than widening it:

> Draw a region. Place landmarks by hand, on both sections. The gate does the rest.

No proposed landmarks, no guided landmarks. **Proposing is circular** — asking a matcher to
suggest correspondences is the same automatic step that already failed, wearing a human's
signature; and at cohort scale a per-pair proposal pass is slow enough to be impractical.
The backends (`propose_landmarks`, `guide_landmark_candidates`, `suggest_moving_landmark`)
stay unreachable from the UI, and `legacy/webui_guided_landmarks.js` keeps the retired
implementation. `tests/test_ui_wiring.py::test_the_manual_landmark_path_offers_no_suggestions`
still guards this.

This is the answer to a question § 13 could not settle — though the evidence for it needed
correcting, and the correction changes the reason.

**"42 of 54 returned NO_MATCHES" was one pair, the worst one, on a control arm.** That figure
is `LL478_junction_10X_3` alone. Over all 14 pairs in
`roi_certification_neighbourhood_results.json`:

| arm | pairs | windows | certifying | NO_MATCHES | DEFORMED |
|---|---|---|---|---|---|
| `identity` (control — no provisional registration) | 12 | 503 | 24 % | 20 % | **56 %** |
| `register_similarity` (production) | 2 | 82 | 2 % | 38 % | **60 %** |

So the dominant failure is **DEFORMED, not NO_MATCHES** — LoFTR *did* find correspondences and
they *did* agree on a similarity; the error was simply too large. Manual landmarks therefore
help less than "no matches" implies for the majority case. What helps is what the ROI workflow
does: shrink the window until a similarity fits. Right conclusion, wrong reason.

Note also that only **2 of 14 pairs** ever ran the production path, so the production failure
profile is essentially unmeasured (§ 14.5).

And the promotion path is already in `serial_registration.py`:

```
accuracy_um <= 5.0 µm                              → CERTIFIED
else a spatial subset passes, hull ≥ 10% of field  → LOCALLY_CERTIFIED
else enough radius band remains                    → RADIUS_LIMITED
```

`_RADIUS_FLOOR_FACTOR` does **not** decide that — a point worth stating plainly, because
§ 13's calibration might otherwise be read as a route to promoting pairs. It is not.
LOCALLY_CERTIFIED needs a spatial subset of landmarks agreeing to within the gate over
≥ 10 % of the field, which is a person placing good landmarks in a good region. What § 13's
calibration would change is only what a RADIUS_LIMITED pair may *claim*, not its verdict.

### 14.2 The gap: a matched window nobody could move

`certSeedMovingRoi` copies the reference ROI onto the moving section, scaled by
`ref_px / mov_px` so it covers the same PHYSICAL area rather than the same pixel count
(250 px is 188 µm at 0.7519 but 160 µm at 0.641). Its comment claimed the operator "only
ever translates it, never resizes it" — but **there was no translate control and no resize
control**. The only way to change it was to redraw the polygon by hand, which destroys both
the shape and the matched physical area the seed exists to establish. The seed is also
centred on the moving image, which is almost never where the corresponding tissue sits.

### 14.3 What was built

Direct manipulation on the moving pane: grab anywhere inside the window to move it, grab a
corner handle to resize it. Scaling is **uniform about the centroid**, so the window keeps
its drawn shape — a per-vertex edit would let it be distorted into something that no longer
corresponds to the reference window at all. The collapse case is clamped (0.05×–20×) so a
grab that lands on the centroid cannot reduce the window to a point.

`lmDown` resolves three competing gestures and the order is the whole correctness: **active
draw tool → moving window → pan**. Claiming the window before the draw tool makes the
moving pane undrawable; claiming pan first makes the window immovable. Neither failure
raises anything, so it is pinned by a test.

The ROI badge now reports physical size: `both windows set · moving 2.00× reference`, or
`· matched` within 2 %. Resizing is allowed rather than prevented — only the operator can
see which tissue corresponds — but silently drifting from the reference window's area is
exactly what is worth showing.

**Why resizing cannot bias a verdict:** `movRoi` never reaches the transform or the
certification. Those come from the hand-placed landmark pairs and from the REFERENCE ROI,
which is what `certify_landmarks` is sent. A test asserts `movRoi` is never included in that
call, so the aid cannot be promoted to evidence by a later edit.

### 14.4 Verified

Driven in-browser: seeded window matches the reference area exactly; move shifts by the
exact delta with area unchanged; a corner drag to 2× distance gives exactly 2.000× linear
scale with the centroid fixed; the collapse clamp holds. Gesture priority checked on all
five paths — inside the window drags it, outside pans, an active draw tool wins over both,
LoFTR mode does not hijack the moving pane, and the reference pane still pans. No console
errors; suite green.

**Not done:** exercised with synthetic geometry, not on real slides — how well the seeded
window lands on corresponding tissue in practice is unmeasured. Nothing in the spatial
statistics was touched (§ 13.2 remains diagnosed and unfixed).

---

## 15. Session 2026-08-02c — the spatial statistic, audited on real tissue

§ 13 diagnosed the band decomposition and could not fix it, and left the radius floor
calibrated only as a censored upper bound on synthetic blobs. This session closes both,
finds three further defects on the way, and re-runs the one real result to see what changes.

The common thread: **every previous spatial calibration ran on a Gaussian-mixture
substrate whose structure sits at 180 µm — above the null's 75 µm bandwidth, so the null
absorbed it and everything looked easy.** Real tissue has structure below the bandwidth.
`validation/spatial_substrates.py` replaces it with real segmented cells (OASIS's own LL477
detections; Keren TNBC as an independent tissue and modality), imposing association by
**thinning** so every coordinate is a real cell — generating recruited points from a Gaussian
would have put synthetic coordinates back into a "real tissue" test.

### 15.1 The band decomposition was broken two ways, and is fixed

`validate_band_decomposition.py`, 50 draws × 199 permutations, three substrates:

| null | statistic | size | contact→coinfil leak | power |
|---|---|---|---|---|
| reweighted | **bands** *(shipped)* | 0.24 | **0.98** | 0.06 |
| dense_morphology | bands | 0.00 | **1.00** | 0.00 |
| dense_morphology | bands_pcf | 0.02 | 0.66 | 0.44 |
| dense_morphology | bands_annulus | 0.04 | 0.04 | 0.06 |
| **dense_morphology** | **bands_ring** | **0.02** | **0.06** | **0.38\*** |

\* the power column is the WORST cell across all substrates and both truths, and 0.38 is
the **synthetic** substrate's regional cell. On the two REAL substrates the same combination
gives contact 1.00 and regional 0.60–0.90 at the same 2.0× enrichment (§ 15.8). Quoting 0.38
for real tissue would understate the test.

The shipped statistic fails in *both* directions, and the second failure was not predicted
by § 13: a **regional-only truth is detected in its own band at 0.06**. Same cumulative
memory running backwards — a regional truth depletes contact scale, which drags L−r down
across 20–50 µm and cancels the real excess. So `bands` both over-fires upward and goes
blind to what it is supposed to find.

Three candidates were added, all computed from the same null draws. `bands_pcf` (DCLF on
g(r)) has no cumulative memory but `np.gradient`'s central difference smears across the band
boundary. `bands_annulus` (K(hi)−K(lo)) cancels everything below the band exactly, but a
truth filling only part of a wide band is cancelled by the depleted remainder. **`bands_ring`
— DCLF on the per-bin ring density — has none of the three problems** and is what
`_BAND_STATISTIC` now selects. `bands` stays in the payload and drives nothing.

**Anti-hallucination checks**, because "K is cumulative" is an explanation and an explanation
can be right about the symptom and wrong about the cause:

* the algebra predicts the contaminating excess falls as r⁻¹ — **fitted −0.98**;
* g(r) recomputed by direct annulus counting instead of differentiating K agrees at
  **r = 0.92**, so it is a property of the pattern, not of `_pcf_from_k`;
* independent bivariate Poisson recovers **K_AB(r) = πr² to 4 %**.

Two harness bugs were found and fixed first, both of which would have flattered the wrong
answer: the "both scales" truth ran at 1.8× enrichment while the single-band truths ran at
3.9× (a flat boost does not equalise enrichment when annuli differ in occupancy —
`solve_boost` inverts for it), and the contact truth sat mostly *below* the band under test.

**A claim withdrawn before it was made.** The reweighted null *is* anti-conservative on real
tissue (size 0.24). But the architecture pre-flight classifies all three substrates as
`dense_tissue` (scale 28–34 µm against a 150 µm `min_ok`), so production never selects it
there. The measurement **validates that routing** rather than exposing a defect.

### 15.2 One constant was two constants, pulling opposite ways

`_RADIUS_FLOOR_FACTOR` was doing two jobs: what radius may be **quoted**, and whether a pair
is analysed **at all** (in `_certify_fitzpatrick_west` the analysability side is the direct
`else` of `band_ok`, deciding DEFORMED vs RADIUS_LIMITED). They are now separate constants
and were calibrated separately, on real tissue with the corrected statistic, sweeping ε to
80 µm — the earlier sweep stopped at 20 and censored every answer.

| | evidence | ε\* | factor |
|---|---|---|---|
| interpretation floor | leakage 0.00–0.03 at **every** ε to 80 µm | ≥80 (censored) | ≤0.25 → **set 1.0** |
| analysability gate | power 1.00 → 0.68 (ε=10) → 0.28 (ε=15) → 0.10 (ε=30) | **10 µm** | **5.0** |

The floor is **not** set to 0.25. A reporting boundary should not rest on "no leakage
observed": you cannot resolve a separation below your own registration error, so 1.0 is the
resolution limit — 4× more conservative than the data requires, 3× looser than the value it
replaces.

**The gate TIGHTENS**, which contradicts the guess made earlier in the session that a lower
factor would rescue DEFORMED windows. On the real 10X windows it reclassifies **84 of 123
(68 %)** from RADIUS_LIMITED to DEFORMED. Those pairs were being analysed by a test with
~0.25 power, and a null result from an underpowered test reads as evidence of absence.
Meanwhile the loosened floor takes contact-scale claims from **8 % to 100 %** of the windows
that remain. Net: fewer pairs answered, but the ones answered can make the claim the tab
exists for.

Size is unaffected throughout (0.00–0.03 at every ε): registration error still cannot invent
a finding.

### 15.3 `caution` routed to the weak null, and saturation aimed at it

`valid = worst in ("ok", "caution")` sent marginal architecture to the reweighted primary
with a "treat with care" string. `validate_saturated_marker_null.py` measured what that
costs — A held at LL477's real 3 % CD8 rate, B swept to 95 % of all cells, always chosen
independently of A so every claim is a false positive:

| B fraction | 5 % | 20 % | 50 % | 80 % | 95 % |
|---|---|---|---|---|---|
| reweighted (Keren) | 0.00 | 0.33 | 0.67 | 0.92 | **1.00** |
| dense_morphology | 0.00 | 0.08 | 0.08 | 0.00 | **0.00** |

The levels where LL477 lands in `caution` are exactly where the reweighted null's false
**cell-scale-engagement** rate reaches 0.17–0.25. A warning string does not undo a 5×
inflated false-positive rate on the headline claim, so `caution` now routes to the dense
fallback (`valid = worst == "ok"`).

**The two failures compound, which is how this hid.** Saturating a marker makes its pattern
resemble the whole cell population, which *raises* the estimated architecture scale out of
`dense_tissue` and into `caution` — so the worst input is precisely the one routed to the
weaker null. Under the null production now selects, saturation costs **power** but does not
manufacture an engagement claim.

### 15.4 DEFORMED was unreachable

`band_ok` covers TRE up to `max_radius·(1−band_frac)/floor_factor` = 100·0.5/3 = **16.67 µm**,
and the DEFORMED arm below it additionally required TRE ≤ 15 — so the window [16.67, 15] was
**empty whenever `image_wh` was supplied**, i.e. on every production call. Measured: TRE
10.7 → RADIUS_LIMITED, TRE 16.8 → NOT_CERTIFIABLE, DEFORMED never issued.

Not cosmetic. NOT_CERTIFIABLE tells the operator *"the landmarks do not agree on a single
transform; insufficient correspondence"* — false for landmarks fitting one similarity to
16.8 µm, and it points them at adding correspondences when the problem is deformation, on
the manual path v1 ships with. DEFORMED is now the direct `else`, matching
`_certify_fitzpatrick_west`. Both verdicts block analysis identically, so the diagnosis
changes and nothing else. The gap widens as the floor factor drops, so it had to be closed
before § 15.2.

### 15.5 The one real result survives; a floor never arrived

`validate_ll477_reanalysis.py` reproduces the shipped LL477 run exactly — n_a = 38, n_b = 34,
window 243,200 µm², IoU 0.530 — before comparing anything. Two wrong reconstructions were
rejected by that guard first (the ROI is not the analysis window; and `transform_polygon`
silently returns the *untransformed* polygon when `scale_ref` is missing, which cost n_b
34 → 15).

**The co-infiltration claim survives.** All four statistics agree: `bands` p = 0.033 (what
shipped), `bands_pcf` 0.019, `bands_annulus` 0.005, `bands_ring` 0.034 (what ships now).
Colocalization stays "none" under all four. The recalibrated constants also leave the pair
analysable and contact-claimable (cell error 2.352 µm), so nothing about the one real result
is retroactively invalidated.

Found while checking it: the run recorded `floor_um: null` and
`contact_scale_resolved: false` for a pair registered to **2.352 µm**. `run_pipeline` reads
`min_interpretable_radius_um` off the certification, and the LoFTR-ROI path records
`cell_error_um` and stops — so None was passed through and the headline claim was silently
withheld from one of the best-registered pairs in the repo. The floor is a pure function of
the cell error and is now derived when the key is absent.

### 15.6 The failure-mode record, corrected

"**42 of 54 returned NO_MATCHES**" was one pair, the worst one, on the `identity` control
arm. Over all 14 pairs the dominant failure is **DEFORMED (56 %), not NO_MATCHES (20 %)** —
LoFTR did find correspondences and they did agree on a similarity. Manual landmarks
therefore help less than "no matches" implies for the majority case; what helps is what the
ROI workflow does, shrink the window until a similarity fits. Right conclusion, wrong reason.

### 15.7 What is still not done

* **DONE.** The production-arm re-run across all 14 pairs (598 windows) — only 2 of 14 had
  ever used it:

  | verdict | production arm | identity arm, re-scored |
  |---|---|---|
  | CERTIFIED | 16 (3 %) | 3 |
  | RADIUS_LIMITED | 44 (7 %) | 34 |
  | DEFORMED | 444 (74 %) | 466 |
  | NO_MATCHES | 94 (16 %) | 101 |
  | **analysable** | **60/598 (10 %)** | **37/503 (7 %)** |

  Scored under the same constants the production arm is **better** than the identity
  control, which is the expected direction and corrects an early single-pair read that
  suggested the opposite (`LL477_Liver_4X_1` alone gives 3 of 54, but `LL479_Liver_10X_2`
  gives 15 and `LL477_Liver_4X_2` gives 19). `LL478_junction_10X_3` — the pair behind the
  retired "42 of 54" figure — still returns 0 certifying and 39 NO_MATCHES, so that pair
  genuinely is the bad one. `LL477_x4_2_scale` yields no windows, as a scale-bar image
  should.

  **WINDOW COUNTS ARE NOT REGION COUNTS — corrected after a user spotted it.** The sweep
  grids probes at R/2 spacing with radius R, so adjacent probes overlap by 50 % and one
  certifiable area yields many "certified windows". LL479_Liver_10X_2's 7 CERTIFIED windows
  have a minimum pairwise centre separation of 158 px against a 316 px probe radius — they
  are seven samples of **one** region, which is exactly what the app displays. Clustering
  the windows by actual overlap:

  | | windows | distinct regions |
  |---|---|---|
  | CERTIFIED | 16 | **9** |
  | usable (CERT + RADIUS_LIMITED) | 60 | **13** |

  So the cohort holds roughly **one usable region per pair**, occasionally two, and two
  pairs with none — not the 60 the window count suggests. A high window count means the
  region is LARGE (LL477_Liver_4X_2: 19 windows, 1 region), not that there are many. The
  10 % window figure is a sampling density and must not be read as "10 % of the slide is
  usable".

  The headline remains: **DEFORMED dominates at 74 %**, so the limiting factor is tissue
  deformation against a similarity model, not correspondence-finding.

* Superseded note, kept for the record: only 2 of 14 pairs had ever used the production arm.
  The comparison against the old identity numbers needs care, because those predate the
  § 15.2 tightening. **Re-scoring the identity arm under the new constants** (its stored
  per-window `cell_p90` makes this exact, with no re-run needed) gives:

  | verdict | old factor 3.0 | new factor 5.0 |
  |---|---|---|
  | CERTIFIED | 3 | 3 |
  | RADIUS_LIMITED | 118 | 34 |
  | DEFORMED | 0 | 84 |
  | **analysable** | **121 (100 %)** | **37 (31 %)** |

  So the tightening, not the arm, explains most of any drop — that is the baseline the
  production arm must be read against.
* Power is characterised at **one** effect size (2.0× enrichment). A power curve versus
  effect size is what would let a reader judge "power 0.38" properly.
* ε\* for the interpretation floor is still **censored** at 80 µm.
* § 12.5's contradiction between § 8 and the shipped behaviour is still open, and now needs
  re-deciding against these results rather than the pre-§ 12.4 ones.
* Association is imposed, never observed: the substrates are real, the truths are not.

### 15.8 Power versus effect size

`validate_band_power_curve.py`. A single power number is uninterpretable: 0.38 is fine or
disqualifying depending entirely on how large the effect had to be to reach it. The shipped
combination (`dense_morphology` + `bands_ring`), swept over per-band enrichment:

| enrichment | 1.25 | 1.5 | 1.75 | 2.0 | 2.5 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|---|
| ll477 contact | 0.40 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ll477 regional | 0.00 | 0.07 | 0.60 | 0.93 | 1.00 | 1.00 | 1.00 |
| keren contact | 0.40 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| keren regional | 0.07 | 0.27 | 0.40 | 0.67 | 1.00 | 1.00 | 1.00 |

**80 % power at 1.42× enrichment for contact scale, 1.9–2.2× for regional.** The contact
band — the tab's headline claim — is the *more* sensitive of the two, which is the useful
direction.

The asymmetry is geometric, not a defect: the regional band (20–50 µm) is wide and the
regional truth (30–40 µm) fills only part of it, so the surrounding depleted radii dilute
the signal. A regional truth that filled its band would behave like the contact one.

**What this is for.** It is not a claim about what enrichment real CD8/TIM-3 tissue carries
— nothing here measures that. It is what makes a NULL result readable: "no association
detected" means something different at power 0.9 than at 0.3, and without this curve there
is no way to say which regime a run was in. Any reported absence should name the effect the
test could have found.

### 15.9 A certificate that claimed field the landmarks never spanned

Found while fixing § 15.4 and deliberately deferred, then investigated.

`landmark_register_and_verify` returned **CERTIFIED with `roi_polygon = None` — the whole
field — for any landmark set that fit well**, with no spatial-support requirement. A
similarity is mathematically determined by ≥2 distinct collinear points, so the fit is not
degenerate and the residuals are legitimately ~0. That is exactly what makes it dangerous:
**a residual can only measure the transform where the landmarks are.**

Demonstrated with a deformation chosen to be invisible to the landmark set — a vertical
scaling about the same line the landmarks sit on:

| scale k | verdict | fit residual | claimed TRE | **realized field TRE p90** |
|---|---|---|---|---|
| 1.02 | CERTIFIED | 0.0 µm | 0.0 µm | **11.7 µm** |
| 1.05 | CERTIFIED | 0.0 µm | 0.0 µm | **29.3 µm** |
| 1.20 | CERTIFIED | 0.0 µm | 0.0 µm | **117.0 µm** |

The certificate claims 0.0 µm while cells are displaced by up to 117 µm. Not optimistic —
**blind by construction**, because the landmarks lie on the deformation's invariant line.
This is the precise failure a fail-closed tool exists to prevent.

**The evidence was already in the repo and gated nothing.** `coverage_frac` is computed
(`_hull_area(ref)/field_area`), reads 0.0 for the collinear set, is displayed in the UI as
*"fraction of the fixed tissue field spatially supported by the landmark layout"*, and is
used only to rank guided candidates. There was also an inversion: **LOCALLY_CERTIFIED
required `roi_frac ≥ min_roi_frac` while field-wide CERTIFIED required nothing** — the
stronger claim carried the weaker support requirement.

**The fix needed no new constant and no new threshold**, because the sibling path already
did the right thing: `_certify_fitzpatrick_west` sets `roi_polygon = eval_roi` with the
comment *"the hull IS the certified window"*. Only the legacy LOO path extrapolated. It now
matches:

* `coverage_frac ≥ min_roi_frac` → **CERTIFIED over the landmark hull**, and the reason says
  so. Inside that hull the claimed error is honest — for the ±120 px layout the field-wide
  error is ~28 µm while the hull's own error is a few µm, which is what the certificate now
  reports.
* below it → **NOT_CERTIFIABLE**, not RADIUS_LIMITED. Falling through to the deformation
  cascade was wrong twice: it produced a whole-field verdict from landmarks spanning none of
  it, and printed *"held-out TRE is 0.0 µm, above the ≤5.0 µm gate"*, which is neither true
  nor the problem. The pair is not deformed; the evidence is too thin.

**Cost to real work: none.** Real ANHIR landmark sets cover 0.41–0.67 of the field
(median 0.57), far above the 0.10 floor, and LL477 certified through the FW path, which
already behaved this way.

**Still open:** RADIUS_LIMITED keeps the whole field by design, so it carries a milder form
of the same extrapolation — at coverage 0.29 it reports an 8.7 µm floor where the realized
field error is 28 µm. That is a weaker claim than CERTIFIED and was left in scope-discipline.

---

## 16. Session 2026-08-02d — the matcher: measured, then shopped

The operator's verdict after seeing three auto-found regions fail was *"loftr
correspondences officially suck"*. This section establishes how much of that is true, how
much is the certificate failing to notice, what the physical ceiling is regardless of
matcher, and what the literature actually offers. Nothing here is implemented yet; § 16.8 is
a plan and § 16.9 is the gate it has to pass first.

### 16.1 The certificate is blind to a spun fit, and the blindness scales with window size

The three production sweeps already in the repo (`validation/roi_production_arm_{4X,10X,20X}.json`,
`arm = register_similarity`) record the matrix of every window that reached a certifying
verdict. Nobody had ever looked at what those matrices *say*. Decomposing each into rotation
and isotropic scale:

| arm | window radius | windows | reached a verdict | physically implausible | rate | \|rot\| med / max | scale min..max | n_corr median |
|---|---|---|---|---|---|---|---|---|
| 4X | 600 µm | 162 | 26 | 0 | **0 %** | 0.1° / 3.1° | 0.986 .. 1.020 | 95 |
| 10X | 238 µm | 324 | 24 | 12 | **50 %** | 2.5° / 58.4° | 0.641 .. 1.245 | 30 |
| 20X | 139 µm | 112 | 10 | 10 | **100 %** | 14.0° / 32.3° | 0.537 .. 1.123 | 12 |

*implausible* = |rotation| > 5° or |scale − 1| > 0.05. These are serial sections from one
block, imaged on one scanner at one objective. A 32° rotation or a 0.54 scale is not tissue.

Every 20X window, itemised, with what the certificate claimed about it:

| pair | verdict | n | rotation | scale | **claimed cell error p90** |
|---|---|---|---|---|---|
| LL478_junction_20X_1 | CERTIFIED | 8 | 4.0° | 0.797 | 3.68 µm |
| LL478_junction_20X_1 | RADIUS_LIMITED | 12 | 14.4° | 0.537 | 8.20 µm |
| LL478_junction_20X_1 | RADIUS_LIMITED | 12 | 14.8° | 0.560 | 9.50 µm |
| LL478_junction_20X_2 | CERTIFIED | 11 | 19.2° | 0.678 | 4.32 µm |
| LL478_junction_20X_2 | RADIUS_LIMITED | 9 | 16.7° | 1.018 | 5.19 µm |
| LL480_Liver_20X_2 | RADIUS_LIMITED | 13 | 7.4° | 1.123 | 6.19 µm |
| LL480_Liver_20X_2 | RADIUS_LIMITED | 9 | 32.3° | 0.592 | 7.24 µm |
| LL480_Tumor_20X_3 | RADIUS_LIMITED | 13 | 11.8° | 0.717 | 9.13 µm |
| LL480_Tumor_20X_3 | CERTIFIED | 16 | 13.6° | 0.894 | 4.80 µm |
| LL480_Tumor_20X_3 | CERTIFIED | 9 | 5.4° | 1.051 | **2.32 µm** |

The last row is the clearest statement of the defect: a window certified at **2.32 µm**
whose transform shrinks the section by 5 % and rotates it by 5.4°.

**This is § 15.9 again with a different cause.** There the certificate was blind because the
landmarks were collinear — *"a residual can only measure the transform where the landmarks
are"*. Here the correspondences are neither collinear nor few enough to be degenerate; they
are **clustered and contaminated**. A similarity fitted to 9 clustered points with blunders
has a small residual *at those 9 points* and is unconstrained everywhere else. The residual
gate cannot see it, by construction, and no threshold on the residual ever will.

The monotone column is the finding: **the failure is a function of how much tissue the
window spans, not of the matcher's average quality.** At 600 µm the same matcher on the same
slides is flawless. The certification design — shrink the window until the error gate passes
— walks straight into the regime where its own evidence stops constraining it.

### 16.2 Three separate things were being called "LoFTR sucks"

1. **A new cross-check refusing fits that previously certified.** Regions that certified at
   06:10 now return DEFORMED. That is not new breakage becoming visible; it is old breakage
   becoming visible, and § 16.1 says the old certificates were the wrong ones.
2. **Genuinely poor correspondences on this cohort.** Real, and quantified in § 16.5 —
   13.6 % gross on LL477 against 1.1 % on HyReCo's CD8↔CD45.
3. **An estimator that lets a handful of blunders determine four parameters.** Addressed
   by constraining rotation and scale to the whole-field transform (measured: constrained
   residual 0.83–0.99 against free 0.89–1.20 on clustered correspondences with blunders,
   and it recovers the true translation 24.95 / −11.81 against a true 25 / −12).
   **Currently half-wired** — `M_local` is constrained but `landmark_register_and_verify`
   still fits its own unconstrained similarity to the same points, so the transform that is
   *reported* and the transform that sets the *verdict* are different matrices. On the test
   case the verdict's matrix still carries 19.8°.

Only (2) is a matcher problem. (1) and (3) are ours, and (3) is the one that turns a bad
correspondence set into a confident wrong number.

### 16.3 The ceiling nobody's matcher crosses

Before shopping, the honest bound. The field's own controlled comparison of **consecutive
versus re-stained** sections (Jansen *et al.*, *J. Med. Imaging* 10(6):067501, 2023;
HyReCo, 86 slide pairs, 5 404 manual landmarks, 4 µm cuts):

| section relationship | best achieved median landmark error |
|---|---|
| re-stained (same physical section, bleached and re-stained) | **0.9 µm** |
| consecutive (adjacent cuts) | **6.5 µm** |
| ANHIR consecutive | 24.1 µm |

and their limiting statement, which is about biology and not about software:

> an MTRE of 1.0 µm allows nucleus-level alignment, which is infeasible in serial sections
> where the same nucleus is often not present on the next slide

The two current best methods land in the same place from opposite directions. **RegWSI**,
winner of ACROBAT 2023 (Wodziński *et al.*, *Comput. Methods Programs Biomed.* 250:108187):
HyReCo re-stained **0.59 µm**, HyReCo consecutive **4.96 µm**. **CORE** (Nov 2025):
re-stained **0.41 ± 1.27 µm**, consecutive **4.35 ± 3.67 µm**.

**LL477/LL478/LL480 CD8↔TIM-3 are consecutive sections.** So the floor for this data is
~4.4–5.0 µm even with the world's best pipeline and a 24 GB GPU, and OASIS's own certified
windows land at 4.8–9.8 µm cell error. **We are within roughly 1–2× of the state of the art
on the same class of problem.** A matcher swap cannot move a 4.4 µm floor, and the contact
band this app tests (0–10 µm) sits *on* that floor. That is the single most important
paper-grade sentence in this section, and it reframes the target: the win available is not
lower error, it is **refusing the windows where the error is secretly enormous**.

### 16.4 What the field's best pipelines actually use

| pipeline | matcher / primitive | reported accuracy | notes |
|---|---|---|---|
| **RegWSI** (ACROBAT 2023 winner) | **SuperPoint + SuperGlue**, *not* fine-tuned, run **multi-scale and multi-angle**, keeping the rotation/scale with the most matches; then instance-optimisation with local NCC | ANHIR MMrTRE 0.0017; ACROBAT median 137 µm; HyReCo 4.96 / 0.59 µm | RTX 3090, <2 min at 8192² |
| **DeeperHistReg** | framework wrapping the above | "podium at ANHIR, won ACROBAT" | CC-BY-SA, pip + Docker, A6000 |
| **CORE** (2025) | **XFeat** for global refinement, then **nuclei centroids + shape-aware rigid + Coherent Point Drift** | ANHIR AArTRE 0.0034; consecutive 4.35 µm; Cyc-IF 1.76 µm | A100; coarse stage 12–14 s |
| **VALIS** (*Nat. Commun.* 2023) | VGG features + BRISK, rigid then non-rigid, plus micro-registration | state of the art on ANHIR at publication | the most widely deployed |
| **DFBR** / TIAToolbox | VGG16 features from three layers, then SimpleITK non-rigid | "comparable to the ANHIR winner" when swapped into its first two stages | |
| **ZeroReg3D** (2025) | zero-shot keypoint matching + optimisation-based affine/non-rigid | — | explicitly argues learned registration "suffers from limited generalizability" on consecutive histology |

Three things follow, and all three are actionable.

**(a) The challenge winner uses a detector-based sparse matcher, not a detector-free one.**
OASIS's own conclusion that "detector-based matching failed on this data" was reached with
SIFT+mutual-NN, then DISK+LightGlue. It was never tested with SuperPoint or ALIKED, which
are different detectors with far more permissive firing.

**(b) The winner's real trick is not the matcher, it is the search.** SuperPoint/SuperGlue
are not rotation- or scale-invariant, so RegWSI *brute-forces* over rotations and
resolutions and keeps the hypothesis with the most matches. OASIS does a single pass at a
single scale and a single orientation. Given § 16.1 shows our failures manifest as wrong
rotation and wrong scale, an explicit search over rotation and scale is the most directly
targeted fix in this entire section.

**(c) Both 2025 pipelines fall back to nuclei, not pixels, for the fine stage.** CORE and
the Warwick point-set paper (Jeyasangar *et al.*, MICCAI 2024) both register **nuclei
centroids** with CPD, because hematoxylin counterstain is present in *both* sections whereas
DAB is not. **OASIS already segments every nucleus in both sections.** This primitive is
sitting unused in the repo, and it is stain-invariant by construction — precisely the
property LoFTR lacks here.

### 16.5 The matcher measurements we already have, restated

From `validation/matchers_on_cohort_results.json` (LL477 CD8↔TIM-3, three pairs, 800 px
crops — what `certify_local_roi` actually runs) and `hyreco_field_blunders_results.json`:

| matcher | LL477 matches per pair | LL477 gross frac | ANHIR blunders | verdict |
|---|---|---|---|---|
| LoFTR (shipped) | 144 / 77 / 632 | 0.083 / 0.221 / 0.103 | 2.10 % | only one that works on cohort |
| DISK + LightGlue | 0 / 0 / 35 | — / — / 0.629 | **0.34 %** | dead on target tissue |
| DeDoDe + LightGlue | 633 / 130 / 3019 | 0.371 / 0.669 / 0.185 | — | plentiful and wrong |
| SIFT + LightGlue | 0 / 0 / 0 | — | — | dead |
| KeyNet + HardNet | 87 / 89 / 92 | **1.00 / 1.00 / 0.989** | — | totally wrong |

And the slide-quality control that stops this being a LoFTR indictment: on HyReCo's
professionally prepared **CD8↔CD45**, the same LoFTR returns **2 481–5 168 correspondences
at 1.3–5.5 % gross**; on HyReCo **CD8↔HE** it degrades to 152–1 106 at 1.3–53 %. Same
matcher, same stain class, same magnification — **an order of magnitude difference driven by
the slides, not the algorithm** (§ 12, and the commit *"HyReCo says the blunders are our
slides, not the matcher"*).

### 16.6 The candidates, with licences — the filter that eliminates the obvious answer

OASIS is MIT and headed for JOSS, so licence is a hard gate, not a footnote.

| candidate | venue | type | licence | CPU-viable | already installed? |
|---|---|---|---|---|---|
| **MatchAnything-ELoFTR** | arXiv 2501.07556 (2025) | semi-dense, cross-modality pre-trained | **Apache-2.0** (HF weights) | **yes — 16 M params, HF documents CPU** | no (needs `transformers`) |
| Efficient LoFTR | CVPR 2024 | semi-dense | **Apache-2.0** | 2.5× faster than LoFTR | no |
| XFeat | CVPR 2024 | sparse + semi-dense | **Apache-2.0** | **designed for CPU**, real-time VGA | **yes — `kornia.feature.XFeat`** |
| ALIKED (+ LightGlue) | IEEE TIM 2023 | sparse | **BSD-3-Clause** | light | **yes — `kornia.feature.ALIKED`** |
| DeDoDe v2 | 3DV 2024 | sparse | MIT (DINOv2 parts Apache-2.0) | moderate | **yes** (measured: 37–67 % gross here) |
| RoMa / RoMa v2 | CVPR 2024 / arXiv 2511.15706 | dense | **MIT** (DINOv2/v3 Apache-2.0) | no — 303 ms *on GPU* | no |
| MASt3R | ECCV 2024 | 3D-grounded dense | CC-BY-NC (check) | no | no |
| **SuperPoint + SuperGlue** | CVPR 2018/2020 | sparse | **non-commercial research only (Magic Leap)** | yes | **excluded** |

**The ACROBAT winner's exact recipe is unusable by OASIS.** SuperPoint and SuperGlue are
licensed for non-commercial academic research only, and that restriction propagates to
LightGlue *when run with SuperPoint weights*. This is not hypothetical fussiness — it is why
DeeperHistReg's own licence reads "CC-BY-SA **with exceptions for optional deep learning
models**". LightGlue with **ALIKED, DISK or SIFT** weights is unencumbered; with SuperPoint
it is not. Any future note recommending "just use SuperPoint like the winner" should be
answered with this row.

Also worth recording: **kornia 0.8.3, already a dependency, ships ALIKED, XFeat, DeDoDe,
LoFTR, LightGlue and LightGlueMatcher.** Two of the strongest permissively-licensed
candidates cost zero new dependencies.

### 16.7 The one candidate with direct, published histology evidence

**MatchAnything** (He *et al.*, arXiv 2501.07556) pre-trains detector-free matchers on
~800 M synthetically cross-modalised pairs, so the network learns to match *structure*
rather than appearance — which is exactly the CD8-vs-TIM-3 problem, where the shared signal
is hematoxylin morphology and the differing signal is chromogen.

It reports **ANHIR cross-stain histology** directly, against the challenge's own winners:

| method | Average-Average rTRE | Average-Median rTRE |
|---|---|---|
| Elastix | ~0.095 | ~0.078 |
| MEVIS (ANHIR 1st) | ~0.088 | ~0.072 |
| AGH (ANHIR 2nd) | ~0.087 | ~0.070 |
| DeepHistReg | ~0.082 | ~0.062 |
| **RoMa + MatchAnything** | **~0.061** | **~0.047** |
| **ELoFTR + MatchAnything** | **~0.058** | **~0.045** |

with a stated **55.3 % relative improvement for ELoFTR over its own base weights** on this
task, and single-weight generalisation across eight unseen cross-modal tasks. Inference cost
is unchanged from the base architecture (ELoFTR ~40 ms at 640×480 on GPU).

Practical route: the weights are on HuggingFace as `zju-community/matchanything_eloftr`
under **Apache-2.0**, 16 M parameters, and are wired into `transformers` as
`AutoModelForKeypointMatching` with post-processing that returns per-match confidence — HF's
own tutorial states the model runs on CPU. Confidence per correspondence is separately
useful to us: it is a filter the current pipeline does not have.

**Caveat recorded honestly:** the GitHub repository `zju3dv/MatchAnything` has **no LICENSE
file** (GitHub's API reports no detected licence); only the HuggingFace weight card states
Apache-2.0. Before adopting, that needs resolving — the weights are the part we would ship.

### 16.8 What to do, in priority order, with what each does and does not fix

1. **Finish wiring the constrained fit through certification.** The verdict and the reported
   transform must be the same matrix. Fixes: confident wrong certificates from spun local
   fits. Does not fix: bad correspondences.
2. **Reject a window whose transform disagrees with the field.** § 16.1 gives a directly
   calibrated gate — at 4X, where correspondences are plentiful, *nothing* exceeds 3.1° or
   2 % scale. That is an empirical tolerance, not an invented constant. Fixes: the entire
   20X table above. Costs: coverage, which is the correct trade for a fail-closed tool.
3. **Search over rotation and scale (the RegWSI recipe).** Cheapest large win, no new
   dependency, no licence question, and aimed exactly at how our failures present.
4. **Add MatchAnything-ELoFTR as a second matcher arm** and A/B it on the existing harness
   (`validate_matchers_on_cohort.py` already runs synthetic-warp + real-pair, which is the
   right two-test design and is what killed DISK).
5. **Nuclei-centroid point-set registration as an independent arm** (CORE / Warwick recipe),
   using OASIS's own segmentations. Stain-invariant by construction; also gives a *second
   opinion* transform, and two independent estimates that agree is far stronger evidence
   than one estimate with a small residual.
6. **MAGSAC++ (`cv2.USAC_MAGSAC`) in place of the hand-rolled robust fit** — marginalises
   over the inlier threshold instead of committing to one. Marginal on 9 points; do it last,
   if at all.

### 16.9 The gate this has to pass first

Standing rule: **any change to the spatial association statistics must be validated before
implementing.** Steps 1, 2, 3 and 6 touch registration, not the statistic, but step 2 changes
*which windows produce numbers at all*, so it changes what the paper can claim. It needs the
same treatment the null models got: a measured effect on ANHIR (where expert landmarks give
truth) plus the LL477 cohort, reported as coverage lost against blunders caught, before it
ships. Steps 4 and 5 are new arms and must be measured on both the synthetic-warp and
real-pair tests before either becomes a default.

### 16.10 What this section does not settle

* **No new matcher has been run.** Every number in § 16.6 and § 16.7 is from the literature
  or from vendor documentation. The only measurements on *our* tissue are the ones already
  in the repo, restated in § 16.5.
* **The RUBIK benchmark PDF exceeded the fetch limit**, so its comparison table is
  summarised second-hand and is not cited for any specific number.
* **ANHIR rTRE is relative to image diagonal**, so the MatchAnything table is not directly
  comparable in µm to the HyReCo numbers in § 16.3. Do not mix them in the paper.
* **"Mismatched" (arXiv 2408.16445) is a standing warning against exactly the reasoning in
  § 16.7**: it evaluated 20 matchers and found leaderboard rank a poor predictor
  out-of-domain — SP-LG-GIM drops from 0.560 mAA in-domain to 0.161 out-of-domain. A
  published ANHIR win is evidence that MatchAnything is worth *testing* on LL477. It is not
  evidence that it will work on LL477.
* **The ceiling in § 16.3 is not removable by any item in § 16.8.** Consecutive sections are
  ~4.4 µm at best. If the biology needs finer than that, the answer is re-stained sections,
  not a better matcher.
