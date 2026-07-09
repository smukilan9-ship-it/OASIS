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

Two entry points share one core and produce identical results.
- **CLI**: `run_pipeline.py --config cfg.yaml --mode {quant|spatial}`
- **Desktop UI**: `app.py` (pywebview) → `webui/api.py` + `webui/index.html`; the API
  writes a config and shells to `run_pipeline.py`.

| Module | Role |
|---|---|
| `run_pipeline.py` | Orchestrator: QuPath/InstanSeg segmentation, quant, spatial driver |
| `pixel_size_util.py` | µm/px from burned-in scale bar; per-image resolution |
| `cell_expansion.py` | Membrane markers: cytoplasmic-ring DAB + completeness cutoffs |
| `webui/calibration.py` | Fit per-marker membrane cutoffs from hand-labelled cells |
| `registration.py` | Thumbnail loading, hematoxylin deconvolution, SITK helpers |
| `serial_registration.py` | Serial registration, landmark certification, auto-propose, NGF |
| `spatial_stats.py` | Cross-type K/g/L, three nulls, DCLF test, cohort FDR |
| `overlay.py` | Segmentation / density / association figures |
| `file_matcher.py` | Pair matching by filename stain tokens |
| `restained_coexpression.py` | Separate same-section restained tab (not this flow) |

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
  grid-seed → RANSAC similarity → consistent matches → local-NCC snap + per-point
  confidence; coverage-first → ROI fallback. Pre-fills the canvas; operator verifies.
  Proposals are consistent *by construction* — human confirmation makes them valid; they
  never certify alone.
- **Production `landmark_register_and_verify`**: operator landmarks define a least-squares
  similarity; accuracy on **held-out** points (independent-annotator set if given, else
  leave-one-out). Four verdicts:
  - `CERTIFIED` — n ≥ 6, held-out TRE median ≤ 5 µm, fit-residual ≤ 5 µm
  - `LOCALLY_CERTIFIED` — only a subset passes; analyse that ROI (hull ≥ 10 %)
  - `DEFORMED` — landmarks exist but no similarity fits within tolerance
  - `NOT_CERTIFIABLE` — too few correspondences (NOT evidence sections are unrelated)
  A failed pair is reported, never warped. LOO is fit-unbiased but **single-annotator**.

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

**Dense-tissue status**: the shipped primary remains the 75 µm reweighted null, and
it is trusted only when the per-image architecture pre-flight says the tissue field is
coarser than the bandwidth. Dense fields (ℓ̂ around 35–45 µm in LL477) are fail-closed:
they may report CSR/co-infiltration, but not a robust cell-scale engagement claim. A
separate dense candidate was calibrated in `validation/validate_public_codex_dense_null.py`
on public Schürch CRC CODEX architecture templates: homogeneous CSR over-rejected true
nulls (≈10–25%); a total-cell morphology-conditioned candidate with a **10–30 µm**
DCLF band and **2 µm** support jitter controlled H0 at 3.7–6.7% with planted-positive
power 1.0. A rendered-pixel bridge (`validate_dense_null_image_derived_morphology.py`)
then recovered morphology from synthetic H-DAB-like hematoxylin pixels (median field
correlation 0.939) and kept H0 at 3.7–6.3% with power 1.0 for the same **10–30 µm /
2 µm** candidate. A real LL477 demonstration then ran the candidate on completed
certified H-DAB bundles using all reference-section OASIS detections as morphology
support: x10_1 p=0.007, x10_3 p=0.024, and sparse x10_2 was skipped (10 TIM-3 positives
inside the window). This is **not shipped** until the candidate is wired into production
with provenance, ROI handling, sparsity gates, and reviewer-facing wording.

**Robustness verdict** (never a single null's significance):
`robust` (reweighted-significant → cell-scale engagement) · `csr_only` (CSR-only →
co-infiltration) · `none` · `mixed`.

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
- **Dense-tissue null exploration** — smaller 35–45 µm reweighted bandwidths and
  square-tile conditioning were rejected. Public Schürch CRC CODEX calibration on real
  dense cell-coordinate architecture promoted one morphology-conditioned candidate
  (`10–30 µm`, total-cell field jitter `2 µm`). Rendered CODEX H-DAB-like pixels then
  showed image-derived nuclei morphology can recover the field and preserve calibration.
  A real LL477 demonstration ran on two usable certified CD8/TIM-3 pairs and skipped the
  sparse third pair. It is still not production because the candidate has not been wired
  into the app with gates/provenance/ROI handling.
- **Registration** — TRE vs **ANHIR/CIMA expert landmarks**; best certified real pair
  (lung-lesion Cc10↔proSPC) LOCALLY_CERTIFIED at 3.66 µm ROI. HyReCo blocked (233 GB+login).
  No public two-marker same-section DAB set exists.
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
