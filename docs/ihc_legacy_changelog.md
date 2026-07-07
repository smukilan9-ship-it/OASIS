# IHC Analyzer — Technical Reference

A living technical document describing the entire current state of the IHC
Analyzer: a desktop + CLI tool for automated immunohistochemistry analysis of
H‑DAB brightfield tissue images. It has two original pipelines plus one gated
same-section validation workflow:

1. **Quantification** — per‑image DAB‑positive cell counting (the validated core).
2. **Spatial Association** — population‑level cross‑type spatial statistics
   between two markers on **registered serial sections** (e.g. CD8 vs TIM‑3).
3. **Restained same-section co-expression validation** — a separate, fail-closed
   workflow for same-cell analysis only when correspondence and marker thresholds
   are independently certified.

> **Scientific honesty (read this first).** The Spatial Association pipeline
> measures whether two cell *populations* occupy the same tissue microregions
> more than expected by chance. It does **not**, and cannot, establish
> single‑cell **co‑expression**. The codebase was deliberately renamed away from
> "coloc"/"coexpression" language for exactly this reason (see Changelog).

---

## 1. Why serial sections cannot establish co‑expression

CD8 and TIM‑3 are stained on **different physical sections** of the same block.
Three independent reasons make a single‑cell "this cell is CD8⁺ **and** TIM‑3⁺"
claim impossible from serial sections:

1. **Z‑gap.** Adjacent sections are different cells (typically 4–5 µm apart). A
   cell in section A is not the same cell in section B; at best a *neighbour* is.
2. **TIM‑3 is not CD8‑restricted.** TIM‑3 is expressed on exhausted CD8⁺ and
   CD4⁺ T cells, Tregs, NK cells, and myeloid/dendritic cells. A TIM‑3⁺ object
   near a CD8⁺ object need not be a T cell at all.
3. **Compartment mismatch.** CD8 is measured nuclear‑adjacent (membranous,
   validated nuclear proxy), TIM‑3 is membranous and must be measured in the
   cytoplasmic ring. The two signals live in different compartments.

The honest, defensible statistic is therefore **cross‑type spatial
association**: a population‑level point‑pattern analysis with no per‑cell pairing
claim. True single‑cell co‑expression requires **same-section** data (multiplex
imaging such as CyCIF/CODEX/mIF/Xenium, or a certified restained same-section
workflow). The Restained tab implements the software path for that stronger
endpoint, but it remains fail-closed unless shared cell coordinates and marker
thresholds are independently certified; it does not rescue any serial-section
CD8/TIM‑3 co-expression claim.

---

## 2. Architecture

```text
                       ┌─────────────────────────────┐
   Raw IHC images ───► │  QuPath (headless) + InstanSeg │  nucleus segmentation
                       └──────────────┬──────────────┘
                                      │  CSV / GeoJSON / summary JSON
                                      ▼
        ┌───────────────────────────────────────────────────────────┐
        │                       run_pipeline.py                       │
        │   config, groovy gen, QuPath subprocess, orchestration      │
        ├───────────────────────────┬───────────────────────────────┤
        │   QUANTIFICATION           │   SPATIAL ASSOCIATION           │
        │   run_pipeline()           │   run_spatial_association_pipeline()
        │                            │                                 │
        │   overlay.generate_overlay │   registration.compute_registration
        │   dashboard.generate_all   │   cell_expansion.measure_cytoplasm_dab
        │                            │   spatial.run_spatial_association
        │                            │     └─ spatial_stats: cross_k_function,
        │                            │        cross_k_all_nulls (reweighted primary + CSR diagnostic + DCLF), A∩B mask
        │                            │   overlay: 2× segmentation, density, plot
        └───────────────────────────┴───────────────────────────────┘
                                      ▲
                                      │  pywebview JS bridge
                       ┌──────────────┴──────────────┐
                       │   app.py + webui/api.py +    │
                       │   webui/index.html (desktop) │
                       └─────────────────────────────┘
```

### Module roles

| Module | Role |
|---|---|
| `app.py` | Desktop entry point; creates the pywebview window, binds `webui.api.API`. |
| `run_pipeline.py` | Orchestrator. Config loading, Groovy generation, QuPath subprocess, the two pipelines, cytoplasm post‑processing, pixel‑size resolution, result persistence. |
| `spatial.py` | Active `run_spatial_association()` (loads positive centroids, registers, runs Ripley's K). Also holds **deprecated/unused** legacy MNN functions (`match_layers`, `spatial_permutation_null`, `run_coloc`, `generate_qc_overlay`) kept only for history. |
| `spatial_stats.py` | The statistics: `cross_k_function`, `cross_k_all_nulls` (production default = reweighted‑primary + homogeneous‑CSR baseline + DCLF + robustness verdict; the legacy three‑null set is retired — §7/§15.3), `cross_k_null` (legacy single‑null, used by validation), intersection‑window helpers (`estimate_tissue_polygon`, `transform_polygon`, `intersection_window`, `filter_points_in_polygon`). |
| `cell_expansion.py` | Voronoi‑clipped cytoplasm‑ring DAB measurement for membrane markers; stores ring geometry. |
| `registration.py` | Rigid alignment cascade (SimpleITK MI → ORB/SIFT → phase → identity) on the hematoxylin channel; `transform_centroids`. |
| `overlay.py` | Quant cell‑boundary overlay; spatial segmentation overlays, consolidated density heatmap, association plot; shared `_draw_cell_boundaries`. |
| `file_matcher.py` | Pairs serial‑section images by filename (two‑folder / single‑folder); excludes `*_scale` calibration images. |
| `pixel_size_util.py` | Pixel‑size resolution chain incl. burned‑in scale‑bar detection. |
| `dashboard.py` | Quantification HTML dashboard + Excel workbook. |
| `webui/api.py` | pywebview ↔ Python bridge; launches pipelines, streams logs, loads results. |
| `webui/index.html` | Single‑file UI (Setup, Quantification, Spatial Association, Settings). |

---

## 3. Quantification pipeline

For each image (`run_pipeline()` → `run_single_image()`):

1. **Pixel size** resolved (Section 5).
2. A Groovy script is generated and run in **QuPath headless** with **InstanSeg
   `brightfield_nuclei`** for nucleus detection (`BRIGHTFIELD_H_DAB`, full‑image
   annotation, `tileDims` 512, configurable device `mps`/`cuda`/`cpu`).
3. Each detection's **DAB: Mean** OD is compared to a threshold; cells are
   classified Positive/Negative. Exports: `*_detections.csv`,
   `*_detections.geojson`, `*_summary.json`.
4. `overlay.generate_overlay` draws cell boundaries (positive **red**, negative
   **green**) on the original image.
5. `dashboard.generate_all_outputs` builds an HTML dashboard + Excel workbook.

**Per‑stain DAB thresholds.** The threshold is matched to the stain by filename
substring (`stain_thresholds`), with per‑image UI overrides taking priority:

| Stain | DAB OD threshold |
|---|---|
| CD8 | **0.20** |
| TIM‑3 | **0.10** |
| (fallback) `dab_threshold` | 0.20 |

> **CORRECTION (audit A2, 2026-06-21):** the "~90% agreement" figure is **NOT
> supported by any shipping artifact** — `validation/validate_segmentation.py` requires
> manual GeoJSON ground truth that is not in the repo, and no saved output produces a
> 90% number. Treat quantification agreement as **UNVERIFIED** until a data-backed
> F1/κ from an actual manual-count run exists. (The HNSCC nuclear-segmentation F1
> 0.776–0.847 in §21.3 is a *different* dataset/endpoint and does not back this claim.)

Current validation status: no saved shipping artifact currently supports a
single manual-count agreement percentage for the original quantification cohort.
The quantification path is reproducible and auditable, but its manual-count
accuracy must be reported as **unverified** until a traceable ground-truth run
produces a data-backed metric such as object F1, Cohen's κ, or count correlation.
Confidence is flagged LOW when `total < 50`, positivity `< 0.1%`, or `> 95%`.

> The quantification pipeline's behaviour is frozen — the spatial work below
> never alters it.

---

## 4. Experimental membrane / cytoplasm measurement (`cell_expansion.py`)

> **Future work only (2026-06-21).** The implementation remains in the codebase,
> but all membrane controls have been removed from the Quant and Spatial UIs.
> Production runs preserve the original QuPath classification. A TIM-3 trial changed
> 61 QuPath-positive cells (0.44%) into 4,433 positives (32.0%), demonstrating that
> the ring threshold was not calibrated and must not be used clinically or
> scientifically yet.

InstanSeg segments **nuclei**, and QuPath measures DAB **inside the nucleus** —
the wrong compartment for a **membranous** stain (CD8, TIM‑3), whose signal sits
in a ring *outside* the nucleus. `measure_cytoplasm_dab()` re‑measures DAB in the
**cytoplasmic ring** (expanded cell − nucleus) and reclassifies.

- **Voronoi clip (critical).** Each nucleus is expanded by `expansion_um`
  (default **2.0 µm**) then **clipped to its Voronoi cell** via exact
  perpendicular‑bisector half‑plane intersection against nearby nuclei
  (`scipy.cKDTree` neighbours + `shapely`). This guarantees an expanded cell can
  never cross the midline into a neighbour and steal its membrane DAB — essential
  in dense lymphocyte infiltrate, mirroring QuPath's `detectionsToCells`.
- **Exact QuPath H‑DAB stain vectors** (so recomputed OD matches QuPath's
  `DAB: Mean`): H = `[0.721, 0.646, 0.249]`, DAB = `[0.532, 0.656, 0.535]`,
  Residual = `[0.539, −0.750, 0.384]`; background `[255,255,254]`;
  `OD = −log10((I+1)/bg)`.
- **Outputs** written back into the GeoJSON: cytoplasm‑based classification,
  three DAB means (nucleus / cytoplasm / cell), and **`cyto_polygon`** — the
  Voronoi‑clipped expanded‑cell boundary, stored so the segmentation overlay can
  draw the actual membrane compartment that was measured.
- **Gating.** Inactive from the UI so the original QuPath classification is
  preserved. The dormant config/API path is retained only for controlled future
  validation using manually labelled membrane-positive and membrane-negative cells.

### 4.1 Membrane‑completeness classification (faint membranous markers, e.g. TIM‑3)

**Problem.** The ring **mean** (`cytoplasm_dab_mean`) is the wrong decision
statistic for a *faint* membranous stain. A membranous signal sits on a **thin
arc** of the membrane, not across the whole ring; averaging dilutes that arc by
the empty ring area. For strong CD8 the mean still clears threshold; for faint
TIM‑3 it collapses **below** threshold → real positives are lost. Lowering the
threshold cannot fix it: the mean has already merged "faint concentrated arc"
(real positive) and "diffuse low background" (true negative) into the same
number, so any threshold trades lost positives for false positives.

**Fix — classify on membrane *completeness*, not the ring mean** (the standard
membranous‑IHC paradigm, cf. HER2/PD‑L1). `measure_cytoplasm_dab` now also emits,
per cell:
- **`cytoplasm_dab_p90`** — 90th percentile of ring OD (the brightest arc;
  undiluted by the empty ring), affine‑calibrated to QuPath scale like the means.
- **`membrane_pos_frac`** — fraction of ring pixels whose **calibrated** OD
  exceeds a pixel‑level threshold `membrane_pix_thr`. Computed in raw space via
  the inverse of the parity affine, so no per‑pixel recalibration is needed.

A cell is positive iff `membrane_pos_frac ≥ membrane_frac_min` **and** (optional
guard) `cytoplasm_dab_p90 > membrane_p90_thr` (`run_pipeline._apply_cytoplasm_measurement`).
The classifier is **opt‑in**: it activates only when `membrane_pix_thr` **and**
`membrane_frac_min` are set in config; otherwise the legacy `mean > threshold`
path is unchanged. The existing parity gate (corr ≥ 0.90, slope > 0, MAE ≤ 0.015
OD vs QuPath nuclear `DAB: Mean`) still runs and keeps the OD *channel*
calibrated, so percentiles/fractions of it are meaningful.

**Per‑image deconvolution + DAB‑dominance gate (2026‑07 hardening).** The fixed
QuPath stain vectors and fixed white point mis‑deconvolve tone‑cast slides — on
CRC‑ICM TIM‑3 the counterstain bled into the DAB channel so the background DAB OD
itself sat at 0.23–0.47 and a flat 0.1 cut flagged ~99 % of tissue. Two changes
(both **default‑on**, params `estimate_stains` / `dab_dominance_gate`):
- **Per‑image stain vectors + white point** (`_estimate_stain_vectors`, Macenko on
  OD; `_estimate_background`, per‑image 99th‑pct white point) replace the fixed
  vectors when the estimate is non‑degenerate; a degenerate estimate falls back to
  the fixed vectors. *Gotcha fixed:* naive Macenko collapsed both stain vectors
  into one whenever the projected‑OD angles wrapped around ±π — corrected by
  orienting the SVD plane toward the data mean and sign‑flipping whole vectors
  (never per‑component `abs`, which folds opposite directions together).
- **DAB‑dominance gate**: a positive ring pixel must exceed `membrane_pix_thr`
  **and** be more DAB than hematoxylin (`DAB_OD > H_OD`), removing dark‑counterstain
  false positives at low OD. `_od_channels` now returns the H channel too;
  `keep_ring_values` also emits `ring_h_values` so `tune_membrane_threshold.py`
  applies the identical gate and fitted cutoffs transfer unchanged.

On the real CRC‑ICM TIM‑3 image whose nuclear path calls 100 % positive, this turned
the ring `membrane_pos_frac` from a saturated median **0.90** (unthresholdable) into
a real distribution (median **0.10**), and the parity calibration *improved*
(corr 1.000, MAE 0.0024). Cutoffs still require labels — the fix makes the signal
*separable*, tuning decides where to cut.

**Tuning (`validation/tune_membrane_threshold.py`).** The three cutoffs are
**fit to hand‑labelled cells**, never guessed. Workflow: label ~50+ TIM‑3 cells
per class in QuPath (**including faint/borderline cases**), export GeoJSON, then
the harness (1) re‑measures the ring keeping per‑pixel ring OD, (2) anchors
`membrane_pix_thr` to the 99th percentile of **negative**‑cell ring pixels
(= "brighter than background"), (3) builds ROC curves for `membrane_pos_frac` and
`p90` vs the labels and reports **AUC** plus operating points at max‑Youden and at
a target sensitivity (default 0.95 — losing positives is the failure of record).
**AUC is the go/no‑go**: high AUC + a clear gap → data‑backed cutoffs; AUC ≈ 0.5 →
no cutoff exists and the signal is too faint to call (a finding, not a tuning
failure). Validate on held‑out cells before production use — the data‑backed
F1/κ the §3/§9a audit notes flag as missing.

> **Mechanism check (synthetic).** On a constructed cohort where every positive
> carries a faint partial DAB arc (positive‑cell ring means ≈ 0.072, all **below**
> the TIM‑3 0.10 ring threshold), the legacy ring‑mean classifier scored **0/50
> sensitivity** (every positive lost), while the completeness classifier scored
> **50/50 sensitivity, 50/50 specificity** (AUC 1.000). This confirms the code
> path and the statistic; real‑data separation will be lower and is what the
> labelled‑cell tuning measures.

**Real‑data validation (2026‑07).** The method is validated three ways:
- **TIM‑3 (the target marker).** 281 positive / 318 negative cells hand‑labelled across
  4 CRC‑ICM images (browser tool `validation/make_tim3_label_tool.py`). Leave‑one‑image‑out
  CV with a **fixed** `membrane_pix_thr 0.30` / `membrane_frac_min 0.14`: held‑out **F1 0.93**
  (precision 0.90, recall 0.97) on the three adequately‑stained slides, **0.80 pooled** incl.
  the faint 92290_IM. `membrane_pos_frac` (with the DAB>H gate) beat `cytoplasm_dab_p90`
  (F1 0.77), validating the gate. Cutoffs in `validation/membrane_cutoffs.yaml`.
- **CD8 (membranous, IF ground truth).** Per‑cell held‑out **F1 0.76 / AUC 0.89** on 18 k CD8
  cells (local HNSCC‑mIF‑mIHC set; `validation/validate_membrane_cd8_hnscc.py`). Chromogen is
  AEC not DAB (no membranous‑DAB+IF set exists — DAB is un‑strippable), so this validates the
  membranous *method*.
- **Ki67 (nuclear, IF ground truth).** Pipeline‑level **F1 0.81** on 41 k DeepLIIF cells
  (`validation/deepliif_pipeline_validation.py`).

**Callability is gated by staining quality** (see the membrane‑quality gate, §19.4): faint
tissue where the ring background approaches the pixel threshold over‑calls (92290_IM held‑out
F1 0.30) and is flagged low‑confidence rather than reported.

---

## 5. Pixel‑size resolution (`pixel_size_util.py`)

Resolved per image, highest priority first:

1. **Per‑image override** (`pixel_overrides`, set by the UI / manual entry).
2. **TIFF/OME metadata** (`PhysicalSizeX`, SVS `MPP`, TIFF XResolution).
3. **UI session default** (`default_pixel_size` when `_pixel_size_from_ui`).
4. **Filename parsing** (`x10`, `20x`, … → standard table: 4×=2.5, 10×=1.0,
   20×=0.5, 40×=0.25, 60×=0.165, 100×=0.10 µm/px).
5. **Interactive prompt** (expert mode only).
6. **Fallback** 0.5 µm/px.

**Scale‑bar extraction.** `_detect_scale_bar()` reads a **burned‑in 100 µm scale
bar** in the bottom 15% strip: Otsu threshold → horizontal morphological open →
connected components → longest horizontal segment; confident when the bar is
`>50 px` and `<40%` of image width; `pixel_size = 100 / bar_length_px`. Images
whose stem ends `_scale` are calibration‑only and are **never** analysed
(`is_scaled_image`). The Spatial Association tab resolves one session value up
front (`resolve_pixel_size`: manual > scale image > session > scale‑bar‑in‑image
> 0.5) and injects per‑image `pixel_overrides`.

---

## 6. Spatial Association pipeline

Driver: `run_spatial_association_pipeline()` (CLI `--mode spatial`; `--mode coloc`
is a deprecated alias). Pairs come from `file_matcher` (UI) or `spatial_pairs`
in config.

Per pair:

1. **Segment both images** through the quantification path (per‑stain thresholds;
   TIM‑3 gets cytoplasm measurement).
2. **Register** TIM‑3 → CD8 (`registration.compute_registration`) on the
   **hematoxylin** channel (shared structural signal across stains). Cascade:
   **SimpleITK Mattes mutual information (rigid)** → **ORB/SIFT + RANSAC** →
   **phase correlation** → **identity**, each sanity‑gated. `transform_centroids`
   maps TIM‑3⁺ centroids into CD8 image space.
3. **A∩B intersection tissue window** (`estimate_tissue_polygon` ×2 +
   `transform_polygon` + `intersection_window`): Otsu tissue masks are estimated
   for **both** the CD8 and TIM‑3 images (downsampled grayscale → Otsu (tissue
   darker) → morphology), **preserving internal holes/lumens** (`RETR_CCOMP`; holes
   are *not* filled — an empty lumen is not tissue). B's mask is transformed into
   CD8 space with the registration transform, and the analysis window is the
   **intersection** `A_tissue ∩ B_tissue`. This single window bounds the K‑function
   area normalization, the observed points (cells outside the intersection are
   excluded and the count logged), **and** all null sampling. Falls back to A‑only
   Otsu, then the points' bounding box, with a logged warning. *Why intersection:*
   regions present in only one section (folds, tears, missing tissue) carry no
   cross‑section information — measuring A against B's absence there is an artifact.
4. **Cross‑type Ripley's K** under the calibrated reweighted‑primary + homogeneous‑CSR
   baseline nulls (`spatial.run_spatial_association` → `spatial_stats.cross_k_all_nulls`),
   Section 7. (The earlier three‑null design is retired — §15.3.)
5. **Registration QC gate** (fail‑closed): objective residual/overlap metrics
   decide whether the statistics are trustworthy; invalid pairs are flagged, not
   hidden (Section 7).
6. **Four output figures** (Section 8), persisted to
   `spatial_association_results.json` and per‑pair `{sample}_spatial_association.json`.

### Result schema (per pair, in JSON)

```jsonc
{
  "sample_id": "...", "stain_a": "CD8", "stain_b": "TIM3",
  "registration_method": "simpleitk", "pixel_size_ref_um": 0.5,
  "tissue_area_um2": 1234567.0, "tissue_mask_method": "otsu_intersection",
  "intersection_overlap_iou": 0.78,
  "registration_qc": { "valid": true, "status": "valid", "reason": "...",
                       "residual_error_um": 1.4, "tissue_overlap_fraction": 0.83 },
  "statistics_valid": true,
  "spatial_association": {
    "per_marker": { "CD8": {"positive": N}, "TIM3": {"positive": M} },
    "tissue_area_um2": ..., "tissue_mask_method": "otsu_intersection",
    "association": {
      "CD8__TIM3": {                       // neutral key — NOT "CD8+TIM3+"
        "radii_um": [...], "K_observed": [...], "g_observed": [...], "L_minus_r": [...],
        "n_a": N, "n_b": M, "n_a_excluded": 3, "n_b_excluded": 5,
        "primary_null": "reweighted",
        // top-level null_*/global/p_values mirror the PRIMARY reweighted null
        "null_lower_L": [...], "null_upper_L": [...], "global": { ...DCLF... },
        "nulls": {
          "reweighted":    { "bandwidth_um": 75.0, "method": "reweighted_kinhom",
                             "global": { "significant": true,
                                         "direction": "association", ... } },
          "homogeneous":   { "null_lower_L": [...], "p_values": [...],
                             "global": { "significant": false, "global_p_dclf": 0.31,
                                         "direction": "none", ... } }
        },
        "robustness": {
          "verdict": "robust",            // robust | csr_only | none | mixed/legacy
          "direction": "association",
          "summary": "Significant association under the calibrated reweighted inhomogeneous cross-K ...",
          "per_null_significant": { "reweighted": true, "homogeneous": false }
        },
        "seg_a": "...", "seg_b": "...", "consolidated": "...", "association_plot": "..."
      }
    }
  }
}
```

---

## 7. The statistics (`spatial_stats.py`)

All point inputs are in CD8 image‑space pixels; radii are in pixels internally
and converted to microns for output.

### Cross‑type Ripley's K
`K_ab(r) = (A / (N·M)) · Σ_{i,j} 1[dist(aᵢ, bⱼ) ≤ r]`, where `A` is tissue area,
`N`,`M` the population sizes. Pair counting uses `scipy.cKDTree.count_neighbors`
(all radii in one pass). The estimator is **uncorrected** (no edge correction);
this is acceptable because the null uses the *same* estimator and window, so the
edge bias cancels in significance testing.

### Pair correlation function
`g_ab(r) = (1 / (2πr)) · dK/dr` (finite difference). Dimensionless; under spatial
independence `g = 1`. `g(0)` is undefined → emitted as `null` in JSON.

### L‑function (the association curve)
`L_ab(r) = √(K_ab(r)/π)`; the plotted curve is **`L_ab(r) − r`**. Under
independence it is 0; **positive ⇒ association** (co‑clustering), **negative ⇒
segregation**.

### Null models (`cross_k_all_nulls`)
> **CORRECTION (audit A3, 2026-06-21):** the "three null models" design described in
> this section and in the §6 result schema is **RETIRED** (it was anti-conservative
> under shared tissue preference). **Production default is
> `nulls=("reweighted","homogeneous")`** (`spatial_stats.py`): the size-controlled
> **intensity-reweighted inhomogeneous cross-K is the PRIMARY** and drives the verdict;
> **homogeneous CSR is a diagnostic baseline** that flags shared-preference artifacts
> (`csr_only`). The homogeneous+inhomogeneous(resampling)+toroidal trio below is kept
> only for the calibration scripts and **never gates** a result. See §15.3 for the
> retirement, §15.4 for the calibration. The text below is retained for history.

Production runs use two reported null components, not the retired three-null vote:

1. **Intensity-reweighted inhomogeneous cross-K (`reweighted`) — PRIMARY.** A and B
   intensities are estimated with leave-one-out Gaussian kernels, and each
   A-B pair is weighted by `1/(λ_A(a_i) · λ_B(b_j))`. Under independence after
   accounting for shared tissue architecture, the expected curve returns to
   πr². The global DCLF p-value from this reweighted curve drives the headline
   verdict. The calibrated bandwidth is **75 µm** (§15.4), chosen to sit above the
   10-50 µm interaction band so broad tissue architecture is controlled without
   absorbing short-range cell-cell association.
2. **Homogeneous CSR (`homogeneous`) — diagnostic baseline only.** B is sampled
   uniformly inside the A∩B tissue window. This intentionally weak null is shown
   because it reveals the classic failure mode: if CSR is significant but the
   reweighted primary is not, the result is labelled `csr_only`, meaning the
   apparent association is likely explained by shared tissue preference rather
   than a defensible cross-type interaction.

The earlier **homogeneous + resampling-Kinhom + toroidal** design is retained in
the code only for calibration and historical diagnostics. It is not the production
gate because §15 shows that the resampling Kinhom and toroidal shift were
anti-conservative under a realistic shared-preference null. Those retired nulls
remain scientifically useful as a record of what failed and why, but they no
longer justify a biological claim.

**Why not random labeling.** The textbook one‑image null permutes type labels among
the pooled points. It is **invalid here**: CD8 and TIM‑3 come from two *different
physical serial sections*, so a label is inseparable from which section the cell
was segmented in — swapping labels invents cells that were never observed and
destroys the per‑section intensity that is the whole comparison. The production
null instead preserves the serial-section design and asks whether cross-type
proximity remains after first-order tissue preference is controlled.

**Robustness reporting.** A `robustness` verdict is now primary-driven:
`robust` means the calibrated reweighted primary is significant in a direction;
`csr_only` means only homogeneous CSR is significant and the result should be read
as shared-preference artifact; `none` means no significant production finding; and
`mixed` is reserved for diagnostic/legacy cases with extra nulls. The result's
top-level `global`, `null_*`, `K_observed`, and `L_minus_r` fields mirror the
**reweighted** primary so the plot/QC/UI show the statistically defensible curve.

### Registration QC gate (fail‑closed)
Before the statistics are presented as valid, `compute_registration_qc` measures
the **residual alignment error** (median/90th‑percentile distance of matched
hematoxylin feature points after the chosen transform) and the **tissue‑overlap
fraction**; `evaluate_registration_qc` then marks the pair `valid` only when the
residual is below half the DCLF band lower bound (**< 5 µm**, warning 5–10 µm,
invalid ≥ 10 µm or identity fallback or overlap < 0.5). Invalid pairs still produce
all diagnostic images but their statistics are flagged unreliable in JSON and UI
and excluded from the "robust" tally — never silently presented as valid.

### Global DCLF envelope test (the significance call)
The per‑radius envelope test, ORed across ~50 radii, has an inflated family‑wise
false‑positive rate. The single yes/no call is therefore a **Diggle–Cressie–
Loosmore–Ford rank envelope test** over the whole L−r curve:

```
u = Σ_r ( L(r) − L̄(r) )²        (∝ ∫(L−L̄)² dr; constant dr cancels in the rank)
global_p_dclf = (1 + #{ u_null ≥ u_obs }) / (1 + n_perm)
```

`u` is computed for the observed curve and every null curve relative to the
**pooled (observed + nulls) mean**, giving exact exchangeability under H₀ → a
uniform p‑value. **One‑sided** variants use only positive deviations
(`global_p_association`) or negative deviations (`global_p_segregation`);
`direction` is whichever is smaller when significant. `significant = global_p_dclf
< 0.05`.

The test is restricted to a **biologically relevant band `[dclf_rmin_um=10,
dclf_rmax_um=50]` µm** (parameters): below ~one cell diameter, hard‑core
exclusion (two centroids cannot coincide) forces L−r negative for non‑biological
reasons; above ~50 µm the curve reflects tissue architecture rather than
cell–cell association.

---

## 8. Output figures (`overlay.py`)

Four images per pair (segmentation overlays reuse the quant `_draw_cell_boundaries`):

1. **`{sample}_A_segmentation.png`** — CD8 nuclear boundaries on the CD8 image:
   negatives **green**, positives **red** (translucent fill). Stats box: stain,
   total, positive, positivity %.
2. **`{sample}_B_segmentation.png`** — TIM‑3 on its image: negatives **green**,
   positives **blue**, with **brighter/vivid** colours; draws the **nucleus**
   (thin) **and the cytoplasm ring** (`cyto_polygon`) so the measured membrane
   compartment is visible. Stats box adds expansion µm.
3. **`{sample}_consolidated.png`** — dual‑channel KDE density heatmap on the
   registered (CD8) space over a faded grayscale tissue backdrop: CD8⁺ density =
   **green**, TIM‑3⁺ density = **blue**, overlap reads **cyan**. Legend + a
   Ripley's‑K stats box. (`scipy.ndimage.gaussian_filter`, bandwidth 30 µm.)
4. **`{sample}_association_plot.png`** — the statistical figure (matplotlib):
   L−r vs r, shaded 95% null envelope, independence line at 0, annotated with the
   **global DCLF p**.

**Colour convention:** CD8 positive = **red**, TIM‑3 positive = **blue**,
negatives = **green**, density overlap = **cyan**.

---

## 9. Validation

### (a) Statistical correctness — `validation/validate_cross_k.py`, `validate_dclf.py`
- **Exact match:** a brute‑force O(N·M) cross‑K reproduces the `cKDTree` estimator
  to floating‑point precision (max abs diff 0.0).
- **Closed form:** for independent Poisson patterns, the (edge‑corrected toroidal)
  K(r) converges to **π·r²** within Monte‑Carlo error; the uncorrected estimator
  is biased low at large r as expected, which is why significance uses a simulated
  null, not the theoretical line.
- **Null calibration:** under CSR the per‑radius false‑positive rate is ≈0.05.
- **DCLF calibration:** under CSR `global_p_dclf` is uniform — P(p≤0.05)=**0.045**,
  mean p≈0.515. On clustered patterns p≈0.002 with `direction=association`; on
  separated patterns p≈0.002 with `direction=segregation`.

### (b) Real‑data biological validation — `validation/validate_real_data.py`
> **CORRECTION (audit A4, 2026-06-21):** the figures in this section use the **RETIRED
> homogeneous-CSR null** (`validate_real_data.py` calls `cross_k_null`), not the
> production primary. The production-primary (reweighted) re-run gives materially
> different, lower numbers (e.g. CD8↔CD4 → ~50% robust; CD8↔tumour segregation largely
> disappears) — see **§15.7**. Read §15.7, not this table, for the current method.

Run on the **Schürch et al. 2020 CRC CODEX** single‑cell table (Mendeley DOI
`10.17632/mpjzbtfgfr.1`, CC BY 4.0; **258,385 cells, 140 TMA spots**), using the
authors' own validated cell‑type labels (`ClusterName`) and per‑cell `X:X`/`Y:Y`.
Each spot is an independent ~1920×1440‑px core; every qualifying spot is run
through `cross_k_null` and aggregated. Full traceable output is saved to
`validation/real_data_validation_output.txt`.

**Conditions — historical statistic validation, not the current production
primary.** The run uses the same point-pattern family (cross-type K / L−r + DCLF)
but the retired homogeneous-CSR null (`cross_k_null`), so it validates the basic
curve/significance machinery on real biology without validating the current
reweighted production gate:

- **Pixel size `0.3775 µm/px`** — the published **nominal** CODEX resolution for
  this dataset (Keyence microscope, 20× objective). It is *stated*, not derived
  from the table (the coordinate export carries no calibration field). This makes
  the DCLF band a genuine **10–50 µm (≈ 26.5–132.5 px)**. *(An earlier run set
  `pixel_size=1.0`, which silently collapsed the band to 10–50 raw pixels
  ≈ 3.8–18.9 µm — see "Hard‑core artifact" below.)*
- **Radii** `0–100 µm` in `2 µm` steps, converted to px exactly as the pipeline
  does; **`n_perm=1000`, `seed=0`**; the per‑spot verdict is a direction‑resolved
  **global DCLF** call on the 10–50 µm band (not a per‑radius OR), but under the
  old homogeneous-CSR null.
- **Tissue mask — a deliberate, documented difference.** The pipeline derives its
  mask from the CD8 *image* via Otsu; CODEX gives cell **coordinates** directly,
  with no brightfield image to threshold. We therefore bound the area and the CSR
  null with the **convex hull of all cells in the spot**. This is an intentional
  substitution for the *image front end only*: we are validating the **statistic**
  (cross‑type K / L−r + CSR null + DCLF) on real point patterns of known spatial
  structure, not the Otsu step. Once a polygon exists the statistic treats it
  identically however it was obtained, and the hull is the *same* window for the
  observed pattern and every null replicate, so the area/edge normalization
  cancels in significance exactly as with an Otsu polygon. It is **not** claimed to
  reproduce the Otsu mask numerically.

**Results (corrected conditions).** Per‑spot global DCLF, reported
direction‑specifically:

| Comparison | N spots | Expected | mean L−r @15/30/50 µm | DCLF **association** | DCLF **segregation** |
|---|---|---|---|---|---|
| CD8⁺ ↔ CD4⁺ T cells | **100** | association | **+3.9 / +8.9 / +12.1** | **86/100 (86%)** | **0/100 (0%)** |
| CD8⁺ ↔ tumour cells | **107** | segregation | **−2.9 / −4.5 / −6.5** | 12/107 (11%) | **62/107 (58%)** |
| CD8⁺ ↔ Tregs (FOXP3⁺) | **31** | co‑infiltration | **+2.8 / +6.4 / +8.4** | **21/31 (68%)** | 0/31 (0%) |

(N = spots with ≥ 30 CD8⁺ **and** ≥ 30 of the target type; **no spot cap** — every
qualifying spot is run. One‑sided p<0.05 counts are slightly higher than the
two‑sided‑significant direction counts: CD8–CD4 88% association; CD8–Treg 74%
association; CD8–tumour 58% segregation / 15% association.)

The statistic reproduces the known biology in **direction**: CD8⁺/CD4⁺ T cells
associate (mean L−r > 0, no spot called segregating), and CD8⁺/tumour segregate
(mean L−r < 0, segregation the dominant call). The numbers are **lower and more
honest than previously reported** (the old table headlined 94% / 91% / 77%): those
figures came from a per‑radius "significant at *any* radius" OR over a band that
**started at r = 0** and ran at the wrong pixel scale, which conflated hard‑core
exclusion with biology (next paragraph). The CD8↔tumour control is also genuinely
**mixed‑sign**: 11% of spots are called *association*, consistent with real
biology — CD8⁺ T cells do infiltrate tumour nests in some MMR‑deficient/inflamed
cores — so a clean 100% segregation result would not be expected and is not
claimed.

**Hard‑core artifact — confirmed fixed.** At sub‑cell‑diameter distances two cell
centroids physically cannot coincide, so L−r is forced negative there for reasons
that have nothing to do with biological segregation; this is a **known artifact**,
which is why the DCLF band's lower bound is **10 µm** (above one cell diameter).
In the earlier run, `pixel_size=1.0` put the band at 10–50 *pixels* (≈ 3.8–18.9 µm),
dragging the hard‑core zone *into* the test. Reproducing that condition on the
biologically **positive** CD8↔CD4 control, the old per‑radius metric flagged
**94% "positive" and 71% "segregation" on the same 35 spots simultaneously** — a
mixed‑sign result the old one‑line summary omitted. With the corrected pixel size
and the 10 µm lower bound, **segregation is called in 0/100 CD8↔CD4 spots**: the
artifact is gone, and what remains (86% association) is biology.

> **TIM‑3 availability.** No openly downloadable multiplex dataset that contains
> **both CD8 and TIM‑3** with published spatial statistics could be verified.
> Schürch CRC has CD8 but **no TIM‑3** (its checkpoints are PD‑1/PD‑L1/LAG‑3/
> VISTA/ICOS/IDO‑1). The Phillips CTCL CODEX has both but is "available on
> request" only. So results‑to‑results verification against a published CD8+TIM‑3
> Ripley's K analysis is not currently possible; the two validations above are
> the rigorous substitute.

### (c) Historical null‑model validation — `validation/validate_null_models.py`
This was the original constructed-pattern check for the now-retired three-null
design. It remains useful as a teaching/control artifact, but §15 supersedes its
conclusion because later calibration showed that the resampling-Kinhom and
toroidal nulls were anti-conservative under realistic shared tissue preference.

| Constructed truth | Homogeneous CSR | Inhomogeneous (Kinhom) | Toroidal shift |
|---|---|---|---|
| **Shared preference** — A and B each cluster independently in the *same* distributed compartments, **no** cross‑attraction | **association (FALSE +)** | **n.s. (correct)** | **n.s. (correct)** |
| **Genuine cross‑attraction** — B sits beside A | association ✓ | association ✓ | association ✓ |
| **Segregation** — B avoids A's neighbourhood | segregation ✓ | segregation ✓ | segregation ✓ |

At the time, the first row appeared decisive: homogeneous CSR false-positive on
shared tissue preference, while the two structure-preserving nulls did not. The
later §15 calibration is stronger evidence and reverses the operational takeaway:
the old structure-preserving nulls pass this simple constructed case but fail a
broader shared-preference null, so they are retained only as historical controls.

### (d) Registration QC gate — `validation/validate_registration_qc.py`
A well‑aligned synthetic pair is marked `valid`; an unregistrable pair and a forced
identity fallback are marked `invalid` (statistics flagged unreliable). Confirms the
fail‑closed gate fires.

---

## 10. Alternatives considered and rejected (design rationale)

Every major methodological choice, and why the alternative was rejected:

| Choice made | Alternative rejected | Why |
|---|---|---|
| **Spatial association** (population‑level) | **Single‑cell co‑expression** ("this cell is CD8⁺ *and* TIM‑3⁺") | Serial sections are different physical slices: a **Z‑gap** (~4–5 µm) means a cell in A is not the cell in B; **TIM‑3 is not CD8‑restricted** (also on CD4/Treg/NK/myeloid); and the signals live in **different compartments** (CD8 nuclear‑proxy, TIM‑3 membranous). No per‑cell pairing is defensible (§1). |
| **Cross‑type Ripley's K** | **Mutual‑nearest‑neighbour (MNN) matching** | MNN *pairs* a CD8 cell to a TIM‑3 cell, which reads as co‑expression — the exact claim serial sections cannot support. Ripley's K is **population‑level** (no pairing), and additionally reveals **the spatial scale** of any association via the whole L−r curve. (The legacy MNN code is kept only as a marked‑deprecated artifact.) |
| **Reposition one population** (CSR‑type nulls) | **Random labeling** (permute type labels among pooled points) | The two types come from **different physical sections**; a label is inseparable from its section, so relabeling invents never‑observed cells and destroys per‑section intensity. Invalid here. |
| **Calibrated reweighted primary + CSR diagnostic** | **Homogeneous CSR alone** or the retired **three-null robustness vote** | Homogeneous CSR alone is too weak: shared tissue preference can look like association. The later three-null vote was also anti-conservative under realistic shared preference (§15). Production therefore uses the size-controlled **intensity-reweighted inhomogeneous cross-K** as the gate, with homogeneous CSR retained only as a `csr_only` warning flag. |
| **Calibrated fixed µm bandwidth** for the reweighted primary (75 µm, LOO) | **Scott's/Silverman's data‑adaptive rule** or the retired 50 µm resampling-Kinhom sweep | Data-adaptive rules are driven by global point spread and can over-smooth multimodal tissue architecture toward uniform. The production bandwidth is fixed by calibration (§15.4), explicitly reported, and paired with the §15.5 architecture-scale caveat. |
| **DCLF global envelope test** | **Per‑radius p‑values ORed across radii** | ORing ~50 per‑radius decisions inflates the family‑wise false‑positive rate far above 0.05. DCLF reduces the whole curve to one deviation statistic ranked against the nulls — one honest p‑value, validated uniform under CSR (§9a). Per‑radius p‑values are still emitted, but only drive the plotted envelope. |
| **A∩B intersection mask** | **Single‑image (A‑only) mask** | A region present in only one section (fold/tear/missing tissue in B) has no B cells by construction; scoring A against that absence manufactures "segregation". Only the intersection carries valid cross‑section information. Internal holes/lumens are preserved (not tissue). |
| **Rigid/affine registration** + **fail‑closed QC gate** | **Non‑rigid (deformable) registration** | Non‑rigid warping of sparse, differently‑stained serial sections risks **hallucinating** alignment that fits noise. Rigid/affine is conservative; its residual error is *measured* and the QC gate **restricts claims to scales above the residual** (invalid ≥ 10 µm). Non‑rigid is noted as future work. |
| **Voronoi‑clipped cytoplasm ring** for membrane markers | **Fixed‑radius expansion** (un‑clipped) | In dense lymphocyte infiltrate a fixed expanded ring crosses into neighbouring cells and **steals their membrane DAB**. Clipping each expanded nucleus to its Voronoi cell guarantees it never crosses the midline to a neighbour (mirrors QuPath `detectionsToCells`). |
| **InstanSeg** (`brightfield_nuclei`) | **Cellpose** | InstanSeg ships an H‑DAB brightfield nuclei model that runs **headless inside QuPath** with the exact stain‑vector pipeline already used for DAB OD, giving one consistent toolchain and reproducible measurements. Swapping in Cellpose would add a second segmentation stack and re‑introduce stain‑handling/threshold mismatches for no accuracy gain on these nuclei. |

---

## 11. Defensible claims

| Can state | Cannot state |
|---|---|
| "CD8⁺ and TIM‑3⁺ populations show significant spatial association/segregation over the 10–50 µm band under the calibrated intensity-reweighted inhomogeneous cross-K (DCLF p<0.05), with homogeneous CSR reported only as a shared-preference diagnostic." | "These cells co‑express CD8 and TIM‑3." |
| "CD8⁺ cells co‑cluster with / are segregated from population X at the invasive front (within the A∩B tissue intersection, residual alignment < *r*)." | "*This* CD8⁺ cell is also TIM‑3⁺." |
| "Population‑level cross‑type Ripley's K / g(r), calibrated reweighted primary plus CSR baseline, global DCLF envelope test." | "X% of CD8⁺ cells are TIM‑3⁺ double‑positive." |
| "Membrane DAB measured in a Voronoi‑clipped cytoplasmic ring." | Any per‑cell pairing across serial sections. |

---

## 12. Current state & known limitations

- **Registration is rigid/affine, not non‑rigid.** Serial sections can deform
  (folds, tears, stretch) that a rigid/affine transform cannot correct; severe
  local deformation degrades alignment. The **fail‑closed QC gate** (§7) bounds
  this: pairs whose residual error reaches the analysis scale are marked invalid.
- **The reweighted primary depends on the first-order intensity estimate.** The
  calibrated 75 µm bandwidth assumes tissue architecture is coarser than the
  10–50 µm interaction band; fine cell-scale architecture can still leak into the
  test (§15.5).
- **Homogeneous CSR is diagnostic only.** A CSR-only finding is not a robust
  association claim; it is a warning that shared tissue preference may explain the
  apparent signal.
- **Hard‑core exclusion** dominates L−r at very short radii (< one cell
  diameter); the DCLF band starts at 10 µm to avoid mistaking it for biology.
- **Uncorrected K estimator** (edge bias cancels in the null, but raw K values
  are not directly comparable to π·r²).
- **Tissue window is Otsu‑based intersection**; faint/necrotic tissue or strong
  artefacts can mis‑estimate either mask → A‑only, then bbox fallback (logged).
- **No TIM‑3 in open validation data** (see §9); biological validation used
  CD8↔CD4 / CD8↔tumour as positive/negative controls.
- **True single‑cell co‑expression** is only defensible in the same-section
  Restained tab after manual correspondence certification and validated marker
  thresholds. It is not available for serial-section CD8/TIM‑3 claims.
- Desktop app is **macOS‑targeted**; needs QuPath 0.7.x + InstanSeg installed.

---

## 13. Configuration & running

### Run
```bash
python app.py                                            # desktop UI
python run_pipeline.py --config config.yaml              # quantification
python run_pipeline.py --config config.yaml --mode spatial   # spatial association
# --mode coloc is accepted as a deprecated alias of --mode spatial
```

### Key config fields
| Field | Meaning |
|---|---|
| `qupath_binary`, `instanseg_model` | required tool paths |
| `input_dir` / `output_dir` / `dashboard_dir` | quant I/O (spatial supplies paths via `spatial_pairs`) |
| `dab_threshold`, `stain_thresholds` | global + per‑stain DAB OD thresholds |
| `adaptive_threshold` | per‑image Otsu cut on cell `DAB: Mean` instead of a fixed OD (§19.4); default false |
| `preprocess_normalize` | run seg/measurement on a per‑image white‑balanced copy (§19.4); default false |
| `default_pixel_size`, `pixel_overrides` | pixel size + per‑image overrides; summary also emits `pixel_size_source` / `cells_per_mm2` / `pixel_size_warning` QC |
| `use_cytoplasm_measurement`, `cytoplasm_overrides`, `cell_expansion_um` | membrane measurement (default 2.0 µm) |
| `membrane_pix_thr`, `membrane_frac_min`, `membrane_p90_thr` | membrane‑completeness classifier cutoffs (§4.1); unset ⇒ legacy ring‑mean. Fit with `validation/tune_membrane_threshold.py`. |
| `estimate_stains`, `dab_dominance_gate` | `measure_cytoplasm_dab` args (both default on): per‑image Macenko stain vectors + white point, and the `DAB > H` per‑pixel gate (§4.1) |
| `spatial_pairs` *(alias: `coloc_pairs`)* | pre‑built pairs for `--mode spatial` |
| `max_radius_um` (100), `radius_step_um` (2) | Ripley's K evaluation radii |
| `enable_registration`, `device`, `cleanup_intermediates` | misc |

### Output files (Spatial Association)
- `spatial_association_results.json` — combined results (read by the UI).
- `<sample>/<sample>_spatial_association.json` — per‑pair result.
- `<sample>/<sample>_{A_segmentation,B_segmentation,consolidated,association_plot}.png`.

---

## 14. Changelog / decision log

Major pivots, newest first:

0. **Three null models + A∩B intersection window + fail‑closed registration QC.**
   *(Superseded by §15.3 — the three‑null robustness vote was later found
   anti‑conservative and replaced by the reweighted‑primary + CSR‑baseline design.
   Kept here as the historical pivot it was.)*
   The single homogeneous‑CSR null (which mistakes shared tissue preference for
   association) was joined by an **inhomogeneous (Kinhom)** null and a **toroidal
   shift** null, reported side by side with a `robust`/`csr_only`/`mixed` robustness
   verdict (`cross_k_all_nulls`). The single‑image Otsu mask was replaced by the
   **intersection** of both sections' tissue masks (holes preserved), bounding
   area/observed‑points/null alike. A registration **QC gate** measures residual
   error + tissue overlap and marks under‑aligned pairs' statistics invalid rather
   than presenting them as valid. Validated by `validate_null_models.py` (decisive
   shared‑preference test) and `validate_registration_qc.py`. Quantification
   pipeline untouched.

1. **Terminology reframe → spatial association (honest naming).** Renamed
   "coloc"/"coexpression"/"co‑localization" throughout to spatial‑association
   language so neither the JSON, the code, nor the UI implies single‑cell
   co‑expression. `coloc.py → spatial.py`; `run_coloc_pipeline →
   run_spatial_association_pipeline`; result key `"coloc" → "spatial_association"`;
   association key `"CD8+TIM3+" → "CD8__TIM3"`; `coloc_results.json →
   spatial_association_results.json`; `--mode coloc` and `coloc_pairs` retained as
   **deprecated aliases**. The legacy MNN functions in `spatial.py` keep their
   names but are clearly marked **DEPRECATED/UNUSED**. Pure rename — zero numeric
   or image change.

2. **Global significance fix → DCLF envelope test.** Replaced the
   anti‑conservative per‑radius OR (inflated family‑wise error) with a single
   DCLF rank envelope test over the L−r curve (two‑sided + directional one‑sided),
   restricted to 10–50 µm. Validated uniform under CSR.

3. **Three‑image overlay set + brighter TIM‑3 colours.** Replaced the single
   registered cell map with two segmentation overlays + a dual‑channel density
   heatmap (+ the existing association plot); TIM‑3 overlay uses vivid green/blue.

4. **MNN → cross‑type Ripley's K.** Replaced mutual‑nearest‑neighbour cell
   matching (which implies pairing ⇒ co‑expression) with population‑level
   Ripley's K / g(r) / L‑function and a tissue‑mask‑bounded Monte‑Carlo null —
   the scientifically correct statistic for serial sections.

5. **Membrane / cytoplasm measurement added.** Voronoi‑clipped cytoplasmic‑ring
   DAB for membranous markers (CD8/TIM‑3), fixing the nucleus‑compartment
   mismatch; ring geometry persisted for visualization.

6. **Spatial Association workflow & deterministic core.** Scale‑bar calibration,
   per‑image pixel sizes, filename‑based pairing, persisted null stats; earlier
   move off any AI dependency to a fully deterministic, reproducible pipeline.

---

## 15. Null-model calibration failure, diagnosis, and redesign

A 500-realization calibration harness showed the two "structure-preserving" nulls
that gated the production **robust** verdict were **severely anti-conservative**
under a biologically realistic null. This section records the failure, the
first-principles diagnosis, the redesign, and the proof, for a skeptical reviewer.

### 15.1 The failure (measured, `validate_primary_null_calibration.py`)

H0 = **shared tissue preference, no cross-interaction**: A and B are *independent*
inhomogeneous Poisson processes drawing from the **same** coarse intensity field
(CD8+ and TIM-3+ cells both responding to the same margins/vessels/stroma for their
own reasons). A correctly-sized test rejects at ~5%. Measured P(global DCLF
p <= 0.05), 500 reps:

| Null | shared-preference H0 | uniform-CSR sanity | verdict |
|---|---|---|---|
| homogeneous CSR (baseline) | **1.00** | ~0.05 | anti-conservative (expected) |
| inhomogeneous Kinhom (resampling, primary) | **0.87** | ~0.05 | **ANTI-CONSERVATIVE** |
| toroidal shift | **0.85** | ~0.05 | **ANTI-CONSERVATIVE** |

The uniform-CSR column (all ~0.05) proves the harness is fair. Because **robust**
required *both* structure-preserving nulls and both fired ~85-87% under a true
null, the headline did **not** control the false-positive rate against shared
preference — the exact failure the design was meant to defeat.

### 15.2 Diagnosis (first principles, against the code)

- **Homogeneous CSR** fails as designed: uniform B erases B's intensity → a
  first-order effect reads as interaction. Baseline only.
- **Resampling Kinhom** fails three ways: (1) **plug-in/double-dipping** — lambda_B
  is estimated from the same B being tested; the observed uses the *real* rough B
  while the nulls use draws from a *smoothed* lambda_B, so the observed is biased
  high; (2) **bandwidth (50 um) sits on the test band**, smoothing away the tested
  scale; (3) it uses the **unweighted** K + resampling rather than the
  intensity-**reweighted** Kcross,inhom whose null mean is pi*r^2 independent of the
  shared intensity.
- **Toroidal shift** fails *structurally*: under shared **non-stationary** preference
  a rigid shift moves B's mass **off** the shared architecture, so its operational
  "no association = B at a random offset from A" **erases the shared architectural
  co-location that is the null**. Random-shift assumes stationarity, which tissue
  architecture violates. (Edge wrap on the bbox makes it worse.)

**Biological statement of the correct null.** "No association" = **no preferential
A-B proximity *beyond* the shared architectural response**. That shared response is
the null expectation, not signal; the correct null holds the architecture fixed and
asks only whether A and B are closer than the architecture alone predicts.

### 15.3 Redesign — the calibrated PRIMARY

**Intensity-reweighted inhomogeneous cross-K** (Baddeley-Moller-Waagepetersen),
`cross_k_inhom_reweighted_test`:

```
K_AB^inhom(r) = (1/|W|) Sum_i Sum_j 1[ d(a_i,b_j) <= r ] / ( lambda_A(a_i) * lambda_B(b_j) )
```

Reweighting each pair by 1/(lambda_A*lambda_B) makes E[K] = pi*r^2 under
independence **regardless of the shared first-order intensity** — the architecture
cancels analytically; what remains is interaction beyond it. Two design choices were
required to make it *calibrated*, both validated, not assumed:
1. **Leave-one-out (LOO) kernel intensity** (`_loo_kernel_intensity`) for the
   reweighting — excluding each point's own kernel removes the self-attraction
   plug-in bias (spatstat's `leaveoneout=TRUE`). Without LOO the test was mildly
   anti-conservative under uniform CSR.
2. **Parametric-bootstrap null that holds A fixed, draws B\* from lambda_B, and
   RE-ESTIMATES the intensity from each B\*** — so observed and simulated curves are
   treated symmetrically.

The **DCLF global envelope** (10-50 um) and the **fail-closed registration QC gate**
are unchanged. Homogeneous CSR is retained as a **diagnostic baseline only** (its
"significant under CSR but not the primary" case is the `csr_only` verdict that
flags shared-preference artifacts). The resampling-Kinhom and toroidal nulls are
**retired from the production verdict** (kept callable only for the diagnostic
scripts that document why).

### 15.4 Proof (`validate_reweighted_null.py`, 500 reps, 3 regimes)

Bandwidth chosen by the calibration, not assumed. Winner = **75 um with LOO**:

| regime | result | target | pass |
|---|---|---|---|
| shared-preference size | **0.032** | 0.03-0.07 | YES (was 0.87) |
| uniform-CSR size | **0.064** | 0.03-0.07 | YES |
| power, 7 um attraction | **1.000** | >= 0.80 | YES |
| power, 25 um (mid-band) attraction | **0.992** | >= 0.80 | YES |

Under **shared random, multi-scale architecture** (same field for A and B, the
registered-serial-section regime) the false-`robust` rate is **2-8%** — calibrated
(`validate_internal_controls.py` Control 1).

### 15.5 Limitations (reported prominently — do not paper over)

- **Bandwidth <= architecture scale is REQUIRED.** The reweighting cancels the
  architecture only when the intensity bandwidth (75 um) **resolves** it. If tissue
  structure exists *at or below* the interaction band (<~ bandwidth), the test is
  anti-conservative (empirically: false-`robust` rises to 40-100% when the
  architecture scale drops to 45-70 px). The method therefore **assumes tissue
  architecture is coarser than the 10-50 um cell-cell interaction band** — true for
  margins/vessels/compartments, but it **cannot** separate association from
  preference when meaningful structure exists at the cell scale. The fully-correct
  remedy is a **covariate-conditioned null** built on an externally measured
  architecture surface (margin/vessel/stroma segmentation) — future work, requires
  data we do not yet have.
- **Independent (unrelated) architecture leaks in the statistic alone.** Pairing
  two *unrelated* tissues (independent architectures) can produce false `robust` —
  but such pairs **fail registration and are rejected by the fail-closed QC gate**
  before their statistics are trusted (`validate_registration_qc.py`). The honest
  operating regime of the statistic is **shared** architecture (registered serial
  sections of the same block), where it is calibrated.
- The bandwidth (75 um) is fixed; an adversarial multi-scale intensity could still
  be mis-estimated. The single intensity bandwidth is reported in every result.

### 15.6 Alternatives considered and **rejected**

- **Pooled-label / random-labeling null.** Rejected — a trap. It *appears*
  calibrated on the synthetic harness **only because synthetic A and B are
  exchangeable by construction**; on real serial sections A and B are **not**
  exchangeable (different physical sections/markers/intensities), so relabeling
  invents never-observed cells (Section 7). Passing the synthetic test by exploiting
  its own exchangeability while being biologically invalid is result-shopping.
- **Re-tuning the KDE bandwidth of the old resampling Kinhom.** Rejected: the
  0.5x/1x/2x sweep showed *no* bandwidth was calibrated; the failure was the
  estimator design, not the value.
- **Keeping toroidal "for robustness".** Rejected: ANDing in a null that is
  anti-conservative under the target H0 manufactures false `robust` calls.
- **Adjusting the 0.05 threshold to hit a target.** Rejected as silent tuning.

### 15.7 Real-data behaviour through the new primary (Schürch CODEX)

`validate_real_data_production.py` re-runs the Schürch CRC CODEX controls through
the calibrated reweighted primary (40 spots/control, n_perm=499, 0.3775 um/px,
10-50 um band). The prior numbers move — reported honestly:

| control | old CSR-only path | calibrated reweighted primary |
|---|---|---|
| CD8-CD4 (positive) | 92% "association" | **50% robust association** (mean L-r@15um = +4.0) |
| CD8-Treg (info) | 68% | 10% robust (mean +6.2) |
| CD8-tumour (negative) | 72% segregation | **22% segregation; mean L-r@15um now POSITIVE (+2.6)** |

- **CD8-CD4 (positive control) PASSES.** Real co-infiltration beyond shared
  preference is detected in ~half the spots; the rest of the old 92% was shared
  tissue preference, now correctly demoted to `csr_only`. This is the more honest
  number.
- **CD8-tumour (intended negative control): the "segregation" largely disappears.**
  Two non-exclusive, both-reported reasons: (a) the old segregation was a
  **compartment-scale first-order intensity effect** (immune vs tumour zones) — the
  exact confound the band-limited reweighted test removes — while at the 10-50 um
  cell scale CD8 genuinely infiltrate the **tumour margin**, giving real local
  proximity; (b) tumour nests finer than the 75 um intensity bandwidth can cause
  some reweighting leak (the §15.5 limitation). **Consequence:** CD8-tumour is NOT a
  clean cell-scale negative control for this statistic — compartment-scale
  segregation is real biology but is deliberately not what a band-limited cross-type
  test measures. The appropriate negative controls are cell-scale-exclusive pairs or
  the registration-QC-gated cross-sample swap (see `validation/VALIDATION_DATASETS.md`).

**Net:** the redesign fixes the calibration failure (§15.4) and the positive control
holds, at the cost of revealing that one prior "negative control" result was a
first-order artifact. Per the project's guiding principle, a correct null that
breaks a prior claim beats a flattering null we cannot defend.

---

## 16. Cross-validation against spatstat (R reference)

Our intensity-reweighted inhomogeneous cross-K is a custom Python implementation,
so we cross-validated it against the reference implementation in **spatstat**
(R, `spatstat.explore` 3.8.1) — `validation/validate_spatstat_crossval.py` +
`validation/spatstat_crossval.R`. No production code was modified; this is a
read-compare-report check.

**How inputs were matched (any mismatch makes a comparison meaningless).** Both
tools were fed byte-identical inputs, exported once per case: the same (x,y,type)
points; the same observation window W (same polygon, window areas equal to ratio
1.000000); the identical r-grid; and the identical bandwidth. Everything was done
in **pixels** (pixel_size = 1) to eliminate any um/px unit mismatch. To isolate the
estimator from the intensity, Stage B fed spatstat the **same Python lambda-hat
vectors** via `Kcross.inhom(..., lambdaI, lambdaJ, correction="none")`.

**Stage A — intensity surface.** Our `_loo_kernel_intensity` (leave-one-out
isotropic Gaussian, edge=FALSE) vs spatstat
`density.ppp(sigma=h, kernel="gaussian", leaveoneout=TRUE, edge=FALSE, at="points")`.
Max relative difference per point: **5e-4 to 1.3e-3**. The residual is fully
explained by our 4h Gaussian-tail truncation (exp(-8) ~ 3.4e-4); it is a controlled
approximation, not a bug (widening the cutoff to 6h would drive it to ~1e-8).

**Stage B — the estimator (the decision gate).** Our `_cross_k_inhom_weighted`
K(r)/L(r) vs spatstat `Kcross.inhom` with `correction="none"`, on identical
lambda-hat:

| case | L(r) max rel diff | median K ratio | verdict |
|---|---|---|---|
| CSR (synthetic) | 6.7e-14 | 1.000000 | MATCH |
| clustered, shared field | 1.4e-14 | 1.000000 | MATCH |
| calibration field | 4.9e-14 | 1.000000 | MATCH |
| **real Schurch CODEX spot** (CD8 vs CD4, hull window) | 1.4e-10 | 1.000000 | MATCH |

Our estimator equals spatstat's uncorrected Kcross.inhom to **floating-point
precision** on synthetic AND real inputs, with the K ratio exactly 1 (no hidden
normalization factor). The `border` / `translate` / `isotropic` corrections differ
by 5-74%, which **confirms** our estimator is the uncorrected ("none") one, as
designed (the uncorrected estimator is valid here because the bootstrap null uses
the identical estimator and window, so edge bias cancels in the test).

**Stage C — the test.** Not forced to match numerically: our null is a
per-simulation intensity-re-estimation parametric bootstrap with a DCLF global
envelope; spatstat's `envelope()` is a different simulation procedure. Because the
**estimator** matches exactly (Stage B), any Stage C difference is purely the null
procedure, not an estimator bug. Our null's calibration and power are validated
separately in `validate_reweighted_null.py` (Section 15.4).

**Verdict: KEEP the Python core, cite spatstat as cross-validation.** The estimator
is mathematically identical to the R reference (to numerical precision) on both
synthetic and real point patterns; the only difference (Stage A, ~1e-3) is a
documented Gaussian-tail truncation with no material effect. There is no
implementation bug and no reason to switch the stats core to an Rscript subprocess.

---

## 17. Why no edge correction (the calibration evidence)

The reweighted cross-K uses the UNCORRECTED estimator (no translation/isotropic/
border edge correction), for both the observed K and every null-simulated K. §16
confirmed we compute the uncorrected estimator correctly; this section settles
whether uncorrected is the right CHOICE, by testing whether edge correction changes
the null's calibration. (`validation/validate_edge_correction.py`.)

**Hypothesis.** Observed and null K are both computed with no edge correction, so a
systematic boundary undercount affects BOTH sides of the DCLF rank test and cancels
— making uncorrected safe.

**Method.** We added a translation (Ohser) edge correction — exact for a
rectangular window, weight e = |W|/((Lx−dx)(Ly−dy)) per pair — applied through the
SAME function to observed and null K (symmetric by construction). Our translation
correction agrees with spatstat's `correction="translate"` to ~4e-3 (radius-
dependent; a minor normalization nuance in spatstat's variant — our "none" matches
spatstat at 1e-15, §16), and it is a substantial correction (weights up to ~1.25×
near the boundary). We re-ran the full §15 three-regime calibration (500 reps,
bw=75 µm + LOO), paired (identical seeds/patterns), under no correction vs
translation:

| correction | shared-pref P05 | uniform-CSR P05 | power@7px | power@25px |
|---|---|---|---|---|
| none (production) | 0.032 | 0.064 | 1.000 | 0.992 |
| translation | 0.032 | 0.064 | 1.000 | 0.992 |

(95% CIs identical: shared [0.017, 0.047], uniform [0.043, 0.085].)

**Result.** Calibration and power are UNCHANGED — identical to three decimals on
every metric — despite the correction changing the raw K by up to ~25% near the
boundary. The boundary undercount cancels in the rank-based DCLF test exactly as
hypothesized. Isotropic correction is expected to behave identically by the same
symmetric-cancellation argument (it changes K by a comparable 5–17%, §16); the
decisive test is translation, which we ran.

**Decision: KEEP the uncorrected estimator.** Edge correction does not improve
calibration or power here; adopting it would add cost and a window-shape dependence
(the analytic translation weight is rectangle-specific; real tissue windows are
irregular polygons) for zero statistical benefit. "Edge correction is standard" is
not a reason to adopt it when the calibration is identical with and without it. This
is the documented, paper-grade answer to "why no edge correction": **because, under
our symmetric DCLF null, it provably changes nothing.** No production code changed.

---

## 18. Phase A — serial-section registration redesign + certification (052526)

Scope: 7 H-DAB CD8/TIM-3 serial-section pairs (3 tumor, 4 liver) in `052526`.
Tumor #4 EXCLUDED — its "TIM-3" is byte-identical (MD5 `cbaa06cd…`) to Liver TIM-3
#1 (the only duplicate in the set). All images are 1920×1440 RGB field-of-view
crops (NOT whole-slide); embedded `XResolution` is 96 dpi/inch — no micron
calibration, so pixel size must come from the scale bar / manual value.

### 18.1 Why the legacy registration + QC fails (audit, restated)
Nuclear-texture ORB/SIFT cannot register serial sections (individual nuclei do not
correspond across the z-gap); the QC reused the same nuclear matcher, so it failed
*closed* on well-aligned tissue (residual `None` → "invalid"). SimpleITK ran one
init and returned on its first near-identity stop, blocking fallbacks. Tissue
overlap saturates for large blobs and is blind to the 10–50 µm band. Pixel size
fell through to the filename lookup (10× → 1.00 µm/px) — wrong for this scanner
(true ≈0.75). Net: the legacy path could neither certify good pairs nor catch local
deformation.

### 18.2 Scale-bar finding (corrects the "6% within-pair scale" premise)
Robust detection (longest contiguous solid dark run, voted across thresholds
60/90/120) measures the burned-in 100 µm bar at **exactly 133 px in all six tumor
scale images → 0.7519 µm/px**, within-pair ratio **1.000** (stable across all
thresholds 60–120). The hypothesised ~6% within-pair scale difference (127 vs 135
px) is **not reproduced** — the bars are pixel-identical; the manual read was
end-point ambiguity. The similarity (rotation+translation+uniform-scale) transform
was adopted anyway — it is strictly more general than rigid and *empirically
measures* any within-pair scale from tissue stretch (which a bar cannot capture).
Measured per-pair scale came out **0.998–1.006** → no real within-pair scale
difference. Liver has no bars → 0.7519 µm/px used as a DOCUMENTED manual absolute
reference (the weaker calibration link). A ~6% absolute pixel-size uncertainty only
shifts the analysis band edges ~6% and does not change whether association exists.

### 18.3 Registration redesign (implemented, `serial_registration.py`)
Register on the **low-frequency structural hematoxylin** channel (σ≈12 µm Gaussian
on the Ruifrok–Johnston H channel → suppresses single nuclei, keeps vessels /
sinusoids / lumens / boundaries). **Similarity2D, multi-resolution** (shrink
[4,2,1]) Mattes-MI, launched from **multiple initialisations** (geometry- and
moments-centred × a −10…+10° rotation seed grid) plus phase-correlation and
identity candidates. The registration core is **proven correct**: on a clean
known +10/+8 px self-shift it recovers tx,ty=−10.06,−8.06, scale 1.000, structural
NCC 0.994. MI-metric selection localises tumor translations robustly and
reproducibly (e.g. Tumor_1 tx=−104.8 px ≈ 78 µm field offset).

### 18.4 The hard finding — automated sub-5 µm TRE is NOT reliable on these tissues
Multiple independent QC/TRE metrics were built and each was **falsified by a
deliberate-shift control** (apply a known shift, demand the metric report it):
- **Corner/keypoint landmarks** (goodFeaturesToTrack on the structural channel):
  do not correspond across serial sections; mutual-NN matches are random → median
  residual ≈ tolerance/2 (≈15 µm) regardless of true alignment.
- **Patch phase-correlation residual flow**: ALIASES on the dense quasi-periodic
  texture. A deliberate **30 µm shift reads median ≈0.2 µm** on BOTH tumor and
  liver (region-max <2 µm) — blind to gross misalignment. Disqualified as both a
  selection metric and a TRE.
- **Dense edge Chamfer**: ~200 k edge px → near space-filling → saturates at ≈2 µm
  for any shift. Insensitive.
- **Tissue-outline Chamfer**: monotonic but floored at ≈12–13 µm at best alignment
  because the FOV crops' "outline" is mostly the frame edge (different framing per
  section), not a corresponding biological boundary.
- **Lumen-centroid correspondence** (the only genuinely corresponding structures):
  sparse (n=1–12 per pair) and carries a several-µm *biological* centroid-scatter
  floor (a vessel's lumen changes shape across the z-gap), so it cannot certify
  ≤5 µm even under perfect registration.

Root cause: these tissues are texturally **dense and quasi-periodic**, so every
*dense* intensity/edge/phase metric saturates or aliases and is insensitive to real
10–30 µm misalignment, while the only *sparse* corresponding structures (large
lumens/vessels, true tissue boundary) are too few / too biologically variable for a
µm-precise automated TRE. This is the same class of failure the original audit
diagnosed (QC measuring a non-corresponding signal) — now shown to apply to
automated QC *in general* on FOV-crop serial sections of this tissue.

The registration is sound and visually reasonable (green/magenta + checkerboard
overlays in `validation/phase_a_qc/`); what is NOT reliably available is an
*automated, independent* numeric TRE at the ≤5 µm certification scale. Per the
Phase-A plan (A4 escape hatch), the trustworthy route is a handful of **manual
corresponding structural landmarks** per pair → gold-standard TRE. Awaiting the
user's decision on the certification route before Gate A is frozen. No production
pipeline code changed; the legacy `registration.py` cascade is untouched.

### 18.5 Manual + auto-proposed landmark rounds (gold-standard attempt)
Round 1 — user-clicked landmarks (5/2/8/8/8/8/8 pts for T1/T2/T3/L1–L4): only 2–4
points per pair were mutually consistent under any single similarity; raw median TRE
vs the MI transform was 10–113 µm, dominated by mis-clicks. Where landmarks were
reliable the MI registration was accurate to ~2 µm (Liver_3 pts 1–3; Liver_4 pt 1),
i.e. positive evidence those pairs are well-aligned — but n_consistent < 6, below the
agreed bar. Liver_1 was the exception: all points ~100–144 µm off MI while internally
self-consistent → MI registration itself failed there (~130 px y-error).

Round 2 — auto-PROPOSED landmarks (grid-search lumen overlap + RANSAC, scale=1):
produced internally tight sets (self-residual 1.5–2.8 µm) BUT for Tumor_2 / Liver_2 /
Liver_4 the implied transform disagreed with the image-based MI registration by
137–296 µm with opposite rotation — i.e. *coincidental* consistent matches in the
dense lumen field (a wrong transform still aligns 5–7 of ~70–140 lumens by chance).
Only Tumor_1 had the two independent methods agree (~10 µm). Verification contact
sheets (zoomed CD8|TIM-3 crops per point) showed most proposed points land on generic
parenchyma/tumor texture, not on confirmable shared structures.

**Convergent conclusion.** Across automated intensity/edge/phase metrics (alias /
saturate), automated landmark correspondence (spurious consistent matches), and human
landmarking (too few unambiguous fiducials), these 10× H-DAB FOV crops of fairly
uniform tumor/liver tissue **do not support sub-5 µm registration verification**, and
two pairs show genuine registration ambiguity (Liver_1 MI failure; spurious elsewhere).
Per fail-closed policy and the no-threshold-tuning rule, **Gate A is NOT met on this
dataset** and Phase B (10–50 µm spatial association) must not run on it. The
registration engine itself is sound (clean-shift control recovers known transforms;
Tumor_1 cross-validates ~10 µm) — the limitation is the data's lack of distinctive,
verifiable corresponding structure at this magnification/field framing. To proceed
would require data with resolvable fiducials (higher magnification, fields containing
clear vessels/portal tracts, or registration markers), or accepting a coarser
resolvable scale than 10–50 µm with that caveat stated.

---

## 19. Landmark-driven registration + leave-one-out certification (Phase A, final design)

After §18 showed that automated metrics and even automated landmark proposal cannot
verify ≤5 µm alignment on these FOV crops, Phase A registration is made **user-driven
and self-explaining**: the operator (a domain expert) places confident anatomical
landmarks, those landmarks BOTH drive and validate the registration, and the act of
finding landmarks becomes the analyzability test.

### 19.1 Method (`serial_registration.py`)
- **Landmark-driven transform.** A least-squares **similarity** (rotation + uniform
  scale + translation; Umeyama, `_fit_similarity_ls`) is fitted from the user's
  corresponding points. Similarity is **distance-preserving up to a known scale**, so
  the downstream cross-K metric stays valid — we deliberately do **not** non-rigidly
  warp (a thin-plate-spline would align landmarks perfectly but distort local
  distances and corrupt the spatial statistic).
- **Leave-one-out TRE (`loo_tre`).** For each landmark, refit on the other N−1 and
  predict the held-out one; the held-out error is an **unbiased** accuracy estimate
  (fitting and scoring on the same points is optimistic — this removes that bias).
  Needs N ≥ 3 to be defined, which is part of why the certification floor is N ≥ 6.
- **Fit residual** (all-landmark median residual under the single similarity) is the
  **non-rigid-deformation diagnostic**: large residual on points the user is confident
  about ⇒ the sections deform locally / don't correspond, *not* a fixable misalignment.

### 19.2 Three-way verdict (`landmark_register_and_verify`)
| verdict | condition | meaning / action |
|---|---|---|
| **CERTIFIED** | n≥6, LOO-TRE med ≤5 µm, fit-residual ≤5 µm | eligible for Phase B; landmark-fitted transform is used |
| **DEFORMED** | n≥6, but LOO-TRE / fit-residual >5 µm (≤15 µm) | real correspondences, local non-rigid deformation — NOT certified, NOT warped |
| **NON-CORRESPONDING** | n<6, or LOO-TRE ≫ tolerance (>15 µm) | sections don't carry enough corresponding structure — analysis would be meaningless |

Thresholds derive from the ≤5 µm criterion + the serial-section z-gap floor; they are
fixed, never tuned to force a pass. Certified pairs' transforms →
`phase_a_qc/certified_transforms.json` for Phase B; `validation/phase_a_finalize.py`
emits the table. Result on the operator's first landmark round: Tumor_1/2
NON-CORRESPONDING (too few points — operator judged them "not the same slice"),
Tumor_3/Liver_1/Liver_2 NON-CORRESPONDING (LOO-TRE 19–38 µm), Liver_3/Liver_4 DEFORMED
(LOO-TRE ~12 µm). None certified — the truthful outcome.

### 19.3 Is this biologically sound?
Yes, with two caveats baked in. Expert anatomical landmarks are the field-standard
ground truth for histology registration (cf. ANHIR). (1) **Resolution floor:** even
perfect registration correlates markers in *different physical planes*, valid only
where architecture is ~z-invariant; the smallest meaningful association scale ≈
z-gap + TRE, so a certified pair is read with more confidence at 20–50 µm than
10–20 µm (the §B2 identifiability point, made concrete). (2) **No non-rigid warp** —
keep the transform rigid/similarity so the cross-K metric is preserved; a pair that
*needs* heavy warping is flagged DEFORMED, not "fixed". The decisive virtue: when the
operator cannot find corresponding structures, the pair is transparently
NON-CORRESPONDING — the user sees *why* spatial association would be meaningless rather
than assuming the software is immature.

### 19.4 Quantification tab — biomarker thresholds; adaptive/normalize/QC; membrane mode (re-enabled, validated)
The Quantification UI exposes required stain identity and optional per-marker filename
tokens/DAB thresholds. Membrane/cytoplasm-ring measurement was previously retired after an
un-calibrated TIM-3 trial produced 4,433 positives (32.0%) vs the original 61 (0.44%). It is
now **re-enabled** — the failure was un-calibrated deconvolution + no ground truth, both since
fixed (§4.1). A **"Membranous marker"** toggle enables the
Voronoi-clipped cytoplasm-ring measurement; when the stain is **TIM-3** it applies the
data-backed completeness cutoffs (`membrane_pix_thr 0.30`, `membrane_frac_min 0.14`;
`validation/membrane_cutoffs.yaml`), otherwise the ring is measured with legacy mean
classification until that marker's cutoffs are fit. A **membrane-quality gate** flags faint /
low-contrast slides (positive rate >50% **or** threshold-minus-background margin <0.03) as
low-confidence and shows an amber banner — the validated failure mode (92290_IM: 50% positive,
margin 0.016, held-out F1 0.30) is caught rather than silently reported.

**Added controls (2026-07, validated on real data through the CLI).** Three options,
wired UI → `webui/api.py` → `run_pipeline.py` so the desktop app and CLI stay in sync:
- **Adaptive threshold** (`adaptive_threshold`). Instead of a fixed OD cut, each image is
  classified at a **per-image Otsu** cut computed in-Groovy from that image's own cell
  `DAB: Mean` distribution — adapts to per-slide stain intensity. Skips the fixed
  per-image override when on; the summary records the actual threshold used. On real
  CRC-ICM TIM-3 this moved nuclear positivity from a saturated 100 % (fixed 0.1) to a
  plausible ~18–26 %.
- **Per-image stain normalization** (`preprocess_normalize`). Runs segmentation/measurement
  on a per-image **white-balanced** copy (`_normalized_copy`) — corrects tone/illumination
  to a per-image white point **without rescaling DAB**. Best-effort; skips WSIs >400 MB and
  falls back to the original on any failure. Overlays/naming still key off the original.
- **Pixel-size QC.** The summary now carries `pixel_size_source`, `cells_per_mm2`, and a
  `pixel_size_warning` flag (raised when the pixel size is a silent default fallback **or**
  the cell density is implausible, <100 or >20 000 cells/mm²). The results view shows the
  source under each pixel-size cell, a density readout, a **Low-confidence** badge, and an
  amber banner when any image is flagged — guarding the silent failure where a wrong pixel
  size looks fine on counts but wrecks cell-level accuracy (DeepLIIF validation: 0.5 µm/px
  gave near-perfect cell *counts* yet 50 % of cells misaligned; 0.25 µm/px was correct).

### 19.5 Independent review corrections (Codex) — final certification design
An external review (Codex) endorsed the landmark-driven direction and supplied
corrections that are now implemented:

- **Verdict renamed + 4 states.** The old `NON-CORRESPONDING` overclaimed (failing to
  find correspondences is not positive evidence the sections are unrelated — repetitive
  tissue alone can cause it). Verdicts are now **CERTIFIED / LOCALLY_CERTIFIED /
  DEFORMED / NOT_CERTIFIABLE**. `LOCALLY_CERTIFIED` certifies only a spatially-coherent
  ROI (convex hull of the passing landmarks, ≥10 % of field) and Phase B is restricted
  to that ROI — we do **not** claim field-wide accuracy where no landmark exists.
- **LOO honesty fix.** Leave-one-out is *fit-unbiased* (a point never sits in the
  transform that predicts it) but **NOT annotator-independent** — all points share one
  annotator's selection bias. The docstring/report now say so. An **independent
  validation set** (ideally a second annotator, ANHIR-style) is supported
  (`landmarks_val.json`); when present, TRE is the held-out error of that set. With one
  annotator we have LOO-grade validation and report it as such — not a gold standard.
- **Coverage reported**; ~12–16 well-spread landmarks wanted for paper-grade (6 only
  fits a similarity; on these crops 12–16 is unreachable, which is itself the
  NOT_CERTIFIABLE signal — a quality tier, not a software failure).
- **Registration-uncertainty sensitivity (`registration_perturbation_sensitivity`).**
  Phase B perturbs the certified transform within its measured TRE (translation σ =
  TRE/px, rotation σ = TRE/field-radius) and re-runs the cross-K/DCLF; a stable
  direction+significance across perturbations ⇒ supported, a flip ⇒ inconclusive, and a
  radius comparable to the TRE is not interpretable. This is at least as important as
  the single ≤5 µm gate. (Validated on synthetic stats: stable→agree 1.0, fragile→0.38.)
- **Blinded landmarking.** `landmark_tool.html` shows greyscale by default (`Blind: ON`)
  so points mark hematoxylin structure, not DAB positivity — no confirmation bias.
- **Forced overrides quarantined** (product rule): any forced exploratory alignment is
  watermarked and excluded from certified pair/cohort statistics — landmarks certify,
  they never act as a subjective permission switch.

**Terminology for any positive result:** *"projected cross-section spatial concordance
between CD8⁺ and TIM-3⁺ populations within anatomically corresponding serial-section
regions"* — explicitly NOT same-cell co-expression, exhaustion, cell–cell interaction,
or same-plane proximity. Record **section thickness + any skipped-section gap** as
metadata: consecutive-section registration carries an inherent biological error floor
(local structures differ between planes) and is substantially harder than registering a
restained copy of the *same* section. **Acquisition priority:** restaining the same
physical section (stain → image → strip → restain → image) removes the z-gap floor,
makes registration near-trivial, and enables genuine same-cell co-expression — it
outranks higher magnification as the next-data recommendation.

### 19.6 Recorded future work at the time of §19 (partly superseded)
This subsection is retained as the historical roadmap from the §19 pass. The first
item below was later shipped in §22.3; the remaining items are still recorded as
future validation/workflow improvements.
- **In-app landmark picker in the Spatial tab — SHIPPED later (§22.3).** Promote the
  standalone blinded `landmark_tool.html` + `phase_a_finalize.py` flow into the UI:
  per-pair point placement → live held-out TRE + 4-state verdict badge + coverage →
  certified / locally-certified pairs feed Phase B automatically; failed pairs show
  the reason; forced overrides watermarked and excluded from certified statistics.
- **Phase-B wiring of the perturbation sensitivity test + ROI window** — call
  `registration_perturbation_sensitivity` around the cross-K/DCLF on certified pairs,
  restrict the analysis window to a LOCALLY_CERTIFIED ROI, and auto-attach the
  resolution-floor caveat (z-gap + TRE bounds the smallest interpretable radius).
- **Second-annotator validation path in the UI** (`landmarks_val.json` already
  supported by `phase_a_finalize.py`).

### 19.7 Quantification tab redesign + validation dataset (this round)
- **Quant tab** rebuilt to mirror the Spatial tab: two modes — **Single image**
  (`image_whitelist` of one file; new ~4-line filter in `run_pipeline.py`) and **Batch
  (folder)**. Starting presets preserve QuPath nuclear measurement: CD8 → 0.20 OD,
  TIM-3 → 0.10 OD. Membrane-ring measurement is an explicit experimental option,
  not a default.
- **Validation dataset = HyReCo** (IEEE DataPort, CC-BY-SA 4.0): CD8 + consecutive +
  re-stained serial sections with 690 dual-expert-verified landmarks. Chosen as the
  single most relevant public set; ANHIR/ACROBAT considered but less specific (no CD8,
  general tissue). HyReCo has **no TIM-3** (no public CD8/TIM-3 set exists) → validates
  the registration/certification METHOD, not the CD8↔TIM-3 biology. Certification works
  on landmark coordinates, so only HyReCo's small landmark CSVs are needed.
  `validation/validate_hyreco.py` ships a self-test (PASS: clean→CERTIFIED sub-µm,
  scrambled→NOT_CERTIFIABLE); real-CSV run pending download. **→ See §20:** the HyReCo
  real-CSV run is blocked by access (233 GB+ login-gated, no landmark-only mirror); the
  real-data method validation was completed on the open ANHIR/CIMA expert landmarks instead.

---

## 20. Public-data certification run — method validation + end-to-end on real data (Jun 2026)

Goal of this round (closes the open question from §18-19): find public dataset(s) that
(1) can be CERTIFIED by our §19 landmark-driven pipeline and (2) support cross-type spatial
association, then run Phase B end-to-end on anything that certifies — our own 052526 pairs
having failed to certify (§18.5). Scientific honesty over a clean result: certify on TRUE
accuracy, never tuned passes; keep method-validation and biological-claim strictly separate.

### 20.1 Certification criterion (UNCHANGED — restated, no thresholds tuned)
Per §19.2 / §19.5, on the landmark COORDINATES alone (`landmark_register_and_verify`):
- **CERTIFIED** — n ≥ 6 landmarks, held-out TRE median ≤ 5 µm, fit-residual ≤ 5 µm.
- **LOCALLY_CERTIFIED** — only a spatially-coherent subset passes (≥ 6 good points whose
  convex hull is ≥ 10 % of the field); Phase B is restricted to that ROI.
- **DEFORMED** — real correspondences but TRE / fit-residual in (5, 15] µm (local non-rigid
  deformation; not warped, not certified).
- **NOT_CERTIFIABLE** — n < 6 or TRE ≫ tolerance (> 15 µm).
Transform is a **similarity only** (no warp) so the cross-K distance metric stays valid.
Held-out TRE is annotator-INDEPENDENT when a second annotator's landmarks are supplied,
else leave-one-out. These thresholds are fixed; nothing below was tuned to force a pass.

### 20.2 Dataset survey (web-verified Jun 2026)
Separated by PURPOSE — never blurred.

**Purpose A — METHOD validation (registration benchmarks with expert landmark GT):**
| dataset | modality / markers | landmark GT | license | access | verdict for us |
|---|---|---|---|---|---|
| **ANHIR / CIMA** (Borda/dataset-histology-landmarks) | consecutive multi-stain (lung-lesion CD31·Cc10·proSPC·Ki67·HE; lung-lobes; mammary ER·PR·CNEU·HE) | ~80 pts/img, expert, **2 annotators** on some pairs | CC-BY 4.0 | **OPEN, no login** (landmark CSVs in GitHub) | **USED — the real-data anchor** |
| **HyReCo** (IEEE DataPort) | consecutive **+ re-stained** serial (CD8·HE·Ki67·CD45·PHH3) | 690 dual-expert-verified | CC-BY-SA 4.0 | **login + 233 GB+ ZIPs only** (landmarks bundled inside huge archives) | **BLOCKED** — chosen in §19.7 but the real-CSV run is infeasible here (see 20.4) |
| **ACROBAT 2022/23** (grand-challenge) | HE↔IHC breast (ER·PGR·HER2·KI67) | 54k pts / 13 annotators, but **val/test landmarks confidential** (Docker submission) | research-use | gated (join + Docker) | not usable for offline certification; markers not immune cell-types |

**Purpose B — end-to-end biology (two cell-type markers on certifiable sections):**
No public set pairs **CD8 + TIM-3/HAVCR2** on registrable serial/re-stained sections (the
original §9 finding holds). Substitute second markers vs CD8, ranked, and the set supplying
each (all from ANHIR-family or the multiplex sets in `validation/VALIDATION_DATASETS.md`):
CD4 (Schürch CODEX / ANHIR-COAD) > FOXP3/Treg (Schürch) > CD68 macrophage (ANHIR-COAD) >
PD-1/PD-L1 (Keren MIBI) > CD20 B-cell (HuBMAP) > panCK/tumor (Keren/Jackson). ANHIR-COAD is
the only landmarked set carrying **CD8 + CD4/CD68** on the same series, but its images +
landmarks are grand-challenge-login-gated. The freely-OPEN landmarked sets (ANHIR/CIMA lung
& mammary) carry no T-cell marker, so the best OPEN Purpose-B-shaped pair is a **two-distinct-
cell-population** lung pair (Cc10 club cells vs proSPC type-II pneumocytes; or CD31 endothelium
vs proSPC) — a METHOD demonstration, explicitly not an immune-pair claim.

**True-multiplex CD8 + TIM-3 (gives the biology but BYPASSES our §18-19 registration):**
Phillips 2021 CTCL **CODEX** (56-plex, incl. Tim-3 + CD8; access on-request) and Nirmal/Lin
2022 primary-melanoma **CyCIF** (CD4·CD8·TIM3, HTAN-public). Single-section multiplex → all
markers already co-located, so they do **not** exercise the serial-section registration the
pipeline exists to solve; recorded as confirmatory biology only, not a pipeline test.

### 20.3 Method validation on REAL expert landmarks — `validation/validate_anhir_landmarks.py`
83 stain-pairs across 9 ANHIR/CIMA series certified on the real expert landmark coordinates
(pixel size = ANHIR Table I native µm/px ÷ stored scale: lung-lesion 0.348, lung-lobes 1.274,
mammary 2.294 µm/px). These are **consecutive** sections and ANHIR is by construction a
**non-rigid** benchmark, so the truthful expectation — a warp-free similarity cannot reach
≤ 5 µm on deformed pairs — is exactly what we see:
- **verdict counts: 3 LOCALLY_CERTIFIED, 80 NOT_CERTIFIABLE, 0 globally CERTIFIED.**
- TRE ranges from **10.4 µm** (best, lung-lesion_1 Cc10↔proSPC) to ~500 µm (mammary, lowest-
  res + most deformable). The high-res 40× lung-lesion series dominate the low-TRE end.
- **Annotator-INDEPENDENT** check (lung-lesion_3 He↔proSPC, fit on annotator PS, scored on
  annotator JB's independent landmarks): held-out TRE median 45 µm → NOT_CERTIFIABLE — the
  similarity genuinely cannot register that consecutive pair, confirmed across two experts.
The method **correctly withholds CERTIFIED from every pair that needs non-rigid warping** and
never manufactures a pass; the positive arm (clean correspondences → CERTIFIED sub-µm) is held
by `validate_hyreco.py --selftest` (PASS). Results → `validation/anhir_certification_results.json`.

### 20.4 HyReCo (the §19.7 anchor) — status: real-CSV run still NOT completed, honestly
HyReCo is genuinely the most on-point set (it has CD8 AND a near-zero-z-gap re-stained pair
that *should* CERTIFY). But verified Jun 2026, IEEE DataPort distributes it **only** as
233.36 / 273.73 / 232.58 GB login-gated BigTIFF ZIPs with the landmark CSVs bundled inside —
there is **no landmark-only download and no public mirror**. Extracting the small CSVs would
require an IEEE account + a multi-hundred-GB download. So §19.7's "pending real-CSV run"
**remains pending due to access friction**, and the real-data method validation was carried by
the open ANHIR/CIMA landmarks (20.3) instead — scientifically equivalent for the METHOD
(certification operates on coordinates), differing only in that ANHIR/CIMA ships no re-stained
(≈0-z-gap) pair, which is why nothing in 20.3 reaches the CERTIFIED tier.

### 20.5 Best Purpose-B candidate — certification verdict (Step 4)
Most-promising OPEN Purpose-B-shaped pair = **lung-lesion_1 Cc10 ↔ proSPC** (two distinct lung
epithelial populations, 40× / 0.348 µm/px, 78 shared landmarks):
- **Verdict: LOCALLY_CERTIFIED.** 18 of 78 expert landmarks agree on a single similarity to
  ≤ 5 µm within a spatially-coherent ROI (~17 % of field); **ROI residual median 3.66 µm,
  p90 4.89 µm.** Two further lung-lesion_1 pairs locally-certify (Cc10↔Ki67 ~12 %; Ki67↔proSPC
  ~25 %). All other series/pairs: NOT_CERTIFIABLE. No thresholds tuned.
This is the §19.5 four-state design behaving exactly as intended: it does **not** claim
field-wide accuracy, but it identifies a real ROI where consecutive-section registration is
certifiably ≤ 5 µm.

### 20.6 Phase B on the certified pair (Step 5) — `validation/validate_phase_b_certified.py`
Phase-B cross-K consumes per-cell coordinates of two markers in the registered frame; the
production path takes those from upstream segmentation (geojson/CSV) and the cross-K/DCLF
engine is already validated on **real** pre-segmented cells (Schürch CODEX,
`validate_real_data*.py`) and synthetically (§7/§15/§16). What this round adds on real data is
the **fail-closed registration-QC gate** (`run_pipeline.evaluate_registration_qc`) driven by
the measured landmark residual:
- lung-lesion_1 Cc10↔proSPC **certified ROI (3.66 µm) → status `valid`** (gate ADMITS Phase B).
- same pair, **whole field (10.4 µm) → `invalid`**; lung-lesion_3 Cc10↔proSPC (**21 µm, the
  pair whose images the repo actually ships) → `invalid`** (gate REJECTS).
A real cross-K/DCLF on a CERTIFIED pair was **not** run: the one certified registration
(lung-lesion_1 ROI) has **no freely-accessible images** (CIMA/ANHIR images are Kaggle/login-
gated; only lung-lesion_3 images ship in the landmark repo, and that pair is gate-rejected).
This is a **data-access** limit, not a method failure, and we did not fabricate a cell detector
on coarse 5 % images to force a statistic the gate would reject. Resolution floor for any future
run here: smallest interpretable radius ≈ z-gap + TRE (consecutive z-gap unknown; ROI TRE ~3–5
µm) ⇒ interpret only ≳ 20–50 µm.

### 20.7 Plain readout — does the full pipeline produce a certified end-to-end result on real data?
**Registration / certification half: YES (locally).** On real, open, expert dual-annotator
landmarks the §19 method certifies a real consecutive-section ROI to ≤ 5 µm (lung-lesion_1
Cc10↔proSPC, ROI residual 3.66 µm) and correctly refuses every pair that needs non-rigid
warping — validated honestly, no tuning. **End-to-end cross-K half: NOT on freely-accessible
data** — the certified pair's images are paywalled and the only image-shipped pair is gate-
rejected; the statistic itself is already validated on real Schürch cells. HyReCo, the ideal
anchor (would certify + has CD8), is blocked behind a 233 GB+ login.
**What this establishes:** the certification METHOD works on real expert ground truth, and the
registration→gate chain runs end-to-end and fails closed correctly on real transforms.
**What it does NOT establish:** any CD8↔TIM-3 biology, or even any immune-pair association — the
certified pair is Cc10 vs proSPC (lung epithelium), a method demonstration. It does **not**
substitute for our own **re-stained CD8/TIM-3** acquisition, which (per §19.5) removes the z-gap
floor, makes registration near-trivially CERTIFIED, and is the only path to the actual
biological claim. Next-data recommendation is unchanged: restain the same physical section.

---

## 21. Restained same-section co-expression tab + real nuclear-GT run (Jun 2026)

Purpose of this round: add a separate path for data that has already followed the §19.5
acquisition recommendation — image one physical section, strip it, and restain the **same
section**. This is deliberately not folded into the serial-section Spatial Association
pipeline. Same-section data permit the stronger endpoint of **same-cell co-expression**:
segment nuclei once, reuse those exact cell coordinates on both restains, and ask whether each
cell is A-only, B-only, double-positive, or double-negative. Registration, landmark
certification, cross-K, registration perturbation, and the serial-section z-gap resolution
floor are therefore inapplicable and are not run.

### 21.1 Isolated implementation (no existing analysis algorithm changed)

The UI gained a **Restained** tab, wired as an additive extension. Its analysis code is in
`restained_coexpression.py`, its API adapter in `webui/restained_api.py`, its UI in
`webui/restained_coexpression.js`, and its focused validator in
`validation/validate_restained_coexpression.py`. Existing Quantification, registration,
Spatial Association, segmentation, threshold, and production-config implementations were not
modified. The two existing UI files only receive the minimal loader hooks needed to expose the
new module.

The run contract is fail-closed:

- Input = one hematoxylin reference + marker-A AEC image + marker-B AEC image from the
  **same physical section**, already in the same pixel coordinate frame. Single-section and
  suffix-matched batch modes are supported (defaults: `_Hematoxylin`, `_CD8`, `_FoxP3`).
- Every input, including an optional expert mask, must have exactly identical pixel dimensions.
  A mismatch stops the run with the dimensions in the error. The result JSON records
  `registration.performed=false`; the tab never silently registers or rescales an image.
- The existing QuPath/InstanSeg `brightfield_nuclei-0.1.1` path segments the hematoxylin
  reference **once** at the entered pixel size. Those unchanged polygons are then measured on
  both AEC images. CD8 defaults to a 2 µm Voronoi-clipped membrane ring; FOXP3 defaults to the
  nucleus. Either compartment is explicitly selectable.
- AEC is measured by fixed colour deconvolution (H vector `[0.650, 0.704, 0.286]`; AEC vector
  `[0.274, 0.680, 0.680]`). AEC thresholds are mandatory blank inputs: no DAB threshold is
  copied from Quant/Spatial, and no biological default is implied.
- Outputs include per-cell CSV, measurement/classification GeoJSON, a four-class overlay
  (gray neither, red A-only, blue B-only, magenta double-positive), per-tile JSON, and combined
  JSON. Per tile the report includes the 2×2 contingency table, double-positive expectation
  under independence, enrichment ratio, Fisher odds ratio/p, and phi coefficient; batch
  Fisher p-values receive Benjamini-Hochberg FDR correction. Cells for which either compartment
  is not measurable are labelled `UNMEASURED` and excluded from the contingency denominator.

### 21.2 Faint-nucleus preprocessing — isolated, fixed, and auditable

Only this new tab has a faint-nucleus preprocessing stage. It deconvolves the hematoxylin
optical-density channel with the fixed H/AEC vectors, stretches positive H OD between its fixed
1st and 99th percentiles, and reconstructs an H-only RGB image with target maximum OD 1.2. The
original image is never overwritten; the exact intermediate PNG and percentile values are saved
in the run output/result JSON. There are intentionally no tuning sliders. The stage is an
explicit toggle so raw-vs-preprocessed performance can be checked against a supplied expert
mask instead of assumed from appearance.

### 21.3 Real-data segmentation validation — HNSCC-mIF-mIHC-comparison v2

The real run used three 512×512 mIHC tiles from Case 1:
`Case1_M1_0_0`, `Case1_M1_0_1`, and `Case1_M1_1_0` (0.5 µm/px), each with its hematoxylin,
CD8 AEC, FOXP3 AEC, and dataset nuclear mask. This dataset is the same-physical-section case:
mIF was imaged first and the section was stripped/restained as AEC mIHC. Its supplied masks
encode blue nuclear interiors with green boundaries and were technician-corrected and
pathologist-approved; they are used here as real segmentation ground truth, not merely a visual
reference. Validation uses one connected blue component per reference nucleus, global centroid
matching at the existing fixed 5 µm tolerance, plus pixel Dice/IoU. QuPath parent annotations
are explicitly excluded from detection counts and masks.

No model, pixel size, preprocessing constant, matching tolerance, or threshold was tuned. The
raw rows are the user's existing InstanSeg exports; the preprocessed rows are new headless runs
through the unchanged brightfield model:

| tile | condition | GT / predicted | precision | recall | F1 | pixel Dice |
|---|---|---:|---:|---:|---:|---:|
| Case1_M1_0_0 | raw H | 387 / 386 | 0.837 | 0.835 | 0.836 | 0.665 |
| Case1_M1_0_0 | preprocessed H | 387 / 439 | 0.768 | 0.871 | 0.816 | 0.662 |
| Case1_M1_0_1 | raw H | 422 / 370 | 0.938 | 0.822 | 0.876 | 0.756 |
| Case1_M1_0_1 | preprocessed H | 422 / 381 | 0.906 | 0.818 | 0.859 | 0.751 |
| Case1_M1_1_0 | raw H | 334 / 272 | 0.945 | 0.769 | 0.848 | 0.758 |
| Case1_M1_1_0 | preprocessed H | 334 / 314 | 0.898 | 0.844 | 0.870 | 0.783 |
| **combined** | **raw H** | **1143 / 1028** | **0.902** | **0.811** | **0.854** | — |
| **combined** | **preprocessed H** | **1143 / 1134** | **0.850** | **0.843** | **0.847** | — |

The honest finding is mixed: preprocessing recovered 37 additional true nuclei (FN 216→179;
recall +3.2 percentage points) but added 69 false detections (FP 101→170; precision −5.2
points), leaving combined F1 slightly lower (0.854→0.847). It materially helped the third tile,
slightly hurt the other two, and is **not established as an accuracy improvement**. Its value is
an isolated, reproducible recall/precision trade-off that can be accepted or disabled per run,
with expert-mask metrics displayed when ground truth is supplied. The raw method remains a
strong baseline; faint staining was real, but stronger contrast alone does not solve every
missed nucleus without increasing false positives.

### 21.4 Verification + precise completion claim

`validation/validate_restained_coexpression.py` passes its synthetic end-to-end checks for H/AEC
channel separation, preprocessing/dimension preservation, same-cell measurement, four-way
classification, expert-mask detection metrics, CSV/GeoJSON/overlay output, bundle discovery,
combined JSON/FDR, and fail-closed dimension mismatch. The pre-existing
`validation/validate_segmentation.py --selftest` also still passes. The new tab was loaded and
interacted with in a browser inspection: single/batch switching, required AEC inputs, suffixes,
preprocessing state, and layout were present with no console errors. On the three real HNSCC
tiles, the complete software path ran: preprocessing → unchanged InstanSeg → real expert-mask
validation → shared-coordinate AEC measurement → per-cell co-expression/statistics → artifacts.

**What this establishes:** the new same-section end-to-end software path works on real CD8 +
FOXP3 AEC images, uses the dataset's real nuclear ground truth, and reports segmentation
performance honestly. Same-section restaining removes the serial-section correspondence/z-gap
bottleneck: no registration was needed, and exact shared nuclear coordinates could be reused.

**What is still not established:** the pilot's marker classifications are not a CD8↔FOXP3
biological result because the temporary 0.20 AEC OD values used to exercise artifact generation
were not manually/pathologist validated. The UI therefore supplies no default and requires the
operator's independently chosen AEC thresholds. Once those CD8 and FOXP3 thresholds are supplied
and validated, the tab can make the corresponding same-cell co-expression report without any
registration step. This remains CD8↔FOXP3 substitute biology, **not CD8↔TIM-3**; the latter still
requires the intended same-section restained CD8/TIM-3 cohort.

### 21.5 Three-tile threshold-stability run (completion addendum)

After the technical pilot, a candidate AEC cutoff was derived from the measured cell-value
distributions without changing segmentation or preprocessing. On `Case1_M1_0_0`, the CD8
ring distribution separated at 0.194 OD by Otsu and 0.191 OD by a two-component log-Gaussian
mixture, so **CD8 = 0.19 AEC OD (2 µm ring)** was carried forward. FOXP3 at the temporary
0.20 cutoff visibly overcalled weak gray nuclei (96/439 positive); its high-signal Otsu split
was 0.465 OD, so **FOXP3 = 0.47 AEC OD (nucleus)** was carried forward. The corresponding mIF
channels were used only as an orthogonal sanity check (not ground truth): at these cutoffs,
same-coordinate agreement with an independently Otsu-split mIF channel was directionally
consistent. No manual/pathologist marker-positive labels are supplied by the dataset, so these
remain **provisional image-derived thresholds**, not validated clinical cutoffs.

The same fixed preprocessing, 0.5 µm/px, compartments, thresholds, and 5 µm segmentation-GT
tolerance were then run on all three tiles with their matching expert nuclear masks:

| tile | cells | CD8+ | FOXP3+ | double+ | Fisher p | segmentation P / R / F1 | pixel Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| Case1_M1_0_0 | 439 | 28 (6.38%) | 11 (2.51%) | 0 | 1.000 | 0.768 / 0.871 / 0.816 | 0.662 |
| Case1_M1_0_1 | 381 | 24 (6.30%) | 15 (3.94%) | 1 | 1.000 | 0.906 / 0.818 / 0.859 | 0.751 |
| Case1_M1_1_0 | 314 | 16 (5.10%) | 5 (1.59%) | 0 | 1.000 | 0.898 / 0.844 / 0.870 | 0.783 |
| **three-tile total** | **1134** | **68 (6.00%)** | **31 (2.73%)** | **1 (0.09%)** | **1.000 pooled** | **0.850 / 0.843 / 0.847** | — |

Every cell had both AEC measurements (`UNMEASURED = 0`). Per-tile Fisher p-values were all
1.0, hence their Benjamini-Hochberg q-values are all 1.0. The descriptive pooled table was
`[[double+=1, CD8-only=67], [FOXP3-only=30, neither=1036]]`; independence predicts 1.86
double-positive cells, pooled odds ratio = 0.515 and Fisher p = 1.0. Therefore this three-tile
run contains **no evidence of CD8↔FOXP3 same-cell enrichment** at the provisional cutoffs.
The CD8-positive fraction was stable (5.10–6.38%); FOXP3 was low and more variable
(1.59–3.94%). These are three tiles from one case, not three independent patients, so the
pooled cell table is descriptive and must not be treated as patient-level replication.

**Compartment is load-bearing (re-confirmed Jun 2026).** A re-run of `Case1_M1_0_0` with the
correct compartments (CD8 in the 2 µm membrane ring, FOXP3 in the nucleus) reproduced the
table above exactly — CD8⁺ 28, FOXP3⁺ 11, **double⁺ 0, Fisher p = 1.0, φ = −0.04**, no
co-expression. Mis-measuring the membrane marker CD8 in the **nucleus** compartment on the
same tile fabricates a strong false co-expression (double⁺ 11, φ = 0.44, p ≈ 2e-11). The
membrane/nucleus compartment choice therefore materially decides the result and must match
the marker's biology; it is validated, not cosmetic.

**Presentation/demo note.** If a slide or demo shows the `Case1_M1_0_0` nuclear CD8/FOXP3
2×2 table (439 cells, 51 CD8⁺, 11 FOXP3⁺, 11 double⁺, Fisher p≈1.85×10⁻¹¹, φ≈0.442), read it
as a software/statistics demonstration at supplied nuclear thresholds, **not** as the
biologically valid CD8 compartment analysis. The scientific compartment-corrected result for
membranous CD8 is the ring/nucleus table above.

**Final completion status:** YES for the end-to-end software demonstration on real same-section
images: raw images → isolated preprocessing → unchanged InstanSeg → real expert nuclear-mask
validation → shared-coordinate CD8/FOXP3 AEC measurement → per-cell co-expression → per-tile
test + across-tile FDR → CSV/GeoJSON/overlay/JSON. Segmentation has real pathologist-approved
ground truth and achieved combined F1 0.847 under the fixed preprocessed protocol. NO for a
fully ground-truthed marker-positivity/biological validation: the dataset does not provide
pathologist CD8/FOXP3 cell labels, and the AEC thresholds are image-derived. It validates the
pipeline execution and nuclear segmentation, demonstrates provisional CD8↔FOXP3 analysis, and
does not establish CD8↔TIM-3 biology.

### 21.6 All-image external validation and validity-gate audit (268 tiles, eight patients)

The three-tile pilot was expanded without changing the model, preprocessing constants, pixel
size, compartments, matching tolerance, or AEC thresholds. The isolated reproducibility harness
is `validation/validate_hnscc_restained_all.py`; its complete output is
`/Users/mukilan/Desktop/hnscc_restained_all_validation_20260621`. The run used fixed 0.5 µm/px,
the unchanged `brightfield_nuclei-0.1.1` model, the §21.2 H-only preprocessing, 5 µm centroid
matching, CD8 = 0.19 AEC OD in a 2 µm Voronoi-clipped ring, and FOXP3 = 0.47 AEC OD in the
nucleus. No all-image result was used to tune a constant or threshold.

#### Dataset integrity and exact ground-truth boundary

The local v2 package contains 1,608 mIF PNGs, 1,336 mIHC PNGs, and 268 segmentation PNGs
(3,212 total), rather than the 3,216 PNGs listed by TCIA. Four mIHC CD8 files are absent:
`Case2_M2_0_0`, `Case2_M2_0_1`, `Case2_M2_1_0`, and `Case2_M2_1_1`. Consequently, nuclear
segmentation and FOXP3 concordance were evaluated on all 268 tiles; CD8 concordance and complete
CD8/FOXP3 co-expression were evaluated on 264 tiles. No substitute files were fabricated.

Across all masks there were 95,519 blue connected nuclear interiors and zero red pixels. The
masks therefore provide expert-corrected nuclear segmentation reference shapes, but no released
CD8- or FOXP3-positive cell class. This agrees with the paper's more limited claim: DAPI masks
were corrected by a technician and approved by a pathologist, while its classified-mask example
used CD3 channel intensity. The paper's assay concordance is a pixel-level mIF-foreground versus
mIHC-AEC-background analysis, not pathologist CD8/FOXP3 cell labels. Marker-classification
agreement below is therefore explicitly a non-expert mIF intensity sensitivity analysis, not
marker ground truth.

#### End-to-end execution and expert-mask nuclear segmentation

All 268 raw hematoxylin images completed preprocessing and headless InstanSeg, and all 264
complete bundles completed AEC measurement, four-way same-cell classification, per-tile Fisher
testing, across-tile BH FDR, and CSV/GeoJSON/overlay/JSON export. Against the released nuclear
masks, the all-image object totals were:

| endpoint | result |
|---|---:|
| reference / predicted nuclei | 95,519 / 85,336 |
| matched TP / FP / FN | 70,142 / 15,194 / 25,377 |
| micro precision / recall / F1 | 0.822 / 0.734 / **0.776** |
| per-tile F1, median (IQR; range) | 0.786 (0.680–0.841; 0.178–0.919) |
| per-tile pixel Dice, median (IQR; range) | 0.696 (0.595–0.759; 0.206–0.835) |

This is materially weaker and more heterogeneous than the three-tile Case 1 pilot (F1 0.847),
so the run was stopped for failure-tail investigation before biological interpretation. Visual
inspection and an independent zero-lag structure diagnostic showed that part of the measured
error is a reference-correspondence problem: high-pass mIF DAPI versus deconvolved mIHC
hematoxylin correlation had median 0.794 (IQR 0.718–0.835), but ranged from −0.078 to 0.898.
It strongly tracked apparent segmentation F1 (Spearman ρ = 0.740, p = 8.84×10⁻48). The ten
lowest correlations were all Case 6 tiles; several showed displaced/deformed DAPI-mask versus
hematoxylin nuclei. Other failure-tail tiles had sparse DAPI/masks despite many visible
hematoxylin nuclei (for example `Case8_T3_0_1`: 43 reference versus 154 predicted). Thus the
all-image F1 is the honest observed agreement, but it cannot be attributed entirely to InstanSeg:
the released DAPI-derived reference is not uniformly complete or in the same local coordinate
frame as the mIHC hematoxylin target. No post-hoc transform or tile-exclusion cutoff was fitted.

#### Marker localization versus marker classification

The paper-style pixel analysis supports genuine AEC signal localization overall. mIF Otsu
foreground had higher mean AEC OD than background in 261/264 CD8 tiles and 266/268 FOXP3 tiles.

| marker | tiles | pixel AUC, median (IQR; range) | foreground−background AEC OD, median (IQR) |
|---|---:|---:|---:|
| CD8 | 264 | **0.939** (0.694–0.964; 0.409–0.999) | 0.272 (0.037–0.352) |
| FOXP3 | 268 | **0.777** (0.656–0.922; 0.411–1.000) | 0.122 (0.032–0.481) |

This validates the technical premise that the AEC channel generally marks the same regions as
paired mIF, particularly for CD8. It does not create binary cell ground truth. For a prespecified
sensitivity analysis, each mIF image used the public DeepLIIF automatic marker rule (90% of the
non-zero 0.1th-to-99.9th percentile range) and the cell maximum in the same CD8-ring/FOXP3-nucleus
compartments. On the 68,961 expert nuclei that matched an InstanSeg detection, agreement with
the fixed AEC calls was only moderate:

| marker | precision | recall | specificity | F1 |
|---|---:|---:|---:|---:|
| CD8 | 0.487 | 0.539 | 0.961 | **0.512** |
| FOXP3 | 0.673 | 0.455 | 0.991 | **0.543** |

The high specificity but low-to-moderate sensitivity/F1 means the provisional AEC thresholds
cannot be promoted to validated marker cutoffs. The mIF proxy is itself intensity-derived and
is not a pathologist label; these numbers quantify threshold sensitivity, not clinical accuracy.

#### Co-expression result and why it fails the biological validity gate

The complete 264-tile software output contained 83,783 detected cells: 5,230 CD8+, 1,962
FOXP3+, and 361 double-positive. The pooled descriptive table was
`[[361, 4869], [1601, 76952]]`, giving 2.95-fold more double positives than independence,
odds ratio 3.56, Fisher p = 9.83×10⁻78, and phi = 0.078. Eighteen of 264 per-tile Fisher tests
had BH q < 0.05. These values are recorded for reproducibility but **must not be interpreted as
CD8/FOXP3 biology**: pooling cells is pseudoreplication, binary marker cutoffs lack expert ground
truth, and the apparent signal includes correspondence failures.

The decisive negative control is the most significant tile, `Case2_S3_1_1`: 215 cells, 35 CD8+,
31 FOXP3+, and 31 double-positive (BH q = 5.46×10⁻31). Its CD8 and FOXP3 mIF/AEC pixel AUCs
were only 0.558 and 0.497, segmentation F1 was 0.414, and direct inspection showed grossly
non-corresponding tissue content across the hematoxylin and two marker PNGs. The extreme
co-expression is therefore an image-correspondence artifact. Dimension equality alone did not
certify shared cell coordinates. Even the independent mIF proxy showed pooled enrichment
(odds ratio 2.29), but because it is not expert-classified and inherits the same pairing/
threshold limitations, it does not rescue a biological claim.

**Final all-image verdict.** **YES**: the complete Restained software path executed on every
available public bundle, produced auditable artifacts, and was tested against real expert nuclear
masks and paired mIF images. **PARTIAL**: AEC localization is technically supported (strong for
CD8, more variable for FOXP3), while observed all-image nuclear agreement was F1 0.776 and was
confounded by non-uniform DAPI-mask↔hematoxylin correspondence. **NO**: this release does not
validate CD8/FOXP3 binary cell classification or same-cell biological co-expression, and it
cannot be used to claim CD8/TIM-3 biology. The all-image result also qualifies the three-tile
pilot's “no registration needed” statement: same-section restaining removes the serial-section
z-gap, but stripping, deformation, cropping, and imperfect supplied co-registration can still
break exact cell correspondence. A research claim therefore requires independently certified
same-cell alignment plus expert marker-positive labels (or a prospectively locked and separately
validated marker-threshold protocol); equal dimensions and same-section provenance are
insufficient.

---

## 22. Stabilization pass — enforcing the validity rules in the app (2026-06-21)

This pass closes the gap the audit (`audit_20260621.md`) identified between the
documented honesty standard and the shipping app. **Staged: validity first**; the
in-app landmark picker and the full visual workflow redesign are deferred (see
"Remaining"). No scientific threshold was tuned; the dirty worktree's unrelated
changes were preserved.

### 22.1 What changed
**Stage 1 — Spatial fail-closed on certification (audit A1/B1).** Every spatial
result now carries an explicit `certification` block (`status:"not_performed"`,
`is_certified:false`, `method:"legacy_automated_registration_qc"`) at both the
per-pair and per-association level (`run_pipeline.py`). The Spatial UI renders an
always-on amber banner: "Not landmark-certified (registration QC only) … known-
unreliable on FOV-crop serial sections (§18.4); 'valid' is a weaker claim than
CERTIFIED" (`webui/index.html`). The app can no longer present a legacy-QC pass as
§18–20 certification.

**Stage 2 — Restained correspondence gate (audit B4).** Co-expression is now
**fail-closed**: `run_bundle` requires explicit **manual** correspondence
certification (`correspondence_certified`) before segmentation/statistics; uncertified
tiles return a BLOCKED result with no co-expression (`restained_coexpression.py`). A
new advisory `structural_correspondence_diagnostic` (zero-lag hematoxylin NCC across
the three captures) is reported per tile — **no tuned cutoff** (honesty rule); it
informs the operator's manual decision. The UI adds a required "I certify shared cell
coordinates" checkbox + a §21.6 caveat, blocks the run if unchecked, and renders
BLOCKED tiles in red (`webui/restained_coexpression.js`). The result separates
software-execution / segmentation / marker-threshold / biological validity.

**Stage 0 — traceability/provenance (audit A2–A8).**
- A2: the unsupported "~90% segmentation agreement" claim is flagged UNVERIFIED in §3.
- A3/B2/B5: the retired "three null models" design is corrected to the
  reweighted-primary + CSR-baseline in §7 (banner), `spatial.py` docstring,
  `index.html` (headline + comment), and `learn.md` §20.
- A4: §9b banner — its table uses the retired CSR null; see §15.7.
- A5: `_reweight_run.log` / `_reweight_run2.log` prepended with SUPERSEDED banners
  (kept, not deleted).
- A6/A7: provenance now stamps `reweight_bandwidth_um` (75.0), `null_seed` (0), and
  the **architecture-scale assumption** (`architecture_scale_assumption_um`,
  `architecture_scale_measured:false`); the "robust" UI interpretation carries the
  §15.5 architecture caveat.
- A8: cohort **BH/FDR** across per-pair DCLF p-values is now applied and written to
  `spatial_cohort_fdr.json`, with a caveat that uncertified pairs must not contribute
  to a cohort claim.

**Launcher hardening.** `app.py` now fails fast with an actionable message (missing
deps + the venv path) on the wrong interpreter, instead of a raw `ModuleNotFoundError`.

### 22.2 Tests run (all PASS)
- `validation/validate_stabilization_gates.py` (new): provenance A7/A6 fields; cohort
  BH/FDR A8; Restained fail-closed BLOCK without certification + diagnostic
  discriminates corresponding (min NCC 1.00) vs non-corresponding (−0.005) B4;
  Restained imports cleanly (no Shapely error on the gate path).
- Regression: `validate_registration_qc.py` (all cases pass), `validate_hyreco.py
  --selftest` (pass), `validate_phase_a` finalize (still 4-state, none certified).
- `py_compile` clean across all touched modules; `node --check` clean on the app JS
  and `restained_coexpression.js`.
- Launcher: wrong interpreter → actionable error + exit 1; venv → dependency check
  passes.

### 22.3 Remaining (deferred to the next pass)
- **Stage 3 — DONE (shipped after this note was first written).** The in-app landmark
  picker is now in the Spatial tab: "2 · Landmark certification" step with place-landmark
  UI and "Fit transform & certify" (`webui/index.html`), backed by
  `API.prepare_landmark_pair` / `API.certify_landmarks` → `landmark_register_and_verify`
  (`webui/api.py`), and statistics are gated on it (`require_landmark_certification:true`,
  consumed at `run_pipeline.py`). The operator can now CERTIFY in-app. (What remains is
  *biological*: no CD8/TIM-3 pair has yet reached CERTIFIED — a data problem, §19.6/§20.7.)
- **Stage 4:** full staged Setup→Inputs→Certification→Segmentation→Classification→
  Statistics→Results visual redesign with input previews/overlays and dependency
  pre-checks; console-hidden-by-default.
- Real-CSV HyReCo / second-annotator TRE; a data-backed quantification agreement
  number to replace the retired "~90%".

### 22.4 Publication-readiness
**Not publication-ready for an external CD8/TIM-3 or CD8/FOXP3 biological claim** — and
this pass does not assert any such biological validation. What changed is that the app
now **enforces** that discipline: uncertified spatial results cannot be presented as
certified, the Restained path is fail-closed on correspondence, provenance is complete,
and stale descriptions are corrected. Defensible for **internal / method-development**
use. The blocking gap to external use is genuinely certifiable/registrable data
(ideally restained same-section) plus independently validated marker-positive labels or
locked thresholds, as already documented in §19.6 and §20.7.
