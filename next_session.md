# Next session — live handoff

**Living doc. Updated after every major step.** Read top-to-bottom before touching code.
Last updated: **2026-07-13** (Spatial UX round 2: drawing/zoom, calibration, LoFTR layout, tooltips).

## ⚑ Most recent work (2026-07-13) — Spatial UX round 2 (user feedback)
All browser-verified on serve.py :8765, committed with no Claude co-author trailer.
- **`fbed110`** — association L−r plot (`oasis/reporting/overlay.py`): annotation switched from
  the old `Robustness: ROBUST` label to the two-band interaction box (colocalization 10–20 µm +
  co-infiltration 20–50 µm, each verdict+p), outlined by the strongest resolvable finding; legacy
  results (no `interaction`) fall back to the old label.
- **`53b8747`** — Spatial tab UI (`oasis/webui/index.html`):
  - **Drawing (priority):** `spatialCertPoint` now clamps to image bounds (edge-inclusive, QuPath-like
    — drawing past an edge snaps to the boundary instead of being dropped). `lmDown` uses
    `setPointerCapture` so a drag continues past the stage edge. Live freehand renders a smooth
    Catmull-Rom `<path>` while drawing (it was gated on `w.loftrMode` and therefore INVISIBLE during
    freehand before) via new `loftrSmoothPath()`; samples per ~3 on-screen px.
  - **Zoom (priority):** `lmWheel` now zooms proportional to the wheel delta (`Math.exp(-dy*0.0015)`,
    deltaMode-normalised, clamped 0.6–1.6/event) and coalesces the overlay redraw to one paint/frame
    (`lmScheduleRender`) — fixes runaway trackpad zoom.
  - **Pixel calibration:** reworded as an optional run default; per-image override panels always
    expanded (collapse + `spatialAssocToggleOverride` removed); `buildOverride` reads the manual
    input directly (blank = default); per-image scale image takes precedence. New `spatialOvManualChanged`.
  - **Removed** the legacy `Import expert landmarks` button + `spatialCertImport()` (backend
    `api.certify_expert_landmarks` kept for validation harnesses).
  - **LoFTR region controls** regrouped into a two-path layout (Recommended auto-certify vs.
    draw-your-own) with visible inline descriptions.
  - **Removed every `?` help-tip** and the collapsible `Parameters used` / `What this result means`
    dropdowns — `renderParameterDetails`/`renderMeaningDetails` now render always-open blocks
    (`.output-details-title`). 0 `.help-tip` spans remain app-wide.
- Push command: `cd "/Users/mukilan/PycharmProjects/ihc-original copy" && git push origin main`.
- NOT yet done: a real end-to-end run to see all this on live images (needs QuPath/InstanSeg + a
  certified pair); the interactive drag/zoom feel is logic-verified but should get a real click-test.

## ⚑ Prior work (2026-07-12) — Spatial UI wired (validation-gated)
After the full validation pass came back all-green (see next section), wired the Q3/Q4 science
into the Spatial tab. Both changes are **layout/report only — the science is untouched**. All
browser-verified on the served build (serve.py :8765), committed with no Claude co-author trailer.
- **Commit `452bef1`** — backend: `run_pipeline.py` console diagnostics aligned to Q3 taxonomy
  (print-only; the removed `underpowered_insufficient_positives` branch mislabeled sparse as dense).
- **Commit `b33e54c`** — Spatial **Results report** rewired to the two-band `interaction`:
  per-pair short-range colocalization (10–20 µm) + co-infiltration (20–50 µm), each
  attraction/segregation/shared-compartment-only/none, with the **not_resolvable** guardrail when
  the registration floor exceeds the band. Sparse pairs show an **UNDERPOWERED (all-cell support
  null)** banner and still report segregation; (near-)absent markers get a dedicated **abundance/
  absence** card. Two-scale summary cards + rewritten legend; the `spatialBandwidthStatusLabel`
  + results bandwidth banner no longer reference the removed status. Null table + plots + QC/cert
  gating preserved. Verified against robust / sparse-underpowered / marker-absent / QC-invalid /
  legacy (no-interaction) result shapes.
- **Commit `d38bca3`** — Spatial **guided wizard**: config view is now a functional 4-step wizard
  (Inputs → Certify registration → Bandwidth check → Settings & run) with a clickable stepper,
  Back/Next, per-step what/why hints, resets to step 1 on entering config. Removed the stale
  "Guided certification / automatically proposes an ROI" copy; refreshed bandwidth + analysis-param
  copy to the two-band/Q3 language. Verified all 4 steps navigate + batch/single toggle still works.
- Key UI contract (for future edits): per-pair association `a.interaction.{colocalization,coinfiltration}`
  (verdict ∈ attraction/segregation/csr_only/none/not_resolvable; band_um; label; summary; resolvable);
  `p.spatial_validity.worst_status` ∈ {ok,caution,dense_tissue_bandwidth_invalid,
  underpowered_sparse_marker,marker_absent,architecture_not_estimable}. Wizard JS: `spatialWizardGoTo(n)`.
- **NEXT:** (a) run one real pipeline run to see the new report on live data (needs QuPath/InstanSeg
  segmentation + a certified pair); (b) the planned repo restructure to mirror the real codebase;
  (c) optional: QuPath separation (assessed viable). Tutorial overlay was explicitly deferred by the user.

## ⚑ Most recent work (2026-07-12 late) — FULL VALIDATION PASS before UI wiring
Ran the whole spatial-pipeline validation surface to gate the Q3/Q4 UI wiring. **Everything green.**
- **Unit tests:** `pytest tests/` = 46 pass (incl. 2 slow null-calibration markers), 1 skip (missing local HNSCC data).
- **Q3/Q4-touching harnesses:** `validate_certification_roi_bandwidth` (new sparse/absent taxonomy), `validate_radius_floor`, `validate_architecture_scale`, plus a written **interaction-contract check** (confirms `cross_k_all_nulls` now emits `interaction.colocalization` + `interaction.coinfiltration`, back-compat `robustness` intact) — all PASS.
- **Core stats:** `validate_null_models`, `validate_cross_k`, `validate_dclf`, `validate_edge_correction` (byte-identical to baseline), `validate_registration_qc`, `validate_stabilization_gates`, `validate_internal_controls` — all PASS.
- **Real data:** `validate_dense_null_real_ll477` (5/5 LL477 pairs tested, 0 skipped), `validate_deformation_estimator` — PASS.
- **exit=2 harnesses are EXPECTED, not regressions:** `primary_null_calibration` (documents the KNOWN anti-conservatism of structure-preserving nulls under shared preference), `reweighted_null` ("no bandwidth passes" — the committed `.txt` baseline is STALE: it was made with `NREAL=500,bw[50,60,75]`; the current committed script runs `NREAL=300,bw[50,75,100,150]`), and the candidate-null SCREENING studies (`dense_reweighted`/`compartment_conditioned`/`morphology_conditioned`, all intentional REJECTs). **Proof of no regression:** the `.py` files are clean vs HEAD, my session's commits never touched them, overlapping-bandwidth numbers match within MC noise, and every calibration *classification* is identical. Regenerated MC artifacts were restored to HEAD.
- **End-to-end Q3 confirmed in code:** a sparse marker (5–29 positives) routes to the all-cell support null, runs UNDERPOWERED, and sets `statistics_valid=True` (not fail-closed); a near-absent marker (<5) becomes a `marker_absent` abundance finding.
- **One defect found & fixed (backend, print-only):** `run_pipeline.py` console diagnostics still branched on the removed `underpowered_insufficient_positives` and mislabeled a sparse case as "dense tissue" — aligned to the Q3 taxonomy (data was already correct). **Uncommitted.**
- **NEXT (now unblocked):** wire the Q3/Q4 outputs into the Spatial UI — guided wizard (task 4) + research-grade Results report surfacing the two-band `interaction`, sparse UNDERPOWERED flag, and `marker_absent` finding (task 5). UI contract map: `a.interaction.{colocalization,coinfiltration}`, `p.spatial_validity.worst_status` ∈ {ok,caution,dense_tissue_bandwidth_invalid,underpowered_sparse_marker,marker_absent,architecture_not_estimable}. index.html:4331 still checks the REMOVED status — fix in task 5.

## ⚑ Prior work (2026-07-12) — Spatial tab fixed
- **Root cause found & fixed:** the restructure moved `webui/` → `oasis/webui/` one level
  deeper, but `PROJECT_DIR = Path(__file__).parent.parent` was NOT updated, so it resolved to
  `<root>/oasis` instead of the repo root. `PROJECT_DIR/"run_pipeline.py"` therefore didn't
  exist, and every spatial/calibrate subprocess died with an empty stdout → the UI showed the
  **misleading** "segmentation may have failed" (both the 75 µm pre-flight AND the full spatial
  run share this path). Segmentation itself was never broken (verified: QuPath/InstanSeg
  segments 2407 cells in 15 s).
- **Fixed in 3 files** (same off-by-one, all moved into `oasis/webui/`): `api.py:42`,
  `calibration.py:12`, `restained_api.py:15` → now `.parent.parent.parent` (repo root). Also
  fixed `restained_api.py` to invoke the moved module as `-m oasis.restained.restained_coexpression`.
  Also hardened api.py's pre-flight to report the subprocess **returncode + stderr** instead of
  the misleading "segmentation failed" (it previously read only stdout).
- **Verified:** re-ran the exact pre-flight path headlessly on the LL477 pair → now returns a
  real bandwidth verdict (`status: ok`). Changes are **uncommitted** in the working tree.
- **QuPath separation: VIABLE** — see the assessment note in §10 below.
- Note: ACROBAT was **deprioritized** — it has no offline ground truth, so it can't validate
  LoFTR/the gate offline. ANHIR remains the validated benchmark. §2/§5 below are ACROBAT context
  kept for reference only; the active task is the Spatial/Quant build.

> An older, superseded handoff lives at `legacy/next_session.md` (v1, registration-gate
> deep dive). Keep it for the scientific arc; this file is the current source of truth.

---

## 0. TL;DR — where we are right now

- **Branch:** `rebuild/spatial-quant` @ `0a98a2c` — **local only, not pushed.** All the
  restructure + Spatial rebuild lives here.
- **Spatial tab: rebuilt and browser-validated.** LoFTR global-first flow is live
  (`certify_spatial_auto`), manual landmarks demoted to Advanced, propose/guide legacy
  removed. Proven end-to-end on LL477 → 1 region `LOCALLY_CERTIFIED` @ 2.35 µm.
- **Codebase: restructured** into the `oasis/` package (see §4). Verified: whole tree
  compiles, pytest collects, serve.py boots, UI renders in browser.
- **Claude-as-contributor: REMOVED (done).** `origin/main` is `464cc15`, the de-attributed
  merge; `fix-attribution` is byte-identical to it. No git action pending.
- **Dataset to make LoFTR research-grade: ACROBAT (validation split).** The validation
  source-points CSV is now **in hand and verified to match the WSIs already on Expansion**
  (see §2). This unblocks building the register→predict pipeline.
- **NEXT: build the ACROBAT validation register→predict→submission pipeline** (see §5).

---

## 1. Immediate next action

Build the ACROBAT validation pipeline (§5, step by step). It runs on data **already on disk** —
no further download needed. The only thing that needs the user is the final grand-challenge
*submission* (their account), which yields the official median-error number.

---

## 2. Datasets — exact state on disk

### ACROBAT (the one that matters now)
- **Validation source points — HAVE IT, VERIFIED:**
  `/Users/mukilan/Downloads/acrobat_validation_points_public_1_of_1.csv`
  - 5,020 source points, 100 cases (ids 0–99), stains ER/PGR/HER2/KI67.
  - Columns: `anon_id, anon_filename_he, anon_filename_ihc, point_id, ihc_x, ihc_y,
    mpp_ihc_10X, mpp_he_10X, ihc_antibody, he_x, he_y`.
  - `he_x`/`he_y` are **empty on all rows by design** — these are the H&E coords we register
    and predict, then submit. Targets stay server-side.
  - Source points (`ihc_x, ihc_y`) are on the **IHC (moving)** image.
- **Validation WSIs — HAVE THEM:** `/Volumes/Expansion/oasis_datasets/acrobat/valid.zip`
  (23.4 GB, 201 WSIs, verified intact). Only 1 pair currently extracted to `valid/`
  (`0_HE_val.tif`, `0_KI67_val.tif`) — the rest are still zipped.
- **CSV↔disk mapping (verified):** CSV case 0 → `0_HE_val` + `0_KI67_val`; on disk →
  `0_HE_val.tif` + `0_KI67_val.tif`. **Same stems.** CSV names `.ndpi` (original Hamamatsu),
  disk is `.tif` (SND redistribution) → pipeline must **swap the extension** on lookup.
- **Pixel sizes:** per-row `mpp_*_10X ≈ 0.91` µm/px; full per-level pyramid mpp in
  `/Volumes/Expansion/oasis_datasets/acrobat/df_acrobat_meta.csv` (openslide reads 9 levels,
  level-0 ≈ 0.907 µm/px, confirmed).
- **TEST split points also on disk but NOT what we use now:**
  `/Users/mukilan/Downloads/acrobat_test_points_public_1_of_1.csv` (14,920 pts, `_test.ndpi`,
  303 cases). Pairs with the *test* WSIs (not downloaded). Park it — only relevant if we later
  submit to the test leaderboard.
- **No offline TRE:** ACROBAT never releases targets. A landmark-validated number comes only
  from a grand-challenge submission (user account). Offline we produce predictions + the
  submission CSV + per-case cert verdicts.

### ANHIR / CIMA (already validated against, no download needed)
- Images on Expansion; two-annotator landmarks at
  `~/oasis_validation_datasets/CIMA_ANHIR/inputs/annotations/` (lung-lesion_3, mammary-gland_1/2
  have PS + JB).
- Gate calibration PASSED (ratio 0.96/1.03/1.10, coverage 89–93%).
- LoFTR whole-image = worse than manual (expected ceiling); in-ROI rescues matching but ANHIR
  tissue is too deformed to yield a *positive* cert — which is exactly why ACROBAT matters.

### LL477 (the real CD8↔TIM-3 pair)
- Images `~/Desktop/assets/{cd8_input,tim3 input}/LL477_*_3.tif`, px 0.7519.
- `certify_spatial_auto` → 1 region `LOCALLY_CERTIFIED` @ 2.35 µm. No independent annotator
  (that's the gap ACROBAT fills).

---

## 3. Validation status

| check | result |
|---|---|
| gate calibration (`validate_fw_anhir_calibration.py`) | ✅ PASS 0.96/1.03/1.10, cov 89–93% |
| LoFTR whole-image on ANHIR | done — worse than manual (ceiling), 0 matches on mammary @41µm/px |
| LoFTR in-ROI on ANHIR (`validate_local_roi_loftr.py`) | done — rescues matching (28–65 corr), gate stays honest (DEFORMED) |
| `certify_spatial_auto` on LL477 | ✅ LOCALLY_CERTIFIED @ 2.35 µm (headless + browser) |
| ACROBAT WSI read (openslide) | ✅ reads pairs, 9 levels, 0.907 µm/px |
| ACROBAT register→predict pipeline | ❌ NOT BUILT — this is the next work |
| pytest | collects 44 tests (last full run pre-restructure: 43 pass/1 skip) |

Env: torch 2.13.0 + kornia 0.8.3 in `.venv` (CPU/MPS). LoFTR weights need
`export SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi;print(certifi.where())')`.

---

## 4. Codebase layout (post-restructure)

```
oasis/
  common/     pixel_size_util · registration · file_matcher
  quant/      cell_expansion
  spatial/    serial_registration · spatial_stats · spatial · loftr_matcher
  reporting/  overlay · dashboard
  restained/  restained_coexpression
  webui/      api · calibration · restained_api · index.html
app.py · serve.py · run_pipeline.py        (entrypoints at root)
validation/  (registry refs harnesses by bare filename — do NOT move)
legacy/      (dead code: send_chat, propose/guide, stale docs + manifest)
```
- Run the browser-driveable app: `.venv/bin/python serve.py` → http://127.0.0.1:8765
  (after any Python change, RESTART; after HTML/JS only, just reload).
- Desktop app (`app.py`, pywebview) is untouched and still works.

---

## 5. THE PLAN — ACROBAT validation pipeline (do in order)

Goal: register each ACROBAT validation pair, map the IHC source points onto H&E, write the
predicted `he_x`/`he_y` in submission format. Then (user) submit for the official number.

**Step A — extract the validation WSIs.** Unzip `valid.zip` fully into
`/Volumes/Expansion/oasis_datasets/acrobat/valid/` (only 1 pair extracted so far). ~100 cases,
each `N_HE_val.tif` + one-or-more `N_{ER,PGR,HER2,KI67}_val.tif`.

**Step B — write the pipeline module** (new, committed — e.g.
`validation/validate_acrobat_registration.py` + any helper in `oasis/spatial/`):
  1. Parse the validation CSV; group points by `anon_id` + IHC filename.
  2. For each case: resolve on-disk `.tif` by swapping the CSV's `.ndpi` stem→`.tif`.
  3. Load H&E (fixed) + IHC (moving) at a chosen pyramid level via openslide; track the
     level's µm/px. **Landmark bookkeeping is load-bearing** — CSV coords are at 10× (level-0
     ≈ 0.91 µm/px); rescale to whatever level LoFTR runs at. LoFTR is NOT scale-invariant.
  4. Register IHC→H&E (LoFTR/`certify_spatial_auto`, in-ROI at a fine level).
  5. Apply the transform to each source `(ihc_x, ihc_y)` → predicted `(he_x, he_y)` in
     **level-0 / original 10× pixel coords** (undo any level rescale before writing).
  6. Write predictions in ACROBAT submission format (confirm exact spec on grand-challenge —
     typically a CSV of predicted x,y per `point_id`).

**Step C — sanity-check offline.** No GT targets, so validate plumbing not accuracy: predicted
points land inside the H&E tissue, transforms are non-degenerate, per-case cert verdicts from
`certify_spatial_auto` are sensible. Optionally hold out a few source points and check
round-trip consistency.

**Step D — submission (needs user).** User is set up on grand-challenge (ACROBAT 2022 phase —
confirm it still accepts submissions; else 2023, same landmark corpus). Upload the predictions
→ server returns official median error = the landmark-validated IHC number for the paper.
Do NOT submit to a leaderboard autonomously — user confirms/drives the upload.

**Then:** resume Spatial polish (recent-pairs list) → Quant tab rebuild.

---

## 6. Pending user actions
- **ACROBAT grand-challenge submission** (Step D) — their account; I build everything up to
  the upload.
- Nothing else. (Git attribution fix is already live on `origin/main`.)

---

## 7. Traps / must-remember
- **Extension swap:** CSV = `.ndpi`, disk = `.tif`, same stem. Don't fail the lookup on it.
- **Landmark scale bookkeeping is the #1 accuracy risk.** Get 10×↔level rescale exactly right,
  both directions (load and write-back). This silently ruined the earlier coarse smoke-test.
- **`rebuild/spatial-quant` is unpushed** — all rebuild work is local. Push when the user asks.
- **The circularity rule (from v1 handoff):** never let correspondences be selected by, or a
  bound tuned on, the transform under test. Run `residual_field_assay` on any new matcher.
- **After Python changes, restart serve.py** — a stale in-memory API caused a phantom
  "no such method" on certify last session.
- **No offline ACROBAT accuracy** — the honest deliverable offline is predictions + verdicts;
  the validated number needs the submission.
- **No Claude co-author trailer on this repo, ever** (user pushes with their own token).

---

## 8. Reproduction / key commands
```bash
cd "/Users/mukilan/PycharmProjects/ihc-original copy"
export SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi;print(certifi.where())')

.venv/bin/python serve.py                                   # browser app @ :8765
.venv/bin/python validation/validate_fw_anhir_calibration.py  # gate calibration
.venv/bin/python -m pytest tests/ -q

# ACROBAT inputs
#   points (validation): /Users/mukilan/Downloads/acrobat_validation_points_public_1_of_1.csv
#   WSIs:                 /Volumes/Expansion/oasis_datasets/acrobat/valid.zip  (+ valid/)
#   pyramid meta:         /Volumes/Expansion/oasis_datasets/acrobat/df_acrobat_meta.csv
```

---

## 10. QuPath separation viability (assessed 2026-07-12) — VIABLE

QuPath's ENTIRE role is one step, the generated groovy (`generated_pipeline.groovy`), invoked
in exactly one function (`run_pipeline.run_single_image` → the `qupath_binary` subprocess). It:
read image → `setImageType(BRIGHTFIELD_H_DAB)` (stain vectors) → run **InstanSeg**
(`brightfield_nuclei-0.1.1`) with `makeMeasurements` → per-nucleus `DAB: Mean` → threshold →
export GeoJSON/CSV/summary. **Everything downstream** (cell_expansion, overlay, spatial,
restained, calibration) only CONSUMES those files — no binary needed.

Each QuPath capability already has a Python equivalent, most already in this repo:
- image read → `openslide`/`tifffile` (already used); H-DAB deconvolution → `extract_hematoxylin`
  + Macenko in `cell_expansion.py` (already used).
- **InstanSeg** → the open-source `instanseg-torch` Python package (Apache-2.0, same
  `brightfield_nuclei` weights, torch/MPS — MPS is available here). **Not yet installed** in
  `.venv` (would need `pip install instanseg-torch`).
- per-nucleus DAB:Mean, thresholding, cell expansion/Voronoi, GeoJSON/CSV export → all already
  reimplemented in Python (`cell_expansion.py`, `overlay.py`).

**Verdict: viable, and ~70% already done.** The work is to replace the single groovy
segmentation step with a Python InstanSeg segmenter that emits the SAME GeoJSON/CSV/summary
schema, leaving all consumers untouched. **Biggest risk = measurement parity, not segmentation:**
`cell_expansion.py` currently *anchors* its recomputed DAB to QuPath's exported `DAB: Mean`
(parity gate at ~L342-347, scale anchor at ~L475-479). Standalone, the Python DAB measurement
must BECOME the primary and be revalidated so the calibrated membrane cutoffs don't shift.
Secondary risk: model-version parity (pin the same InstanSeg weights) and WSI format coverage
(openslide covers .tif/.ndpi/.svs/.png — their formats; exotic formats would still want
BioFormats). **Not committing to it — this is the viability read you asked for.**

## 9. Changelog (append newest on top)
- **2026-07-12 (later)** — Spatial STATS upgrade (Q3+Q4), committed `163c208` on main:
  - **Q3** — architecture ℓ̂ now from the all-cell support field (tissue property), not
    per-marker positives; power graded separately (adequate ≥30 / sparse 5–29 / absent <5).
    Asymmetric fields (one rich, one sparse) run the all-cell support null flagged
    UNDERPOWERED instead of fail-closing; <5 → absence finding. `worst_status` gained
    `underpowered_sparse_marker` + `marker_absent`. **LL477 (TIM-3 n=15) now runs** (was
    fail-closed). Files: `precheck_bandwidth_within_window`, `_build_precheck_null_plan`,
    full-run `dense_info` loop in spatial.py; `_SPARSE_MIN_POSITIVE=5`.
  - **Q4** — two-band DCLF from the same null envelope: **short-range colocalization
    [10–20µm]** + **co-infiltration [20–50µm]**, each with attraction/segregation/none via
    new `_assess_interaction` → `res["interaction"]`. Short band gated `not_resolvable` by
    the registration floor (no false contact claim across serial sections). Old
    `robustness` kept for back-compat. spatial_stats.py: band constants + `bands` in
    `_null_summary_from_k` + `_assess_interaction` + `cross_k_all_nulls` wiring.
  - Verified: pytest green; validate_certification_roi_bandwidth all PASS (new taxonomy);
    LL477 end-to-end. Reweighted-null calibration re-run confirms no drift (core stat
    byte-identical — only additive fields). **UI still shows the OLD single verdict** — the
    Results report + wizard (tasks 4/5) surface `interaction`/sparse/absence next.
  - **Terminology**: use "short-range colocalization" (proximal association) vs
    "co-infiltration"; directions "attraction"/"segregation". NOT "engagement/contact"
    (can't claim single-cell contact across two serial sections).
- **2026-07-12** — Fixed the Spatial-tab failures (75 µm pre-flight + full spatial run): root
  cause was `PROJECT_DIR` off-by-one after the restructure (pointed at `oasis/` not repo root),
  so the `run_pipeline.py` subprocess couldn't be found. Fixed api.py/calibration.py/
  restained_api.py + hardened the pre-flight's error reporting. Verified headlessly. Assessed
  QuPath separation = viable (§10). Uncommitted.
- **2026-07-11** — Located + verified the ACROBAT **validation** source-points CSV; confirmed
  1:1 stem match to `valid.zip` WSIs on Expansion. Distinguished it from the test-points CSV.
  Confirmed Claude-removal already live on `origin/main`. Created this living handoff.
- **2026-07-11 (earlier)** — Spatial rebuild done + browser-validated (LoFTR global-first,
  manual→Advanced, propose/guide removed); `oasis/` restructure committed; ACROBAT valid.zip
  downloaded + WSI read proven. Commits c8b0ba6…0a98a2c on `rebuild/spatial-quant`.
```
