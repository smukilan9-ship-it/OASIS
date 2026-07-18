# VALIS-rigid as a registration engine — integration scope + MVP

Status: **MVP BUILT (2026-07-18).** Spike de-risked, MVP implemented and verified.

## MVP — what was built & verified
- `oasis/spatial/valis_worker.py` — runs in the isolated `~/valis_runtime` venv (subprocess only,
  never imported by the main process — verified). Returns the RIGID moving→ref similarity in
  ORIGINAL px via `warp_xy_from_to(non_rigid=False)` on probe points (the coordinate-clean path).
- `oasis/spatial/valis_engine.py` — main-venv bridge: subprocess → transform → `assert_distance_
  preserving` (invariant enforced main-side) → **structural patch-residual certification** (Option 1:
  stain-robust, reuses `serial_registration.patch_residual_flow`/`lumen_tre`; stain-agnostic overlap
  mask so cross-modal H&E↔IHC still overlaps). `valis_register_and_certify` (image paths) +
  `register_crops_and_certify` (ROI arrays).
- `oasis/spatial/loftr_matcher.certify_local_roi(..., valis_fallback=False)` — VALIS branch inserted
  BETWEEN LoFTR-fails and the landmark fallback; maps the crop-local transform back to full frame.
  Default off ⇒ byte-identical behaviour (verified). `api.certify_local_roi_multi` reads
  `payload["valis_fallback"]`.

**Spike findings that shaped the MVP:**
- ✅ VALIS recovers dense cross-modal matches where LoFTR gets 0 (mammary H&E↔IHC: 509–871).
- ✅ Transform accurate in original coords (held-out rTRE 0.0022–0.008, verified end-to-end through
  the wired `certify_local_roi`).
- ❌ Feeding VALIS's raw `MatchInfo.matched_kp{1,2}_xy` to the LOO gate is NOT clean — they live in
  VALIS's internal feature frame (~396 px, per-slide) that doesn't map to original via documented
  shapes (conversions off ~1.34×); `error_df.rigid_rTRE` is ~8× the true error. → **Option 1
  (structural cert) chosen** instead of the LOO gate. Its residual TRACKS true error (region-max
  181µm vs true 189µm; 74 vs 67µm), so it certifies honestly without landmarks.
- Note: on ANHIR-25pc (coarse 3–9 µm/px) nothing certifies sub-5µm — correct/fail-closed; OASIS's
  real CD8/TIM-3 use case (~0.5 µm/px) is where it passes.

**Remaining (v1):** UI toggle "Use VALIS-rigid where LoFTR fails"; provenance surfacing;
env-detection greying; a `validation/registry.py` entry (certify-honesty on cross-modal ROIs at a
finer scale); optional persistent JVM-warm worker for multi-ROI speed; wire the flag into the other
`certify_local_roi` call sites (probe/partition/global) beyond the primary draw-ROI path.

---

## Original scope (retained for reference)
 Motivated by the ANHIR
benchmark (`validation/valis_bench/RESULTS.md`, ihc.md §7.1): VALIS-rigid is faster than our
LoFTR pass (29 s vs 51 s/pair), distance-preserving, and — decisively — **produces
correspondences on cross-modal H&E↔IHC stains where LoFTR returns 0 matches** (LoFTR: lung
95–100%, mammary/breast/kidney 0%).

## 1. Objective (one line)
Add VALIS-**rigid** as an optional registration engine that feeds the **existing** certification
gate, specifically to recover ROIs on **different/cross-modal stains where LoFTR fails** — without
changing the gate, the ROI-certification workflow, or cross-K.

## 2. Non-negotiable constraints
- **The gate and ROI framework do NOT change.** VALIS produces correspondences/a transform; OUR
  `landmark_register_and_verify` (Fitzpatrick–West) still decides WHETHER and the multi-ROI
  fan-out still maximises certified ROIs. VALIS never bypasses the gate.
- **Rigid only.** Only VALIS's rigid transform + its RANSAC-filtered rigid correspondences ever
  cross into the pipeline. Its non-rigid warp is never requested and never used (§6, cross-K).
- **Main `.venv` is never touched.** `valis-wsi` cannot even import on the repo's Python 3.14
  (2021-era native stack). ⇒ **in-process import is impossible; a subprocess bridge is mandatory.**

## 3. Architecture — subprocess bridge (the only viable option)
Mirror the DeepLIIF isolation pattern. A small worker script runs in the isolated
`~/valis_runtime/venv` (py3.11, `valis-wsi` 1.2.0, libvips); the main pipeline shells out to it.

**Worker** `oasis/spatial/valis_worker.py` (executed by the isolated interpreter, NOT imported):
- input (argv/JSON): `{ref_crop_path, mov_crop_path}` — two ROI-crop images written to a temp dir.
- runs `Valis(..., non_rigid_registrar_cls=None, crop="reference").register()` (rigid only).
- extracts, from the rigid registrar's `MatchInfo`, `matched_kp2_xy` (ref) and `matched_kp1_xy`
  (moving) — the **actual correspondences** — plus the rigid matrix.
- output (stdout JSON): `{ok, rigid_matrix (mov→ref, crop coords), ref_xy[], mov_xy[], n, secs}`.
- self-contained, `registration.kill_jvm()` in a finally.

**Main-side caller** in `oasis/spatial/loftr_matcher.py` (new `valis_correspondences_in_roi(...)`):
- writes the ref/mov ROI crops to a temp dir, invokes the worker via
  `subprocess.run([VALIS_PY, "-m", "...valis_worker", ...], env={DYLD_LIBRARY_PATH:.../lib})`.
- parses JSON, maps `ref_xy/mov_xy` from crop coords back to full-res reference/moving pixels.
- returns the same dict shape LoFTR's `loftr_correspondences` returns (`ref_points`, `mov_points`,
  `n`, `ok`, `msg`) so downstream code is engine-agnostic.

## 4. Correspondence extraction (the enabler — verified)
`valis.feature_matcher.MatchInfo` fields `matched_kp1_xy` / `matched_kp2_xy` hold the matched
keypoints VALIS used to fit the rigid. These are real, independently-localised points in both
images → they are exactly what the gate needs for a LOO/held-out TRE. (Same self-consistency
caveat as LoFTR: RANSAC-filtered to a rigid, so LOO-TRE is optimistic; the ANHIR held-out-landmark
validation is what breaks that circularity — see §11.)

## 5. Insertion point (a clean add, not a rewrite)
`loftr_matcher.certify_local_roi` (loftr_matcher.py:250) already has the exact branch:
LoFTR-in-ROI → **if `< min_matches`** → landmark fallback → gate. Insert VALIS **between** LoFTR
and the landmark fallback:

```
c = loftr_correspondences(roi crops...)          # try LoFTR first (default, fast where it works)
if not enough matches and engine allows VALIS:
    v = valis_correspondences_in_roi(roi crops)   # cross-modal recovery
    if v.ok: ref_pts, mov_pts, source = v..., "valis_rigid_in_roi"
if ref_pts is None: ... existing landmark fallback ...
# UNCHANGED from here:
M_local = sr._fit_similarity_robust(mov_pts, ref_pts)
sr.assert_distance_preserving(M_local, "valis_rigid_in_roi")   # invariant enforced main-side
cert = sr.landmark_register_and_verify(ref_pts, mov_pts, px, ...)   # SAME gate
```
`certify_local_roi_multi` (api.py:807) and the auto-ROI finder need **no change** — they call
`certify_local_roi`, which now has one more engine in its fallback chain.

## 6. Config + UI
- Config/engine selector: `registration_engine = "loftr"` (default) | `"loftr+valis"` (LoFTR,
  VALIS on failure) | `"valis"` (VALIS-rigid primary). Default keeps current behaviour exactly.
- UI: one toggle in the registration/Spatial step — *"Use VALIS-rigid where LoFTR fails
  (cross-modal stains)"*. Off by default. Disabled (greyed, with reason) if `~/valis_runtime`
  is absent.

## 7. Safety
- `assert_distance_preserving` is called main-side on whatever the worker returns, BEFORE the gate
  — so a misbehaving worker can never inject a non-similarity.
- The worker requests rigid only (`non_rigid_registrar_cls=None`); non-rigid is never computed.
- Engine + version recorded in the certification result (`source: "valis_rigid_in_roi"`,
  `valis_version`) so provenance shows which engine earned each certified ROI. The gate's sub-5 µm
  standard is identical regardless of engine.

## 8. Performance
- ~29 s/ROI including JVM warmup (benchmark). Acceptable as a **fallback** (cross-modal ROIs only),
  but slow for many ROIs. Optimisation (phase 2): a **persistent worker** — keep one JVM-warm
  process alive and stream ROI requests over stdin/stdout, amortising startup (matcher weights +
  JVM load dominate). Not needed for MVP.

## 9. Failure handling / env detection
- Missing `~/valis_runtime` or import failure → VALIS engine reported unavailable; pipeline falls
  back to current behaviour (LoFTR → landmark). Never crashes the main pipeline.
- Worker timeout / non-zero exit / bad JSON → treated as "no matches", continue to landmark
  fallback. All logged (ties into the diagnostics-log plan, `docs/diagnostics_log_plan.md`).

## 10. Provenance/reporting
Per-ROI certification result gains `engine` and `valis_version`; the Spatial-tab provenance and
any exported report state which engine produced each certified transform.

## 11. Validation (before it can be trusted)
- Extend `validation/valis_bench`: for the cross-modal pairs (mammary/breast/COAD H&E↔IHC), run
  VALIS-in-ROI → gate, and confirm **gate-certified VALIS ROIs achieve their claimed TRE on the
  held-out expert landmarks** (i.e. the gate is as honest fed VALIS as fed LoFTR). Add a
  `validation/registry.py` entry.
- Regression: with the engine set to `"loftr"` (default), byte-for-byte identical behaviour to
  today (no accidental path change).

## 12. Phasing
- **MVP:** worker + `valis_correspondences_in_roi` + the `certify_local_roi` fallback branch +
  config flag (no UI). Validate on cross-modal ANHIR ROIs. Per-ROI subprocess (accept ~29 s).
- **v1:** UI toggle + provenance + env-detection greying + diagnostics-log wiring.
- **v2 (optional):** persistent JVM-warm worker for multi-ROI speed; optional VALIS-rigid as a
  provisional/global candidate cross-checked against `register_similarity`.

## 13. Non-goals
Not VALIS non-rigid; not replacing LoFTR as default; not changing the gate, ROI drawing/auto-
suggest, multi-ROI partitioning, cross-K, or the "maximise certified ROIs" strategy; not adding
`valis-wsi` to the main `.venv`.

## 14. Effort & risks
- Effort: MVP ~0.5–1 day (worker + caller + branch + a validation run); v1 ~+0.5 day (UI/provenance).
- Risks: (a) exact registrar→MatchInfo access path is a small spike (fields verified; the container
  attribute is not); (b) VALIS on a *small* ROI crop may find fewer matches than on the whole slide
  — mitigate by padding the crop or, if sparse, using VALIS's global rigid + gate on whatever
  matches exist; (c) subprocess latency (addressed by the persistent-worker phase).
