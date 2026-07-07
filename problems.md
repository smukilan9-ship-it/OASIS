# OASIS — Problems & Roadmap

What's still open, weak, or unfinished, and what we have to work toward.

**Last reconciled against the code: 2026-06-21**, after the stabilization pass. This
file is verified against the *current* source, not just the audit — every audit finding
(`audit_20260621.md`, A1–A8 / B1–B7) was checked against the files and the status below
reflects what the code actually does now.

---

## Honest one-line status

The stabilization pass closed **all** open audit findings, and in-app landmark
certification is now wired end-to-end (UI → `certify_landmarks` →
`landmark_register_and_verify`). The app now **enforces** its honesty discipline:
fail-closed registration certification, fail-closed Restained correspondence, complete
provenance, corrected docs. **Still NOT publication-ready for an external CD8–TIM-3 (or
CD8–FOXP3) biological claim** — not because of a code gap now, but because **no such
pair has ever been CERTIFIED** and the quantification agreement number is still
unverified. Defensible for internal / method-development use: yes.

---

## ✅ Resolved in the 2026-06-21 stabilization pass (verified in code)

These were the P0/P1 blockers in the prior version of this file. All confirmed fixed:

| Audit | What was fixed | Where (verified) |
|---|---|---|
| **A1/B1** | Certification wired into production; spatial run is fail-closed on certification; every result JSON stamped `certification` status | `webui/api.py:411,590` `run_pipeline.py:1041,1269,1299` |
| **A1/B7** | **In-app landmark picker now exists** ("2 · Landmark certification" step, place-landmarks UI, "Fit transform & certify") — *ahead of* what `ihc.md` §22.3 still calls "deferred" | `webui/index.html:809–832`, `webui/api.py:366,406` |
| **A2** | "~90%" flagged UNVERIFIED with a correction block | `ihc.md:115–118, 1604` |
| **B4** | Restained correspondence gate is fail-closed (blocked by default) + advisory `structural_correspondence_diagnostic` (zero-lag hematoxylin NCC) + UI "I certify shared coordinates" checkbox | `restained_coexpression.py:435,486`, `webui/restained_coexpression.js` |
| **A6/A7** | Provenance stamps `reweight_bandwidth_um` (75), `null_seed` (0), `architecture_scale_assumption_um`, `architecture_scale_measured:false`; "robust" UI carries §15.5 caveat | `run_pipeline.py:826,827,832–837` |
| **A3/B2/B5** | Retired three-null design corrected to reweighted-primary + CSR-baseline | `spatial.py:484`, `learn.md:685`, `ihc.md:290` |
| **A4** | §9b table flagged as retired-CSR-null with pointer to §15.7 | `ihc.md:426, 1608` |
| **A5** | Stale `_reweight_run*.log` prepended "SUPERSEDED — DO NOT CITE" banners | `validation/_reweight_run*.log` |
| **A8** | Cohort BH/FDR across per-pair DCLF p-values, written to `spatial_cohort_fdr.json` | `run_pipeline.py:1334–1355` |
| — | New regression harness proving the gates fire | `validation/validate_stabilization_gates.py` |

---

## Open problems — what to work toward

### P0 — Blocks an external / publication biological claim

**1. No CD8/TIM-3 (or CD8/FOXP3) pair has ever been CERTIFIED.**
- The app now *enforces* certification, but in practice certification has never passed
  for the target biology — `validate_phase_a` finalize is "still 4-state, none
  certified" (`ihc.md` §22.2). So the headline biological claim still cannot be made.
- **Target.** Acquire genuinely registrable data — ideally **restained same-section**
  CD8/TIM-3, not serial sections — and drive at least one pair to CERTIFIED (or
  LOCALLY_CERTIFIED on a defined ROI). This is the real blocker, and it is a data
  problem, not a code problem (§19.6, §20.7).

**2. Quantification agreement is still UNVERIFIED ("~90%" has no backing).**
- The number is flagged UNVERIFIED but still sits in the §3 prose (`ihc.md:122`), and no
  manual ground-truth file ships to produce a real figure. The only seg ground truth is
  HNSCC *nuclear* masks (F1 0.776) — a different endpoint, not CD8/TIM-3 DAB quant.
- **Target.** Produce a data-backed F1/κ from an actual manual-count run
  (`validate_segmentation.py` already expects `<image>_manual.geojson` inputs) and
  replace the "~90%" prose with it.

### P1 — Real method gaps (disclosed, but only half-guarded)

**3. Architecture scale is disclosed but never *measured*.**
- Provenance stamps `architecture_scale_measured: false` and the UI shows the §15.5
  caveat — but the pipeline still hardcodes `_REWEIGHT_BANDWIDTH_UM = 75.0`
  (`spatial_stats.py:62`) and never estimates the tissue's real architecture scale. Fine
  (cell-scale) architecture → confident false-`robust` with only a generic caveat.
- **Target.** Estimate a characteristic architecture scale (e.g. KDE intensity
  autocorrelation length), set `architecture_scale_measured: true`, and auto-downgrade /
  flag when it approaches the bandwidth — turn the disclosure into a real guard.

**4. Restained correspondence guard has no tuned cutoff.**
- By design (the "no tuned threshold" honesty rule), the structural-correspondence NCC is
  *advisory* — it informs a **manual** operator certification, it does not auto-fail.
  Validation shows it discriminates well (corresponding NCC 1.00 vs non-corresponding
  −0.005), but production protection still depends on the operator ticking the box.
- **Target.** Decide whether to keep this purely manual, or calibrate a defensible
  warn/fail band on held-out data so non-corresponding tiles are caught automatically.

**5. Calibrated bandwidth is a single knife-edge value.**
- 75 µm is the only bandwidth that passes; bw=50 leaks (shared 0.083), bw=75 sits near
  the 0.07 ceiling (§15.5 / audit A5). Disclosed and reproduces, but fragile.
- **Target.** Either justify 75 µm as tissue-specific (ties to #3) or widen the
  defensible window with a better null.

### P2 — Doc drift & deferred UX

**6. ✅ Doc-drift sweep — DONE (2026-06-21).** `ihc.md` §22.3 now records the in-app
landmark picker as shipped; the stale current-tense "three null models" mentions in §1
(file map), §5 (pipeline steps), §10 (rationale table), and §13 (changelog) were
corrected or given §15.3 forward-pointers. Every remaining mention is now either the
§7 correction banner, retained-for-history text under it, or already marked RETIRED
(`spatial.py:484`, `learn.md:685`). No action remaining.

**7. Stage 4 visual redesign deferred** (§22.3): full staged
Setup→Inputs→Certification→Segmentation→Classification→Statistics→Results flow with
input previews/overlays, dependency pre-checks, console-hidden-by-default.

### P3 — Validation / reproducibility debt (from audit §5, "unverifiable from repo")

- **HNSCC Restained outputs live outside the repo** (`/Users/mukilan/Desktop/hnscc_…`) →
  F1 0.776, AUCs, and the Case2_S3_1_1 negative control are not reproducible from the
  repo alone. **Target:** vendor a small reproducible fixture + checked-in output.
- **Full-N calibration** (NREAL=500/NPERM=199) only reproduced in *trend* at reduced N;
  `reweighted_null_output.txt` should be regenerated at documented settings and cited as
  the single artifact.
- **§16 spatstat cross-validation** needs an R + `spatstat.explore` 3.8.1 env to re-run;
  only the Python side + saved output were verified.
- **§20 ANHIR/CIMA** (3 LOCALLY_CERTIFIED / 80 NOT_CERTIFIABLE) backed by
  `anhir_certification_results.json` but not re-run against raw landmark CSVs.
- **QuPath/InstanSeg end-to-end** depends on an external install not in the repo — no
  live-segmentation claim is exercisable here.

---

## Inherent method limitations (not bugs — they bound what can be claimed)

- **Serial sections are different physical slices** → population-/architecture-level
  co-localization only, **never single-cell co-expression**. A true CD8⁺TIM-3⁺
  single-cell claim requires multiplex/restain on one section.
- **Automated sub-5 µm TRE is unreliable** on FOV-crop serial sections (patch-flow
  aliases on a 30 µm shift; §18.4) → manual landmarks are required, which is why
  certification is operator-driven.
- **The reweighted test is anti-conservative** when architecture approaches the
  bandwidth (§15.5) — now disclosed and stamped, but see #3.
- **Public certification is largely blocked**: HyReCo 233 GB+login; cross-K blocked by
  image access; no public registrable CD8+TIM-3 set (§20).

---

## Suggested order of attack

1. **P0-1** (registrable/restained data → first CERTIFIED pair) and **P0-2** (real
   quantification agreement number) — the only two things between here and an external
   claim.
2. **P1-3** (measure architecture scale → real guard) and **P1-4** (decide the Restained
   correspondence policy) — close the two "disclosed but not enforced" gaps.
3. **P2-6** doc sweep, then **P2-7** Stage 4 UX, then **P3** reproducibility debt.

---

*Reconciled against source on 2026-06-21. No production code, config, or thresholds were
changed in producing this file.*
