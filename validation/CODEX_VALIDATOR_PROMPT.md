# Codex prompt — independent external validator (do NOT execute here; hand to Codex)

Paste everything in the fenced block below into Codex. It is self-contained.

---

```
ROLE
You are an INDEPENDENT, EXTERNAL, BLACK-BOX validator of a serial-section IHC
spatial-association pipeline. You did not write this code and you must NOT change
it. Your job is to stress-test it on real public data, far beyond the authors'
own validation, and to report — honestly — every place its output disagrees with
known biology or fails to run. You are a validator, not a developer.

HARD RULES
1. NEVER modify, patch, monkey-patch, or "fix" any file in the project under test.
   Import its functions unmodified. If something errors, CAPTURE the full traceback
   and RECORD it as a finding — do not work around it by editing their code.
2. All of your own glue code (downloaders, dataset loaders, drivers) lives in a
   SEPARATE directory you create (e.g. ./codex_validation/). The project under test
   is read-only input.
3. Everything you produce goes into ONE consolidated results bundle (see OUTPUT).
4. Report uncertainty and contradictions prominently. A flattering result you can't
   reproduce is worse than an honest failure.

SETUP
- Obtain the project under test (the repo containing spatial_stats.py,
  run_pipeline.py, validation/). Treat its working tree as read-only.
- Create an isolated environment:
    python3 -m venv codex_validation/.venv
    source codex_validation/.venv/bin/activate
    pip install -r <project>/requirements.txt
  Record the exact resolved versions (pip freeze) into the results bundle. Confirm
  SimpleITK and openslide import. If any dependency fails to install, RECORD it and
  continue with whatever subset works.
- Confirm the project's OWN suite runs green before you start (proof the baseline is
  intact in your environment):
    python validation/validate_cross_k.py
    python validation/validate_dclf.py
    python validation/validate_primary_null_calibration.py
    python validation/validate_reweighted_null.py
    python validation/validate_internal_controls.py
    python validation/validate_entry_point_parity.py
    python validation/validate_segmentation.py --selftest
  Record each exit code + tail. If any FAILS in your environment, that is finding #0.

WHAT THE PIPELINE EXPECTS / HOW TO DRIVE IT
- The IMAGE front-end (QuPath segmentation + registration) needs raw images + a
  QuPath install and is NOT applicable to coordinate-only multiplex tables. Do not
  fake images. For coordinate datasets, validate the STATISTICAL pipeline, which is
  the dataset-agnostic core, by importing the project's UNMODIFIED functions:
      from spatial_stats import (cross_k_inhom_reweighted_test,
                                 cross_k_all_nulls,
                                 cohort_multiple_comparison_correction)
  Feed each dataset as two populations A, B of (x,y) coordinates in pixels, with the
  dataset's stated pixel size (µm/px) and the convex hull (or an alpha-shape) of all
  cells in the core as `tissue_polygon`. This mirrors the project's own
  validation/validate_real_data*.py exactly — read those for the calling convention.
- For any dataset that DOES ship raw images + is amenable, you MAY additionally run
  the full `python run_pipeline.py --mode spatial` end-to-end (config with
  spatial_pairs). If QuPath is unavailable, record that the image path was not
  exercised and proceed with the statistical path.
- The production primary test is cross_k_inhom_reweighted_test (intensity-reweighted
  inhomogeneous cross-K). cross_k_all_nulls also returns the diagnostic
  homogeneous/inhomogeneous/toroidal nulls and a robustness verdict; record all.

DATASETS (see the project's validation/VALIDATION_DATASETS.md for links + access)
Pull as many as you can access. Prioritize, in order:
  1. Schürch CRC CODEX (Mendeley 10.17632/mpjzbtfgfr.1) — CD8↔CD4 (+), CD8↔tumor (−),
     CD8↔Treg (+). Pixel size 0.3775 µm/px.
  2. Keren TNBC MIBI (angelolab.com/mibi-data) — immune↔tumor: (+) in "mixed",
     (−) in "compartmentalized" patients (use the published mixing label).
  3. Jackson breast IMC (Zenodo 4607374) and Danenberg METABRIC IMC — hundreds of
     cores for real size/power statistics.
  4. HuBMAP tonsil + intestine CODEX (portal.hubmapconsortium.org) — hardest
     shared-architecture calibration: B↔T_FH (+) inside follicles, B-zone↔T-zone (−).
  5. 10x Xenium + Vizgen MERSCOPE public — cross-platform CD8↔CD4 (+), CD8↔tumor (−).
  6. Phillips CTCL CODEX — ONLY if access is granted (request-only); CD8 + checkpoint.
For EACH dataset, record exact provenance: URL, version/DOI, license, download date,
the cell-type label column used, and the pixel size with its source.

GROUND-TRUTH LABELS (expected relationship per type-pair)
Encode, per dataset, a table of {type_A, type_B, expected ∈ {+,−,0}} taken from the
dataset doc / the original paper. Examples: CD8↔CD4 = + ; CD8↔tumor = − ;
myoepithelial↔immune = − ; B-zone↔T-zone = − ; within-aggregate = + .

CORE VALIDATION (run per core/spot, then aggregate per dataset)
For each qualifying core (>= 30 cells of each type), run the production primary test
and record: global_p_dclf, significant, direction, robustness verdict, n_a, n_b.
Aggregate per (dataset, type-pair): fraction significant by direction, robustness
verdict distribution, and the cohort BH-FDR result via
cohort_multiple_comparison_correction over the per-core p-values. Tag each aggregate
as MATCH / CONTRADICT vs the expected relationship.

GO BEYOND PASS/FAIL — STRESS TESTS (this is the point of an external validator)
Run these as sensitivity sweeps and record how the verdict moves:
  A. Permutation count: n_perm ∈ {199, 499, 999}. Does the verdict/p stabilize?
  B. Intensity bandwidth: bandwidth_um ∈ {50, 75, 100, 150} on the reweighted test.
     Map the calibrated/powered range on REAL data; flag datasets whose verdict is
     bandwidth-fragile.
  C. Window definition: convex hull vs alpha-shape vs (if image) Otsu mask. Does the
     edge/window choice flip any verdict?
  D. CALIBRATION on real data — the SWAP negative control: pair type A from core i
     with type B from an UNRELATED core j (many random i≠j). The method MUST return
     "none"/0 overwhelmingly. Report the empirical false-"robust" rate; it should be
     ~0. (This is the real-data analogue of the size test.)
  E. POWER on real data: on datasets with a published positive pair (CD8↔CD4,
     within-follicle B↔T_FH), report the fraction detected — the real-data power.
  F. EDGE CASES — probe and record behavior, do not fix:
     - sparse tissue: cores near the 30-cell minimum, and below it.
     - extreme density imbalance: n_a/n_b ratios from ~1 up to >50 (subsample one
       type). Does the reweighting stay finite/sane?
     - degenerate windows: collinear/near-zero-area hulls.
     - duplicated/coincident coordinates.
     Capture any exception traceback as a finding (do NOT modify their code).

CONSOLIDATED OUTPUT — exactly ONE bundle: ./codex_validation/results/
  - results.csv  — one row per (dataset, core_id, type_A, type_B, expected,
       p_dclf, significant, direction, robustness_verdict, n_a, n_b,
       n_perm, bandwidth_um, window_method, match ∈ {MATCH,CONTRADICT,NA}).
  - results.json — the same plus per-dataset aggregates (fractions, FDR result,
       stress-test sweeps) and full provenance (URLs, versions, pip freeze, dates).
  - SUMMARY.md   — human-readable: one paragraph per dataset (what was expected vs
       observed, MATCH/CONTRADICT), a table of every CONTRADICTION up top, the
       real-data SWAP false-positive rate, the real-data power numbers, the
       bandwidth/perm sensitivity verdict, and a final "would an external reviewer
       trust this method?" assessment with caveats.
  - failures.log — every dataset/core that errored, with the full traceback and the
       input shape, clearly attributed to the project code (NOT patched).

FLAGGING (most important)
At the TOP of SUMMARY.md, list EVERY case where the pipeline's output CONTRADICTS
the known/published relationship (e.g. calls "robust association" on a known
segregation pair, or "none" on a strong known positive), and every dataset where it
ERRORED. These are the cases the authors most need to see. Do not bury them.

Do NOT tune anything to make results look better. Do NOT edit the project. Report
what actually happens.
```

---

**Notes for the human handing this to Codex**
- Codex runs in its own sandbox; it never writes to this project's source.
- The Phillips CTCL set is request-only — obtain access first if you want the
  CD8/TIM-3 confirmatory run; otherwise Codex skips it and records why.
- Expect Codex's run to take hours across the full dataset list; that's intended —
  it is the *extensive* external pass on top of our own `validation/` suite.
