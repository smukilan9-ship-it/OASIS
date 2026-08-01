"""
registry.py — the single source of truth for every OASIS validation.

Each record documents the scientific claim, why it matters, the dataset(s) and
external tools it needs, its assumptions and limitations, how to read the result,
and how to run it. The CLI runner (validation/run.py) and the desktop Validation
tab both render from this list, so the docs a reviewer reads and the thing that
actually executes can never drift apart.

Categories mirror the real OASIS pipeline stages:
  statistical -> registration -> segmentation -> quantification ->
  spatial_association -> end_to_end

runner kinds:
  {"kind": "script",  "script": "<file in validation/>", "argv": [...]}
  {"kind": "pytest",  "node": "<pytest node id>"}
runtime_tier:  instant | short | long
external_deps: subset of {"qupath", "instanseg", "R"}  (preflight-checked)
datasets:      dataset names from validation/datasets/datasets.yaml (may be empty)
"""
from __future__ import annotations

CATEGORIES = [
    ("statistical",         "Statistical Validation"),
    ("registration",        "Registration Validation"),
    ("segmentation",        "Segmentation Validation"),
    ("quantification",      "Quantification Validation"),
    ("spatial_association", "Spatial Association Validation"),
    ("end_to_end",          "End-to-End Validation"),
]

VALIDATIONS = [
    # ── Statistical ──────────────────────────────────────────────────────────
    {
        "id": "cross_k",
        "title": "Cross-type Ripley's K estimator",
        "category": "statistical",
        "claim": "The cross-type K estimator is computed correctly.",
        "purpose": "Check the cKDTree cross-K against exact brute-force pair counting "
                   "and known analytic limits on fixed synthetic point patterns.",
        "why": "Every spatial-association verdict is a function of this estimator; if K "
               "is wrong, every downstream p-value and verdict is wrong.",
        "datasets": [],
        "assumptions": "Points in a rectangular window; pixel size fixed at 1 for the check.",
        "limitations": "Reference-free (no R): validates internal consistency + analytic "
                       "limits, not an external reference implementation (see spatstat_crossval).",
        "interpretation": "PASS = estimator matches brute force to ~float epsilon.",
        "expected": "All checks PASS (max error ~1e-12).",
        "runner": {"kind": "script", "script": "validate_cross_k.py"},
        "runtime_tier": "instant", "external_deps": [],
    },
    {
        "id": "radius_floor",
        "title": "Registration error — size, power, and the radius floor",
        "category": "statistical",
        "claim": "Residual registration error costs the cross-K test power, never validity, "
                 "so a deformed serial-section pair may be analysed; its error sets the "
                 "smallest resolvable inter-cell distance, not permission to run.",
        "purpose": "Displace B by Gaussian error ε (clipped to the analysis window, as the "
                   "pipeline does) and measure (A) the false-positive rate under "
                   "independence, (B) detection of a weak true association, and (C) whether "
                   "raising the DCLF band floor to k·ε improves power.",
        "why": "The ≤5 µm landmark-certification gate withholds the whole spatial analysis "
               "from a pair with ~2 µm RMS of pervasive elastic deformation — which every "
               "serial section has. If error cannot manufacture a finding, that gate "
               "withholds a valid conservative result rather than preventing a wrong one. "
               "This is the evidence behind the RADIUS_LIMITED verdict.",
        "datasets": [],
        "assumptions": "Landmark-driven (cell-blind) transform; Gaussian isotropic error; "
                       "points displaced outside the analysis window are dropped, as "
                       "run_spatial_association does. Homogeneous-CSR null, rectangular window.",
        "limitations": "Synthetic patterns, homogeneous null only. Says nothing about "
                       "INTENSITY-driven non-rigid warps, which optimise on a signal "
                       "correlated with cell density and COULD manufacture association.",
        "interpretation": "PASS = size stays ≈α at every ε (error cannot invent a finding); "
                          "power declines gracefully; band clipping does not help, so the "
                          "radius floor is a reporting boundary, not a gate on the statistic.",
        "expected": "Size ≈0.00–0.05 for ε up to 20 µm; power ~0.5 → ~0.3; clipping never "
                    "raises power.",
        "runner": {"kind": "script", "script": "validate_radius_floor.py"},
        "runtime_tier": "long", "external_deps": [],
    },
    {
        "id": "deformation_estimator",
        "title": "Patch-flow deformation estimator — negative result, and its containment",
        "category": "registration",
        "claim": "serial_registration.measure_deformation cannot measure tissue deformation "
                 "and must never gate a certification verdict.",
        "purpose": "Run measure_deformation on a real LL477 CD8/TIM-3 pair under (a) the "
                   "certified landmark similarity, (b) an identity transform leaving the "
                   "sections ~106 µm apart, and (c) a known uniform translation. Then "
                   "inject a fabricated deformation dict into landmark_register_and_verify "
                   "and confirm the verdict does not move.",
        "why": "A cell-level certification statistic, cell_registration_error = "
               "sqrt(estimation² + model²), was proposed to escape the σ floor of "
               "leave-one-out landmark TRE. Its model term needs an independent, "
               "image-based deformation measurement. This is the evidence that the one we "
               "had does not work, so the statistic was NOT adopted.",
        # Resolved by the script itself from ~/Desktop/assets; it prints SKIP when absent.
        "datasets": [],
        "assumptions": "Real H-DAB liver serial sections (LL477 CD8 / TIM-3, 0.7519 µm/px); "
                       "structural_channel blurred at "
                       "σ≈12 µm; 128 px patches, Hann-windowed cv2.phaseCorrelate.",
        "limitations": "Demonstrates blindness on one real pair and by mechanism (the blur "
                       "removes the high-frequency content a displacement estimator needs). "
                       "It does not prove NO image-based deformation estimator can work — "
                       "only that phase-correlation patch flow, NCC template matching and "
                       "gradient-magnitude phase correlation on this channel all fail, as "
                       "does the tol-censored lumen_tre.",
        "interpretation": "PASS = the estimator still reads ≈0 for an unregistered pair "
                          "(i.e. it is still blind, as documented) AND no supplied "
                          "deformation dict changes the verdict or the accuracy basis. "
                          "FAIL means someone re-wired it into certification, or replaced "
                          "it with something that works — either way, re-derive before use.",
        "expected": "certified ≈0.14 µm, identity ≈0.22 µm, 48.8 µm shift ≈0.18 µm; verdict "
                    "RADIUS_LIMITED with basis leave_one_out_landmark_tre in both arms.",
        "runner": {"kind": "script", "script": "validate_deformation_estimator.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "dclf",
        "title": "DCLF global test — calibration & power",
        "category": "statistical",
        "claim": "The DCLF envelope test has correct size and real power.",
        "purpose": "Under CSR the global p is ~Uniform(0,1) (≈5% false positives); under "
                   "genuine association it detects with high power and correct direction.",
        "why": "The DCLF p-value is the significance statement OASIS reports; it must not "
               "over-reject under randomness nor miss real association.",
        "datasets": [],
        "assumptions": "Independent A/B under the null; band-limited to 10–50 µm.",
        "limitations": "Monte-Carlo calibration at finite n_perm; synthetic patterns only.",
        "interpretation": "PASS = uniform p under CSR AND high power with correct direction.",
        "expected": "Both calibration and power/direction PASS.",
        "runner": {"kind": "script", "script": "validate_dclf.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "null_models",
        "title": "Null models — shared-preference discrimination",
        "category": "statistical",
        "claim": "The reweighted inhomogeneous null separates cell-scale engagement from "
                 "mere shared tissue preference; homogeneous CSR does not.",
        "purpose": "Run constructed patterns (shared preference, true engagement, "
                   "independence) through cross_k_all_nulls and check the verdicts.",
        "why": "The whole 'co-infiltration vs engagement' framing depends on the "
               "reweighted null not calling shared preference 'robust'.",
        "datasets": [],
        "assumptions": "Architecture scale coarser than the 10–50 µm interaction band.",
        "limitations": "Synthetic architecture; the real per-image architecture scale is "
                       "not measured (see reweighted_null caveat).",
        "interpretation": "PASS = all scenarios produce their correct verdict.",
        "expected": "All scenarios PASS.",
        "runner": {"kind": "script", "script": "validate_null_models.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "edge_correction",
        "title": "Edge-correction cancellation",
        "category": "statistical",
        "claim": "Omitting analytic edge correction is valid because the boundary bias "
                 "cancels between observed and null (both use the same uncorrected estimator).",
        "purpose": "A/B the estimator with and without a translation edge correction at "
                   "fixed seeds and show identical calibration.",
        "why": "Justifies the deliberate 'no edge correction' design decision (research/ihc.md §17).",
        "datasets": [],
        "assumptions": "Same window/estimator for observed and null.",
        "limitations": "Demonstrated on the translation correction; not every edge scheme.",
        "interpretation": "PASS = calibration identical to reported decimal places.",
        "expected": "Corrected vs uncorrected calibration match.",
        "runner": {"kind": "script", "script": "validate_edge_correction.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "primary_null_calibration",
        "title": "Primary null — calibration under shared preference",
        "category": "statistical",
        "claim": "The production primary null holds its false-positive rate under a "
                 "realistic shared-preference null hypothesis.",
        "purpose": "Decisive calibration of the inhomogeneous/reweighted primary against "
                   "the shared-preference regime that fools homogeneous CSR.",
        "why": "The headline 'robust association' comes from this null, not CSR.",
        "datasets": [],
        "assumptions": "Bandwidth (75 µm) exceeds the tissue architecture scale.",
        "limitations": "Mildly anti-conservative near the bandwidth; calibrated at a single "
                       "bandwidth (disclosed in research/ihc.md §15.5).",
        "interpretation": "PASS = shared-preference false-positive rate within tolerance.",
        "expected": "Shared-preference rate ~0.03 at bw=75 µm.",
        "runner": {"kind": "script", "script": "validate_primary_null_calibration.py"},
        "runtime_tier": "long", "external_deps": [],
    },
    {
        "id": "reweighted_null",
        "title": "Reweighted inhomogeneous null — 3-regime proof",
        "category": "statistical",
        "claim": "The reweighted null passes shared-preference H0, uniform H0, and "
                 "engagement power simultaneously at the shipped bandwidth.",
        "purpose": "Monte-Carlo rate calibration across regimes (the ship/no-ship gate).",
        "why": "This is the operating-characteristics evidence for the primary statistic.",
        "datasets": [],
        "assumptions": "Architecture > bandwidth; fixed seeds.",
        "limitations": "Long-running Monte-Carlo; single-bandwidth window is a knife-edge.",
        "interpretation": "PASS = all three regimes within their rate tolerances.",
        "expected": "SHIP verdict at bw=75 µm.",
        "runner": {"kind": "script", "script": "validate_reweighted_null.py"},
        "runtime_tier": "long", "external_deps": [],
    },
    {
        "id": "architecture_scale",
        "title": "Architecture-scale operating characteristics & gate",
        "category": "statistical",
        "claim": "The reweighted null is size-controlled only when tissue architecture is "
                 "coarser than the bandwidth; the runtime gate enforces this.",
        "purpose": "Monte-Carlo size/power of the reweighted test vs the measured "
                   "architecture scale ℓ̂, and validation of the ℓ̂ estimator + the "
                   "≥2×bandwidth validity gate.",
        "why": "Directly answers the top statistical reviewer objection (audit A6): the "
               "75 µm bandwidth assumption was disclosed but unmeasured — this turns it "
               "into a measured, calibrated guard against false 'robust' verdicts.",
        "datasets": [],
        "assumptions": "Log-Gaussian architecture; engagement planted in the 10–50 µm band; "
                       "anti-conservativeness depends on intensity contrast.",
        "limitations": "Derived threshold is contrast-dependent; gate (2×bandwidth) is a "
                       "deliberately conservative default, re-derive at paper-grade sims.",
        "interpretation": "type-I should fall below α as ℓ̂ grows; the gate flags fields "
                          "whose ℓ̂ is too small to trust a 'robust' call.",
        "expected": "Anti-conservative below bandwidth; size-controlled + powered above the "
                    "derived threshold (~2×bandwidth); estimator monotonic.",
        "runner": {"kind": "script", "script": "validate_architecture_scale.py"},
        "runtime_tier": "long", "external_deps": [],
    },
    {
        "id": "public_codex_dense_null",
        "title": "Dense morphology-conditioned null — public CODEX calibration",
        "category": "statistical",
        "claim": "A candidate dense-tissue null can control false positives on real "
                 "CRC tissue architecture templates without using biological marker "
                 "pairs as ground truth.",
        "purpose": "Use Schürch CRC CODEX cell coordinates as real dense architecture, "
                   "simulate known independent and planted-associated marker pairs, "
                   "and calibrate morphology-conditioned dense-null candidates.",
        "why": "Dense LL477 fields fail the 75 µm architecture gate; this tests whether "
               "a marker-independent total-cell morphology field is a plausible next "
               "primary null instead of shipping an uncalibrated 35-45 µm bandwidth.",
        "datasets": ["codex_crc"],
        "assumptions": "CODEX coordinates approximate total-cell architecture; simulated "
                       "A/B populations are known-truth null/positive controls; this is "
                       "coordinate-level calibration, not image segmentation validation.",
        "limitations": "Does not validate H-DAB/hematoxylin morphology extraction by "
                       "itself; production dense fallback also requires the rendered-"
                       "image bridge, real serial-section demonstration, and runtime "
                       "gates.",
        "interpretation": "PASS-like result = candidate worth pursuing; any H0 over-rejection "
                          "means do not use. Current focused result supports the shipped "
                          "gated fallback: 10-30 µm, 2 µm total-cell support jitter.",
        "expected": "Homogeneous CSR over-rejects; the candidate controls H0 near 5% and "
                    "retains planted-positive power.",
        "runner": {"kind": "script", "script": "validate_public_codex_dense_null.py"},
        "runtime_tier": "long", "external_deps": [],
    },
    {
        "id": "dense_null_image_morphology",
        "title": "Dense null — rendered H-DAB morphology extraction",
        "category": "statistical",
        "claim": "The dense morphology-conditioned candidate remains calibrated when "
                 "the marker-independent morphology field is recovered from rendered "
                 "H-DAB-like hematoxylin pixels rather than oracle coordinates.",
        "purpose": "Bridge public CODEX coordinate calibration to image-derived OASIS "
                   "morphology extraction by rendering real cell architectures, detecting "
                   "nuclei from hematoxylin pixels, and re-running known-null/planted "
                   "dense-null calibration.",
        "why": "The public CODEX coordinate null is not enough by itself; production needs "
               "lambda_M(x) from images. This checks that the image-derived morphology "
               "field is not the failure point before real LL477 validation/runtime gates.",
        "datasets": ["codex_crc"],
        "assumptions": "Rendered H-DAB-like nuclei are a controlled bridge, not real DAB "
                       "serial-section images. Marker truth is simulated over real CODEX "
                       "architecture.",
        "limitations": "Does not validate real LL477 H-DAB staining or section artifacts "
                       "by itself; those are covered by the real LL477 demonstration and "
                       "runtime certification/ROI gates.",
        "interpretation": "Current focused result: 10-30 µm / 2 µm image-derived nuclei "
                          "morphology passes screen (worst H0 0.063, power 1.0, median "
                          "field correlation 0.939). Real serial-section validation remains.",
        "expected": "Image-derived morphology controls H0 near 5%, preserves planted-positive "
                    "power, and recovers the coordinate morphology field.",
        "runner": {"kind": "script", "script": "validate_dense_null_image_derived_morphology.py"},
        "runtime_tier": "long", "external_deps": [],
    },
    {
        "id": "dense_null_real_ll477",
        "title": "Dense null — real LL477 serial-section demonstration",
        "category": "spatial_association",
        "claim": "The dense morphology-conditioned candidate can run on completed real "
                 "LL477 OASIS serial-section bundles with sparse-pair exclusion.",
        "purpose": "Apply the 10-30 µm / 2 µm support-jitter candidate to certified real "
                   "CD8/TIM-3 H-DAB pairs using OASIS all-cell detections as the "
                   "marker-independent morphology support.",
        "why": "Public CODEX and rendered-image calibration are necessary but not enough; "
               "the candidate also has to behave on the user's actual serial-section data.",
        "datasets": [],
        "assumptions": "Reads completed local LL477 result bundles under Desktop; not a "
                       "known-null calibration because LL477 biology is not ground truth.",
        "limitations": "Only two usable pairs; one sparse pair skipped. Significant calls "
                       "are real-use demonstrations, not publication-grade biological proof.",
        "interpretation": "Current result: x10_1 p=0.007, x10_3 p=0.024 under the 10-30 µm "
                          "dense candidate; x10_2 skipped for only 10 TIM-3 positives.",
        "expected": "Usable certified pairs run; sparse pair is skipped; dense mode remains "
                    "gated by certification, support count, ROI/window, and provenance.",
        "runner": {"kind": "script", "script": "validate_dense_null_real_ll477.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "dense_scaffold_keren_external",
        "title": "Dense scaffold — Keren external-support check",
        "category": "spatial_association",
        "claim": "For three dense Keren TNBC pseudo-IHC fields, the dense "
                 "morphology-conditioned verdict is not driven merely by using "
                 "OASIS's own all-cell scaffold.",
        "purpose": "Compare the UI-path dense-null result using OASIS all-cell "
                   "support against the same positives/window using an external "
                   "Keren mask-derived support scaffold.",
        "why": "This directly attacks the dense-scaffold circularity objection: "
               "if the null support is extracted by OASIS itself, a reviewer can "
               "ask whether the support scaffold manufactured the dense verdict.",
        "datasets": [],
        "assumptions": "Requires the local Keren pilot artifact folder "
                       "`~/Desktop/OASIS_keren_tnbc_validation` or "
                       "`OASIS_KEREN_TNBC_VALIDATION_DIR`; same-section MIBI was "
                       "rendered into pseudo-IHC fields and run through the same "
                       "Spatial backend used by the UI.",
        "limitations": "Only three FOVs; same-section multiplex-derived pseudo-IHC, "
                       "not serial-section registration; checks a completed pilot "
                       "artifact rather than re-downloading/re-rendering the raw "
                       "4+ GB dataset.",
        "interpretation": "PASS = OASIS-scaffold and external-scaffold dense-null "
                          "calls agree in significance, direction, and robust verdict "
                          "for p13/p16/p32.",
        "expected": "All three fields remain robust segregation under the external "
                    "Keren scaffold (p13 p=0.001, p16 p=0.001, p32 p≈0.028).",
        "runner": {"kind": "script", "script": "validate_dense_scaffold_keren_external.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "dense_scaffold_perturbation",
        "title": "Dense scaffold — perturbation sensitivity harness",
        "category": "spatial_association",
        "claim": "The dense-null scaffold sensitivity harness can distinguish strong "
                 "dense verdicts from borderline scaffold-dependent ones.",
        "purpose": "Keep the same Keren CD8/PanCK positives and replace only the "
                   "dense-null all-cell support scaffold using thinning, "
                   "density-biased deletion, local dropout, and centroid jitter.",
        "why": "A dense-null paper cannot just show one scaffold works; it must show "
               "whether the verdict survives plausible support-scaffold errors and "
               "must flag calls that do not survive.",
        "datasets": [],
        "assumptions": "Requires the local Keren pilot artifact folder "
                       "`~/Desktop/OASIS_keren_tnbc_validation` or "
                       "`OASIS_KEREN_TNBC_VALIDATION_DIR`; perturbations test the "
                       "null scaffold only, not marker segmentation or registration.",
        "limitations": "Still a three-field pilot. p32 is intentionally not a clean "
                       "success: it is a cautionary borderline field showing why "
                       "scaffold-sensitivity reporting is mandatory.",
        "interpretation": "PASS = p13/p16 remain stable under all 33 perturbations, "
                          "and p32 is explicitly exposed as scaffold-sensitive rather "
                          "than silently overclaimed.",
        "expected": "p13 and p16: 33/33 stable and significant. p32: 21/33 stable, "
                    "22/33 significant, 1 fail-closed support-gate case.",
        "runner": {"kind": "script", "script": "validate_dense_scaffold_perturbation.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "internal_controls",
        "title": "Internal negative/positive controls",
        "category": "statistical",
        "claim": "Swapped-section (unrelated tissue) shows no association; a planted "
                 "engaged partner does — through the production analysis path.",
        "purpose": "End-to-analysis controls on the real cross_k_all_nulls entry point.",
        "why": "Guards against a pipeline that reports association on unrelated inputs.",
        "datasets": [],
        "assumptions": "Controls constructed to have known ground truth.",
        "limitations": "Constructed inputs, not real serial sections.",
        "interpretation": "PASS = negative control n.s., positive control robust.",
        "expected": "All controls PASS.",
        "runner": {"kind": "script", "script": "validate_internal_controls.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "stabilization_gates",
        "title": "Fail-closed gates & provenance stamps",
        "category": "statistical",
        "claim": "Registration certification is fail-closed, provenance is complete, and "
                 "the honesty gates fire (no uncertified pair reads as certified).",
        "purpose": "Regression test for the certification stamp, provenance fields "
                   "(reweight bandwidth, null seed, architecture assumption), and cohort FDR.",
        "why": "The product's honesty discipline must be enforced in code, not just docs.",
        "datasets": [],
        "assumptions": "Runs before segmentation (no QuPath needed).",
        "limitations": "Checks the gates fire; does not exercise live registration.",
        "interpretation": "PASS = every gate fires as specified.",
        "expected": "All stabilization-gate checks PASS.",
        "runner": {"kind": "script", "script": "validate_stabilization_gates.py"},
        "runtime_tier": "instant", "external_deps": [],
    },

    # ── Registration ─────────────────────────────────────────────────────────
    {
        "id": "registration_qc",
        "title": "Registration QC gate (fail-closed)",
        "category": "registration",
        "claim": "The automated registration QC marks bad alignments invalid and greys the stats.",
        "purpose": "Drive the real compute_registration + compute_registration_qc on "
                   "synthetic H-DAB-like pairs across good/bad alignment cases.",
        "why": "A wrong alignment fabricates the inter-cell distances K consumes; the gate "
               "must refuse those pairs.",
        "datasets": [],
        "assumptions": "Synthetic pairs approximate the QC's decision surface.",
        "limitations": "Automated QC is NOT the §18–20 landmark certification (weaker; "
                       "known-unreliable on FOV-crop serial sections).",
        "interpretation": "PASS = identity/high-residual cases flagged invalid.",
        "expected": "All QC cases PASS (gate fires).",
        "runner": {"kind": "script", "script": "validate_registration_qc.py"},
        "runtime_tier": "instant", "external_deps": [],
    },
    {
        "id": "anhir_landmarks",
        "title": "Landmark TRE vs ANHIR/CIMA experts",
        "category": "registration",
        "claim": "Distance-preserving landmark registration reproduces expert alignment, "
                 "with honest CERTIFIED/LOCALLY_CERTIFIED/DEFORMED/NOT_CERTIFIABLE verdicts.",
        "purpose": "Run landmark_register_and_verify against expert corresponding landmarks.",
        "why": "Validates the registration itself (held-out TRE), not just the downstream stat.",
        "datasets": ["cima_landmarks"],
        "assumptions": "Similarity transform only; held-out landmark TRE ≤ 5 µm to certify.",
        "limitations": "Consecutive sections are a hard case; single-annotator landmark sets.",
        "interpretation": "Read the verdict + held-out TRE per pair.",
        "expected": "Best real pair LOCALLY_CERTIFIED (lung-lesion_1 ~3.66 µm ROI).",
        "runner": {"kind": "script", "script": "validate_anhir_landmarks.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "local_smoothness_filter",
        "title": "Local-smoothness correspondence filter — expert-landmark A/B",
        "category": "registration",
        "claim": "Rejecting LoFTR matches that disagree with their neighbours' displacement "
                 "moves expert-annotated anatomy CLOSER to where the expert put it — so the "
                 "filter removes wrong correspondences, it does not merely flatter its own "
                 "residual.",
        "purpose": "Paired A/B over ANHIR training pairs: fit a similarity from "
                   "loftr_correspondences with local_k=0 (cycle+scale only) and with "
                   "local_k=8, then measure realized TRE at the expert landmarks under each.",
        "why": "The filter SELECTS the points its own residual is measured on, so a smaller "
               "residual is guaranteed and proves nothing. Expert landmarks are the only "
               "quantity it cannot influence — LoFTR never sees them, so the entire set is "
               "held out from both arms and the comparison is like-for-like.",
        "datasets": [],
        "assumptions": "Similarity fit; both arms run at an identical working resolution on "
                       "identical images, so the paired delta isolates the filter.",
        "limitations": "Needs the ANHIR medium-size IMAGES, which are CC BY-NC-SA and "
                       "therefore not bundled with the landmark dataset — point ROOT at a "
                       "local copy. Reported in working pixels, not µm: a paired A/B on "
                       "identical images does not need the per-tissue µm/px, and asserting "
                       "an unverified one would be exactly the bookkeeping error the sibling "
                       "ANHIR harness guards against. ANHIR is H&E/IHC serial histology, not "
                       "CD8/TIM-3 H-DAB; it bounds the filter's behaviour, not its transfer.",
        "interpretation": "Do NOT read the aggregate as an accuracy claim — the filter is a "
                          "no-op wherever the correspondences are already clean, so a large "
                          "overall shift would be the surprising result. Read instead: the "
                          "split by whether arm A was contaminated, and the ASYMMETRY among "
                          "the pairs that move. A filter discarding correct correspondences "
                          "would show symmetric movement, or regressions as large as its "
                          "gains.",
        "expected": "MEASURED, 44 training pairs: no overall shift (Wilcoxon p=0.93; 34/44 "
                    "move by ≤0.5 px). Split by contamination — where arm A was already "
                    "<10 px (n=35) mean Δ −0.04 px, i.e. a no-op; where arm A was ≥10 px "
                    "(n=9) mean Δ −6.97 px, median −1.60 px. Among the 10 pairs that move at "
                    "all: 8 improve, 2 worsen, best −41.7 px vs worst +2.4 px. Mean cull "
                    "9.7%. The worst regression (mice-kidney_1 9_PAS→6_CD31, 6.2→8.6 px at a "
                    "22% cull) is the known cost: on cross-stain pairs the local-continuity "
                    "assumption can discard correct matches.",
        "runner": {"kind": "script", "script": "validate_local_smoothness_anhir.py",
                   "argv": ["--limit", "40"]},
        "runtime_tier": "long", "external_deps": [],
    },
    {
        "id": "phase_b_certified",
        "title": "Phase-B: analysis only on certified ROIs",
        "category": "registration",
        "claim": "Spatial analysis runs only where registration is certified; uncertified "
                 "pairs are refused, not warped.",
        "purpose": "Exercise the certified-ROI gating of the spatial path on CIMA data.",
        "why": "Closes the loop between certification and what the pipeline will analyse.",
        "datasets": ["cima_landmarks", "codex_crc"],
        "assumptions": "Certification thresholds as in research/ihc.md §3.5.",
        "limitations": "Few real certifiable pairs exist publicly.",
        "interpretation": "PASS = only certified ROIs are analysed.",
        "expected": "Certified ROI analysed; others refused.",
        "runner": {"kind": "script", "script": "validate_phase_b_certified.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "loftr_fle_groundtruth",
        "title": "LoFTR fiducial localisation error against a known warp",
        "category": "registration",
        "claim": "The FLE that enters σ_fit² = 2·FLE² + deformation² is measured against "
                 "ground truth, not inferred from the matcher's repeatability.",
        "purpose": "Warp one real H-DAB section by a KNOWN transform (translation, rotation, "
                   "scale, smooth B-spline) and match it against itself. Because the "
                   "displacement is known analytically at every pixel, q − W(p) is the true "
                   "localisation error of each correspondence. Swept across working "
                   "resolutions (is FLE a tissue property or a matcher-grid one?) and across "
                   "appearance mismatches (different DAB density, counterstain, focus, "
                   "illumination, noise).",
        "why": "loftr_fle measures REPEATABILITY under image noise, i.e. precision, not "
               "accuracy — and a lower-bound FLE is mechanically an upper-bound deformation, "
               "so every certification is an upper bound of unknown tightness. This is the "
               "only measurement that says whether that bound is tight. See "
               "research/registration.md.",
        "datasets": [],
        "assumptions": "One real H-DAB field (LL477 at 0.7519 µm/px by default); the warp is "
                       "the ground truth, so no annotator and no second section are needed.",
        "limitations": "Matching a section against a resampled copy of itself cannot "
                       "reproduce the one thing that makes a real pair hard: two serial "
                       "sections are 4 µm apart and contain DIFFERENT CELLS. This measures "
                       "the matcher's floor, which bounds the real FLE from below.",
        "interpretation": "PASS covers the CONTROLS only (resample direction, identity warp "
                          "≈ 0, dense-field inversion). The FLE numbers are the measurement "
                          "and are reported, never asserted — a harness that asserts its own "
                          "answer cannot discover one.",
        "expected": "Controls pass; measured FLE ≈ 0.19–0.25 µm, i.e. the shipped 0.199 µm "
                    "is right and the 4.07 µm residual on the disputed ROI is NOT matcher "
                    "noise.",
        "runner": {"kind": "script", "script": "validate_loftr_fle_groundtruth.py"},
        "runtime_tier": "long", "external_deps": [],
    },
    {
        "id": "matchers_on_cohort",
        "title": "Which matcher works on LL477 H-DAB serial sections",
        "category": "registration",
        "claim": "LoFTR is the only one of five matchers that produces usable "
                 "correspondences on the target cohort; the alternatives either find nothing "
                 "or find mostly-wrong matches.",
        "purpose": "Two tests per pair. A: warp one real section by a known transform and "
                   "match it against itself, so accuracy and blunder rate are exact on this "
                   "tissue. B: the real CD8/TIM-3 cross-stain pair, scored on coverage and "
                   "residual contamination against a robust similarity.",
        "why": "The ANHIR matcher benchmark found DISK+LightGlue 6x cleaner than LoFTR and it "
               "does NOT transfer — on LL477 DISK matches 0-35 where LoFTR gets 77-632. A "
               "benchmark on the wrong tissue answers the wrong question, and the cohort "
               "every disputed number comes from is this one.",
        "datasets": [],
        "assumptions": "LL477 at 0.7519 um/px, downscaled to the 800 px working size "
                       "certification uses; gross = residual over 15 um.",
        "limitations": "Three pairs, one specimen. Test A cannot see cross-stain difficulty "
                       "by construction — that is exactly why test B decides.",
        "interpretation": "Read column B. A matcher that sails through A and dies in B "
                          "handles the texture but cannot bridge two stains, which is the "
                          "whole problem.",
        "expected": "Synthetic: every matcher except SIFT is sub-micron with 0 % gross. Real: "
                    "LoFTR 144/77/632 matches at 8-22 % gross; DISK 0/0/35; SIFT 0/0/0; "
                    "DeDoDe 633/130/3019 but 37-67 % gross; KeyNet ~90 at 100 % gross.",
        "runner": {"kind": "script", "script": "validate_matchers_on_cohort.py"},
        "runtime_tier": "long", "external_deps": [],
    },
    {
        "id": "matcher_blunders",
        "title": "Matcher blunder rate vs expert landmarks — LoFTR, DISK+LightGlue, SIFT",
        "category": "registration",
        "claim": "The correspondence engine's weakness on real serial sections is its BLUNDER "
                 "RATE, not its localisation accuracy, and a learned sparse matcher produces "
                 "roughly 6x fewer blunders at equal expert-landmark accuracy.",
        "purpose": "Hand every matcher the same hematoxylin+CLAHE input OASIS would give it, "
                   "fit a robust similarity from each one's correspondences, and measure "
                   "(a) the fraction of its own residuals exceeding 5x the median — blunder "
                   "contamination — and (b) realized TRE at expert landmarks no matcher saw.",
        "why": "validate_loftr_fle_groundtruth measured 0.22 um localisation error and ZERO "
               "gross errors, and that was used to argue no better matcher could help. It "
               "cannot support that: it matches a section against a resampled copy of itself, "
               "where blunders are nearly impossible by construction. On real LL477 regions "
               "the max/median residual ratio is 6.7-8.8 against ~2.5 for noise, and because "
               "the gate reads a p90 those few wrong matches set the reported cell error.",
        "datasets": [],
        "assumptions": "ANHIR training pairs at 1024 px; residual-based blunder definition "
                       "(>5x median) needs no ground truth and is matcher-agnostic.",
        "limitations": "20 pairs. ANHIR is largely cross-stain rather than serial H-DAB. "
                       "SIFT+LightGlue did not complete. The blunder metric is measured on "
                       "each arm's OWN residuals, so it detects contamination but cannot "
                       "distinguish a blunder from genuine local deformation — which is why "
                       "expert TRE is reported beside it.",
        "interpretation": "A matcher swap is a bigger change than any filter tweak; only a "
                          "large, consistent blunder advantage at no accuracy cost justifies "
                          "it. Read the pairs where a detector-based matcher COLLAPSES.",
        "expected": "DISK+LightGlue 0.34 % mean blunders vs LoFTR's 2.10 % (Wilcoxon "
                    "p = 0.011), equal expert TRE (p = 0.39), but under 40 matches on 4 of 19 "
                    "pairs where LoFTR still finds 47-230.",
        "runner": {"kind": "script", "script": "validate_matcher_blunders_anhir.py"},
        "runtime_tier": "long", "external_deps": [],
    },
    {
        "id": "scale_filter_tolerance",
        "title": "Scale-consistency tolerance — expert-landmark A/B",
        "category": "registration",
        "claim": "The scale filter's absolute tolerance is tighter than the coarse pass can "
                 "physically meet, and flooring it at half the coarse grid stride recovers "
                 "correspondences without costing expert-landmark accuracy.",
        "purpose": "Four arms over ANHIR training pairs — shipped absolute tolerance, filter "
                   "off, and stride-aware at 0.5x and 1.0x of the coarse stride — each "
                   "fitting a similarity from its own correspondences, then measuring "
                   "realized TRE at expert landmarks LoFTR never saw.",
        "why": "tol_px is an absolute distance while the coarse pass matches on an 8/scale = "
               "16 px grid, so the filter demands agreement to a quarter of its own cell. "
               "Measured on real 270 µm ROIs it removes 85–93 % of cycle-consistent matches, "
               "leaving 5 inside the polygon and a NO_MATCHES verdict on a region that had "
               "283 raw matches.",
        "datasets": [],
        "assumptions": "ANHIR training pairs at 1024 px long side; paired per-pair deltas "
                       "against the shipped arm; reported in working pixels.",
        "limitations": "A filter selects the points its own residual is measured on, which is "
                       "why only held-out expert landmarks can answer this. ANHIR is largely "
                       "cross-stain and mostly well-textured — its median pair already has "
                       "427 correspondences, far above the threshold — so the decisive "
                       "evidence is the n<200 subgroup (10 pairs), which is post-hoc even "
                       "though the mechanism predicts it.",
        "interpretation": "A change is justified only by more correspondences AND no "
                          "degradation, since relaxing a filter is the over-certifying "
                          "direction. Read the n<200 subgroup and the WORST-case paired "
                          "delta, not the median.",
        "expected": "Filter off: 4.2x correspondences but TRE degrades on hard pairs (mean "
                    "+8.85 px, worst +48). Stride-aware 0.5x: 1.64x correspondences, TRE "
                    "improves (mean −2.39 px), worst case bounded at +2.13 px. 1.0x is too "
                    "loose (worst +25 px).",
        "runner": {"kind": "script", "script": "validate_scale_filter_anhir.py"},
        "runtime_tier": "long", "external_deps": [],
    },
    {
        "id": "residual_origin",
        "title": "Origin of the certification residual — filter, noise, or deformation",
        "category": "registration",
        "claim": "The residual the gate books as deformation IS deformation: a spatially "
                 "continuous displacement field, not an artefact of the correspondence "
                 "filters and not random localisation error.",
        "purpose": "On the real LL477 CD8/TIM-3 pair, reproduce the disputed certification "
                   "and then (a) sweep the correspondence filters' admission tolerance "
                   "tol_um to test whether the residual is simply what we admit, and (b) "
                   "measure the spatial structure of the residual vectors by semivariogram "
                   "and Moran's I with a permutation null.",
        "why": "cell_error_budget charges the residual to a systematic cell displacement. "
               "That is correct only if the residual is spatially structured. If it were "
               "random, a similarity fitted from n correspondences would average it away "
               "(tre_pred_p90 = 0.091 µm at n=59) and the budget would double-count. The "
               "variogram nugget also yields FLE independently of any synthetic warp.",
        "datasets": [],
        "assumptions": "Local LL477 pair at 0.7519 µm/px; the ROI of research/registration.md "
                       "§ 9; residuals measured after the local similarity fit.",
        "limitations": "Post-fit residuals have had the global linear component absorbed, "
                       "which biases both tests AGAINST detecting structure — conservative "
                       "in the direction that matters. One pair.",
        "interpretation": "PASS covers the reproduction only (n=59, fit_residual 4.144, "
                          "landmark_noise 4.073); if that fails, nothing else is about the "
                          "disputed ROI. The structure measurements are reported, not "
                          "asserted.",
        "expected": "nugget/sill ≈ 0.02 (2 % random), Moran's I +0.60/+0.39 at p ≤ 0.001, "
                    "variogram FLE ≈ 0.41 µm and deformation ≈ 3.95 µm — i.e. the gate's "
                    "attribution is correct.",
        "runner": {"kind": "script", "script": "validate_residual_origin.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "spatstat_crossval",
        "title": "Cross-K agreement with R spatstat",
        "category": "registration",
        "claim": "The inhomogeneous cross-K matches spatstat's reference implementation.",
        "purpose": "Feed byte-identical inputs to OASIS and spatstat (Kcross.inhom, "
                   "correction='none') and compare curves.",
        "why": "External-reference agreement is the strongest evidence the estimator is correct.",
        "datasets": ["codex_crc"],
        "assumptions": "Everything in pixels (pixel_size=1) so no unit mismatch.",
        "limitations": "Requires an R + spatstat.explore environment; SKIPs without it.",
        "interpretation": "PASS = curves agree to ~1e-10 (documented Stage-A ~1e-3 tail).",
        "expected": "Agreement to floating point.",
        "runner": {"kind": "script", "script": "validate_spatstat_crossval.py"},
        "runtime_tier": "short", "external_deps": ["R"],
    },

    # ── Segmentation ───────────────────────────────────────────────────────────
    {
        "id": "segmentation",
        "title": "Segmentation vs manual counts",
        "category": "segmentation",
        "claim": "InstanSeg detection + DAB classification agrees with manual ground truth.",
        "purpose": "Score detections against human-annotated GeoJSON.",
        "why": "The quantification core rests on detection recall/precision; this is the "
               "number the '~90% agreement' claim needs and currently lacks.",
        "datasets": [],
        "assumptions": "User supplies <image>_manual.geojson ground truth.",
        "limitations": "UNVERIFIED — no manual ground truth ships; cannot run without it.",
        "interpretation": "Reports F1/κ once annotations are provided.",
        "expected": "UNVERIFIED until manual annotations are supplied.",
        "runner": {"kind": "script", "script": "validate_segmentation.py"},
        "runtime_tier": "long", "external_deps": ["qupath", "instanseg"],
    },

    # ── Quantification ─────────────────────────────────────────────────────────
    {
        "id": "deepliif_pipeline_validation",
        "title": "Detection + classification vs DeepLIIF IF truth",
        "category": "quantification",
        "claim": "End-to-end detection + positive/negative classification matches IF-derived "
                 "per-cell truth.",
        "purpose": "Score the real pipeline against DeepLIIF SegMask ground truth (41k cells).",
        "why": "IF-derived truth is the closest available orthogonal check on DAB classification.",
        "datasets": ["deepliif"],
        "assumptions": "IF-derived labels proxy true positivity; nuclear marker (Ki67).",
        "limitations": "Multi-step (prep/overlay-gt/score); end-to-end F1 capped by ~0.75 "
                       "detection recall; IF is a proxy (no same-section DAB+IF truth).",
        "interpretation": "Read class-only F1 and end-to-end F1.",
        "expected": "Class-only F1 ≈ 0.81.",
        "runner": {"kind": "script", "script": "deepliif_pipeline_validation.py", "argv": ["score"]},
        "runtime_tier": "long", "external_deps": ["qupath", "instanseg"],
    },
    {
        "id": "native_segmenter_fidelity",
        "title": "In-process InstanSeg reproduces the model and QuPath's deconvolution",
        "category": "segmentation",
        "claim": "Running the InstanSeg TorchScript bundle in-process reproduces the model's own "
                 "reference output exactly, and reproduces QuPath's 'DAB: Mean' colour "
                 "deconvolution to within 0.004 OD.",
        "purpose": "Isolate the two mechanical halves of dropping QuPath — can we run the model, "
                   "and can we reproduce the stain measurement — from segmentation variability.",
        "why": "Every published quantification number was measured through QuPath. Replacing the "
               "segmenter invalidates them unless the replacement demonstrably agrees.",
        "datasets": [],
        "assumptions": "QuPath's fixed BRIGHTFIELD_H_DAB stain vectors and 255 white point "
                       "(the generated Groovy never estimated stain vectors).",
        "limitations": "Check B uses QuPath's own polygons, so it tests colour maths only. "
                       "Check C (end-to-end on one image) is reported, not asserted — it has no "
                       "ground truth to arbitrate a segmentation disagreement.",
        "interpretation": "A must be exact. B must reach r ≥ 0.99, slope 1 ± 0.05, MAE ≤ 0.01 OD.",
        "expected": "A: 336/336 labels, exact. B: r 0.9986, slope 0.991, MAE 0.0035 OD.",
        "runner": {"kind": "script", "script": "validate_native_segmenter.py"},
        "runtime_tier": "short", "external_deps": ["instanseg"],
    },
    {
        "id": "native_segmenter_membrane",
        "title": "Membrane/cytoplasm path on native segmenter output",
        "category": "quantification",
        "claim": "measure_cytoplasm_dab consumes native GeoJSON and produces the same "
                 "population-level membranous call as it did from QuPath output.",
        "purpose": "The membrane path calibrates its Macenko-estimated DAB channel against the "
                   "GeoJSON's 'DAB: Mean' and refuses to run without >=50 references. Confirm "
                   "the anchor is satisfied by native output and the measurements do not shift.",
        "why": "The nuclear parity gate does not exercise the ring/completeness path at all, and "
               "that path is what CD8/TIM-3 membranous calls depend on.",
        "datasets": [],
        "assumptions": "LL477 CD8 with its recorded QuPath export as the control.",
        "limitations": "Matched-cell spread is expected (the two runs segment slightly "
                       "differently); only a systematic shift would indicate a defect. "
                       "validate_membrane_cd8_hnscc is unaffected — it drives from expert masks.",
        "interpretation": "Ring-DAB median shift <= 0.01 OD and membrane-positive rate shift "
                          "<= 0.05 means the marker call is unchanged.",
        "expected": "Runs; internal calibration r 0.989; ring-median Δ 0.0026 OD; "
                    "membrane-positive rate Δ 0.000.",
        "runner": {"kind": "script", "script": "validate_native_segmenter_membrane.py"},
        "runtime_tier": "short", "external_deps": ["instanseg"],
    },
    {
        "id": "native_segmenter_robustness",
        "title": "Native segmenter: determinism, thresholds, degenerate input, tiling",
        "category": "segmentation",
        "claim": "The segmenter is deterministic, reproduces the Groovy Otsu cut exactly, "
                 "returns empty rather than raising on degenerate input, and gives counts "
                 "independent of tile size.",
        "purpose": "Cover the surfaces the parity gates never touch: repeatability, the adaptive "
                   "threshold port, blank/tiny/thin images, and tile-size invariance.",
        "why": "Reproducibility is claimed throughout research/ihc.md; tile size is a performance knob "
               "that must not move a tissue-density statistic; background and edge tiles are "
               "routine inputs in production.",
        "datasets": [],
        "assumptions": "CPU inference; a tissue crop from LL477 as the working image.",
        "limitations": "A genuine SVS/NDPI was unavailable, so the openslide branch of "
                       "_load_rgb_full is NOT validated — only the tiled-TIFF read path.",
        "interpretation": "All five checks must pass; tile-size spread <= 2%.",
        "expected": "Label arrays identical across runs; Otsu identical to the Groovy "
                    "transliteration; 0 objects and no exception on degenerate input; "
                    "tile-size spread 0.7%; CPU/MPS identical labels.",
        "runner": {"kind": "script", "script": "validate_native_segmenter_robustness.py"},
        "runtime_tier": "short", "external_deps": ["instanseg"],
    },
    {
        "id": "native_segmenter_wsi",
        "title": "Whole-slide reading and streaming segmentation",
        "category": "segmentation",
        "claim": "A real pyramidal slide is segmented end to end in bounded memory, with global "
                 "normalisation percentiles that are EXACT rather than estimated, and streamed "
                 "output identical to the in-memory path.",
        "purpose": "The parity gates all ran on flat images that fit in RAM. A 776 Mpx ACROBAT "
                   "slide does not: the float64 optical-density intermediates alone need "
                   "~18.6 GB on a 17.2 GB machine. QuPath streamed tiles, so a non-streaming "
                   "replacement would be a silent capability regression.",
        "datasets": [],
        "assumptions": "ACROBAT valid slides on an external volume; 0.907 µm/px.",
        "limitations": "Runtime is dominated by CPU inference and by read throughput off the "
                       "external drive; the full-slide run is long by nature.",
        "interpretation": "J must be EXACT (histogram vs np.percentile). K must match the "
                          "in-memory path. L must complete in bounded memory.",
        "expected": "J: 0.0 intensity-unit difference (a downsampled level would be off 0.35, a "
                    "grid sample 0.28 — both measured and rejected). K: identical objects, DAB "
                    "MAE 0.00000. L (full 776 Mpx slide): 27,156 nuclei in 32 stripes, peak RSS "
                    "5.37 GB vs a naive 18.6 GB, 49 min on CPU / ~17 min on MPS.",
        "runner": {"kind": "script", "script": "validate_native_segmenter_wsi.py"},
        "runtime_tier": "long", "external_deps": ["instanseg", "openslide"],
    },
    {
        "id": "native_segmenter_deepliif_parity",
        "title": "Native segmenter parity gate vs QuPath (DeepLIIF IF truth)",
        "category": "segmentation",
        "claim": "Swapping QuPath for the in-process segmenter leaves every published "
                 "detection and classification figure unchanged within ±0.03.",
        "purpose": "Re-run the 598-image DeepLIIF IF-truth benchmark with the native segmenter "
                   "and diff it against the recorded QuPath run, through the SAME scorer.",
        "why": "This is the gate for removing QuPath. Fidelity checks show the parts agree; only "
               "this shows the published numbers survive.",
        "datasets": ["deepliif"],
        "assumptions": "Same pixel size (0.25 µm) and DAB threshold (0.2) as the recorded run; "
                       "the QuPath outputs on disk are the ones RESULTS.md reports.",
        "limitations": "Ki67 is nuclear — this validates nuclear segmentation and classification. "
                       "Membranous measurement parity needs the HNSCC CD8 harness.",
        "interpretation": "All five figures within ±0.03 of the QuPath run = QuPath removable.",
        "expected": "det-recall 0.752→0.747, det-precision 0.871→0.876, class-F1 0.809→0.812, "
                    "class-acc 0.928→0.929, e2e-F1 0.666→0.669. All |Δ| ≤ 0.005.",
        "runner": {"kind": "script", "script": "validate_native_segmenter_deepliif.py"},
        "runtime_tier": "long", "external_deps": ["instanseg"],
    },
    {
        "id": "membrane_cd8_hnscc",
        "title": "Membranous CD8 vs HNSCC IF truth",
        "category": "quantification",
        "claim": "The ring/completeness membrane method calls membranous CD8 correctly vs "
                 "IF-derived per-cell truth.",
        "purpose": "Score the hardened cytoplasm-ring completeness classifier on HNSCC tiles.",
        "why": "Membranous markers (CD8/TIM-3) are the flagship quant path; needs held-out proof.",
        "datasets": ["hnscc"],
        "assumptions": "IF-derived CD8 truth; AEC (not DAB) chromogen.",
        "limitations": "IF proxy; AEC not DAB; no membranous-DAB+IF set exists.",
        "interpretation": "Read held-out F1/AUC.",
        "expected": "Held-out F1 ≈ 0.76, AUC ≈ 0.89.",
        "runner": {"kind": "script", "script": "validate_membrane_cd8_hnscc.py"},
        "runtime_tier": "long", "external_deps": ["qupath", "instanseg"],
    },
    {
        "id": "entry_point_parity",
        "title": "CLI ⟷ UI entry-point parity",
        "category": "quantification",
        "claim": "The CLI and desktop UI produce identical spatial-association results.",
        "purpose": "Run the same inputs through both entry points and diff pixel size, "
                   "thresholds, and verdicts.",
        "why": "Reproducibility requires the two front doors to be the same pipeline.",
        "datasets": ["tim3_crc_icm"],
        "assumptions": "Same config resolved by both paths.",
        "limitations": "Uses a representative pair, not the full cohort.",
        "interpretation": "PASS = outputs identical.",
        "expected": "Parity across both entry points.",
        "runner": {"kind": "script", "script": "validate_entry_point_parity.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "tune_membrane_threshold",
        "title": "Membrane cutoff calibration (leave-one-cell-out)",
        "category": "quantification",
        "claim": "Membrane cutoffs are callable with an honest held-out metric, not an "
                 "optimistic in-sample fit.",
        "purpose": "Fit membrane_pix_thr/frac_min from hand-labelled cells and report "
                   "leave-one-cell-out F1/AUC.",
        "why": "DAB is not quantitative; cutoffs must be calibrated per protocol with a "
               "held-out callability gate (AUC ≥ 0.75).",
        "datasets": ["tim3_crc_icm"],
        "assumptions": "Hand-labelled positive/negative cells available.",
        "limitations": "Leave-one-CELL-out (not leave-one-image-out) is optimistic for "
                       "cross-slide transfer.",
        "interpretation": "Read held-out AUC/F1 and the callable verdict.",
        "expected": "TIM-3 held-out AUC ≈ 0.90 (callable).",
        "runner": {"kind": "script", "script": "tune_membrane_threshold.py"},
        "runtime_tier": "short", "external_deps": [],
    },

    # ── Spatial Association ─────────────────────────────────────────────────────
    {
        "id": "real_data",
        "title": "Real-data spatial controls (CODEX)",
        "category": "spatial_association",
        "claim": "On real CODEX data, known biological relationships reproduce as spatial "
                 "association / segregation.",
        "purpose": "Run the spatial statistic on Schürch CODEX marker pairs.",
        "why": "Sanity check on real single-cell coordinates.",
        "datasets": ["codex_crc"],
        "assumptions": "CODEX coordinates as the point pattern.",
        "limitations": "Uses the (retired) homogeneous-CSR null; see real_data_production.",
        "interpretation": "Descriptive association/segregation directions.",
        "expected": "Immune pairs associate; tumour pairs segregate (CSR null).",
        "runner": {"kind": "script", "script": "validate_real_data.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "real_data_production",
        "title": "Real-data through the production primary null",
        "category": "spatial_association",
        "claim": "Through the shipped reweighted primary, the real-data associations are "
                 "materially weaker than the CSR null suggests — honestly reported.",
        "purpose": "Re-run the CODEX controls through cross_k_all_nulls (production).",
        "why": "Shows the production null does not inherit CSR's shared-preference inflation.",
        "datasets": ["codex_crc"],
        "assumptions": "Production null set (reweighted primary + CSR baseline).",
        "limitations": "CODEX coordinates, not serial-section DAB.",
        "interpretation": "Compare robust vs csr_only verdicts against real_data.",
        "expected": "Association fraction drops vs the CSR-null table.",
        "runner": {"kind": "script", "script": "validate_real_data_production.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "restained_coexpression",
        "title": "Same-section restained co-expression",
        "category": "spatial_association",
        "claim": "Same-section restaining supports single-cell co-expression, fail-closed on "
                 "dimension + correspondence.",
        "purpose": "Validate the restained path's gates and per-cell co-expression logic.",
        "why": "Co-expression is only defensible on one physical section; the guards must hold.",
        "datasets": [],
        "assumptions": "Restains share cell coordinates (operator-certified).",
        "limitations": "Correspondence guard is advisory (no tuned cutoff).",
        "interpretation": "PASS = gates fire; co-expression only on corresponding tissue.",
        "expected": "Gates PASS on synthetic bundles.",
        "runner": {"kind": "script", "script": "validate_restained_coexpression.py"},
        "runtime_tier": "short", "external_deps": [],
    },
    {
        "id": "hnscc_restained_all",
        "title": "Restained validation on all HNSCC tiles",
        "category": "spatial_association",
        "claim": "The restained workflow's detections + co-expression match IF truth, and "
                 "the correspondence diagnostic flags non-corresponding tiles.",
        "purpose": "Run the restained path across HNSCC tiles vs released nuclear masks + IF.",
        "why": "Real-tissue exercise of the restained gates and the §21.6 negative control.",
        "datasets": ["hnscc"],
        "assumptions": "Released expert nuclear masks as segmentation truth.",
        "limitations": "AEC not DAB; outputs are large; long-running.",
        "interpretation": "Read detection F1 + the Case2_S3_1_1 negative-control flag.",
        "expected": "Detection F1 ≈ 0.78; non-corresponding tile flagged.",
        "runner": {"kind": "script", "script": "validate_hnscc_restained_all.py"},
        "runtime_tier": "long", "external_deps": ["qupath", "instanseg"],
    },

    # ── End-to-End ──────────────────────────────────────────────────────────────
    {
        "id": "e2e_knownwarp_deepliif",
        "title": "End-to-end B: real-DAB known-warp reconstruction",
        "category": "end_to_end",
        "claim": "Real chromogenic DAB pixels flow correctly through the whole pipeline "
                 "(segmentation → registration → cross-K) at cell scale.",
        "purpose": "Warp a real DeepLIIF IHC panel by a known transform, segment both, "
                   "register, and check the reconstruction TRE + that the verdict is "
                   "recovered only WITH registration (necessity control).",
        "why": "Bounds one side of the untestable real-DAB cell-scale gap (research/ihc.md §10): "
               "real pixels + full pipeline, with a geometric ground truth we can build.",
        "datasets": ["deepliif"],
        "assumptions": "Same-image warp → the two cell populations are identical (trivial, "
                       "maximal association); DeepLIIF ≈ 0.25 µm/px.",
        "limitations": "Same marker, not two different markers (association is trivial) — "
                       "that is Validation A. Registration is the automated path here.",
        "interpretation": "Small reconstruction TRE + registered verdict associated + "
                          "unregistered verdict different = pipeline sound on real pixels.",
        "expected": "Median reconstruction TRE ≤ 5 µm; registered→associated; "
                    "registration necessary on most tiles.",
        "runner": {"kind": "script", "script": "validate_e2e_knownwarp_deepliif.py"},
        "runtime_tier": "long", "external_deps": ["qupath", "instanseg"],
    },
    {
        "id": "keystone_degradation",
        "title": "Serial-section degradation keystone (CODEX)",
        "category": "end_to_end",
        "claim": "The serial-section approximation is sound: a known same-section verdict "
                 "survives being split to pseudo-serial and degraded by a realistic "
                 "registration error.",
        "purpose": "Same-section multiplex truth (CODEX CD8/PD-1) → split → inject "
                   "registration error the size of the measured TRE → verdict must not flip.",
        "why": "The ONLY place true cross-marker association ground truth exists; it is the "
               "cell-scale complement that bounds the untestable real DAB case.",
        "datasets": ["codex_crc"],
        "assumptions": "CODEX coordinates as truth; injected error ≈ measured TRE.",
        "limitations": "CODEX ships as coordinates, not registrable images (point-level).",
        "interpretation": "PASS = engaged/independent/csr_only verdicts all survive degradation.",
        "expected": "All three degradation tests PASS.",
        "runner": {"kind": "pytest", "node": "tests/test_degradation.py"},
        "runtime_tier": "short", "external_deps": [],
    },
]

_BY_ID = {v["id"]: v for v in VALIDATIONS}


def by_id(vid: str) -> dict | None:
    return _BY_ID.get(vid)


def all_ids() -> list[str]:
    return [v["id"] for v in VALIDATIONS]


def by_category() -> list[dict]:
    """[{key, title, validations:[...]}] in pipeline order."""
    out = []
    for key, title in CATEGORIES:
        items = [v for v in VALIDATIONS if v["category"] == key]
        out.append({"key": key, "title": title, "validations": items})
    return out
