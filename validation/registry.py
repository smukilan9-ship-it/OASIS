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
        "why": "Justifies the deliberate 'no edge correction' design decision (ihc.md §17).",
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
                       "bandwidth (disclosed in ihc.md §15.5).",
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
        "id": "phase_b_certified",
        "title": "Phase-B: analysis only on certified ROIs",
        "category": "registration",
        "claim": "Spatial analysis runs only where registration is certified; uncertified "
                 "pairs are refused, not warped.",
        "purpose": "Exercise the certified-ROI gating of the spatial path on CIMA data.",
        "why": "Closes the loop between certification and what the pipeline will analyse.",
        "datasets": ["cima_landmarks", "codex_crc"],
        "assumptions": "Certification thresholds as in ihc.md §3.5.",
        "limitations": "Few real certifiable pairs exist publicly.",
        "interpretation": "PASS = only certified ROIs are analysed.",
        "expected": "Certified ROI analysed; others refused.",
        "runner": {"kind": "script", "script": "validate_phase_b_certified.py"},
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
        "why": "Bounds one side of the untestable real-DAB cell-scale gap (ihc.md §10): "
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
