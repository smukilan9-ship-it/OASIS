# OASIS — Paper Skeleton (JPI methods/tools paper)

**How to use this file:** each section gives (a) its PURPOSE, (b) the POINTS/FACTS to cover
(these are *your* real results — verify each against the current code/JSON before you rely on
it), and (c) a **YOU WRITE** prompt. Write the prose yourself, in your own words. This is a
scaffold, not text to copy. Delete these instructions before submission.

Framing rule for the whole paper: **methods/tools paper, not a biological-findings paper.**
Say that in the abstract, intro, and discussion. LL477 = illustrative proof-of-concept, never a
finding.

---

## Title
- A tool + a method claim. Draft options:
  - "OASIS: a fail-closed, distance-validity-certified pipeline for population-level spatial
    association on serial-section single-plex IHC."
- **YOU WRITE:** pick/adapt one.

## Abstract (structured: Background / Methods / Results / Availability)
- Background: multiplex imaging is costly; serial single-plex IHC is cheap but cannot establish
  single-cell co-expression (different physical slices).
- Methods: segmentation (QuPath+InstanSeg) → distance-preserving registration with held-out-TRE
  certification → cross-type Ripley's K with a null framework that separates co-infiltration
  from engagement.
- Results (headline numbers only): estimator matches R spatstat to ~1e-13; DCLF calibrated ~5%;
  reweighted null size-controlled at 75 µm; registration TRE certified on ANHIR/CIMA; membrane/
  detection F1 vs IF proxies.
- Availability: repo URL, license, DOI.
- **YOU WRITE:** ~200 words, one paragraph per part.

---

## 1. Introduction
- Motivation: spatial context of immune cells (e.g., CD8/TIM-3) in the tumor microenvironment.
- Core honest principle up front: serial sections = different slices → **population-level
  spatial association only, never single-cell co-expression**.
- Contributions (bulleted list — tools papers expect this):
  1. deterministic end-to-end pipeline, no LLM inference in the analysis path;
  2. fail-closed registration certification with held-out TRE;
  3. a null framework separating co-infiltration from cell-scale engagement;
  4. a reproducible, claim-by-claim validation harness.
- **YOU WRITE:** ~4 short paragraphs + the contributions list.

## 2. Related work / Background
- Contrast with commercial tools (e.g., HALO) that use elastic/non-rigid alignment → fabricates
  the inter-cell distances a distance statistic consumes (your key differentiator).
- Cross-type Ripley's K and the inhomogeneous variant (cite Baddeley–Møller–Waagepetersen;
  spatstat as reference implementation).
- DCLF envelope test (cite Diggle–Cressie–Loosmore–Ford).
- **YOU WRITE:** ~3 paragraphs. Get the citations right — verify each.

---

## 3. Methods (the core — one subsection per pipeline stage)

### 3.1 Segmentation & quantification
- InstanSeg nuclear segmentation via QuPath; stain deconvolution; per-cell DAB.
- Membrane mode: cytoplasmic-ring DAB, Voronoi half-plane clipping in dense infiltrate.
- **DAB is not quantitative** → per-slide calibration; leave-one-cell-out AUC gate ≥ 0.75.
- **YOU WRITE:** describe each; state the honesty point about DAB explicitly.

### 3.2 Registration & certification
- Similarity only (rotation + uniform scale + translation); never non-rigid — a warp invents the
  distances the statistic measures.
- Structural low-frequency hematoxylin channel (σ≈12 µm) so shared architecture dominates.
- Held-out TRE: independent second-annotator set if available, else leave-one-out (fit-unbiased).
- Four verdicts: CERTIFIED (n≥6, held-out TRE median ≤5 µm, fit-residual ≤5 µm),
  LOCALLY_CERTIFIED (ROI), DEFORMED, NOT_CERTIFIABLE. Failed pair refused, never warped.
- **YOU WRITE:** explain the four verdicts and *why fail-closed*.

### 3.3 Spatial statistic & null framework  ← spend the most space here
- Cross-type K; L(r)−r (0 independent, + attraction, − segregation); g(r).
- Window = A∩B tissue intersection; no analytic edge correction (bias cancels vs. the null).
- **Two nulls = two questions:**
  - Homogeneous CSR → co-infiltration (weak; almost always true for two immune markers).
  - Reweighted inhomogeneous cross-K (PRIMARY) → engagement; pairs weighted 1/(λ_A·λ_B),
    LOO intensity, parametric bootstrap with per-sim re-estimation; bandwidth 75 µm.
- DCLF global test over 10–50 µm → one p-value, no multiple-comparison inflation.
- Verdict: robust / csr_only / none.
- Dense-tissue fallback (present as a *validated candidate*, with the scaffold-sensitivity
  caveat — do NOT present as a fully-settled co-equal primary): morphology-conditioned null,
  all-cell support + 2 µm jitter, 10–30 µm band, fail-closed gates.
- **YOU WRITE:** this is the heart. Write it so a pathologist follows the *what* and a
  statistician trusts the *how*. Use the plain-English framing you can defend from scratch.

### 3.4 Cohort correction
- Benjamini–Hochberg FDR across certified pairs; never quote the bare-minimum p.
- **YOU WRITE:** one paragraph.

---

## 4. Implementation / software
- Two entry points (CLI + desktop UI) over one shared core → identical results
  (validate_entry_point_parity).
- Determinism; pinned dependencies with the reason (lib drift moves KDE binning/percentiles →
  can flip a verdict).
- Platform (macOS), external deps (QuPath/InstanSeg), config surface.
- (Optional) a clean pipeline architecture figure.
- **YOU WRITE:** ~2 paragraphs; consider a figure.

---

## 5. Validation / experiments  ← present as a CLAIM → VALIDATION → RESULT → CAVEAT table
Pull rows from validation/registry.py. Numbers to include (VERIFY each before use):
- Estimator correctness: brute-force + **spatstat cross-validation ~1e-13**.
- DCLF: ~5% false-positive under CSR + power/direction.
- Reweighted null: 75 µm shippable (shared size ~0.032, uniform ~0.064, power@7px 1.0,
  power@25px ~0.99); **disclose the ~10% anti-conservatism + 75 µm knife-edge here, not in a
  footnote.**
- Registration: TRE vs ANHIR/CIMA; best LOCALLY_CERTIFIED lung-lesion ~3.66 µm ROI.
- Detection/membrane (IF proxies): DeepLIIF class F1 ~0.81; HNSCC membranous CD8 F1 ~0.76,
  AUC ~0.89.
- Dense null: Schürch CODEX calibration (CSR over-rejects ~10–25%; morphology-conditioned H0
  ~3.7–6.7%, power 1.0); rendered-pixel bridge field corr ~0.939, H0 ~3.7–6.3%; Keren scaffold
  stress test (p13/p16 stable, p32 scaffold-sensitive).
- End-to-end bounding suite: keystone degradation (CODEX CD8/PD-1) + Validation B
  (known-warp DeepLIIF, reconstruction TRE ~1.6 µm) [+ Validation A once built].
- **YOU WRITE:** the table + a paragraph per group. Frame the e2e as "we bound the untestable
  case from three sides," per ihc.md §10.

## 6. Results / illustrative application (LL477)  ← proof-of-concept, NOT a finding
- Present the discrimination story, not "3 associations":
  - x10_1: CERTIFIED (TRE ~1.5 µm), **robust** association, p_reweighted ~0.006.
  - x10_3: CERTIFIED (TRE ~4.9 µm), **csr_only** — p_homog ~0.001 but p_reweighted ~0.72
    (co-infiltration, not engagement).
  - x10_2: CERTIFIED (TRE ~2.7 µm), **csr_only** segregation; skipped for the dense null
    (only ~10 TIM-3 positives).
- Explicit caveat: n=8 landmarks/pair, single annotator, one cohort, nothing survives cohort FDR.
- (If you resolve which primary governs each pair, state the per-pair provenance.)
- **YOU WRITE:** tell it as "the method retained 1 of 3 as engagement and correctly declined 2 —
  including saying no." One figure: an association curve with the null envelope.

## 7. Discussion & Limitations  ← this is where the paper earns trust; give it real space
- Serial sections → population-level only, never single-cell co-expression.
- Reweighted null mildly anti-conservative → p≈0.05 treated with caution.
- 75 µm bandwidth calibrated for this tissue class, not universal.
- **Segmentation recall ~0.75 non-randomly thins dense infiltrate** — the one bias no null can
  fix; ideally add a sensitivity analysis.
- Certification single-annotator LOO (until the second-annotator study lands).
- **Yield:** most serial-section FOVs are not globally distance-valid → analysis runs in
  certified sub-regions (report your global/local/fail rates as a number).
- **YOU WRITE:** name each limitation plainly. This section is a strength, not a confession.

## 8. Availability & reproducibility
- Repo URL, LICENSE, archived DOI (Zenodo), dataset sources/licenses, how to run the validation
  harness. AI-use disclosure statement.
- **YOU WRITE:** short, factual.

---

### Cross-cutting checks before submission
- [ ] Every abstract claim traces to a row in §5 (the claims-vs-evidence spine).
- [ ] "No single-cell co-expression / no biological finding" stated in abstract, intro, discussion.
- [ ] Every statistical sentence and every citation independently verified (models hallucinate both).
- [ ] AI-use disclosed per JPI policy.
- [ ] Authorship + data permission settled with the data provider before preprinting.
