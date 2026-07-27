# Competitive analysis — how this pipeline compares (2026-07)

Two parts: (1) segmentation + quantification + membranous markers, (2) the spatial-association
pipeline. Honest throughout — where we win, where we lose, and why our niche exists.

---

## Part 1 — Segmentation, quantification & membranous markers

### The landscape (what each tool actually is)

| Tool | Modality | Segmentation | Compartment/membrane quant | Openness / cost |
|---|---|---|---|---|
| **This pipeline** | Chromogenic **DAB IHC** (brightfield) | InstanSeg (via QuPath), nuclei | Voronoi-clipped cytoplasm ring, per-image stain vectors, DAB>H gate, **completeness** classifier, per-cohort calibration | **Open** (QuPath/Python), free |
| **HALO** (Indica Labs) | Brightfield IHC + mIF (≤5-plex) | HALO AI (pretrained nuclear/membrane) | Nucleus/cytoplasm/**membrane** algorithms (HER2/PD-L1 connectivity), threshold assist | Proprietary, ~$$$/yr |
| **QuPath** (our base) | IHC + mIF | Watershed / StarDist / InstanSeg | Cell expansion + compartment "DAB: Mean"; no membrane-completeness classifier by default | Open, free |
| **Akoya inForm / Phenoptics** | Multiplex IF (Vectra) | Nuclear + membrane (trainable) | Per-marker per-compartment on mIF | Proprietary, $$$ + instrument |
| **Xenium** (10x) | Spatial **transcriptomics (RNA)** | Multimodal (DAPI + boundary stains) | N/A — measures transcripts/cell, not chromogen OD | Instrument $$$$ + high per-sample |
| **Cellpose / StarDist / InstanSeg** | Any (segmentation only) | SOTA nuclei/cell seg | None (segmentation libraries) | Open |

**Key framing:** Xenium and Akoya are *different modalities* (RNA, multiplex IF). The true
head-to-head for "quantify a membranous DAB marker per cell" is **HALO** and **QuPath**.

### Where we match or beat the field

- **Membranous quantification is a first-class, validated path**, not an afterthought.
  - We measure the **Voronoi-clipped cytoplasmic ring** (not the nucleus) — the correct
    compartment for CD8/TIM-3 — mirroring QuPath `detectionsToCells` but with the ring as the
    decision region.
  - We classify on **membrane completeness** (fraction of the ring that is DAB-dominant),
    the HER2/PD-L1 paradigm, rather than a diluted ring mean. HALO's HER2/PD-L1 membrane
    algorithms use the same *idea* (membrane connectivity/completeness); the difference is ours
    is transparent and inspectable, theirs is a proprietary black box.
- **Per-image color deconvolution with auto stain-vector selection.** We estimate H-DAB
  vectors per image (Macenko) *and* fall back to fixed vectors when they deconvolve better
  (parity-checked) — so tone-cast slides (the CRC-ICM green cast, 99%-false-positive failure)
  and faint low-DAB slides both work. Most tools use fixed or single-method deconvolution.
- **A per-pixel DAB-dominance gate** (`DAB_OD > H_OD`) removes the dominant low-OD false
  positive — dark counterstain masquerading as signal — a failure mode that a raw OD threshold
  (QuPath default, and naïve HALO thresholds) does not guard against.
- **Ground-truth validation is built into the method, not assumed.** We validated three ways
  against *IF-derived* per-cell truth: TIM-3 held-out **F1 0.93** (hand-labelled), CD8 vs CD8-IF
  **F1 0.76 / AUC 0.89** (HNSCC), Ki67 vs mIF **F1 0.81** (DeepLIIF, 41k cells). HALO/QuPath ship
  threshold *assistance* but no built-in held-out F1/AUC against orthogonal ground truth.
- **Native, honest calibration.** The Calibrate tab makes the user fit cutoffs to *their own*
  protocol/scanner (with an AUC go/no-go and a "too faint to call" verdict), because DAB is not
  quantitative and cutoffs are cohort-specific. Commercial tools let you *set* thresholds; they
  don't tell you the marker is uncallable on this staining.
- **Quality gating that fails closed.** A faint slide is flagged low-confidence (background-margin
  / implausible positive rate) rather than silently over-called. Pixel-size provenance is surfaced
  and a wrong/default pixel size is flagged.
- **Open and free**, built on QuPath + InstanSeg. HALO is a five-figure annual license; Xenium/Akoya
  are capital instruments.

### Where we lose (be honest)

- **DAB is not quantitative.** Chromogen OD is not stoichiometric; every cutoff is per-protocol.
  mIF (Akoya) and RNA (Xenium) give far more linear, quantitative signal. We mitigate with
  per-cohort calibration, but we can never match IF/RNA quantitativeness on brightfield DAB.
- **Single-plex chromogenic** → one marker per section. HALO mIF does 5-plex; Xenium/Akoya do
  dozens–thousands. True multi-marker phenotyping on one cell is out of reach for serial DAB.
- **Segmentation ceiling.** We inherit InstanSeg on brightfield: ~0.75 detection-recall vs
  DAPI-defined truth (small nuclei + dense clusters missed). HALO AI and Xenium's DAPI-based
  segmentation see nuclei brightfield can't. Our end-to-end F1 is capped by this.
- **No polished enterprise tooling** (LIMS, audit trails, validated regulatory workflows, support)
  that HALO/Visiopharm provide for clinical labs.
- **Membrane path needs a QuPath geojson with `DAB: Mean`** (for the parity gate) — a coupling
  commercial single-vendor stacks don't have.

### Verdict (Part 1)

For **quantifying a membranous DAB marker per cell on serial chromogenic IHC, transparently and
for free, with real validation**, this pipeline is competitive with HALO's membrane module and
ahead of stock QuPath — largely because of the auto stain-vector selection, the DAB>H gate, the
completeness classifier, and the built-in IF-validated calibration. For **multiplex phenotyping,
quantitative signal, or clinical enterprise workflows**, HALO / Akoya / Xenium win by modality and
maturity, not by algorithm.

---

## Part 2 — Spatial-association pipeline

### What we do

Serial-section chromogenic IHC (marker A on slice 1, marker B on slice 2) → **register** the two
sections (SimpleITK/openslide/hematoxylin cascade, with a **fail-closed registration QC** gate:
residual/overlap/quality; identity or residual ≥10 µm → stats marked invalid) → **cross-type
Ripley's K / pair-correlation g(r)** with a **Monte-Carlo null envelope** and three null models
(homogeneous CSR, inhomogeneous Kinhom, toroidal) on the A∩B intersection window. Reports a
population-level L−r curve, and is deliberately **honest that it cannot claim same-cell
co-expression** (the serial-section z-gap breaks cell identity).

### Landscape

| Tool | Spatial model | Serial-section registration | Co-expression claim | Statistics |
|---|---|---|---|---|
| **This pipeline** | Cross-type Ripley's K / g(r), MC null envelope, 3 null models | **Yes** — with fail-closed QC gate | **No same-cell claim** (population association only, z-gap-honest) | Formal spatial point-process stats + null envelope |
| **HALO Spatial** | Nearest-neighbor, proximity, infiltration, density heatmaps | Yes (serial sections supported) | Proximity/counts (not formal co-expression) | Descriptive proximity histograms; less formal null modeling |
| **Xenium** | All markers on **one** section (multiplex) | **Not needed** — no serial registration | **Direct single-cell co-expression** (same section) | Downstream (Squidpy/Giotto) — full point-process stats available |
| **Akoya mIF** | All markers on one section | Not needed | Direct single-cell co-expression | Downstream tools |
| **Squidpy / Giotto (open)** | Ripley's K, co-occurrence, neighborhood enrichment | No (analysis only) | Depends on input | Rich, formal |

### Where we win / are distinctive

- **We solve co-analysis on cheap serial chromogenic IHC** — the most widely available substrate
  on earth. Xenium/Akoya get true single-cell co-expression but require a ~$$$$ instrument and
  fresh/special protocols; we work on archival FFPE DAB slides two labs already have.
- **Serial-section registration with a fail-closed QC gate.** You are right that **Xenium has no
  serial-section registration** — it doesn't need one because it multiplexes on a single section.
  HALO *does* register serial sections, but our differentiator is the **QC that invalidates the
  result when registration is untrustworthy** (identity/residual thresholds, greyed-out UI),
  rather than silently reporting a spatial stat on a bad alignment.
- **Formal point-process statistics with a Monte-Carlo null envelope and multiple null models**,
  plus registration-uncertainty sensitivity (perturb the transform within its measured TRE and
  re-run). HALO's proximity/nearest-neighbor outputs are more descriptive; we are closer to the
  Squidpy/spatstat rigor while staying on chromogenic IHC.
- **Scientific honesty as a feature.** We explicitly refuse the same-cell co-expression overclaim
  that serial sections cannot support (z-gap; a marker-B+ object near a marker-A+ object is not
  proven to be the same cell or even the same cell type). This is a differentiator versus
  proximity tools that let users imply co-expression from adjacency.
- **Public-data method certification** (ANHIR/CIMA landmarks → landmark-driven TRE with
  CERTIFIED/LOCALLY_CERTIFIED/DEFORMED/NOT_CERTIFIABLE verdicts) — we validate the registration
  itself, not just the downstream stat.

### Where we lose

- **We never get true same-cell co-expression.** Xenium/Akoya do, trivially, because every marker
  is on the same cell. Our ceiling is *population-level association*, by physics of serial sections.
- **Registration is the hard limit.** Automated sub-5 µm TRE is unreliable on real serial sections
  (patch-flow aliases on ~30 µm shifts) → we often need manual/expert landmarks; multiplex platforms
  sidestep this entirely.
- **Two markers at a time** (pairwise), vs dozens simultaneously on a multiplex platform.
- **Descriptive richness of a commercial UI** (interactive spatial plots, neighborhood/community
  detection, clustering) is more built-out in HALO and in the Xenium/Squidpy ecosystem.

### Verdict (Part 2)

Our spatial pipeline occupies a **defensible niche the multiplex platforms leave open**: rigorous,
statistically-grounded, registration-QC'd **population-level** spatial association on **cheap,
ubiquitous serial chromogenic IHC**, with hard scientific guardrails against the same-cell
overclaim. If a lab has a Xenium/Akoya instrument, single-section multiplex is strictly better for
co-expression. If a lab has archival DAB slides and a question about spatial association between two
markers, this pipeline does something those instruments don't address and does it honestly.

---

## One-paragraph summary

We are not trying to beat Xenium or Akoya at their game (quantitative multiplex on one section) —
we can't, by modality. We are the **open, validated, honest option for chromogenic DAB IHC**: the
membranous-quantification path is competitive with HALO's proprietary membrane module and ahead of
stock QuPath (auto stain vectors + DAB>H gate + completeness + IF-validated calibration), and the
spatial pipeline brings point-process rigor and fail-closed registration QC to serial-section IHC
while refusing the co-expression overclaim those serial sections can't support.
