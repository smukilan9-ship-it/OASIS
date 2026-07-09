# Public datasets for validating cross-type spatial association

Curated for validating the redesigned spatial-association method (intensity-
reweighted inhomogeneous cross-K; see `ihc.md` §15). Each entry tests whether the
method recovers a **known/published** spatial relationship. We need BOTH:
- **positive controls** — cell-type pairs that genuinely co-localize, and
- **negative/segregation controls** — pairs that occupy separate compartments,

so we can check calibration (no false "robust" on segregated/independent pairs)
AND power (detects real co-localization).

> Our method consumes **per-cell (x,y) coordinates + a type label** per "section".
> For multiplex datasets (all cells on one section) we emulate the serial-section
> setting by treating two cell types as the two "populations" A and B and bounding
> the window by the tissue/core hull — exactly as `validation/validate_real_data*`
> already does for Schürch. Datasets that ship a **single-cell table** (coords +
> phenotype) are far cheaper for heavy batch validation than raw image stacks;
> those are tagged **[batch-friendly]**.

Legend — Rel.: expected relationship. **+** co-localization (positive control),
**−** segregation (negative control), **0** independence (null control).

---

## Tier 1 — single-cell tables, published spatial findings (use these first)

### 1. Schürch et al. 2020 — CRC CODEX  **[batch-friendly]**
- **Link:** Mendeley Data `10.17632/mpjzbtfgfr.1` (CC BY 4.0). Paper: Cell 2020, https://doi.org/10.1016/j.cell.2020.07.005
- **Modality / markers:** CODEX, 56 markers; CD8, CD4, FoxP3 (Treg), tumor, B, macrophage. 258k cells, 140 TMA spots.
- **Why it tests us:** the dataset we already use; clean cross-type controls within the same cores.
- **Dense-null calibration use:** `validation/validate_public_codex_dense_null.py`
  uses the same real CRC cell-coordinate fields as marker-independent architecture
  templates, then simulates known-null and planted-positive marker pairs on top.
  This avoids pretending any biological marker pair is true null ground truth.
- **Expected:** CD8↔CD4 **+** ; CD8↔Treg **+** (weaker) ; CD8↔tumor **−** .
- **Caveat:** **no TIM-3** (checkpoints are PD-1/PD-L1/LAG-3/VISTA/ICOS/IDO-1). Nominal 0.3775 µm/px (stated, not in table).

### 2. Keren et al. 2018 — TNBC MIBI-TOF  **[batch-friendly]**
- **Link:** Angelo Lab https://www.angelolab.com/mibi-data ; paper: Cell 2018, https://doi.org/10.1016/j.cell.2018.08.039 (PMID 30193111). Single-cell table with `immuneGroup`/cell types + x,y.
- **Modality / markers:** MIBI-TOF, 36 proteins; CD8, CD4, tumor (keratin), macrophage, PD-1, PD-L1, IDO.
- **Why it tests us:** the authors **explicitly classify tumors as "mixed" vs "compartmentalized"** with a published mixing score — a ready-made positive/negative split of the SAME comparison (immune↔tumor).
- **Expected:** immune↔tumor **+** in "mixed" patients, **−** in "compartmentalized" patients. CD8↔tumor **−** overall. This is our single best calibrated +/− contrast.

### 3. Jackson et al. 2020 — Breast cancer IMC (Basel/Zurich)  **[batch-friendly]**
- **Link:** Zenodo `10.5281/zenodo.4607374` (single-cell + masks); paper: Nature 2020, https://doi.org/10.1038/s41586-019-1876-x
- **Modality / markers:** IMC, ~35 markers; T cells (CD8/CD4), B, tumor, stroma, vasculature. 100s of cores.
- **Why it tests us:** large core count for batch power/calibration; published "single-cell pathology" neighborhoods.
- **Expected:** CD8↔CD4 **+** ; immune↔tumor often **−** ; T-cell↔B aggregates **+** .

### 4. Danenberg et al. 2022 — METABRIC IMC  **[batch-friendly]**
- **Link:** Zenodo `10.5281/zenodo.6036188` / `7494482` (check record); paper: Nat Genet 2022, https://doi.org/10.1038/s41588-022-01041-y
- **Modality / markers:** IMC, ~37 markers, **693 tumors**; 10 recurrent TME structures published.
- **Why it tests us:** the **largest** breast IMC cohort with named spatial structures → heavy external batch validation; vascular/stromal/leukocyte co-localization patterns to check against.
- **Expected:** lymphocyte↔lymphocyte **+** ; immune↔tumor structure-dependent (**+** in "TLS-like", **−** in "immune-excluded").

### 5. HTAN / Lin et al. 2023 — CRC CyCIF 3D atlas  **[batch-friendly]**
- **Link:** Human Tumor Atlas Network portal https://humantumoratlas.org (HTA chunk; CyCIF single-cell tables); paper: Cell 2023, https://doi.org/10.1016/j.cell.2022.12.028
- **Modality / markers:** t-CyCIF, ~25+ markers; CD8, Treg (FoxP3), tumor, PD-1, PD-L1.
- **Why it tests us:** independent platform (CyCIF) for the same CD8/Treg/tumor controls → cross-platform robustness.
- **Expected:** CD8↔Treg **+** at invasive margin ; CD8↔tumor **−** in immune-excluded regions, **+** at the front.

---

## Tier 2 — the CD8 + TIM-3 question (closest to our actual markers)

### 6. Phillips et al. 2021 — CTCL CODEX (pembrolizumab trial)
- **Link:** paper Nat Commun 2021, https://doi.org/10.1038/s41467-021-26974-6 (NCT02243579); 70 cores, 59 protein channels.
- **Modality / markers:** CODEX; CD8 **and immunoregulatory/exhaustion markers** (PD-1; check channel list for TIM-3/LAG-3).
- **Why it tests us:** the **only** identified multiplex set plausibly containing **both CD8 and TIM-3** — direct biological match to our pipeline's target.
- **Expected:** exhausted-CD8 ↔ tumor **+** in responders' topography; CD8↔checkpoint co-expression patterns.
- **Caveat (critical):** access is **on request / restricted** (as already noted in `ihc.md` §9). Not freely batch-downloadable; flag for the user to request. Use as a confirmatory, not bulk, set.

---

## Tier 3 — architecture controls (decisive calibration cases)

### 7. HuBMAP — Tonsil / Lymph node CODEX  **[batch-friendly]**
- **Link:** https://portal.hubmapconsortium.org (search CODEX tonsil/lymph node; single-cell tables + coords).
- **Modality / markers:** CODEX; CD8/CD4 T, CD20 B, CD21 FDC.
- **Why it tests us:** **textbook architecture** = the hardest calibration test. Germinal-center follicles give B↔T_FH **+** inside follicles, while B-zone↔T-zone are **−** (anatomically segregated). A method that survives tonsil's strong shared architecture is the real proof our reweighted null works.
- **Expected:** B↔CD4-T_FH **+** ; B-zone↔CD8-T-zone **−** .

### 8. HuBMAP — Intestine CODEX  **[batch-friendly]**
- **Link:** https://portal.hubmapconsortium.org (Intestine CODEX, e.g., Hickey 2023 gut atlas).
- **Modality / markers:** CODEX; epithelial, immune, stromal, vascular.
- **Why it tests us:** crypt–villus architecture is a strong **non-stationary** shared-preference field — the exact regime the old nulls failed. Negative controls: epithelial↔lymphoid-aggregate segregation.
- **Expected:** immune-aggregate internal **+** ; epithelium↔immune-follicle **−** .

---

## Tier 4 — independent platforms / spatial transcriptomics (cross-modality)

### 9. 10x Genomics Xenium — public breast & lung panels  **[batch-friendly]**
- **Link:** https://www.10xgenomics.com/datasets (Xenium Human Breast, Human Lung); cell × gene + centroids.
- **Modality / markers:** Xenium in-situ; CD8A, CD4, FOXP3, EPCAM/KRT (tumor).
- **Why it tests us:** independent imaging-transcriptomics platform, high cell counts; positive (T-T) and negative (T-tumor) controls.
- **Expected:** CD8↔CD4 **+** ; CD8↔EPCAM⁺ tumor **−** .

### 10. Vizgen MERSCOPE — public FFPE (breast/lung/CRC)  **[batch-friendly]**
- **Link:** https://info.vizgen.com/ffpe-showcase (MERFISH); cell metadata with x,y.
- **Modality / markers:** MERFISH; immune + tumor lineage genes.
- **Why it tests us:** second transcriptomics platform → guards against modality-specific artifacts.
- **Expected:** lymphocyte↔lymphocyte **+** ; immune↔tumor **−/+** by region.

### 11. Risom et al. 2022 — DCIS MIBI-TOF  **[batch-friendly]**
- **Link:** Angelo Lab https://www.angelolab.com/mibi-data ; paper Cell 2022, https://doi.org/10.1016/j.cell.2021.12.023
- **Modality / markers:** MIBI; myoepithelial, immune, ECM.
- **Why it tests us:** myoepithelial layer ↔ immune is a clean **−** (segregated by the basement membrane); progression cohort.
- **Expected:** myoepithelial↔immune **−** ; immune↔immune **+** .

### 12. Moldoveanu et al. 2022 — Melanoma IMC (ICI)  **[batch-friendly]**
- **Link:** paper https://doi.org/10.1126/sciimmunol.abi5072 ; data via Zenodo/listed in paper.
- **Modality / markers:** IMC; CD8, exhaustion markers, melanoma (SOX10/S100).
- **Why it tests us:** **exhausted CD8 ↔ tumor** is the closest biology to our CD8/TIM-3 question on a freely-available set.
- **Expected:** CD8↔tumor **+/−** by response; exhausted-CD8 enrichment near tumor in responders.

---

## Tier 5 — explicit null / negative-control constructions (size checks)

### 13. Cross-sample SWAP control (any Tier-1 dataset)
- **Construction:** pair cell type A from core *i* with cell type B from an
  **unrelated** core *j*. There is no possible real relationship → method must
  return **0 / "none"** (and never "robust"). This is the cleanest negative
  control and works on every dataset above. (Implemented synthetically in
  `validate_internal_controls.py`; here it runs on real coordinates.)
- **Expected:** **0** (verdict "none") overwhelmingly.

### 14. Allen Brain / MERFISH cortical layers  **[batch-friendly]**
- **Link:** https://alleninstitute.org / Vizgen MERFISH mouse brain receptor map.
- **Modality / markers:** MERFISH; cortical layer markers.
- **Why it tests us:** laminar layers are a strong, clean **−** (adjacent-but-segregated) at a known scale — a non-tumor architecture stress test.
- **Expected:** L4↔L6 markers **−** ; within-layer **+** .

---

## Prioritization for heavy external (Codex) batch validation

1. **Schürch CRC CODEX** — already wired; ground-truth +/−; rerun through the new null.
2. **Keren TNBC MIBI** — published mixed/compartmentalized split = built-in +/− calibration.
3. **Jackson + Danenberg breast IMC** — hundreds of cores → real power/size statistics.
4. **HuBMAP tonsil/intestine CODEX** — hardest shared-architecture calibration test.
5. **Xenium + MERSCOPE** — cross-platform robustness.
6. **Cross-sample SWAP** on all of the above — the universal negative control.
7. **Phillips CTCL** — confirmatory CD8(+TIM-3) only if the user obtains access.

Datasets where our output **contradicts** the published relationship are the ones
we most need surfaced — the Codex prompt (`CODEX_VALIDATOR_PROMPT.md`) must flag
each such case explicitly.
