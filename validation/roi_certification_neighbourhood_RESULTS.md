# Does a certifying ROI hold up when you move the window? (2026-07-28)

**Question.** A local certification says the fitted transform is accurate inside this ROI.
The fit and the residual come from the same correspondences, so nothing in that statement is
checkable from the ROI itself. What *is* checkable: serial-section deformation is smooth, so
a genuine local alignment must certify across a patch of overlapping windows, and those
windows' independently-fitted transforms must place the same point in nearly the same place.
A fit to noise cannot do either.

**Why it had never been measured.** Both prior sweeps excluded the evidence by construction:

- the coarse tiling (`batch.py`) skips any candidate within `1.8R` of a region it already
  kept, so its returned regions are disjoint by definition;
- the exhaustive search (`exhaustive.py`) `break`s on the first window that certifies.

Both therefore reported "one region per pair" for reasons that have nothing to do with the
tissue. Any earlier reading of those counts as evidence of isolation was wrong.

**Method.** `validation/roi_certification_neighbourhood.py`. Full grid at one radius,
windows spaced `R/2` so neighbours share most of their tissue, no early exit, swept over the
tissue bounding box only. Per window: verdict and the locally-refitted transform. Then

- `n_with_a_neighbour` — certifying windows having another certifying window within
  `1.45·step` (the 8-neighbourhood diagonal);
- `overlap_disagree` — for pairs of certifying windows that **overlap**, how far apart their
  two transforms place the midpoint between them. Both windows saw that tissue, so this is
  the fit failing to reproduce itself;
- `distant_disagree` — the same for windows that do not overlap. **This is not error.** Two
  locally-fitted transforms a slide apart are supposed to differ; that difference *is* the
  deformation field, and accommodating it is the whole premise of local certification.

The first version of this harness pooled the two and evaluated every transform at every
certifying centre, reporting 171–334 µm. Those figures were the deformation field wearing the
label "disagreement" and are **retracted**; only `overlap_disagree` is a quality signal.

Cohort: `~/Desktop/Region of interest`, CD8 ↔ TIM-3 serial pairs, 1920 px thumbnails,
`work_max_dim=800`, `local_k=8`, `fle_fast=True` — matching `batch.py` exactly.

## Group A — pairs that found a path only under exhaustive search

Radius and arm are the ones that certified for that pair.

| pair | obj | R µm | certifying | with a neighbour | overlap disagree med |
|---|---|---|---|---|---|
| LL478_junction_10X_3 | 10X | 238 | 2 / 54 | 0 | 21.5 |
| LL478_Tumor_10X_3 | 10X | 238 | 1 / 54 | 0 | — |
| LL478_liver_10X_3 | 10X | 278 | 1 / 28 | 0 | — |
| LL480_Tumor_10X_1 | 10X | 238 | 1 / 54 | 0 | — |
| LL478_junction_20X_1 | 20X | 139 | 1 / 28 | 0 | — |
| LL478_junction_20X_2 | 20X | 139 | 1 / 28 | 0 | — |
| LL480_Liver_20X_2 | 20X | 139 | 1 / 28 | 0 | — |
| LL480_Tumor_20X_3 | 20X | 179 | 1 / 15 | 0 | — |
| **total** | | | **9 / 289** | **0** | |

Eight for eight: a single certifying window, never an adjacent pair. Seven of the eight have
nothing to compare against at all, so no agreement figure exists for them.

## Group B — pairs that certified on the first coarse tiling

| pair | obj | R µm | certifying | with a neighbour | overlap med / max | distant med / max |
|---|---|---|---|---|---|---|
| LL477_Liver_4X_3 | 4X | 600 | 45 / 54 | 45 | **1.7** / 12.5 | 6.7 / 35.4 |
| LL477_Liver_4X_2 | 4X | 600 | 17 / 54 | 16 | **3.6** / 13.1 | 9.8 / 39.4 |
| LL477_Liver_4X_1 | 4X | 600 | 20 / 54 | 20 | **7.1** / 45.9 | 30.9 / 105.1 |
| LL479_Liver_10X_2 | 10X | 260 | 13 / 40 | 12 | **4.1** / 22.8 | 9.3 / 129.9 |
| LL480_Junction_10X_2 | 10X | 260 | 2 / 40 | 0 | **46.7** | — |
| **total** | | | **97 / 242** | **93** | | |

## What this says

**The certification test is not broken.** Where tissue genuinely aligns it says so broadly:
40% of all windows certify in group B against 3% in group A, and **93 of 97** certifying
windows have a certifying neighbour, against **0 of 9**. That contrast is the result. It also
retires the multiple-comparisons worry raised against group A on different grounds: the
difference between the groups is not how many windows were searched.

**And where windows overlap, the fit reproduces itself well.** Four of the five group-B
pairs agree to **1.7–7.1 µm median** on tissue both windows saw — at or inside the 5 µm
cell-error budget the gate already enforces. The certification is not merely passing; it is
passing consistently.

**The distance-dependence is a measurement of deformation, not of error.** Median
disagreement rises from 1.7–7.1 µm between overlapping windows to 6.7–30.9 µm between
distant ones, reaching 130 µm at slide scale. That is the deformation field, and it is why
the method certifies locally rather than globally. Reporting it as a defect — as the first
version of this harness did — mistook the phenomenon for the failure.

**One group-B pair is genuinely out of family.** `LL480_Junction_10X_2` certifies 2 of 40
windows, non-adjacent, and its two overlapping-window transforms differ by **46.7 µm**
against 1.7–7.1 for its peers. This one is worth a look; the rest are not.

**Isolation is not, by itself, evidence of a bad fit.** A researcher who identifies the one
well-preserved island on an otherwise deformed section and draws around it produces the same
signature as group A: one certifying window, failing neighbours. Nothing here distinguishes
those two cases, so **`n_with_a_neighbour` must not become a gate** — it would reject exactly
the use the ROI workflow exists for. Report it; do not veto on it.

## Caveats

- Thirteen pairs of one cohort, one lab, one scanner. Group B is five pairs, three of them
  4× and four of them liver.
- Group B radii come from `batch.py`'s size ladder for that pair, group A radii from
  whichever radius certified. Radii are therefore not matched between the groups, and window
  radius affects both the verdict and the number of windows a slide holds.
- `overlap_disagree` is not a target registration error — there are no independent landmarks
  here. It says two windows that saw the same tissue place it in the same spot; it cannot say
  that spot is correct. Both could be wrong together.
- The 4× pairs agree better than the 10× ones, but no block was imaged at more than one
  objective, so magnification and tissue are confounded.
- `LL477_x4_2_scale` was dropped from group B. It is the same field as `LL477_Liver_4X_2`
  with the scale bar burnt in (different files, same tissue), and it reproduced that pair's
  numbers exactly — a free confirmation that the pipeline is deterministic, but not an
  independent pair. It should also come out of the cohort denominator: 75 pairs, not 76.

## Reproducing

```bash
# group A, per pair, at the radius and arm that certified
.venv/bin/python validation/roi_certification_neighbourhood.py \
    --pairs LL478_junction_10X_3 --radius-um 238 --arm identity
.venv/bin/python validation/roi_certification_neighbourhood.py \
    --pairs LL478_Tumor_10X_3 --radius-um 238 --arm register_similarity

# group B, at batch.py's region_um for that pair (which is a RADIUS, not a diameter)
.venv/bin/python validation/roi_certification_neighbourhood.py \
    --pairs LL477_Liver_4X_1,LL477_Liver_4X_2,LL477_Liver_4X_3 --radius-um 600
.venv/bin/python validation/roi_certification_neighbourhood.py \
    --pairs LL479_Liver_10X_2,LL480_Junction_10X_2 --radius-um 260
```

Full per-window records, including every certifying window's transform:
`validation/roi_certification_neighbourhood_results.json`.
