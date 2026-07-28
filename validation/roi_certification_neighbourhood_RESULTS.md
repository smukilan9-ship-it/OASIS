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
- `disagree_um` — for each pair of certifying windows, the max displacement between their
  transforms evaluated at all certifying centres; reported as median and max over pairs.

Cohort: `~/Desktop/Region of interest`, CD8 ↔ TIM-3 serial pairs, 1920 px thumbnails,
`work_max_dim=800`, `local_k=8`, `fle_fast=True` — matching `batch.py` exactly.

## Group A — pairs that found a path only under exhaustive search

Radius and arm are the ones that certified for that pair.

| pair | obj | R µm | certifying | with a neighbour | disagree med | max |
|---|---|---|---|---|---|---|
| LL478_junction_10X_3 | 10X | 238 | 2 / 54 | 0 | 201.5 | 201.5 |
| LL478_Tumor_10X_3 | 10X | 238 | 1 / 54 | 0 | — | — |
| LL478_liver_10X_3 | 10X | 278 | 1 / 28 | 0 | — | — |
| LL480_Tumor_10X_1 | 10X | 238 | 1 / 54 | 0 | — | — |
| LL478_junction_20X_1 | 20X | 139 | 1 / 28 | 0 | — | — |
| LL478_junction_20X_2 | 20X | 139 | 1 / 28 | 0 | — | — |
| LL480_Liver_20X_2 | 20X | 139 | 1 / 28 | 0 | — | — |
| LL480_Tumor_20X_3 | 20X | 179 | 1 / 15 | 0 | — | — |
| **total** | | | **9 / 289** | **0** | | |

Eight for eight: a single isolated window, never a patch. The one pair with two hits places
the same point 201 µm apart depending on which window you believe.

## Group B — pairs that certified on the first coarse tiling

| pair | obj | R µm | certifying | with a neighbour | disagree med | max |
|---|---|---|---|---|---|---|
| LL477_Liver_4X_3 | 4X | 600 | 45 / 54 | 45 | 17.0 | 67.7 |
| LL477_Liver_4X_2 | 4X | 600 | 17 / 54 | 16 | 25.3 | 74.8 |
| LL477_Liver_4X_1 | 4X | 600 | 20 / 54 | 20 | 73.8 | 170.6 |
| LL479_Liver_10X_2 | 10X | 260 | 13 / 40 | 12 | 28.1 | 189.7 |
| LL480_Junction_10X_2 | 10X | 260 | 2 / 40 | 0 | 333.6 | 333.6 |
| **total** | | | **97 / 242** | **93** | | |

## What this says

**The certification test is not broken.** Where tissue genuinely aligns it says so broadly:
40% of all windows certify in group B against 3% in group A, and **93 of 97** certifying
windows have a certifying neighbour, against **0 of 9**. That contrast is the result. It also
retires the multiple-comparisons worry raised against group A on different grounds: the
difference between the groups is not how many windows were searched.

**But contiguity and agreement are separable, and agreement is the weaker of the two.**
`LL477_Liver_4X_1` certifies 20 contiguous windows whose transforms disagree by 74 µm median
and 171 µm max. `LL479_Liver_10X_2` looks healthy at 28 µm median and reaches 190 µm at its
worst pair. Windows can certify next to each other while describing measurably different
alignments — which is worse than not certifying, because the verdict reads the same.

**One group-B pair fails the test outright.** `LL480_Junction_10X_2` certifies 2 of 40
windows, not adjacent, 334 µm apart. It is indistinguishable from group A despite having
certified on the first tiling. So passing the shipped certification does not by itself imply
a neighbourhood — the property has to be measured.

**Agreement degrades with magnification, on this data.** The three 4× pairs sit at 17–74 µm
median; the two 10× pairs at 28 µm and (failing) 334 µm. Group A is entirely 10× and 20×.
Whether that is magnification or which tissue was imaged at which magnification is not
separable here: no block was swept at more than one objective.

**`n_certifying` alone is not a quality measure.** `LL477_Liver_4X_3` (45/54, 17 µm) and
`LL477_Liver_4X_1` (20/54, 74 µm) differ more in agreement than the window counts suggest.

## Caveats

- Thirteen pairs of one cohort, one lab, one scanner. Group B is five pairs, three of them
  4× and four of them liver.
- Group B radii come from `batch.py`'s size ladder for that pair, group A radii from
  whichever radius certified. Radii are therefore not matched between the groups, and window
  radius affects both the verdict and the number of windows a slide holds.
- Transform disagreement is evaluated at the certifying centres, so it measures spread across
  the certified patch. It is not a target registration error: there are no independent
  landmarks here. It cannot say which window is right, only that they differ.
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
