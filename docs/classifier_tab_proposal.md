# Per-cohort classifier tab — design proposal

Status: **proposal, nothing built.** Written against the code as of `29cd742`.

The shipped default is one fixed DAB cutoff per stain across the cohort, reviewed against
the measured distribution before any output exists (`ihc.md` § 11.4). This proposes the
tier above it: a classifier the user trains **on their own cohort**, from their own
labelled cells, that then calls every image in that cohort.

---

## 1. Why per-cohort, and not a pretrained model

There is no pretrained positivity classifier worth shipping, and the reason is structural
rather than a gap someone will fill next year.

The strongest published attempt is the **Universal Immunohistochemistry Analyzer** (npj
Precision Oncology, Dec 2024), built specifically to generalise across immunostains and
cancer types. On unseen IHC it reaches **κ 0.578** — moderate agreement — trained on
PD-L1/HER2. Against that, this project's own per-cohort membrane calibration reaches
**held-out AUC 0.90 / LOIO F1 0.83** on real CD8, and **AUC 0.93** on the CRC-ICM TIM-3
preset. Per-cohort fitting beats the universal model by a wide margin.

`ihc.md` § 3.3 already states why: *DAB is not quantitative — cutoffs don't transfer across
antibody/scanner.* A universal positivity model would have to be immune to exactly the
variation that breaks fixed cutoffs. The clean way to say it:

> **Segmentation generalises because a nucleus looks like a nucleus. Positivity does not
> generalise because positivity is a property of the assay, not of the morphology.**

That is why InstanSeg ships pretrained and no positivity model does. DeepLIIF is separately
unusable — Apache 2.0 **with Commons Clause** (GitHub reports `NOASSERTION`; not
OSI-approved), incompatible with our MIT licence and with JOSS — and it would not remove
the arbitrary constant anyway: its `PostProcessSegmentationMask.py` classifies on a hard
`seg_thresh=150` applied to a GAN-generated mask, which is *less* defensible to a
pathologist than an optical density. Its legitimate role here is as a ground-truth
generator, which is how § 7 already uses it.

## 2. When the classifier earns its place

Not always, and the tab should say so.

Below roughly 20 slides, the fixed cutoff plus the review step is the better tool: the
operator can eyeball every image, and a cutoff at a stated OD is far easier to defend than
a fitted model. The classifier earns its keep at cohort scale, where per-slide eyeballing
is impractical **but a single consistent rule is still required**.

Its real advantage over a fixed cutoff is not "it is trained" — it is that it can use
features a single global OD cannot express, above all **local background correction**
(§ 4). A fixed cutoff cannot handle a regional staining gradient across one section; a
feature measured relative to neighbouring cells can.

## 3. What already exists

`oasis/webui/calibration.py` is most of this tab already:

| Piece | State |
|---|---|
| Ring measurement of labelled cells | `_measure_labeled` — done |
| Pooling labels across several slides | `fit_multi` — done |
| Held-out honesty metric | `_loo_f1_auc` — **leave-one-CELL-out; this is the flaw** |
| Callability gate | held-out AUC ≥ 0.75 — done |
| Hand-label harness | separate browser tool — done |

### The one thing that must change

`_loo_f1_auc` holds out one **cell** at a time. Cells within a slide share staining run,
illumination, section thickness and operator — they are not independent, so leave-one-cell-out
scores a model on cells whose own slide is still in the training set. It systematically
overstates generalisation.

The evidence is in this project's own numbers: pooled leave-one-**image**-out gives
**F1 0.83**, while the faint slide alone scores **F1 0.30**. Per-image variation is exactly
what leave-one-cell-out hides, and it is the variation that decides whether a cohort-wide
rule holds.

**Leave-one-image-out becomes the headline metric, and the tab must report the per-fold
spread, not a single averaged number.** An average of 0.83 across folds of {0.91, 0.88,
0.30} is a different object from a tight 0.83, and only the spread distinguishes them.

## 4. Features

A classifier can only combine the features it is given. This is the part that decides
whether it works, and it is where the membranous answer lives.

### Shared (both marker types)

| Feature | Why |
|---|---|
| `dab_mean`, `dab_p90` | the signal itself |
| `dab_minus_local_bg` | **the key one.** DAB minus the median DAB of the *k* nearest cells. Cancels regional staining gradients that a global cutoff cannot see. |
| `dab_over_h_frac` | fraction of pixels where DAB exceeds hematoxylin. Removes counterstain cross-talk — the failure that put background OD at 0.23–0.47 in the Phase-1 diagnostic. |
| `hema_mean` | catches over-stained nuclei masquerading as DAB-positive |

### Nuclear only

`nucleus_area`, `nucleus_cell_area_ratio` (QuPath uses the same), `dab_cv` within the
nucleus — punctate versus homogeneous staining.

### Membranous only

| Feature | State |
|---|---|
| `membrane_pos_frac` — completeness at a calibrated pixel OD | exists (`cell_expansion.py`) |
| `ring_mean`, `ring_p90` | exists |
| `ring_minus_nucleus` — arc brighter than the cell's own nucleus | cheap to add |
| **`ring_connectivity`** — longest contiguous positive arc ÷ ring circumference | **does not exist; propose adding** |
| `ring_arc_count` — number of separate positive arcs | propose adding |

`ring_connectivity` is the HER2 lesson. Visiopharm's HER2-CONNECT does not threshold
intensity harder for membranous markers — it skeletonises membrane fragments and scores
**connectivity** on 0–1, cutting at 0.12 and 0.56, and reaches κ 0.86–0.87 against
pathologists. Aperio's Membrane algorithm likewise scores intensity *and* completeness. The
field's answer to membranous staining is geometry, not a better threshold.

This matters because of a measurement already in hand: **ring separability is barely above
nuclear on this tissue — CD8 1.51 vs 1.48, TIM-3 1.50 vs 1.33.** The current ring features
carry little more information than the nuclear ones. Training a classifier on them would
produce a rigorously validated wrapper around weak features: the validation would be honest
and the answer would still be mediocre.

> **Adding `ring_connectivity` is a prerequisite for the membranous path, not an
> enhancement of it.** It should be built and measured *before* the classifier, and it
> should be justified by a measured lift in separability on the labelled set — not adopted
> because the HER2 literature says so.

One caveat carried forward: the faint-TIM-3 failure was diagnosed as a stain **contrast**
floor, not ring placement (`legacy/nuclear_adaptive/`, and the Cellpose investigation).
No feature recovers contrast that was never captured. Connectivity should help clean
staining; it will not rescue the faint slide, and the abstain gate (§ 6) is what handles
that case.

## 5. Model

**Regularised logistic regression, implemented in numpy.** ~8–12 engineered features,
L2 penalty, features standardised on the training folds only.

Reasons, in order:

1. **No new dependency.** `requirements.txt` has no sklearn and the bundle is already
   ~288 MB against a 600 MB CI guard. Logistic regression is ~30 lines of numpy.
2. **Interpretable.** Coefficients are readable, so the tab can state *"this classifier is
   mostly ring completeness and local contrast"*. The DPA requires a pathologist to approve
   the operating point; they can interrogate a weighted sum, not a forest.
3. **Calibrated probabilities**, which the abstain band (§ 6) needs.
4. **Will not overfit at n ≈ 300** labelled cells across a handful of slides. A gradient-boosted
   forest would need sklearn, would overfit at this n, and would buy nothing interpretable.

The existing single-feature threshold fit stays as the degenerate case, so Calibrate's
current behaviour is a special case of the new machinery rather than a parallel path.

## 6. Faint tissue, and why the fixed cutoff still wins there

This is the case the user specifically asked about, and the honest framing is narrow:

> On weak or faint tissue a fixed OD is **safer, not more accurate**. It fails *closed* —
> it under-calls. An adaptive or trained rule fails *open*: it finds the best split of
> noise and manufactures positives.

That is measured, not asserted. The parked GMM at the shipped operating point returned
0.0022 on `Tim3_x10_2` and called **28.8 % positive** against 0.1 % at the fixed cut.
Percentile-normalised thresholding called **99.3 %** positive on `Tim3_x10_1`. A trained
classifier is subject to the same failure if allowed to run on an image unlike anything it
was trained on.

So the classifier gets **two** gates:

**Per-cell abstain.** Predicted probability in a band around 0.5 → abstain, counted and
reported rather than silently forced to a call.

**Per-image applicability gate (the important one).** Before scoring an image, compare its
feature distribution to the training range — median DAB, DAB-over-H fraction, ring contrast.
An image outside that range is **refused**: the classifier does not run, the image falls
back to the fixed cohort cutoff, and it is flagged `staining_quality: low`. A model trained
on well-stained slides must never be permitted to render a confident verdict on a faint one.

## 7. Output contract

The classifier writes **the same `classification` property into the GeoJSON** that the
threshold path writes.

Quant, Spatial and batch then need **zero** changes — `oasis/spatial/spatial.py:88` already
reads `props.classification.name`, and `reclassify.apply_threshold` already owns the
write-back including the detections CSV. The classifier becomes a third way to fill one
contract, not a parallel pipeline.

Provenance travels the same road as the override (§ 11.4): the summary records the
classifier name, its hash, its LOIO metrics, and per-image whether the classifier ran or
the applicability gate refused it.

## 8. Minimum data, and what the tab refuses

| Condition | Behaviour |
|---|---|
| < 3 images | **block** — leave-one-image-out is undefined |
| < 5 images | allow, but state that the held-out estimate rests on very few folds |
| < 50 cells of either class | **block** |
| any fold with one class only | **block** that fold, report it |
| held-out AUC < 0.75 | allow saving, **refuse to apply cohort-wide** |
| no LOIO report | **refuse to apply cohort-wide** |

The user's own instinct here was right: at three images, manual thresholding is genuinely
fine and arguably better, because every slide can be checked by eye. The floor should be
set where the held-out number starts to mean something, not at the point where the
arithmetic stops erroring.

## 9. Build order

1. `ring_connectivity` + `ring_arc_count`, measured against the existing labelled sets.
   **Gate: does separability actually improve?** If not, the membranous path stops here and
   that is a finding worth writing down.
2. Feature extraction module with a fixed, versioned feature contract.
3. Leave-one-image-out harness, replacing `_loo_f1_auc`, reporting per-fold spread.
4. Logistic regression fit + persistence with provenance.
5. Applicability gate.
6. Tab UI, reusing the existing hand-label harness.
7. Wire into batch quant and spatial through the shared `classification` contract.

Steps 1–3 are worth doing regardless: they improve the Calibrate tab that already ships,
and step 1 is the honest test of whether the membranous classifier is worth building at all.

## 10. Where this could be wrong

- **The features may not carry the signal.** Step 1 is designed to find that out early
  rather than after the tab is built.
- **This is parity with QuPath**, which has had trained object classifiers since 0.2. That
  is fine — it is the right design and being second to it does not make it wrong — but it
  should not be presented as the novel contribution in the paper. The novel contribution
  is the spatial association pipeline.
- **Labelling burden is real** and falls on the user, per cohort, per marker. The tab should
  be honest that this is the cost of a defensible answer, and that the fixed cutoff plus the
  review step remains a legitimate choice.
- **A classifier is less interpretable than an OD.** The defensibility comes entirely from
  the held-out validation. If the LOIO report is ever allowed to become optional, the
  classifier is strictly worse than the cutoff it replaced.
