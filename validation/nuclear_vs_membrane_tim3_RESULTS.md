# Nuclear vs membrane on a membranous marker (TIM-3)

**Question:** for a marker expressed on the cell surface, does measuring DAB inside the
nucleus do as well as measuring the cytoplasmic ring — particularly on well-stained tissue?

**Data:** 599 hand-labelled cells (281 positive / 318 negative) across four CRC-ICM TIM-3
fields. Held out **by image**, never by cell. Single-number rules have their cutoff chosen
on the training images and applied to the held-out one, so every arm is scored identically.
Pixel size 0.5 µm/px (CRC-ICM carries no metadata).

Harness: `validation/nuclear_vs_membrane_tim3.py`.

## Held-out result

| arm | AUC | F1 | precision | recall |
|---|---|---|---|---|
| 1. nuclear cutoff — *Membranous OFF* | 0.933 | 0.808 | 0.725 | 0.911 |
| 2. ring-mean cutoff | 0.757 | 0.701 | 0.592 | 0.858 |
| 3. completeness, **auto** pixel threshold | 0.876 | 0.790 | 0.710 | 0.890 |
| 4. nuclear classifier (6 features) | 0.925 | 0.848 | 0.825 | 0.872 |
| **5. membrane classifier (9 features)** | **0.948** | **0.881** | 0.841 | 0.925 |
| 6. completeness, threshold fitted on labels | 0.637 | 0.635 | 0.636 | 0.633 |

Per-image held-out F1:

| arm | 9212046_CT | **92290_IM** | 92625_CT | 92658_IM |
|---|---|---|---|---|
| 1. nuclear cutoff | 0.943 | 0.525 | 0.892 | 0.898 |
| 2. ring-mean cutoff | 0.854 | 0.384 | 0.826 | 0.764 |
| 3. completeness (auto) | 0.920 | 0.458 | 0.904 | 0.889 |
| 4. nuclear classifier | 0.958 | 0.662 | 0.917 | 0.778 |
| 5. membrane classifier | 0.960 | 0.667 | 0.941 | 0.900 |
| 6. completeness (fitted) | 0.793 | 0.449 | 0.726 | 0.571 |

`92290_IM` is the visibly faint slide and is the known failure case.

## What this says

**A nuclear cutoff on a membranous marker is much better than the compartment argument
predicts — AUC 0.933, F1 0.808.** It beats *every* single-number ring rule, including
completeness. The reasoning that nuclear measurement "reads mostly background" for a
surface marker is wrong on this data, and that claim has been removed from the UI. The
likely mechanism is bleed-through: strong membrane staining spills across the nuclear
mask, so nuclear OD tracks membrane positivity by side effect. It is still the wrong
compartment in principle; it is simply not the disaster the principle implies.

**The ring does add information, but only as a combination.** The membrane classifier is
the best arm on every slide and pooled (AUC 0.948, F1 0.881). No single ring number gets
near it: ring-mean is the worst arm in the table, and completeness on its own (0.876) sits
below the plain nuclear cutoff. What the nine features buy is the joint use of
completeness, contiguity, arc count and ring-vs-nucleus contrast.

**The margin is small, and on well-stained slides it is nearly nothing.** Membrane
classifier vs nuclear cutoff, per slide: **+0.017, +0.049, +0.002** F1 on the three
well-stained fields. Against the *nuclear classifier* the pooled gap is +0.033 F1 and
+0.023 AUC on 599 cells across 4 folds — not a decisive margin. So, to answer the question
directly: **on well-stained tissue, nuclear is very nearly enough.** The membranous path
earns its labelling cost on the margin and on principle, not by a large measured gap.

**The self-calibrating pixel threshold is not a compromise — it is the better estimator.**
Arm 3 (auto: median + 3·MAD of the image's own ring pixels) reaches AUC 0.876. Arm 6, the
original approach of fitting the threshold on labelled negatives and carrying it to another
image, reaches **0.637**. A threshold fitted on one slide's staining is a constant applied
to a different slide's staining; recomputing it per image is what makes the feature
transfer at all. Arm 6 also cannot ship, since it needs labels for the slide being scored.

**Nothing works on the faint slide.** Best arm on `92290_IM` is F1 0.667, and the
single-number rules collapse to 0.38–0.53. This is the contrast floor: where the stain does
not clear the ring background, no rule separates the classes. It is a staining outcome, and
it is why the classifier's applicability gate hands such images back to the fixed cutoff.

## Caveats

- Four images, one cohort, one scanner. The per-slide spread (F1 0.46–0.96) is wider than
  every difference between arms, so these are not tight estimates.
- The labels are one person's calls on DAB morphology, not an orthogonal ground truth such
  as immunofluorescence.
- Arm 6 is not comparable to the earlier `tune_membrane_threshold.py` figure (held-out
  F1 0.83 / AUC 0.90): that harness used a different decision rule and a different fold
  protocol. This table's arms are only comparable to each other.
