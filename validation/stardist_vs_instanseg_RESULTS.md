# StarDist vs InstanSeg — brightfield nuclear detection (DeepLIIF)

**Question:** how does StarDist brightfield cell detection fare against the pipeline's
current InstanSeg segmenter, on the same data and the same metric?

**Data:** DeepLIIF testing set, 598 images / **41,428** IF-derived ground-truth cell
centroids (`~/oasis_validation_datasets/DeepLIIF/_generated_outputs/pipeline_validation`).
GT is co-registered-IF-derived, not hand-labelled.

**Metric:** greedy one-to-one centroid matching, 15 px tolerance → detection-recall
(matched/GT), detection-precision (matched/pred), detection-F1. Identical scorer for both.

**Apples-to-apples run conditions:** both detectors run headless in QuPath 0.7.0 via the
same `QuPath script -i` call, `BRIGHTFIELD_H_DAB`, pixel size 0.5 µm, full-image
annotation, same GeoJSON export. Only the detector differs.
- **InstanSeg:** `brightfield_nuclei-0.1.1` (in-domain, purpose-built for brightfield).
- **StarDist:** `dsb2018_heavy_augment.pb` (QuPath StarDist ext v0.6.0) on the deconvolved
  **hematoxylin** channel, `normalizePercentiles(1,99)`, `threshold(0.5)` — the standard
  QuPath route for brightfield/IHC nuclei. Note: this is a **fluorescence-trained** model
  repurposed to brightfield (out-of-domain); TensorFlow has no Python-3.14 build, so the
  native `stardist` package cannot run here — QuPath's bundled TF path was used.

## Headline (default operating point, threshold 0.5)

| detector | det-recall | det-precision | det-F1 | # preds |
|---|---|---|---|---|
| **InstanSeg** (brightfield_nuclei) | 0.752 | **0.871** | **0.807** | 35,752 |
| StarDist (dsb2018 → hematoxylin) | **0.853** | 0.546 | 0.665 | 64,780 |

**InstanSeg wins by +0.142 det-F1.** StarDist finds *more* true nuclei (recall +0.10) but
over-detects massively — 64,780 objects vs ~41k GT — so precision collapses (−0.33).
Paired per-image: **InstanSeg better on 580/598**, StarDist on 18, 0 ties.
Per-tissue det-F1 — Bladder 0.827 vs 0.700; Lung 0.801 vs 0.654.

## Is StarDist just mis-tuned? — post-hoc operating-point sweep

QuPath did not export per-detection probability, so a prob-threshold sweep wasn't possible
from these files; hematoxylin-intensity / area post-filters trace the achievable frontier
(higher-confidence nuclei ≈ higher hematoxylin), as a proxy:

| filter | recall | prec | F1 | #preds |
|---|---|---|---|---|
| none (thr 0.5) | 0.853 | 0.546 | 0.665 | 64,780 |
| **Hematoxylin:Mean ≥ 0.15** | 0.840 | 0.635 | **0.723** | 54,755 |
| Hematoxylin:Mean ≥ 0.25 | 0.608 | 0.765 | 0.678 | 32,955 |
| Hematoxylin:Mean ≥ 0.35 | 0.310 | 0.845 | 0.453 | 15,180 |
| Area ≥ 20 µm² | 0.843 | 0.564 | 0.675 | 61,970 |

- **Best achievable StarDist det-F1 ≈ 0.723**, still **−0.084 below InstanSeg (0.807)**.
- Area filtering barely moves the needle → the excess detections are **genuine spurious
  nuclei calls** (background hematoxylin), not small splitting fragments.

## Verdict

On this brightfield DAB-IHC data, **InstanSeg's purpose-built `brightfield_nuclei` model
beats StarDist at every operating point tested** (0.807 vs best-case 0.723). StarDist's one
edge is higher recall (0.85 vs 0.75) — useful only if downstream precision is cheap to
recover, which here it is not. The comparison is honest but not maximally favorable to
StarDist: it pits an in-domain brightfield model against a repurposed fluorescence model.
Not tested: StarDist's literal RGB H&E model (`he_heavy_augment`), and a true
probability-threshold PR curve (needs a re-run with `.includeProbability(true)`).
