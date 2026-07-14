# Nuclear classifier benchmark — results

Harness: `validation/validate_nuclear_classifier.py`
Data: DeepLIIF **Ki67** (nuclear marker), IF-derived per-cell ground truth, resolved via
`validation_data_dir` (`~/oasis_validation_datasets/DeepLIIF`). Segmentation (InstanSeg)
is held fixed — the pre-generated GeoJSON detections are reused — so the numbers reflect
**classification only**, not detection recall. Each detected cell is matched one-to-one to
a GT cell and scored under every method on the same cells.

Classifier under test: `oasis/quant/nuclear_classify.py` (`classify_nuclear`), a
two-component GMM valley-finder with a BIC + Ashman's-D abstain gate. Numpy EM, no sklearn.

## Headline (all 598 images, 31,154 matched cells, GT positive fraction 0.184)

| method | F1 | precision | recall | accuracy | kappa | abstained |
|---|---|---|---|---|---|---|
| fixed @ 0.20 OD (DAB:Mean) | 0.809 | 0.788 | 0.832 | 0.928 | 0.765 | — |
| **otsu on DAB:Mean (current adaptive)** | **0.793** | 0.861 | 0.735 | 0.929 | 0.751 | — |
| **gmm on DAB:Mean (new, ashman_min=1.25)** | **0.808** | 0.794 | 0.823 | 0.927 | 0.763 | **18 / 598 = 3.0%** |

- abstained-set F1 (gmm, under fixed fallback): **0.617** — the gate withholds the genuinely
  harder images (retained 0.808 vs abstained 0.617).
- Channel separability (ROC-AUC): DAB:Mean **0.933**, Macenko 0.927 (tied).

## Decisions this benchmark drove

1. **Replace the current in-Groovy Otsu adaptive.** It is the weakest method at every sample
   size (recall collapses to 0.735 because it over-thresholds a not-cleanly-bimodal
   distribution). On a *unimodal* marker it is catastrophic — a synthetic all-negative marker
   yields 526/1000 false positives under Otsu, vs a correct **abstain** under the GMM gate.
2. **Channel = QuPath DAB:Mean, not per-image Macenko.** Their ranking power is tied (AUC
   0.933 vs 0.927), but at a usable operating point the GMM on DAB:Mean abstains on ~3–5% of
   images while the Macenko channel abstains on 31–47% for no F1 gain — per-image Macenko adds
   instability, not robustness, on nuclear tissue. Keeps nuclear fully on QuPath.
3. **Operating point ashman_min = 1.25** → 3.0% abstention (rare, as required), while retained
   F1 (0.808) ties the well-calibrated fixed cutoff (0.809).

## The honest caveat

On clean, well-calibrated DeepLIIF Ki67, GMM-on-DAB:Mean and a fixed 0.20 OD cutoff are
statistically indistinguishable in F1. The value of the adaptive path is **not** a raw F1
jump on clean data; it is (a) removing the broken Otsu, (b) needing no manually-tuned
per-marker cutoff, and (c) the abstain safety net that a fixed cutoff lacks. A synthetic
stain-perturbation test (adaptive should hold as a fixed cutoff drifts) would strengthen the
"adaptive > fixed on variable staining" claim and is the recommended next validation.

Operating point sweep (120-image subsample) for reference:

| ashman_min | gmm_dabmean F1 | abstained |
|---|---|---|
| 1.0 | 0.800 | 3/120 (2.5%) |
| 1.3 | 0.803 | 6/120 (5%) |
| 1.6 | 0.807 | 17/120 |
| 2.0 | 0.813 | 56/120 |

Reproduce: `python validation/validate_nuclear_classifier.py 598 --no-macenko --ashman 1.25`
