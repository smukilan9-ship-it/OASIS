# Membrane ring-geometry investigation — would Cellpose help the faint TIM-3 case?

Harness: `validation/investigate_membrane_ring_geometry.py`
Data: `TIM3_CRC_ICM` labeling set (InstanSeg detections + images), resolved via
`validation_data_dir`. `92290_TIM3_IM` is the known faint failure (held-out F1 0.30);
`92658_IM / 92625_CT / 9212046_CT` are adequately-stained controls.

## The question
The faint failure could be (a) a **ring-placement** problem — InstanSeg-nuclei + Voronoi
expansion measures the wrong pixels, and a true cell/cytoplasm boundary (Cellpose-cyto) would
fix it — or (b) a pixel-level **stain-contrast floor** a segmenter cannot touch. Cellpose only
changes *where the ring sits*, so we sweep the ring geometry ourselves (expansion 1→6 µm) and
watch whether the positive-population separability responds.

## Result — separability is flat across the entire geometry sweep

| image | pos_rate (1→6 µm) | ring-DAB separability (Ashman D) | Δ across geometry |
|---|---|---|---|
| **92290_IM (faint)** | 0.67 → 0.72 | 1.66 → 1.71 | **0.05** |
| 92658_IM | 0.35 | 1.54 → 1.69 | 0.14 |
| 92625_CT | 0.41 → 0.37 | 1.75 → 1.78 | 0.03 |
| 9212046_CT | 0.28 | 1.50 → 1.55 | 0.05 |

Sweeping the ring from tight (1 µm) to wide (6 µm) — a 6× change in geometry — moves the
positive/negative separability by **Δ ≤ 0.14 on every image, and 0.05 on the faint one.**
Ring placement is not the lever.

## Verdict: Cellpose will not recover the faint case

Because a 6× change in ring geometry leaves separability essentially unchanged, a different
segmenter that produces a different ring placement cannot recover the missing separability
either. The TIM-3 membranous signal is **weakly separable (Ashman D ≈ 1.5–1.8) on all four
images** — consistent with the user's "biological floor" — and the faint failure is a
pixel-level **stain-contrast** problem (on 92290 the DAB>H dominance saturates at 99.8%, so
positive and negative rings are indistinguishable → over-calling, pos_rate 0.67–0.72), not a
geometry problem. No cell-boundary model fixes uninformative pixels.

Honest caveat: a concentric-expansion sweep is a *proxy* for Cellpose's learned boundary, not
an exact replica. But the flatness across a wide geometry range, plus the dominance saturation
being a per-pixel (geometry-independent) quantity, make ring placement an implausible bottleneck.

## Recommended action (not Cellpose)
Strengthen the membrane-quality **abstain gate** (a partial one already exists in
`run_pipeline._apply_cytoplasm_measurement`, ~line 553): flag/withhold when the per-image
DAB>H dominance saturates or the ring-DAB separability is below threshold, so faint slides are
reported as *uncallable* rather than silently over-called. This is the same fail-closed
philosophy as the nuclear abstain gate and the registration/bandwidth gates. Cellpose remains
a possible future experiment for a *different* problem (whole-cell segmentation in dense
infiltrate), but it is not the fix for the faint-tissue floor.

Reproduce: `python validation/investigate_membrane_ring_geometry.py`
