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

---

## Follow-up (2026-07-14): the faint failure is a FIXABLE stain-separation bug

The ring-geometry result above correctly rules out Cellpose (ring placement is not the lever),
but a deeper stain probe shows the faint failure is **not an irreducible contrast floor** — it
is a fixable deconvolution-*selection* bug.

**Diagnosis.** On 92290 the ring hematoxylin OD is **negative (−0.095)** — physically
impossible (hematoxylin absorbs). Cause: `measure_cytoplasm_dab`'s stain-vector parity
selection picks the **fixed** QuPath vectors on this colour-tinted slide (estimated white point
≈ [237, 249, **203**], not [255,255,254]); the fixed vectors then produce a negative H channel,
so DAB beats H *everywhere* → DAB>H dominance saturates at 99.8% → 67% of cells over-called. The
parity metric is **circular** — it rewards correlation with QuPath's own fixed-vector DAB, so it
keeps the broken channel even though the **estimated Macenko channel is physically valid**
(H = +0.161, H-vector cos 0.993).

**Fix.** Reject stain candidates whose hematoxylin channel is physically invalid (negative
median OD over tissue) *before* parity ranking (`_h_validity` in `cell_expansion.py`).

| 92290 (faint) | before | after |
|---|---|---|
| selected channel | fixed | estimated |
| ring H median | −0.095 | **+0.339** |
| DAB>H dominance | 99.8% | **35.1%** |
| over-call (pos_rate) | 67.4% | **50.2%** |

Controls stay stable (9212046 unchanged; 92658 also de-saturates). The guard is a no-op on
properly-stained tissue (only fires when H is negative). **Caveat:** it changes the selected
channel on 2 of the 4 images the membrane completeness cutoffs were fit on, so `pix_thr 0.30 /
frac_min 0.14` need refitting and F1 (0.934 / 0.76) re-confirming on the corrected channel —
gated on the hand-labelled GeoJSONs.

## Cellpose — empirically ruled out

Installed Cellpose-SAM 4.2.1 and compared whole-cell segmentation to InstanSeg+Voronoi on the
same valid stain channel:

| image | Cellpose-SAM | InstanSeg+Voronoi |
|---|---|---|
| 92290 (faint) | **20 cells**, sep 3.16 | 901 cells, sep 1.74 |
| 9212046 (control) | 1864 cells, sep 1.64 | 2561 cells, sep 1.50 |

On the faint image Cellpose finds 20 of ~901 cells (misses ~98%) — the low contrast that breaks
the classifier breaks Cellpose's detection too. On good tissue it is comparable-but-fewer with
no separability gain. Default RGB input, untuned, but the out-of-box result plus the mechanism
make it a non-fix. (Note: Cellpose pulls `opencv-python-headless`, which shadows the pinned
`opencv-python==4.13.0.92` — removed afterwards to protect the KDE-sensitive opencv pin.)

**Conclusion:** the real membranous lever is the stain-selection guard, not Cellpose.
