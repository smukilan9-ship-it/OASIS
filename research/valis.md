# VALIS in OASIS: what it was for, what we measured, and why it was removed

**Status:** evaluated 2026-07-18 → 2026-07-19, shipped briefly as an optional registration
engine, **removed from the application 2026-07-25**. Code and design documents retained under
`legacy/valis/`; the benchmark harness remains live at `validation/valis_bench/`.

This document exists because the negative result is worth more than the feature would have been.
VALIS is an excellent registration tool and it does the thing we wanted it to do. It was removed
anyway, for a reason specific to what OASIS claims about its own output. That distinction is the
point of this note.

---

## 1. What OASIS needs from registration

OASIS measures **cross-type spatial association** between two cell populations stained on
*serial sections* — e.g. CD8 on section *n* and TIM-3 on section *n+1* — using cross-type
Ripley's K against a null model. The statistic is computed on cell coordinates after mapping one
section into the other's frame.

Two constraints follow, and they drive everything below:

1. **The transform must be distance-preserving.** Ripley's K is a function of inter-point
   distances. A non-rigid warp changes those distances and would manufacture or destroy
   association. `serial_registration.assert_distance_preserving` enforces this at every entry
   point; VALIS's non-rigid warp was never eligible.
2. **The registration must be *certified* at cell scale, not merely good.** A 20 µm residual is
   an excellent whole-slide registration and a catastrophic one for a statistic whose signal
   lives in a 10–50 µm interaction band. OASIS therefore does not ask "is this aligned?" but
   "can I *prove* this is aligned to better than 5 µm inside this analysis window?" — and it
   fails closed when it cannot.

The certification gate is Fitzpatrick–West target-registration-error propagation over
correspondences that were **not** selected for agreeing with the transform under test. OASIS
gets those correspondences from LoFTR (detector-free matching) or from expert-placed landmarks.

## 2. Why VALIS was considered

The motivating gap was measured, not assumed. On the public ANHIR/CIMA benchmark
(`validation/valis_bench/`, 222 pairs with held-out expert landmarks, one shared scorer, both
methods registering from pixels only), LoFTR's correspondence quality splits cleanly by stain
appearance:

| tissue | pairs | LoFTR produced usable matches |
|---|---:|---:|
| lung-lesion | 30 | 30 (100%) |
| lung-lobes | 40 | 38 (95%) |
| mice-kidney | 15 | 9 (60%) |
| gastric | 13 | 7 (54%) |
| COAD | 84 | 41 (49%) |
| **mammary-gland** | 38 | **0 (0%)** |
| **breast** | 1 | **0 (0%)** |
| **kidney** | 1 | **0 (0%)** |

LoFTR fails *outright* on cross-modal H&E ↔ DAB-IHC pairs. This is a representational limit —
H&E and DAB-IHC do not share the local texture LoFTR keys on — not a tuning problem.

VALIS-rigid, by contrast, registered every pair, and on overall accuracy it beat both OASIS paths:

| method | transform | distance-preserving | MMrTRE | mean rTRE | pairs registered |
|---|---|---|---:|---:|---:|
| no registration | identity | — | 0.0522 | 0.0947 | 44 |
| OASIS-LoFTR | similarity | yes | 0.0052 | 0.0073 | 23 / 44 |
| OASIS-structural | similarity | yes | 0.0052 | 0.0579 | 44 |
| **VALIS-rigid** | rigid | yes | **0.0037** | **0.0070** | **44** |
| VALIS-nonrigid | rigid + warp | **no** | 0.0015 | 0.0047 | 44 |

The honest summary of that table: **on the full diversity of ANHIR, VALIS is the better general
registrar.** OASIS-structural has a comparable median but a mean of 0.058 because it fails
catastrophically on cross-modal and large-displacement pairs (HER2→HE 0.40, CD68→CD4 0.45), all
of which VALIS handled at 0.001–0.01. Within OASIS's own regime — similar-stain serial sections,
which is the actual CD8/TIM-3 use case — the two tie (0.0052 vs 0.0036 over 23 pairs, OASIS
better on 14 of 23).

So the hypothesis was reasonable and specific: **use VALIS-rigid to recover the cross-modal pairs
OASIS cannot register at all, and certify the result with the existing gate.**

## 3. What was built

A subprocess bridge, because `valis-wsi` cannot be imported alongside the main pipeline — it
needs Python 3.11 and a JVM, while OASIS runs on Python 3.14. VALIS ran in an isolated
`~/valis_runtime` virtualenv; the main process shelled out to a worker over stdin/stdout JSON and
received only a rigid moving→reference similarity in original pixels, which was re-checked
main-side with `assert_distance_preserving` before use.

The transform itself was obtained the coordinate-clean way: warp probe points through
`warp_xy_from_to(non_rigid=False)` and solve the 2×3 similarity. This matters — see §4.3.

Measured end-to-end on ANHIR lung-lesion (this machine, 2026-07-25): worker completes in ~18 s,
returns 177 correspondences, VALIS's self-reported rigid rTRE 0.053, and a valid similarity
matrix. **The integration worked. Every part of it did what it was designed to do.**

## 4. Why it was removed

### 4.1 The certification-honesty failure (the decisive one)

The first design let VALIS certify its own registrations using a **structural patch-residual**:
measure the displacement field between the reference and the warped moving image on the
low-frequency hematoxylin channel, which is shared across serial sections regardless of stain.
This is attractive because it is stain-robust — it works cross-modal, exactly where LoFTR cannot
supply correspondences.

Validated against held-out ANHIR expert landmarks, it **over-certified by 2–3×**. On
lung-lesion_1 (Cc10→CD31, 0.348 µm/px) every region was marked certified with a structural
cell-error of 0.5–14 µm, while the held-out expert landmarks inside those same regions were
**17–36 µm** off.

Three measured root causes:

1. **A single global rigid cannot reach cell scale on deforming serial sections.** VALIS's own
   clean rigid transform is 24.65 µm median held-out TRE on this pair. LoFTR reaches 3.66 µm on
   the same tissue *only* because it fits a separate local rigid per ROI. VALIS emits one global
   rigid for the whole slide, by design — it is a whole-slide registrar.
2. **Structural residual ≠ anatomical correspondence.** `phaseCorrelate` on the near-identical
   hematoxylin texture of two serial sections locks onto local texture and "explains away" real
   anatomical displacement. The residual *shrinks toward zero as resolution increases*
   (21 µm → 0.3 µm as max_side goes 1200 → 3400) while the true TRE stays 24.65 µm. A
   certification metric that improves when you look closer at a misregistration is not measuring
   registration.
3. **The anatomical truth is not in the pixels.** VALIS's *non-rigid* warp — its best possible
   alignment, and more than OASIS is allowed to use — is still 16.8 µm from the expert landmarks
   on this pair. The correspondence experts encode is physically absent from the image data for
   hard and cross-modal pairs.

Mitigations were tried and were insufficient. A resolution floor (never measure the residual
finer than 2 µm/px) plus a lumen-centroid TRE veto made the residual resolution-stable and
correctly fail-closed on gross errors — a real improvement, and it was kept as a QC — but local
regions still over-passed at cell scale. Using the rigid-vs-non-rigid gap as a
texture-independent signal correlated only 0.61 with true TRE, and points where rigid ≈ non-rigid
(gap ≤ 3 µm) were *still* 24 µm off. Rejected.

### 4.2 Demoting it to "provisional" made it inert

Given §4.1, VALIS was re-scoped: it would supply the **provisional** transform, and certification
would stay with the validated Fitzpatrick–West gate. This is defensible in principle. In practice
it is a contradiction, because the gate needs LoFTR correspondences — and the pairs VALIS exists
to rescue are precisely the pairs where LoFTR returns nothing.

Measured on this machine (2026-07-25), running the application's own `certify_spatial_auto`:

| pair | `engine=loftr` | `engine=valis` |
|---|---|---|
| CD31 ↔ Cc10 (similar stain) | `mode:none`, `DEFORMED`, 0 regions | `mode:none`, `DEFORMED`, 0 regions |
| H&E ↔ CD31 (**cross-modal**) | `mode:none`, `DEFORMED`, 0 regions | `mode:none`, `DEFORMED`, 0 regions |

**Identical outcomes.** Selecting VALIS could not change a verdict on any pair tested, in either
regime, by construction. Its only observable effect on the user was 20–60 s of additional runtime
and an extra QC line. A feature that cannot change an outcome is not a conservative feature; it
is a misleading one, because its presence in the interface implies a capability that does not
exist.

### 4.3 The remaining path was closed off, and closing it was correct

One route survived on paper: feed VALIS's own correspondences (`MatchInfo.matched_kp{1,2}_xy`)
to the Fitzpatrick–West gate, as is done with LoFTR's. This was investigated and is not viable as
built. Those keypoints live in VALIS's internal feature frame — a mixed multi-detector space with
features at both 512 px and 542 px, combined — which does not map to original per-ROI coordinates
through documented shapes (conversions were off by ~1.34×), and `error_df.rigid_rTRE` is roughly
8× the true error. This is why the integration used probe-point warping rather than internal
keypoints in the first place.

### 4.4 It cannot ship in a standalone application

OASIS is being packaged as a self-contained executable. VALIS requires a second Python
interpreter, a JVM, and libvips, referenced through an absolute path outside the bundle. Even had
§4.1–4.3 gone the other way, this would have made it an optional, environment-dependent extra
rather than a shipped capability.

## 5. What was kept

- **The benchmark** (`validation/valis_bench/`) is live and reproducible. It is the evidence for
  where OASIS's registration is and is not competitive, and it is deliberately non-circular:
  both methods register from pixels only, and scoring is on held-out expert landmarks through
  one shared scorer.
- **The scope boundary it established.** OASIS is a specialised serial-section tool for
  similar-stain IHC↔IHC pairs with a fail-closed gate — not a general histology registrar. The
  benchmark draws that line with measured numbers rather than asserting it.
- **The gate calibration result.** Across 44 pairs the gate never certified a bad registration
  (every pass verdict had independent rTRE 0.0016–0.0045) and erred toward over-conservatism.
  This is the behaviour a fail-closed design should exhibit, and VALIS is what let us test it
  against an independent reference.
- **Manual landmarks as the cross-modal answer.** For cross-modal pairs, no automatic method
  supplies anatomically-faithful correspondences, because — per §4.1(3) — the correspondence is
  not in the pixels. Expert landmarks are not a fallback here; they are the only honest path.

## 6. What would bring it back

A single change would make VALIS scientifically useful in OASIS: **use its transform to
accelerate the manual-landmark path rather than to replace it.** VALIS produces a strong global
alignment on cross-modal pairs where `register_similarity` fails outright; wiring that alignment
into landmark proposal would let an operator confirm five correspondences in seconds instead of
hunting for them across two visually dissimilar images. Certification would remain entirely with
the validated gate and the human, so nothing in the honesty argument above is disturbed.

This was recommended in the original research note and never implemented — `propose_landmarks`
still uses `register_similarity`, which is the method that fails on exactly these pairs. It
remains the one genuinely valuable use of VALIS for this application, and it is the form in which
it should return, if it returns.

## 7. Where the code lives

| path | contents |
|---|---|
| `legacy/valis/valis_engine.py` | main-process bridge: subprocess call, invariant check, structural certification |
| `legacy/valis/valis_worker.py` | isolated-runtime worker (Python 3.11 + `valis-wsi`) |
| `legacy/valis/valis_certification_research.md` | the certification-honesty investigation in full |
| `legacy/valis/valis_integration_plan.md` | original integration scope and spike findings |
| `validation/valis_bench/` | **live** — the ANHIR benchmark harness and results |

## References

- Gatenbee, C.D. et al. *VALIS: virtual alignment of pathology image series.* Nat Commun 14, 4530 (2023). https://doi.org/10.1038/s41467-023-40218-9
- Borovec, J. et al. *ANHIR: Automatic Non-rigid Histological Image Registration Challenge.* IEEE Trans Med Imaging 39(10):3042–3052 (2020). https://doi.org/10.1109/TMI.2020.2986331
- Sun, J. et al. *LoFTR: Detector-Free Local Feature Matching with Transformers.* CVPR 2021. https://doi.org/10.1109/CVPR46437.2021.00881
- Fitzpatrick, J.M., West, J.B. *The distribution of target registration error in rigid-body point-based registration.* IEEE Trans Med Imaging 20(9):917–927 (2001). https://doi.org/10.1109/42.952729
