# Is a non-rigid warp inadmissible before a cross-type spatial test?

**Measured 2026-08-04. The answer is no, and the rationale currently recorded in
`research/ihc.md` §3.4 / §6 is not supported by these measurements.**

The hypothesis under test was that non-rigid registration fabricates the inter-cell
distances a cross-type statistic consumes, and that because an intensity-driven warp
optimises on a signal correlated with cell density, it could *manufacture* spatial
association that is not there (`ihc.md` §6, stated there as a conditional — "could").

Four independent measurements say otherwise. One narrow version of the mechanism is
real; it does not reach any null the pipeline actually uses.

## Setup

`export_warps.py` → `arms.py` / `arm_gt.py` / `distortion.py` → `report.py`.

* 20 ANHIR training pairs (stratified, 3 per tissue), 17 usable at ≤ 2.5 µm/px across
  **7 tissue types**. Coarser tissues (lung-lobes at 5.1, mammary at 9.2 µm/px) cannot
  resolve a 0–20 µm contact band at all and are excluded.
* **VALIS 1.2.0** run **once** per pair in the isolated runtime. Both transforms come
  from that one registration, so they differ *only* in whether micro-registration is
  applied: `warp_xy_from_to(..., non_rigid=False | True)`.
* Analysis in **800 µm ROI-scale windows** — the scale the real pipeline certifies at
  (§3.5: R = 260–450 µm) and the only scale where a few hundred cells give a mean
  nearest-neighbour spacing (~25 µm) that puts content in the 10–20 µm contact band.
* A and B are drawn from **each section's own real tissue density**, independently.
* 199 replicates per arm; the pipeline's own `cross_k_all_nulls` and
  `cross_k_dense_morphology_test`, 299 permutations, α = 0.05. Paired design — same A,
  same B, same window, same null seed; only the transform moves. Hence McNemar.

**Pre-flight** (`check_warps.py`, all 17 pairs): rigid interpolation error **0.000 px**
(the rigid stage is exactly affine — affine fit residual 0.0000 px, which is also what
makes the closed-form inverse in Arm 1 exact); non-rigid interpolation error
0.19–0.48 px against a micro-registration displacement of 33–134 µm, i.e. a ratio of
**0.003–0.011**. No result here is an interpolation artefact.

## 1. Distance distortion is negligible at contact scale

`distortion.py`, dense point pairs inside ROI-scale windows, pooled over 17 pairs:

| band (µm) | median \|Δd\|/d | p90 \|Δd\|/d | contracted |
|---|---|---|---|
| 0–10 | **1.3 %** | 8.6 % | 44.4 % |
| **10–20** | **1.3 %** | 8.4 % | 46.0 % |
| 20–50 | 1.2 % | 7.7 % | 46.8 % |
| 50–100 | 1.1 % | 7.2 % | 47.1 % |

Micro-registration changes a 15 µm inter-cell distance by a median of **0.2 µm**, and
contracts as often as it expands (46 %) — there is no systematic pull toward attraction.

The reason is geometric, not incidental: distance distortion depends on the *gradient*
of the displacement field, not its magnitude. VALIS's field displaces points by tens of
microns but is locally smooth, so short distances ride along almost unchanged. **A
smooth non-rigid warp is locally almost rigid.**

## 2. It does not manufacture association

Arm 2 — A and B independent by construction, so every rejection is a false positive
regardless of which transform is "correct". 199 replicates.

| null | claim | rigid | nonrigid | McNemar |
|---|---|---|---|---|
| reweighted (PRIMARY) | contact 10–20 µm | 0.052 | 0.057 | **p = 1** |
| dense_morphology (what every real pair selects, §20.1) | contact | 0.010 | 0.010 | **p = 1** |
| reweighted | global | 0.094 | 0.109 | p = 0.73 |
| dense_morphology | global | 0.016 | 0.010 | p = 1 |
| homogeneous (diagnostic only) | contact | 0.089 | 0.141 | p = 0.087 |
| **homogeneous** | **global** | **0.152** | **0.245** | **p = 0.0051** |

The primary null sits at **0.052** at contact scale — nominal. Under the two nulls the
pipeline actually uses there is **no effect whatsoever**.

The single significant inflation is on the homogeneous CSR null's whole-curve test. That
null is already 3× inflated under the *rigid* transform (0.152 vs α = 0.05) and is
documented in `spatial_stats.py` as a "DIAGNOSTIC baseline only" that never gates a
verdict. Non-rigid makes a bad null worse; it does not make a good null bad.

## 3. The mechanism is real, and it is confined to that one cell

Arm 3 adds a smooth **random** displacement matched to the real micro-registration RMS
*at the same points* — identical magnitude, tissue-blind.

| null / claim | rigid | nonrigid | random |
|---|---|---|---|
| homogeneous / global | 0.152 | **0.245** (p = 0.0051) | 0.172 (p = 0.57) |
| homogeneous / contact | 0.089 | 0.141 (p = 0.087) | 0.083 (p = 1) |
| reweighted / contact | 0.052 | 0.057 (p = 1) | 0.047 (p = 1) |

The inflation is **not** reproduced by a displacement of the same size that does not
know where the tissue is. So the hypothesised mechanism — the warp aligns B's density
with A's, and a null with no morphology term reads co-densification as association — is
**confirmed as a mechanism**. It simply has nothing to bite on once the null conditions
on the observed morphology.

## 4. The harm scales with displacement, not with transform class

Arm 1 imposes a true attraction and pulls it back through the rigid map, so rigid
reproduces it by construction. That design is confounded (it assumes rigid is correct)
and is superseded by §5 — but its **dose-response** is the informative part:

| micro-registration displacement | n | rigid recovers | nonrigid recovers |
|---|---|---|---|
| 5–21 µm | 66 | 47 | **61** |
| 21–55 µm | 66 | 60 | 11 |
| 55–622 µm | 67 | 62 | 0 |

Below ~21 µm of displacement, non-rigid recovers the truth **more** often than rigid.
The damage is a monotone function of how far the warp moves things — which is
registration error — not of the transform's class.

At large displacement the failure mode is severe: 27 / 195 replicates turned a true
attraction into **significant segregation** under `dense_morphology`. That is a sign
error, not a power loss. But it is caused by displacement, and a rigid transform with
the same error would do the same.

## 5. Against expert-landmark truth, the non-rigid warp WINS

`arm_gt.py`, 170 replicates. Truth is ANHIR's expert landmark correspondences — which
no registration ever sees — turned into a local displacement field by
`_predict_local_affine`, with windows centred on landmarks so the field is interpolated
rather than extrapolated. A known association is placed at `B_true`, and the question is
which transform reproduces the verdict a *perfect* registration would give.

* Non-rigid halves registration error: median TRE **23 µm vs 49 µm**, better on
  **139 / 170** replicates.

| null | rigid reproduces truth | nonrigid reproduces truth |
|---|---|---|
| dense_morphology | 0.371 | **0.659** |
| reweighted | 0.253 | **0.388** |

And it stratifies exactly as an accuracy-driven effect must:

| subset | n | rigid | nonrigid |
|---|---|---|---|
| non-rigid more accurate | 139 | 0.317 | **0.691** |
| rigid more accurate | 31 | **0.613** | 0.516 |

**The statistic follows registration accuracy, not transform class.** Whichever
transform puts the cells closer to where they belong gives the right answer more often,
and distance-preservation does not enter.

## Verdict

There is no paper in "non-rigid registration is inadmissible before a spatial
statistic." Four measurements refute it: the distance distortion is 1.3 % at contact
scale, the false-positive rate is unmoved under every null in production use, the one
real inflation lands only on a null already documented as unusable, and against external
ground truth the non-rigid warp produces the *correct* verdict nearly twice as often.

### What this obliges us to change

* `ihc.md` §3.4 — "a warp fabricates the inter-cell distances K consumes" is
  quantitatively wrong at the scale that matters. §6's "non-rigid warp → similarity
  only (warping destroys the measured distances)" rests on it.
* §6's conditional — an intensity-driven warp "could manufacture association" — is now
  measured. It can, but only against a null with no morphology term, which the pipeline
  does not use.
* §7.1's decision to forbid VALIS's non-rigid warp needs a different justification. The
  one that survives is **certifiability**: the Fitzpatrick–West budget is defined for a
  similarity, and there is no equivalent per-cell error bound for a free-form
  displacement field. That is a claim about what can be *certified*, not about what
  distorts distances — and it is the thesis of the A+F paper.

### Limits of this experiment

* ANHIR at 25 pc is a coarse regime: median TRE 23–49 µm, far above the ~4.4 µm serial-
  section floor (§16.3). In a regime where both transforms are already cell-scale
  accurate, the 1.3 % distortion would matter relatively more — though 1.3 % of 15 µm is
  0.2 µm, so a reversal is unlikely.
* ANHIR pairs are cross-stain, including H&E↔IHC. OASIS's regime is same-stain serial
  sections, where LoFTR works and deformation is smaller.
* The imposed association is synthetic (50 % of B within σ = 8 µm of an A cell).
* One registrar. VALIS's micro-registration is smooth; a B-spline with a coarser control
  grid or a stronger regulariser would have a different gradient and could distort more.
* Tissue density is read from a thumbnail (≥ 4000 px), so structure below ~2–13 µm
  depending on the pair is not represented in the sampling.

## Reproduce

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib ~/valis_runtime/venv/bin/python -m validation.nonrigid_bench.export_warps
.venv/bin/python -m validation.nonrigid_bench.check_warps
.venv/bin/python -m validation.nonrigid_bench.distortion
.venv/bin/python -m validation.nonrigid_bench.arms --arm all --reps 12
.venv/bin/python -m validation.nonrigid_bench.arm_gt --reps 10
.venv/bin/python -m validation.nonrigid_bench.report
```
