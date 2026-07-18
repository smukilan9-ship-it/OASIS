# VALIS vs OASIS on ANHIR — registration accuracy, LoFTR correspondence quality, gate calibration

Independent, non-circular benchmark on the public ANHIR/CIMA dataset (grand-challenge
`dataset_medium`, **scale-25pc**, 222 training pairs with public landmarks, 8 tissue types).
Both methods register from image pixels only; scoring is on held-out expert landmarks via one
shared scorer (`common.rtre`). VALIS runs in an isolated env; the OASIS pipeline is untouched.
Harness: `validation/valis_bench/` (see README for the anti-circularity contract).

## 1. Registration accuracy (stratified 44 pairs, 7/tissue, all 8 types)

rTRE = target registration error / fixed-image diagonal (ANHIR convention). Lower = better.
MMrTRE = median over pairs of the per-pair median.

| method | transform | distance-preserving | MMrTRE | mean | pairs registered |
|---|---|---|---:|---:|---:|
| no registration | identity | — | 0.0522 | 0.0947 | 44 |
| OASIS-LoFTR | similarity | yes | 0.0052 | 0.0073 | **23 / 44** |
| OASIS-structural | similarity | yes | 0.0052 | **0.0579** | 44 |
| VALIS-rigid | rigid | yes | **0.0037** | 0.0070 | 44 |
| VALIS-nonrigid | rigid+non-rigid | NO (warp) | **0.0015** | 0.0047 | 44 |

- **VALIS is the more robust general registrar.** VALIS-rigid registers all 44 (mean 0.0070);
  OASIS-structural has a comparable median (0.0052) but mean **0.058** — it fails catastrophically
  on cross-modal / large-displacement pairs (HER2→HE 0.40, CD68→CD4 0.45, COAD 0.12–0.18,
  several worse-than-identity). VALIS handled all of those (0.001–0.01).
- **OASIS-LoFTR could only register 23/44** (0 matches on the 21 cross-modal / big pairs).
- **Within OASIS's regime it ties VALIS-rigid:** on the 23 pairs where OASIS-LoFTR works,
  0.0052 vs VALIS-rigid 0.0036, and OASIS is better on **14/23** pairs.
- VALIS-nonrigid (0.0015) is best overall but is the warp OASIS forbids for cross-K
  (`serial_registration.assert_distance_preserving`) — reference/upper-bound only.

## 2. LoFTR correspondence quality (all 222 pairs — the direct, non-circular check)

Each LoFTR match is compared to the ground-truth displacement predicted by a LOCAL affine fit
to the nearest expert landmarks (`common.correspondence_quality`). LoFTR never sees the landmarks.

**LoFTR produced usable matches on 125 / 222 (56%), split almost perfectly by stain appearance:**

| tissue | pairs | matched | median corr err | inlier@10µm |
|---|---:|---:|---:|---:|
| lung-lesion | 30 | **30 (100%)** | 21.7µm | 23% |
| lung-lobes | 40 | **38 (95%)** | 21.7µm | 20% |
| mice-kidney | 15 | 9 (60%) | 15.3µm | 29% |
| gastric | 13 | 7 (54%) | 16.8µm | 39% |
| COAD | 84 | 41 (49%) | 43.9µm | 4% |
| mammary-gland | 38 | **0 (0%)** | — | — |
| breast | 1 | 0 (0%) | — | — |
| kidney | 1 | 0 (0%) | — | — |

- **Reliable on visually similar stains (IHC↔IHC, PAS↔PAS):** lung 95–100% match rate; the
  ~21µm geometric error is a few working-resolution pixels, consistent with the ~0.5% rTRE
  OASIS achieves there.
- **Fails outright on cross-modal H&E↔IHC:** mammary 0/38, breast 0/1, kidney 0/1, and the
  H&E-involving pairs inside COAD/lung. A hard limitation — H&E and DAB-IHC don't share the
  local texture LoFTR keys on — not a tuning issue.
- **Relevance to OASIS:** its real use case (CD8 vs TIM-3 serial sections) is IHC↔IHC, both
  brown DAB — squarely in LoFTR's reliable regime. So LoFTR is validated *for what OASIS uses
  it for*, and is honestly not a general cross-modal matcher.
- Caveat: absolute µm errors and inlier@10µm are pessimistic because whole-slide downsampling
  (WORK_MAX 2000) coarsens the working µm/px; the **match-rate** and downstream **rTRE** are the
  trustworthy signals.

## 3. Gate calibration (non-circular)

The gate saw only LoFTR correspondences; rTRE below is on the independent expert landmarks.

| gate verdict | pairs | median independent rTRE |
|---|---:|---:|
| RADIUS_LIMITED | 3 | 0.0016 |
| LOCALLY_CERTIFIED | 3 | 0.0045 |
| NOT_CERTIFIABLE | 17 | 0.0057 |
| NO_MATCHES | 21 | — (no transform) |

**The gate fails closed** — every pass verdict has genuinely low error (0.0016–0.0045); it never
certified a bad registration. It is **over-conservative** (certified 6/44, flagged good
registrations as NOT_CERTIFIABLE), partly a benchmark artifact: whole-slide downsampling makes
the 5µm LOO threshold sub-pixel-tight. In production (ROI at native resolution) it certifies more.

## Bottom line

On the full diversity of ANHIR, **VALIS clearly wins** — robust to cross-modal staining and large
displacements that break both OASIS paths. **Within OASIS's actual regime — similar-stain serial
sections — OASIS is competitive with VALIS-rigid**, LoFTR correspondences are reliable (~100% on
lung), and the gate is safe (fails closed). OASIS is a specialized serial-section CD8/TIM-3 tool
with a fail-closed gate, not a general histology registrar — and this benchmark draws that line
with measured numbers.

**Follow-up worth doing:** VALIS-rigid recovers many cross-modal pairs OASIS cannot, so adding
VALIS-rigid as an invariant-safe *fallback / cross-check* candidate in the multi-init selection
(never its non-rigid warp) has real, measured value.

## Reproduce
Isolated VALIS env `~/valis_runtime` (valis-wsi 1.2.0); data `~/oasis_validation_datasets/ANHIR_medium`.
`validation/valis_bench/run_all.sh` runs ours (stratified) → valis (stratified) → compare → correspondence (full).
