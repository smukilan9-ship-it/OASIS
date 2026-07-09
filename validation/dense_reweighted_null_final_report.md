# Dense-Tissue Primary Null Calibration - Final Focused Report

Generated after focused paper-grade calibration runs using:

```bash
.venv/bin/python validation/validate_dense_reweighted_null.py \
  --sims 300 --arch-sims 80 --nperm 199 \
  --bandwidths <candidate> --bands <candidate_band> \
  --short-jitter-um 5 --mid-jitter-um 12 \
  --planted-fraction 1.0 --arch-top-n 1
```

This report aggregates the three focused runs. The per-run JSON/report files are
overwritten by the harness, so this file preserves the decision-relevant results.

## Verdict

No tested 35-45 um dense-tissue primary null is calibrated well enough to ship.

The simple "make the reweighted primary bandwidth smaller" approach does not
currently solve dense tissue. It gives good power, but it remains
anti-conservative under dense shared tissue preference. That means it would still
call "cell-scale association" too often when A and B are independently following
the same dense architecture.

Do not add a dense production preset yet.

## Candidates Tested

### Candidate 1: h = 45 um, DCLF = 10-30 um

Screen:

| Regime | P(p <= 0.05) | Verdict |
|---|---:|---|
| Uniform CSR H0 | 0.023 | borderline / slightly conservative |
| Dense shared-architecture H0 | 0.250 | anti-conservative |
| Intermediate shared-architecture H0 | 0.130 | anti-conservative |
| Banded shared-compartment H0 | 0.497 | anti-conservative |

Power:

| Positive control | Power |
|---|---:|
| 5 um attraction | 0.963 |
| 12 um attraction | 0.907 |

Decision:

```text
REJECT_ANTI_CONSERVATIVE
```

Interpretation:

This candidate has excellent power, but the false-positive rate is far too high
under shared architecture. It is especially bad for banded tissue compartments,
where it rejects almost half of true-null simulations.

Architecture sweep:

| Architecture knob (um) | ell_hat median (um) | Type-I | Power |
|---:|---:|---:|---:|
| 25 | 32.2 | 0.713 | 0.988 |
| 35 | 35.4 | 0.350 | 0.988 |
| 45 | 39.3 | 0.163 | 1.000 |
| 60 | 45.2 | 0.150 | 0.963 |
| 80 | 51.7 | 0.100 | 0.938 |
| 100 | 61.5 | 0.100 | 0.950 |
| 130 | 75.7 | 0.025 | 0.988 |
| 160 | 81.6 | 0.037 | 0.950 |

Approximate implication:

For h=45, type-I becomes acceptable only once measured architecture scale is
around 75-80 um. That is not the dense regime this feature was meant to rescue.

## Candidate 2: h = 35 um, DCLF = 10-30 um

Screen:

| Regime | P(p <= 0.05) | Verdict |
|---|---:|---|
| Uniform CSR H0 | 0.023 | borderline / slightly conservative |
| Dense shared-architecture H0 | 0.147 | anti-conservative |
| Intermediate shared-architecture H0 | 0.087 | borderline |
| Banded shared-compartment H0 | 0.113 | anti-conservative |

Power:

| Positive control | Power |
|---|---:|
| 5 um attraction | 0.897 |
| 12 um attraction | 0.763 |

Decision:

```text
REJECT_ANTI_CONSERVATIVE
```

Interpretation:

This candidate is less inflated than h=45, but it still over-rejects under dense
shared architecture and banded shared compartments. It also misses the >=0.80
power target for 12 um attraction.

Architecture sweep:

| Architecture knob (um) | ell_hat median (um) | Type-I | Power |
|---:|---:|---:|---:|
| 25 | 32.3 | 0.200 | 1.000 |
| 35 | 34.8 | 0.138 | 0.963 |
| 45 | 40.5 | 0.163 | 0.900 |
| 60 | 44.6 | 0.125 | 0.975 |
| 80 | 51.4 | 0.087 | 0.900 |
| 100 | 61.5 | 0.075 | 0.900 |
| 130 | 73.9 | 0.050 | 0.950 |
| 160 | 82.7 | 0.037 | 0.912 |

Approximate implication:

For h=35, type-I becomes acceptable only around ell_hat 74-83 um. Again, this
does not rescue truly dense architecture around ell_hat 30-50 um.

## Candidate 3: h = 40 um, DCLF = 5-20 um

This was tested because a narrower DCLF band is more scientifically compatible
with a smaller dense-tissue bandwidth.

Screen:

| Regime | P(p <= 0.05) | Verdict |
|---|---:|---|
| Uniform CSR H0 | 0.040 | pass |
| Dense shared-architecture H0 | 0.150 | anti-conservative |
| Intermediate shared-architecture H0 | 0.113 | anti-conservative |
| Banded shared-compartment H0 | 0.200 | anti-conservative |

Power:

| Positive control | Power |
|---|---:|
| 5 um attraction | 0.983 |
| 12 um attraction | 0.843 |

Decision:

```text
REJECT_ANTI_CONSERVATIVE
```

Interpretation:

This is the most scientifically reasonable dense-scale test because it uses a
smaller interaction band, 5-20 um. It passes uniform CSR and has good power, but
it still fails the actual problem: shared dense architecture. It is not safe as a
primary null.

Architecture sweep:

| Architecture knob (um) | ell_hat median (um) | Type-I | Power |
|---:|---:|---:|---:|
| 25 | 32.8 | 0.412 | 1.000 |
| 35 | 35.3 | 0.163 | 1.000 |
| 45 | 38.7 | 0.175 | 0.975 |
| 60 | 45.2 | 0.150 | 0.988 |
| 80 | 52.7 | 0.113 | 1.000 |
| 100 | 61.1 | 0.062 | 0.988 |
| 130 | 78.1 | 0.025 | 0.963 |
| 160 | 84.8 | 0.050 | 0.975 |

Approximate implication:

For h=40 with a 5-20 um DCLF band, type-I is acceptable only once ell_hat is
roughly 78-85 um. This is still not a true dense-architecture setting.

## Cross-Candidate Summary

| Candidate | Dense H0 p05 | Banded H0 p05 | Short power | Mid power | Decision |
|---|---:|---:|---:|---:|---|
| h45, 10-30 | 0.250 | 0.497 | 0.963 | 0.907 | reject |
| h35, 10-30 | 0.147 | 0.113 | 0.897 | 0.763 | reject |
| h40, 5-20 | 0.150 | 0.200 | 0.983 | 0.843 | reject |

All three candidates fail because their false-positive rates under shared
architecture exceed the target interval of 0.03-0.07.

The important pattern:

```text
smaller h improves local power,
but does not fix shared-architecture false positives in dense tissue.
```

## Scientific Conclusion

The proposed dense-tissue extension cannot currently be framed as:

> A validated dense-tissue primary null using h = 35-45 um.

That would be false.

The defensible conclusion is:

> A validation harness was built and run. Focused paper-grade calibration rejected
> h=35-45 um dense reweighted null presets because they remained anti-conservative
> under dense shared tissue architecture.

## Why This Failed

The reweighted null estimates intensity with a smoothed first-order surface. In
dense tissue, the architecture and the tested neighborhood structure are too close
in scale. A smaller bandwidth helps the intensity estimate follow dense structure,
but then the estimator also approaches the cell-neighborhood scale being tested.

The result is a tradeoff:

1. If h is large, dense shared architecture leaks through as false association.
2. If h is small, the model risks absorbing real interaction or becoming unstable.
3. In these simulations, h=35-45 did not find a stable middle ground.

## What This Means For OASIS

Do not add a dense-primary toggle to production yet.

For current OASIS:

1. Keep the 75 um primary null for coarse-enough architecture.
2. Keep dense tissues marked as not trustworthy for cell-scale reweighted claims
   when the architecture preflight fails.
3. Still allow homogeneous CSR/co-infiltration reporting, clearly labeled as the
   weaker shared-compartment finding.
4. Do not report "robust cell-scale association beyond architecture" in dense
   tissues until a different null passes calibration.

## Next Method To Try

A simple smaller-bandwidth reweighted K is probably not enough. The next serious
dense-tissue route should be one of:

1. Region-stratified / compartment-conditioned null:
   - segment tissue into micro-compartments or density strata,
   - randomize B within matched strata,
   - preserve local cell density without assuming one global kernel bandwidth.

2. Matched local toroidal/block-shift null:
   - shift B within local tiles or certified ROI blocks,
   - avoid global wraparound,
   - preserve local architecture more faithfully.

3. Conditional random-label style null only for true same-section multiplex data:
   - valid for multiplex/same-section coordinates,
   - not valid for serial-section IHC because the two cell sets come from different
     physical sections.

4. Covariate-adjusted point-process model:
   - explicitly include tissue covariates or learned compartment maps,
   - test residual cross-type interaction after covariate adjustment.

For the serial-section OASIS pipeline, option 1 is probably the most defensible
next step.

## Production Recommendation

Current status:

```text
standard primary null, h=75 um: keep
dense primary null, h=35-45 um: reject for now
```

Paper wording:

> OASIS includes a calibrated 75 um reweighted primary null for tissue fields whose
> architecture is sufficiently coarse. Dense fields that fail the architecture-scale
> preflight are not assigned a robust cell-scale association verdict. Pilot and
> focused calibration of 35-45 um dense-bandwidth candidates did not achieve
> acceptable false-positive control under dense shared-architecture nulls.

