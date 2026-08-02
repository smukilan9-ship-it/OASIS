# Real LL477 Dense-Null Candidate Test

This is a real-use test of the dense candidate on completed OASIS serial-section bundles. It is not statistical calibration because LL477 has no known-null ground truth.

## Summary

- Tested pairs: 1
- Skipped pairs: 0
- Candidate 10-30 um association calls: 1

## Pair Results

| Pair | Status | A+ in window | B+ in window | Support cells | Existing OASIS | Existing p(reweighted/CSR) | Dense candidate 10-30 p | Direction | Significant | Notes |
|---|---|---:|---:|---:|---|---|---:|---|---|---|
| LL477_CD8_x10_3__roi0 | tested | 69 | 97 | 2830 | csr_only | None/0.04695 | 0.002 | association | True | peak 12.0 um; TRE None um |

## Interpretation

- A skipped sparse pair is not evidence against association; it is an insufficient-events QC result.
- A significant result here is a real-use demonstration of the candidate, not a calibrated biological claim.
- Production dense mode remains gated: 75 µm must fail, landmark certification/window/support/sparsity gates must pass, and provenance must record the switch.
