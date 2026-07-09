# Real LL477 Dense-Null Candidate Test

This is a real-use test of the dense candidate on completed OASIS serial-section bundles. It is not statistical calibration because LL477 has no known-null ground truth.

## Summary

- Tested pairs: 2
- Skipped pairs: 1
- Candidate 10-30 um association calls: 2

## Pair Results

| Pair | Status | A+ in window | B+ in window | Support cells | Existing OASIS | Existing p(reweighted/CSR) | Dense candidate 10-30 p | Direction | Significant | Notes |
|---|---|---:|---:|---:|---|---|---:|---|---|---|
| LL477_CD8_x10_1 | tested | 236 | 52 | 11611 | robust | 0.00599/0.001 | 0.007 | association | True | peak 30.0 um; TRE 1.513 um |
| LL477_CD8_x10_2 | skipped | 72 | 10 | 5721 | csr_only | 0.17882/0.03197 |  |  |  | requires certified pair, >= 30 positives per marker inside window, >= 500 support cells |
| LL477_CD8_x10_3 | tested | 59 | 75 | 2685 | csr_only | 0.72128/0.001 | 0.024 | association | True | peak 16.0 um; TRE 4.898 um |

## Interpretation

- A skipped sparse pair is not evidence against association; it is an insufficient-events QC result.
- A significant result here is a real-use demonstration of the candidate, not a calibrated biological claim.
- Dense mode should still remain fail-closed until this candidate is wired into production with provenance, ROI, sparsity, and architecture gates.
