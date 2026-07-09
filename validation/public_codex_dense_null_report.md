# Public CODEX Dense-Null Calibration

This report calibrates candidate dense-tissue nulls on real CRC CODEX cell-coordinate architecture.
It does not validate OASIS segmentation, serial-section registration, or a production dense mode.

## Dataset

- Source table: `/Users/mukilan/oasis_validation_datasets/CODEX/inputs/CRC_clusters_neighborhoods_markers.csv`
- Spots used: 60
- Pixel size used only for coordinate conversion: 0.3775 um/px
- Simulated marker counts: A=75, B=75
- Simulations per spot: 5
- Permutations per test: 199
- Generator sigmas: `2,5,10` um
- Candidate null sigmas: `2` um
- Planted fraction: 0.75

## Calibration Table

| Method | Band (um) | Null sigma | Generator H0 | p<=0.05 | 95% CI | Verdict | Power 5um | Power 12um |
|---|---:|---:|---|---:|---|---|---:|---:|
| homogeneous_csr | 10-30 | none | gen_2um | 0.223 | [0.176, 0.271] | anti_conservative | 1.000 | 1.000 |
| homogeneous_csr | 10-30 | none | gen_5um | 0.250 | [0.201, 0.299] | anti_conservative | 1.000 | 1.000 |
| homogeneous_csr | 10-30 | none | gen_10um | 0.167 | [0.124, 0.209] | anti_conservative | 1.000 | 1.000 |
| homogeneous_csr | 5-20 | none | gen_2um | 0.213 | [0.167, 0.260] | anti_conservative | 1.000 | 1.000 |
| homogeneous_csr | 5-20 | none | gen_5um | 0.187 | [0.143, 0.231] | anti_conservative | 1.000 | 1.000 |
| homogeneous_csr | 5-20 | none | gen_10um | 0.103 | [0.069, 0.138] | borderline | 1.000 | 1.000 |
| total_cell_morphology_jitter | 10-30 | 2 | gen_2um | 0.067 | [0.038, 0.095] | pass | 1.000 | 1.000 |
| total_cell_morphology_jitter | 10-30 | 2 | gen_5um | 0.037 | [0.015, 0.058] | pass | 1.000 | 1.000 |
| total_cell_morphology_jitter | 10-30 | 2 | gen_10um | 0.057 | [0.030, 0.083] | pass | 1.000 | 1.000 |
| total_cell_morphology_jitter | 5-20 | 2 | gen_2um | 0.080 | [0.049, 0.111] | borderline | 1.000 | 1.000 |
| total_cell_morphology_jitter | 5-20 | 2 | gen_5um | 0.050 | [0.025, 0.075] | pass | 1.000 | 1.000 |
| total_cell_morphology_jitter | 5-20 | 2 | gen_10um | 0.037 | [0.015, 0.058] | pass | 1.000 | 1.000 |

## Candidate Decisions

| Band (um) | Null sigma (um) | Worst H0 p<=0.05 | Minimum planted-association power | Decision |
|---:|---:|---:|---:|---|
| 10-30 | 2 | 0.067 | 1.000 | passes_screen_needs_real_image_morphology |
| 5-20 | 2 | 0.080 | 1.000 | do_not_ship_anti_conservative |

## Interpretation

- The known-null simulations use real CODEX cell layouts as tissue architecture templates, then independently draw A and B from the same marker-independent total-cell field.
- Homogeneous CSR is expected to over-reject because it ignores dense architecture.
- A morphology-conditioned candidate is only acceptable if every tested generator H0 stays near 5% while planted positives retain useful power.
- Passing this harness would still not be enough to ship dense mode: OASIS also needs real-image morphology extraction validated on H-DAB/hematoxylin-derived fields.
- Therefore this report can promote a candidate to further validation, but it cannot by itself make the dense null production-ready.
