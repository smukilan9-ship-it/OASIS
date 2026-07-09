# Dense Null: Image-Derived Morphology Validation

This validation renders real CODEX cell-coordinate architecture into hematoxylin-like pixels, extracts a morphology field from the rendered image, and compares it with the oracle coordinate morphology field.

## Storage

The harness is lean by default: generated images are kept in memory. It writes one JSON report, one Markdown report, and optionally one example PNG.

## Dataset And Settings

- Source table: `/Users/mukilan/oasis_validation_datasets/CODEX/inputs/CRC_clusters_neighborhoods_markers.csv`
- Spots used: 60
- Render pixel size: 0.5 um/px
- Simulated marker counts: A=75, B=75
- Simulations per spot: 5
- Permutations per test: 199
- Null sigmas: `2` um
- Generator sigmas: `2,5,10` um

## Morphology-Recovery Metrics

- Field correlation mean/median: 0.9314 / 0.9389
- Detected/true nuclei ratio mean/median: 0.8382 / 0.8436
- Median nearest detected nucleus distance: 0.2335 um

## Calibration Table

| Method | Band (um) | Null sigma | Generator H0 | p<=0.05 | 95% CI | Verdict | Power 5um | Power 12um |
|---|---:|---:|---|---:|---|---|---:|---:|
| image_derived_nuclei_morphology | 10-30 | 2 | gen_2um | 0.063 | [0.036, 0.091] | pass | 1.000 | 1.000 |
| image_derived_nuclei_morphology | 10-30 | 2 | gen_5um | 0.050 | [0.025, 0.075] | pass | 1.000 | 1.000 |
| image_derived_nuclei_morphology | 10-30 | 2 | gen_10um | 0.037 | [0.015, 0.058] | pass | 1.000 | 1.000 |
| image_derived_nuclei_morphology | 5-20 | 2 | gen_2um | 0.073 | [0.044, 0.103] | borderline | 1.000 | 1.000 |
| image_derived_nuclei_morphology | 5-20 | 2 | gen_5um | 0.067 | [0.038, 0.095] | pass | 1.000 | 1.000 |
| image_derived_nuclei_morphology | 5-20 | 2 | gen_10um | 0.027 | [0.008, 0.045] | borderline | 1.000 | 1.000 |
| oracle_coordinate_morphology | 10-30 | 2 | gen_2um | 0.067 | [0.038, 0.095] | pass | 1.000 | 1.000 |
| oracle_coordinate_morphology | 10-30 | 2 | gen_5um | 0.040 | [0.018, 0.062] | pass | 1.000 | 1.000 |
| oracle_coordinate_morphology | 10-30 | 2 | gen_10um | 0.043 | [0.020, 0.066] | pass | 1.000 | 1.000 |
| oracle_coordinate_morphology | 5-20 | 2 | gen_2um | 0.077 | [0.047, 0.107] | borderline | 1.000 | 1.000 |
| oracle_coordinate_morphology | 5-20 | 2 | gen_5um | 0.047 | [0.023, 0.070] | pass | 1.000 | 1.000 |
| oracle_coordinate_morphology | 5-20 | 2 | gen_10um | 0.027 | [0.008, 0.045] | borderline | 1.000 | 1.000 |

## Decisions

| Method | Band (um) | Null sigma (um) | Worst H0 p<=0.05 | Min power | Decision |
|---|---:|---:|---:|---:|---|
| oracle_coordinate_morphology | 10-30 | 2 | 0.067 | 1.000 | passes_screen |
| oracle_coordinate_morphology | 5-20 | 2 | 0.077 | 1.000 | do_not_ship_anti_conservative |
| image_derived_nuclei_morphology | 10-30 | 2 | 0.063 | 1.000 | passes_screen |
| image_derived_nuclei_morphology | 5-20 | 2 | 0.073 | 1.000 | do_not_ship_anti_conservative |

## Interpretation

- Oracle-coordinate morphology is the statistical upper bound.
- Image-derived morphology is the actual blocker for a dense production null.
- Dense mode should ship only if image-derived morphology controls H0 near 5%, preserves planted-positive power, and recovers the morphology field well enough.
- Passing rendered CODEX is still not final production validation on real LL477 H-DAB serial sections; it is the required bridge before real-pair demonstration.
