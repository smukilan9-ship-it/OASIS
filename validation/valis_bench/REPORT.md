# VALIS vs OASIS — serial-section registration on ANHIR/CIMA held-out landmarks

rTRE = target registration error relative to the fixed-image diagonal (ANHIR convention). Lower is better. **No method ever sees the expert landmarks** — they score registration driven purely by image pixels. MMrTRE = median over pairs of the per-pair median rTRE.

**Scope:** 44 directed pairs across 25 tissue set(s): COAD_01, COAD_05, COAD_08, COAD_10, COAD_12, COAD_18, COAD_20, breast_1, gastric_1, gastric_3, gastric_5, gastric_6, gastric_7, gastric_9, kidney_3, lung-lesion_1, lung-lesion_2, lung-lesion_3, lung-lobes_1, lung-lobes_2, lung-lobes_3, lung-lobes_4, mammary-gland_1, mammary-gland_2, mice-kidney_1.

## Aggregate accuracy

| method | transform | distance-preserving? | MMrTRE | mean MrTRE | pairs scored |
|---|---|---|---:|---:|---:|
| (no registration) | identity | — | 0.0522 | 0.0947 | 44 |
| **OASIS** loftr | similarity (LoFTR) | yes | 0.0052 | 0.0073 | 23 |
| **OASIS** auto | similarity (structural) | yes | 0.0052 | 0.0579 | 44 |
| VALIS rigid | rigid | yes | 0.0037 | 0.0070 | 44 |
| VALIS nonrigid | rigid+non-rigid | NO (warp) | 0.0015 | 0.0047 | 44 |

## Apples-to-apples: OASIS similarity vs VALIS-rigid (both cross-K-safe)

On the 23 pairs where BOTH produced a transform: OASIS-LoFTR median rTRE 0.0052 vs VALIS-rigid 0.0036. OASIS better on 14/23 pairs.

## Gate calibration (non-circular)

Our gate saw only the LoFTR correspondences; the rTRE below is on the independent expert landmarks. A trustworthy gate should show LOW rTRE for pass verdicts and HIGH for fail verdicts.

| gate verdict | pairs | median independent rTRE | pairs with a transform |
|---|---:|---:|---:|
| LOCALLY_CERTIFIED | 3 | 0.0045 | 3 |
| NOT_CERTIFIABLE | 17 | 0.0057 | 17 |
| NO_MATCHES | 21 | — | 0 |
| RADIUS_LIMITED | 3 | 0.0016 | 3 |

## Per-pair rTRE (median)

| pair | N | init | OASIS-LoFTR | OASIS-auto | VALIS-rigid | VALIS-nonrigid | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| S3→S1 | 61 | 0.1401 | — | 0.1797 | 0.0101 | 0.0081 | NO_MATCHES |
| S5-v1→S2 | 92 | 0.0071 | 0.0030 | 0.0036 | 0.0034 | 0.0015 | NOT_CERTIFIABLE |
| S6→S3 | 68 | 0.1593 | — | 0.1244 | 0.0036 | 0.0020 | NO_MATCHES |
| S7→S4 | 77 | 0.0366 | — | 0.0049 | 0.0044 | 0.0032 | NO_MATCHES |
| S6→S5 | 71 | 0.0504 | 0.0021 | 0.0019 | 0.0024 | 0.0007 | NOT_CERTIFIABLE |
| S7→S2 | 103 | 0.0546 | 0.0066 | 0.0070 | 0.0058 | 0.0033 | NOT_CERTIFIABLE |
| S7→S5 | 83 | 0.1516 | — | 0.1521 | 0.0052 | 0.0010 | NO_MATCHES |
| HER2→HE | 79 | 0.4202 | — | 0.3997 | 0.0052 | 0.0022 | NO_MATCHES |
| CD68→CD4 | 77 | 0.0457 | 0.0047 | 0.0025 | 0.0032 | 0.0009 | NOT_CERTIFIABLE |
| CD68→CD4 | 63 | 0.4529 | — | 0.4517 | 0.0016 | 0.0009 | NO_MATCHES |
| CD68→CD4 | 79 | 0.4243 | — | 0.4227 | 0.0063 | 0.0061 | NO_MATCHES |
| EBV→CD68 | 79 | 0.4172 | — | 0.3595 | 0.0057 | 0.0060 | NO_MATCHES |
| EBV→CD4 | 115 | 0.0329 | 0.0006 | 0.0005 | 0.0007 | 0.0004 | RADIUS_LIMITED |
| CD68→CD4 | 61 | 0.0570 | — | 0.0019 | 0.0023 | 0.0007 | NO_MATCHES |
| CD68→CD4 | 155 | 0.1178 | 0.0015 | 0.0017 | 0.0017 | 0.0007 | NOT_CERTIFIABLE |
| PAS→MAS | 80 | 0.0204 | — | 0.0029 | 0.0102 | 0.0088 | NO_MATCHES |
| c10-5-les1→D31-3-les1 | 78 | 0.0396 | 0.0067 | 0.0055 | 0.0061 | 0.0042 | NOT_CERTIFIABLE |
| i67-7-les1→35-He-les1 | 78 | 0.0297 | 0.0127 | 0.0069 | 0.1264 | 0.1021 | NOT_CERTIFIABLE |
| c10-5-les2→D31-3-les2 | 96 | 0.0448 | 0.0065 | 0.0089 | 0.0080 | 0.0053 | NOT_CERTIFIABLE |
| i67-7-les2→c10-5-les2 | 96 | 0.0512 | 0.0045 | 0.0044 | 0.0042 | 0.0019 | LOCALLY_CERTIFIED |
| SPC-4-les2→i67-7-les2 | 96 | 0.0213 | 0.0056 | 0.0049 | 0.0054 | 0.0030 | NOT_CERTIFIABLE |
| i67-7-les3→c10-5-les3 | 80 | 0.0407 | 0.0064 | 0.0066 | 0.0069 | 0.0028 | LOCALLY_CERTIFIED |
| SPC-4-les3→i67-7-les3 | 80 | 0.0338 | 0.0082 | 0.0080 | 0.0068 | 0.0037 | NOT_CERTIFIABLE |
| zd1-2-cd31→-Izd1-1-HE | 98 | 0.0781 | 0.0052 | 0.0086 | 0.0057 | 0.0011 | NOT_CERTIFIABLE |
| zd1-6-ki67→-Izd1-1-HE | 98 | 0.0870 | — | 0.0300 | 0.0089 | 0.0045 | NO_MATCHES |
| zd2-4-cc10→-Izd2-1-HE | 107 | 0.0087 | 0.0057 | 0.0064 | 0.0067 | 0.0013 | NOT_CERTIFIABLE |
| zd1-2-cd31→-Izd1-1-HE | 80 | 0.0362 | 0.0018 | 0.0017 | 0.0022 | 0.0009 | NOT_CERTIFIABLE |
| zd1-6-ki67→-Izd1-1-HE | 80 | 0.0446 | 0.0244 | 0.0035 | 0.0033 | 0.0026 | NOT_CERTIFIABLE |
| -3-Pro-SPC→zd2-2-cd31 | 89 | 0.0280 | 0.0036 | 0.0047 | 0.0041 | 0.0007 | NOT_CERTIFIABLE |
| zd2-6-ki67→zd2-4-cc10 | 89 | 0.0250 | 0.0105 | 0.0040 | 0.0036 | 0.0012 | NOT_CERTIFIABLE |
| U_A4926-4L→E_A4926-4L | 77 | 0.0580 | — | 0.0035 | 0.0023 | 0.0011 | NO_MATCHES |
| R_A4926-4L→E_A4926-4L | 77 | 0.0250 | — | 0.0013 | 0.0012 | 0.0007 | NO_MATCHES |
| E_A4926-4L→R_A4926-4L | 76 | 0.1040 | — | 0.0273 | 0.0039 | 0.0030 | NO_MATCHES |
| U_A4926-4L→E_A4926-4L | 76 | 0.0259 | — | 0.0058 | 0.0034 | 0.0016 | NO_MATCHES |
| R-A4962-4L→E_A4926-4L | 76 | 0.0532 | — | 0.0041 | 0.0047 | 0.0042 | NO_MATCHES |
| R_A4926-4L→E_A4926-4L | 76 | 0.0227 | — | 0.0056 | 0.0053 | 0.0040 | NO_MATCHES |
| R_A4926-4L→R-A4962-4L | 76 | 0.0344 | — | 0.0024 | 0.0028 | 0.0015 | NO_MATCHES |
| 3_PAS→2_aSMA | 132 | 0.0831 | — | 0.0930 | 0.0015 | 0.0012 | NO_MATCHES |
| 5_PAS→3_PAS | 132 | 0.1213 | 0.0432 | 0.0027 | 0.0024 | 0.0014 | NOT_CERTIFIABLE |
| 6_CD31→5_PAS | 132 | 0.0969 | — | 0.0482 | 0.0017 | 0.0009 | NO_MATCHES |
| 8_CD31→3_PAS | 132 | 0.1280 | — | 0.1275 | 0.0018 | 0.0015 | NO_MATCHES |
| 8_CD31→6_CD31 | 132 | 0.1127 | 0.0016 | 0.0015 | 0.0016 | 0.0010 | RADIUS_LIMITED |
| 9_PAS→5_PAS | 132 | 0.0638 | 0.0011 | 0.0012 | 0.0013 | 0.0009 | LOCALLY_CERTIFIED |
| 9_PAS→8_CD31 | 132 | 0.0792 | 0.0016 | 0.0017 | 0.0017 | 0.0006 | RADIUS_LIMITED |
