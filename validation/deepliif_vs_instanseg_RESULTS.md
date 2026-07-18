# DeepLIIF vs InstanSeg — nuclear segmentation (HNSCC expert masks)

**Question:** is DeepLIIF's model genuinely a better nuclear segmenter than InstanSeg?
Decided on an **independent** expert-labelled set — deliberately NOT DeepLIIF's own test
distribution (that would be circular home-turf; see `stardist_vs_instanseg_RESULTS.md` for
the parallel InstanSeg-vs-StarDist comparison).

**Data:** HNSCC mIHC (`~/Desktop/HNSCC_raw_dataset`, local, not redistributable),
**268 tiles** 512×512. Input = the `*_Hematoxylin.png` component only. Expert nuclear masks
in `inputs/Segmentation/` (blue = nucleus interior, green = separating ring, black = bg).
**91,173** expert-labelled nuclei total. Harness: `validation/score_hnscc_deepliif_vs_instanseg.py`.

**Run conditions — identical for both detectors:** input = hematoxylin only, pixel size
**0.5 µm**, **adaptive threshold OFF**, DAB threshold **0.35** (moot — hematoxylin carries no
DAB; this is a pure nuclear-segmentation test). Same expert GT, same scorer.
- **InstanSeg:** QuPath 0.7 `brightfield_nuclei` (the shipped segmenter), headless.
- **DeepLIIF:** `DeepLIIF_Latest_Model` (Zenodo 4751737, 2.9 GB) run via `deepliif test`
  in an **isolated** `~/deepliif_runtime/venv` (py3.11) — inference-only deps at modern
  versions + `deepliif --no-deps`, WSI-only `bioformats`/`javabridge` stubbed, CPU, ~14 s/tile.
  The project `.venv` (pinned opencv 4.13 / numpy) is untouched. Native `deepliif` cannot
  run on the project's Python 3.14 (2021-era stack), hence the isolated env.

**Scoring:** hematoxylin image + every mask are the SAME registered tile → no alignment
transform needed. Nucleus instances = connected components of the blue(+red) interior
(green rings split touching nuclei). "Allow for shifts" = greedy 1-to-1 centroid match at
6/10/15 px (detection F1) + ±3 px dilation (tolerant pixel F1). Both detectors scored
identically against the same GT.

## Results (268 tiles, 91,173 GT nuclei)

| detector | det-F1 @6px | @10px | @15px | pixel-F1 ±3px | pred/GT count |
|---|---|---|---|---|---|
| **InstanSeg** | **0.772** | **0.797** | **0.817** | **0.823** | 0.85 |
| DeepLIIF | 0.485 | 0.592 | 0.650 | 0.691 | 0.97 |

InstanSeg: recall 0.71→0.75, precision 0.84→0.89 (6→15 px).
DeepLIIF: recall 0.48→0.64, precision 0.49→0.66.
**InstanSeg wins decisively at every tolerance** (+0.17…+0.29 det-F1, +0.13 pixel-F1).

## What's happening
DeepLIIF finds a *similar total count* (ratio 0.97) but **localises poorly and
over-segments background** — on hematoxylin-only input it generates a dense wall-to-wall
nuclear field, painting nuclei into stroma/empty regions the expert and InstanSeg leave
blank. InstanSeg tracks the expert mask closely. So DeepLIIF's failure is placement + false
nuclei, not count. (Montage: `~/Desktop/HNSCC_seg_comparison/comparison_montage.png`.)

## Caveat
DeepLIIF is trained on **full IHC RGB** (it internally infers hematoxylin/DAPI→seg).
Hematoxylin-only is the controlled same-input design (both detectors see the identical
nuclear signal) but is **off-distribution for DeepLIIF** and handicaps it — the background
hallucination is characteristic. Fair reading: *on the hematoxylin/nuclear signal the OASIS
pipeline actually uses, InstanSeg segments substantially better than DeepLIIF.* Not tested:
DeepLIIF on full IHC RGB (a different question); and DeepLIIF cannot do membranous
classification (CD8/TIM-3) regardless of segmentation accuracy.

**Decision: InstanSeg stays the segmenter** — better here, and (unlike DeepLIIF's generative
IHC→IF inference) it delegates only nucleus *geometry* to a network while the marker call
stays a deterministic DAB-OD measurement (§1, §3.2).
