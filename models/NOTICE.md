# Third-party model weights redistributed with OASIS

## InstanSeg `brightfield_nuclei-0.1.1`

Vendored here so OASIS builds and runs without QuPath, and so the segmentation
result is reproducible from a checkout alone. Bit-identical to the copy QuPath's
InstanSeg extension downloads.

| | |
|---|---|
| Licence | **Apache-2.0** (declared in `rdf.yaml` and `brightfield_nuclei_README.md`) |
| Upstream | https://github.com/instanseg/instanseg |
| `instanseg.pt` SHA-256 | `0c724da169d507d8dd2e206ad3675a0b1566824eee6e20ad9a0ba916311d9dd0` |

Apache-2.0 permits redistribution provided the licence and attribution travel
with the work; both are preserved in `brightfield_nuclei-0.1.1/`.

### Cite InstanSeg, not just OASIS

Work using OASIS's segmentation depends on this model and should cite it:

> Goldsborough, T. et al. (2024) InstanSeg: an embedding-based instance
> segmentation algorithm optimized for accurate, efficient and portable cell
> segmentation. *arXiv*. https://doi.org/10.48550/arXiv.2408.15954

> Goldsborough, T. et al. (2024) A novel channel invariant architecture for the
> segmentation of cells and nuclei in multiplexed images using InstanSeg.
> *bioRxiv*, 2024.09.04.611150. https://doi.org/10.1101/2024.09.04.611150

### Training-data licences

The upstream README notes that **the user is responsible for ensuring the model
is used in accordance with the licences of its source datasets**. That obligation
passes through to anyone using OASIS. The model was trained on:

| Dataset | Licence | Source |
|---|---|---|
| tnbc_2018 | CC BY 4.0 | https://zenodo.org/records/3552674 |
| lynsec | CC BY 4.0 | https://zenodo.org/records/8065174 |
| nuinsseg | CC BY 4.0 | https://zenodo.org/records/10518968 |
| ihc_tma | CC BY 4.0 | https://zenodo.org/records/7647846 |
| consep | Apache-2.0 | https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet |

### Which files are vendored, and why

`instanseg.pt` and `rdf.yaml` are required at runtime — `rdf.yaml` carries the
`scale` (0.5 µm/px) and `scale_range` preprocessing the segmenter reads.
`test-input.npy` and `test-output_instance_segmentation.npy` are the upstream
reference pair used by the model-fidelity validation, which asserts OASIS
reproduces the published output exactly. `cover.png` and the ImageJ `.ijm` macros
are not used by OASIS and are not vendored.
