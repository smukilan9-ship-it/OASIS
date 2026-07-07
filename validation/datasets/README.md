# Validation datasets

Datasets are **not committed to the repo**. They live in one consolidated tree at
`validation_data_dir` (default `~/oasis_validation_datasets`), with raw **inputs**
separated from generated **outputs**, one README + checksum per dataset.

`datasets.yaml` in this directory is the machine-readable registry (source, license,
citation, checksum, structure). The tooling:

```bash
python -m validation.datasets.verify           # presence + checksum status table
python -m validation.datasets.acquire            # dry-run consolidation plan
python -m validation.datasets.acquire --apply     # consolidate (move within volume)
python -m validation.datasets.verify --write       # pin checksums after consolidation
python -m validation.datasets.acquire --download NAME  # fetch an open dataset / show instructions
```

## Resolution order (`validation_data_dir`)

1. env `IHC_VALIDATION_DATA_DIR`
2. `~/.ihc_analyzer/setup.yaml` → `validation_data_dir`
3. default `~/oasis_validation_datasets`

Per-dataset, a validation reads `<root>/<DIR>/inputs/…`. Missing datasets skip their
tests/validations with a message naming the exact dataset and source.

## Datasets

| name | dir | what it validates | redistributable |
|---|---|---|---|
| `codex_crc` | `CODEX/` | keystone degradation, real-data spatial, spatstat cross-val | yes (CC BY 4.0) |
| `cima_landmarks` | `CIMA_ANHIR/` | registration TRE vs expert landmarks | yes (landmarks) |
| `deepliif` | `DeepLIIF/` | cell detection + classification vs IF truth | no (research use) |
| `hnscc` | `HNSCC/` | membranous CD8 + restained co-expression vs IF | **no** (TCIA login) |
| `tim3_crc_icm` | `TIM3_CRC_ICM/` | TIM-3 membrane callability (hand-labelled) | yes (Mendeley) |

Restricted datasets (`redistributable: no`) are documented but never uploaded or
committed; obtain them from the `source_url` in `datasets.yaml` and place them under
`<root>/<DIR>/inputs/`.

## The keystone works from the CODEX table

`tests/test_degradation.py` (the End-to-End cell-scale validation) needs only
`codex_crc`. It is redistributable — fetch it with
`python -m validation.datasets.acquire --download codex_crc` if a fresh clone lacks it.
