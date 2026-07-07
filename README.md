# OASIS

Desktop and command-line tooling for automated immunohistochemistry (IHC)
analysis of H-DAB/DAB-stained tissue images. The pipeline uses QuPath with
InstanSeg for nucleus detection, exports cell-level results, draws overlays,
and generates dashboard and Excel summaries.

The project is designed for research workflows where repeated manual cell
counting is slow or inconsistent. It keeps microscope-specific settings local
and does not require committing image data, API keys, or per-machine paths.

## Features

- Batch DAB-positive cell quantification from brightfield IHC images.
- QuPath headless execution with InstanSeg `brightfield_nuclei`.
- Fixed DAB optical-density threshold with configurable pixel size.
- GeoJSON cell-boundary export and OpenCV overlay rendering.
- HTML dashboard and Excel workbook generation.
- pywebview desktop UI for setup, experiment management, analysis, results,
  and optional AI-assisted result discussion.
- Spatial-association workflow for paired serial-section stains using image
  registration and population-level cross-type Ripley's K analysis (NOT
  single-cell co-expression, which serial sections cannot establish).

## Architecture

```text
Raw images
  -> QuPath headless + InstanSeg
  -> CSV / GeoJSON / JSON exports
  -> Python overlays, dashboard, Excel, spatial association
  -> pywebview desktop UI
```

## Requirements

- Python 3.10 or newer.
- QuPath 0.7.x with the InstanSeg extension installed.
- InstanSeg `brightfield_nuclei-0.1.1` model downloaded locally.
- macOS is the currently targeted desktop environment.

Optional AI chat support uses either:

- `GEMINI_API_KEY` for Gemini.
- `ANTHROPIC_API_KEY` for Claude.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp config.example.yaml config.yaml
```

Edit `config.yaml` with local paths for:

- `input_dir`
- `output_dir`
- `dashboard_dir`
- `qupath_binary`
- `instanseg_model`

The desktop app also stores user setup in `~/.ihc_analyzer/`.

## Run The Desktop App

```bash
python app.py
```

## Run The Quantification Pipeline

```bash
python run_pipeline.py --config config.yaml
```

The pipeline scans `input_dir` for supported image files, runs QuPath
headlessly, and writes results to `output_dir` and `dashboard_dir`.

## Run Spatial Association

Spatial association is launched from the desktop UI, or from the command line
with a config that includes `spatial_pairs`:

```bash
python run_pipeline.py --config config.yaml --mode spatial
```

(`--mode coloc` is accepted as a deprecated alias of `--mode spatial`.)

Pairs are processed stain-by-stain and registered into a shared coordinate
space. Population-level spatial association is then measured with the cross-type
Ripley's K / pair-correlation g(r) functions against a tissue-mask-bounded Monte
Carlo null, with a global DCLF envelope test for significance. This is a
population statistic and does **not** assert single-cell co-expression — serial
sections cannot establish co-expression (different Z-planes, TIM-3 is not
CD8-restricted, membrane-vs-nuclear compartments).

## Validation & Reproducibility

Every scientific claim is validated by a registered validation, runnable from the
desktop **Validation** tab or the command line — same runner, same report bundles.

```bash
python -m validation.run --list                 # list validations + dataset/preflight status
python -m validation.run cross_k                  # run one validation
python -m validation.run all --tier instant        # run all no-dataset statistical checks
python -m validation.datasets.verify              # dataset presence + checksum table
python -m validation.datasets.acquire --apply       # consolidate datasets into the tree
```

Each run writes a paper-grade bundle to `validation_reports/<id>/<timestamp>/`
(`report.json` with metrics, status, expected result, software + git SHA, dataset
checksums, timing; `run.log`; any plots) — suitable for supplementary material.

Datasets live in one consolidated tree at `validation_data_dir`
(default `~/oasis_validation_datasets`, override via `~/.ihc_analyzer/setup.yaml`
or the `IHC_VALIDATION_DATA_DIR` env var), with raw **inputs** separated from
generated **outputs** and a per-dataset README + license + checksum. Datasets are
never committed; see `validation/datasets/README.md` and `datasets.yaml` for
sources, licenses, and citations. Restricted datasets (e.g. HNSCC/TCIA) are
documented but never redistributed. Missing datasets skip their validations with a
message naming the exact dataset and source.

## Configuration

`config.example.yaml` contains safe defaults and placeholders. Do not commit
your local `config.yaml`; it may include private paths, sample names, or output
locations.

Important fields:

- `dab_threshold`: DAB mean OD threshold for positive classification.
- `default_pixel_size`: microns per pixel used when no image metadata override
  is available.
- `device`: InstanSeg device, such as `mps`, `cuda`, or `cpu`.
- `cleanup_intermediates`: remove CSV, GeoJSON, logs, and metadata after
  summary outputs are created.

## Repository Hygiene

The `.gitignore` excludes local secrets, virtual environments, generated QuPath
scripts, analysis outputs, and large microscopy image formats. Keep raw datasets
and machine-specific files outside Git.
