# OASIS

Desktop and command-line tooling for automated immunohistochemistry (IHC)
analysis of H-DAB/DAB-stained tissue images. The pipeline runs the InstanSeg
`brightfield_nuclei` model in-process for nucleus detection, exports cell-level
results, draws overlays, and writes one CSV results table per run.

The project is designed for research workflows where repeated manual cell
counting is slow or inconsistent. It keeps microscope-specific settings local
and does not require committing image data, API keys, or per-machine paths.

## Features

- Batch DAB-positive cell quantification from brightfield IHC images.
- In-process InstanSeg `brightfield_nuclei` segmentation (no external tooling).
- Fixed DAB optical-density threshold with configurable pixel size.
- GeoJSON cell-boundary export and OpenCV overlay rendering.
- One CSV results table per run, one row per image, with the per-image quality
  warnings attached to the row they belong to.
- pywebview desktop UI for setup, experiment management, analysis, and results.
- Spatial-association workflow for paired serial-section stains using image
  registration and population-level cross-type Ripley's K analysis (NOT
  single-cell co-expression, which serial sections cannot establish).

## Architecture

```text
Raw images
  -> in-process InstanSeg (TorchScript)
  -> CSV / GeoJSON / JSON exports
  -> Python overlays, CSV results table, spatial association
  -> pywebview desktop UI
```

## Requirements

- Python 3.10 or newer.
- InstanSeg `brightfield_nuclei-0.1.1` model — **ships with OASIS**, nothing to download.

### Platform support

| Platform | Prebuilt app | From source |
|---|---|---|
| macOS, Apple Silicon (arm64) | ✅ | ✅ |
| **macOS, Intel (x86_64)** | ❌ | ❌ |
| Windows x64 | not yet | ✅ (see note) |
| Linux x86_64 / aarch64 | not yet | ✅ (see note) |

**Intel Macs are not supported, and this cannot be worked around.** PyTorch no longer
publishes macOS x86_64 wheels — its macOS wheels are arm64 only. Without a PyTorch wheel
there is no segmenter, so neither the app nor a source install can run. This is an upstream
constraint, not a choice; it would change only if PyTorch resumed Intel Mac builds.

Running from source on Windows or Linux needs one extra step, because `pywebview` uses a
different GUI backend on each platform:

- **Windows** — `pip install pythonnet`; the WebView2 Runtime is built into Windows 11 and
  installable on Windows 10.
- **Linux** — install the GTK backend's system packages, e.g. on Debian/Ubuntu:
  `sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.1`

The command-line pipeline (`run_pipeline.py`) needs none of this — it has no GUI
dependency, and `device: mps` falls back to CPU automatically on non-Apple hardware.

Segmentation runs the bundled InstanSeg model in-process. No external segmenter
is needed. The removed QuPath/Groovy arm was validated as equivalent on a
598-image benchmark before removal (`research/ihc.md` §7); the code is kept for
reference in `legacy/qupath/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Edit `config.yaml` with local paths for:

- `input_dir`
- `output_dir`
- `dashboard_dir`
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

The pipeline scans `input_dir` for supported image files, segments them
in-process with InstanSeg, and writes the results table and per-image exports to
`output_dir`. `dashboard_dir` holds the overlays and the run logs.

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

Dense tissue is gated explicitly. When the per-image architecture check fails
because the tissue is genuinely fine/dense, OASIS automatically attempts the dense
morphology-conditioned primary null: B* is sampled from marker-independent
reference-section all-cell detections inside the certified analysis window, plus
2 µm jitter, and tested over the 10–30 µm DCLF band. This fallback runs only when
registration is landmark-certified, a real analysis window exists, and minimum
positive/support counts pass. Sparse/underpowered fields are not called "dense";
they remain fail-closed as not tested, with the reason recorded in JSON/UI. This is
still population-level spatial association, not same-cell co-expression. The dense
fallback is scaffold-sensitivity checked: Keren TNBC pseudo-IHC pilot validations
show p13/p16 remain stable under external-scaffold substitution and 33 perturbations,
while borderline p32 is correctly flagged as scaffold-sensitive. Dense calls near the
decision boundary should be reported with that sensitivity context.

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

The `.gitignore` excludes local secrets, virtual environments, generated
scripts, analysis outputs, and large microscopy image formats. Keep raw datasets
and machine-specific files outside Git.

## Building the desktop bundle

```bash
./packaging/build.sh
```

Produces `dist/OASIS.app` (~900 MB unpacked). The bundle is a release asset and is never
committed. See `packaging/OASIS.spec` for the freeze recipe and the bundle-only failures it
works around.

### Opening an unsigned build on macOS

OASIS is not currently code-signed or notarized, for the same reason QuPath is not — it
needs a paid Apple Developer membership. macOS will refuse to open it on first launch:

- **macOS 15 and later** — the app is reported as "damaged". Open **System Settings →
  Privacy & Security**, scroll to the message about OASIS, and choose **Open Anyway**, then
  confirm.
- **macOS 14 and earlier** — right-click the app and choose **Open**, then confirm at the
  "unidentified developer" prompt.

This is a Gatekeeper policy for unsigned applications, not a sign that the download is
corrupt. `build.sh` prints the signing and notarization commands for when a certificate is
available.

## Contributing, issues, and support

- **Report a bug or ask a question:**
  <https://github.com/smukilan9-ship-it/OASIS/issues>
- **Contributing guidelines:** [CONTRIBUTING.md](CONTRIBUTING.md) — note the extra
  requirements for any change that affects a reported number.
- **Code of conduct:** [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

Please do not attach patient images or identifiable data to issues.

## License and citation

OASIS is released under the MIT License ([LICENSE](LICENSE)).

The bundled InstanSeg `brightfield_nuclei` weights in `models/` are Apache-2.0 and
carry their own attribution and citation requirements, including the licences of
the datasets they were trained on — see [models/NOTICE.md](models/NOTICE.md). Work
using OASIS's segmentation should cite InstanSeg as well as OASIS.
