# OASIS — TODO / next steps

Living checklist of what's left after the validation-infrastructure refactor.

## Push & GitHub (in progress)
- [ ] Push `main` to `https://github.com/smukilan9-ship-it/OASIS.git`.
  - Repo must exist and be **empty** (no README/license/gitignore) to avoid a push conflict.
  - Auth: use a **fine-grained PAT** with `Contents: read/write` on OASIS (or a classic PAT
    with `repo` scope). If the macOS keychain holds a stale/read-only token:
    ```bash
    printf 'protocol=https\nhost=github.com\n\n' | git credential-osxkeychain erase
    git push -u origin main    # paste the PAT when prompted for password
    ```
- [ ] Confirm on GitHub that **smukilan9-ship-it is the sole contributor** (all commits are
      authored `smukilan9@gmail.com`; add that email under GitHub → Settings → Emails so it
      links to the account). No Claude/co-author trailers remain.

## Troubleshooting / diagnostics log (design held — not built)
- [ ] Build the structured per-run diagnostics log per **`docs/diagnostics_log_plan.md`**:
      `diagnostics.json` + `troubleshooting.md` harvested from the verdicts/warnings the
      pipeline already computes (nuclear reason/quality, membrane quality, registration
      TRE/cert, cross-K/bandwidth verdicts, pixel-size source). Additive, no science change.
      Feeds the future **Troubleshooting tab** ("what happened / what to do next").
  - Open: raw stdout/stderr retention policy (deferred); build scope order (quant-first vs all).

## Datasets (shipping)
- [ ] Decide hosting for the large/restricted datasets and send the links:
  - DeepLIIF (~4 GB), TIM-3 CRC-ICM (~1 GB) → Google Drive / other host.
  - HNSCC (~1 GB) → **document-only** (TCIA login; not redistributable).
  - CODEX (213 MB) + CIMA (1.6 MB) are redistributable → can ship or auto-download.
- [ ] Once links exist, wire download URLs into `validation/datasets/datasets.yaml` so the
      Validation-tab "Download" buttons and `python -m validation.datasets.acquire --download <name>`
      fetch into `~/oasis_validation_datasets/<NAME>/inputs`.
- [ ] (Optional) Later: move datasets to Zenodo for a DOI (paper stage).

## Repo polish before/at first release
- [ ] Add a `LICENSE` (pick one: MIT / BSD-3 / Apache-2.0).
- [ ] Add a short "Datasets" hosting section + a one-line install/run quickstart to README.
- [ ] Consider `pyproject.toml` + pinned lockfile and a minimal CI (pytest on push) —
      strengthens reproducibility for the paper.

## Validation framework follow-ups (nice-to-have)
- [ ] Enrich `report.json` metrics: add a `##METRICS## {json}` print line to key validation
      scripts (cross_k, dclf, keystone, membrane_cd8_hnscc, deepliif) so cards show numbers,
      not just PASS/FAIL. Logs + provenance already captured.
- [ ] Fix the stale `.claude/launch.json` `tim3-label` preview server (points at the moved
      `~/Desktop/tim3 data/labeling`).

## Open scientific items (from the audit — for the paper, not blocking the tool)
- [ ] Real quantification agreement number (nuclear + membrane) via `validate_segmentation.py`
      with manual GeoJSON — replaces the removed "~90%" claim.
- [x] Measure per-image architecture scale → runtime guard (DONE: `estimate_architecture_scale`
      + `validate_architecture_scale.py` + spatial-path gate; ihc.md §7).
- [ ] Report leave-one-**image**-out calibration alongside leave-one-cell-out.

---

# ▶ NEXT SESSION TASK: Build Validation A (`validate_e2e_render_codex.py`)

**This brief is self-contained — a fresh session with no prior context can execute it.**

## Context (what OASIS is, in one paragraph)
OASIS is a deterministic pipeline for **cross-type spatial association** on serial-section
single-plex H-DAB IHC (e.g. CD8 vs TIM-3). It segments nuclei (QuPath + InstanSeg),
registers two serial sections (similarity only, never warping), and measures whether two
cell **populations** are spatially associated beyond chance via **cross-type Ripley's K**
with a reweighted-inhomogeneous null (population-level, NOT single-cell co-expression).
Read `ihc.md` first (esp. §4 statistics, §7 validation, §10 end-to-end suite).

## Why A exists (the scientific gap it closes)
Every cell-scale validation is a **proxy** because real-DAB cell-scale ground truth for two
*different* markers on corresponding sections cannot exist (serial sections = different
slices). We bound the gap from three sides (`ihc.md §10`):
- **Keystone** (`tests/test_degradation.py`, done): real cross-marker truth, but *coordinates* only.
- **B** (`validation/validate_e2e_knownwarp_deepliif.py`, done): real DAB *pixels* + full pipeline, but *trivial* same-marker truth.
- **A** (THIS TASK): full pipeline on **real cross-marker truth**, with *synthetic* pixels.
A + B + keystone jointly bound the untestable real case. A supplies the missing corner:
full pixel pipeline (segment → register → cross-K) driven by genuine cross-marker truth.

## What A does (design)
Render real CODEX cross-marker cells into **cell-scale brightfield H-DAB tiles**, run the
FULL real pipeline, and check the recovered verdict matches the known CODEX verdict.
1. Load the CODEX table (dataset `codex_crc`, already installed). Pick the largest spot;
   gate two markers (CD8 = `CD8 - cytotoxic T cells:Cyc_3_ch_2`, PD-1 =
   `PD-1 - checkpoint:Cyc_12_ch_4`) at the 80th percentile — SAME as `tests/test_degradation.py`.
2. **Ground-truth verdict** = run `spatial_stats.cross_k_all_nulls` on the raw CODEX
   coordinates (this is what the keystone already trusts) → `truth_verdict`.
3. **Render two brightfield tiles** at cell scale (target ~0.5 µm/px):
   - tile_A: a light H&E-like background; draw a **hematoxylin nucleus blob** at every
     CD8 cell centroid, and brown **DAB** (nuclear or a membrane ring) for CD8+ cells.
   - tile_B: same, for the PD-1 cell centroids.
   - CODEX coords are in µm-ish pixels; scale coords → tile pixels so the field is a few
     thousand px and the 10–50 µm interaction band spans tens of px (cell-scale, NOT the
     <2 px mistake that killed the old image e2e).
4. Apply a **known misregistration** to tile_B (rotation+translation, like B's `_warp`).
5. Run the **real pipeline** on both tiles (InstanSeg via `run_pipeline --mode quant`) →
   cell centroids. Register tile_B→tile_A (`serial_registration.register_similarity`).
6. Apply recovered transform → cross-K (`cross_k_all_nulls`) → `recovered_verdict`.
7. **PASS** if `recovered_verdict == truth_verdict` across engaged/independent/csr_only
   regimes AND survives the injected registration error (mirror the keystone's 3 regimes:
   real pair; planted-engaged partner; planted-independent partner).

## Exact steps
1. Copy the skeleton/patterns from **`validation/validate_e2e_knownwarp_deepliif.py`**
   (Validation B) — reuse its `_setup()`, `_segment()`, `_warp()`, `_apply()`, `_verdict()`
   almost verbatim (the segment→register→cross-K plumbing is identical).
2. Copy CODEX loading + marker gating + the 3-regime construction from
   **`tests/test_degradation.py`** (`_largest_spot`, the CD8/PD-1 columns, the engaged/
   independent partner synthesis).
3. Write a **`render_brightfield(centroids, marker_pos_mask, size_px, pixel_size_um)`**
   helper: white background (≈245,245,245); for each cell draw a hematoxylin disc
   (bluish, radius ~3–4 px) via `cv2.circle`; for marker-positive cells overlay DAB brown
   (≈110,64,32) as a nuclear fill or a ring. Keep it simple and deterministic.
4. New file **`validation/validate_e2e_render_codex.py`** with `main()` returning 0/1,
   printing `##METRICS## {json}` (truth vs recovered verdict per regime, reconstruction
   TRE, n cells) and saving a comparison figure to `os.environ.get("OASIS_REPORT_DIR",".")`.
5. **Register** in `validation/registry.py` under category `end_to_end` (copy the
   `e2e_knownwarp_deepliif` record; set `datasets: ["codex_crc"]`,
   `external_deps: ["qupath","instanseg"]`, `runtime_tier: "long"`).
6. Update `ihc.md §10` table: flip row **A** from ⏳ to ✅ with the measured result.

## Reuse map (paths)
- Segment/register/cross-K plumbing: `validation/validate_e2e_knownwarp_deepliif.py`
- CODEX load + gating + regimes: `tests/test_degradation.py`
- Statistic: `spatial_stats.cross_k_all_nulls(a, b, radii, area_px, pixel_size_um, n_perm, seed)`
  → `["robustness"]["verdict"]` ∈ {robust, csr_only, none, mixed}
- Registration: `serial_registration.register_similarity(ref_rgb, mov_rgb, pixel_size_um)`
  → `{matrix (2x3, moving→ref), struct_dice, success}`; apply with `(M @ [pts,1].T).T`
- Dataset path: `from validation.datasets import resolve; resolve.resolve("codex_crc")`
  (returns the CODEX CSV file path)
- Run/verify: `python validation/validate_e2e_render_codex.py --tiles/--nperm ...`, then
  `python -m validation.run e2e_render_codex` (report bundle → `validation_reports/`).

## Gotchas
- **Resolution is the whole point**: render so the 10–50 µm band spans tens of px, else you
  repeat the tissue-scale mistake that got the old e2e deleted. Verify pixel_size_um flows
  into `cross_k_all_nulls` correctly.
- InstanSeg won't segment overly cartoonish blobs well — make nuclei look nucleus-like
  (soft edges, slight size variation) or validate detection count vs input count first.
- The comparison must be verdict-level (truth vs recovered), not exact-count.
- Datasets live at `~/oasis_validation_datasets` (`validation_data_dir`); `codex_crc` is
  redistributable and already present. QuPath+InstanSeg must be installed (preflight gates it).
