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

## Redesign later (removed on purpose)
- [ ] Image-based end-to-end degradation/recovery validation — removed (tissue-scale data
      was scientifically inappropriate). Redesign on a proper **cell-scale** two-marker
      dataset. The CODEX coordinate keystone (`tests/test_degradation.py`) remains.

## Open scientific items (from the audit — for the paper, not blocking the tool)
- [ ] Real quantification agreement number (nuclear + membrane) via `validate_segmentation.py`
      with manual GeoJSON — replaces the removed "~90%" claim.
- [ ] Measure per-image architecture scale → turn the 75 µm bandwidth assumption into a
      runtime guard (currently disclosed, not measured).
- [ ] Report leave-one-**image**-out calibration alongside leave-one-cell-out.
