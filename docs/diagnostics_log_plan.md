# Troubleshooting / diagnostics log — plan (HELD, not yet built)

Status: **design only.** Do not wire into the pipeline until confirmed.
Owner decision pending: raw-stdout/stderr retention policy (deferred, "discuss later").

## Goal
A per-run, structured, self-explaining diagnostic log that (a) contains *exactly*
what is needed to troubleshoot a run, (b) is optionally included in the output, and
(c) is consumed by a future **Troubleshooting tab** that reads it and tells the user
"what happened and what to do next."

## Key insight (why this is low-risk)
The pipeline already **computes** almost everything a troubleshooter needs — it is
just scattered as one-off `print()`s and verdict fields, then flattened into raw
`_stdout.log` / `_stderr.log` text dumps (`run_pipeline.py:386-391`) that a tab
cannot parse. Existing signals to harvest:
- nuclear classifier → `reason` / `quality` / `abstain` / `separability`
  (`oasis/quant/nuclear_classify.py`)
- membrane → `membrane_quality_warning` + positive-rate (`run_pipeline.py:563`)
- registration → LOO-TRE / residual / certification verdict
  (`oasis/spatial/serial_registration.py`)
- spatial stats → cross-K `verdict`, bandwidth `status`, dense-invalid
  (`oasis/spatial/spatial_stats.py`)
- pixel size → source + `pixel_size_warning`

So the log is a **consolidation layer**, not new science. It normalizes these into
one typed event stream. **No science / thresholds change.**

## Design principle — "extremely easy to troubleshoot"
Every event is self-explaining and carries the four things a human (or the tab) needs:
`what` happened · `why` (with the exact measured numbers that fired it) · `next`
(actionable fix) · a stable `code` the tab maps to a fix + doc anchor.

## Structure

### `diagnostics.json` (one per run) + human-readable `troubleshooting.md`

**A. Run header** (context that explains everything downstream)
- Resolved config: pixel size **+ source** (manual / scale-bar / global default),
  DAB threshold + source, adaptive-nuclear on/off, membrane on/off, mode.
- Environment: QuPath version, InstanSeg model, Python/venv, optional-dep presence
  (SimpleITK, openslide, LoFTR).
- Per-stage timings, cell/tile counts, overall status
  (`ok` / `ok_with_warnings` / `blocked`).

**B. Typed event stream** — each entry:
```json
{ "stage": "...", "level": "info|warn|error|blocker", "code": "STABLE_CODE",
  "image": "file or pair", "what": "plain sentence",
  "why": "measured cause", "evidence": { "numbers": "that fired it" },
  "next": "actionable fix", "auto_recoverable": true, "doc": "ihc.md#anchor" }
```

**C. Event catalog** (grounded in real failure points in the code)

| Stage | Codes | Evidence captured |
|---|---|---|
| Input/match | `IMAGE_UNREADABLE`, `NO_PAIR_MATCH`, `CHANNEL_MISSING` | filenames, matcher score |
| Calibration | `PX_MISSING` (blocker), `PX_USED_DEFAULT` (warn), `SCALEBAR_NOT_FOUND`, `SCALE_MISMATCH` | px value+source, scale-bar px, fitted-scale vs bar ratio |
| Segmentation | `QUPATH_FAILED`, `QUPATH_TIMEOUT`, `SEG_EMPTY`, `SEG_LOW_COUNT` | exit code, stderr tail, cell count |
| Nuclear | `NUCLEAR_ABSTAIN`, `NUCLEAR_LOW_QUALITY`, `NUCLEAR_FIXED_FALLBACK` | separability vs 1.25, cell count, method, reason |
| Membrane | `MEMBRANE_LOW_QUALITY`, `MEMBRANE_ABSTAIN` | positive-rate, faint-contrast Δ |
| Registration | `REG_IDENTITY_FALLBACK`, `REG_RESIDUAL_HIGH`, `REG_SCALE_IMPLAUSIBLE`, `REG_NOT_DISTANCE_PRESERVING`, `REG_LANDMARKS_INSUFFICIENT`, cert verdict (`CERTIFIED` / `LOCALLY_CERTIFIED` / `RADIUS_LIMITED` / `NOT_CERTIFIED`) | residual median/p90 vs 10µm, LOO-TRE vs 5µm, n landmarks vs 6, similarity defect |
| Spatial stats | `TISSUE_MASK_EMPTY`, `BANDWIDTH_UNRELIABLE` / `_CAUTION`, `CROSSK_CSR_ONLY`, `CROSSK_NOT_RESOLVABLE`, `DENSE_BANDWIDTH_INVALID` | bandwidth vs ℓ̂, verdict, p-value, radius band frac |

**D. "Option to include in output"** — a toggle mirroring the working-files toggle
(`include_diagnostics`, default ON). When on, `diagnostics.json` + `troubleshooting.md`
land in the results folder; the future tab reads the JSON. The tab's
"what happened / what to do next" is literally rendering `what` / `next` grouped by
severity.

## Touch points when built (all additive)
`run_pipeline.py` (quant single/batch) · spatial pipeline (registration + stats) ·
`oasis/webui/api.py` (`_load_results` to surface it) · one UI toggle in the Quant/Spatial
output options.

## Open decisions (deferred)
1. Raw `_stdout/_stderr.log` retention: structured-only-with-raw-fallback vs always-keep-raw.
   ("discuss later")
2. Build scope order: quant-first vs all-stages-at-once.
