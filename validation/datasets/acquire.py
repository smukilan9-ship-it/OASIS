"""
acquire.py — consolidate datasets into the clean validation_datasets/ tree.

Goal: one canonical layout with raw INPUTS separated from generated OUTPUTS, so
another researcher (and the standalone app) finds everything the same way.

  <root>/<dir>/inputs/               raw dataset (read-only)
  <root>/<dir>/_generated_outputs/   anything that was tangled into the raw folder
  <root>/<dir>/README.md             generated per-dataset doc
  <root>/README.md                   master index + exact tree
  <root>/.migration_manifest.json    every move performed (for reversal)

Safety:
  * DRY-RUN BY DEFAULT — prints the plan; touches nothing without --apply.
  * MOVE within the same volume (instant, no duplication). Cross-volume falls
    back to copy (kept explicit in the plan output).
  * Generated outputs are STASHED into _generated_outputs/, never deleted.
  * A reversal manifest records every source->dest move.

Usage:
  python -m validation.datasets.acquire                # dry-run plan
  python -m validation.datasets.acquire --apply         # perform consolidation
  python -m validation.datasets.acquire --readmes       # (re)generate READMEs only
  python -m validation.datasets.acquire --download NAME  # fetch an open dataset
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

from . import resolve as R

# Per-dataset migration recipe. `raw` items go to inputs/; everything else in the
# source folder is stashed to _generated_outputs/. `extract_zip` unpacks an archive
# into inputs/ when its target dir is absent. Names/globs are matched at the top
# level of the legacy source folder.
MIGRATION = {
    "codex_crc": {
        "source_is_file": True,               # a single CSV, not a folder
        "raw": ["CRC_clusters_neighborhoods_markers.csv"],
    },
    "cima_landmarks": {
        "raw": ["*"],                          # whole public_landmarks folder is raw
    },
    "deepliif": {
        "raw": ["DeepLIIF_Testing_Set"],
        # phase2_RESULTS.md, pipeline_validation/, *_pred_vs_truth.jpg,
        # phase2_validation_harness.py -> _generated_outputs/
    },
    "hnscc": {
        "raw": ["*"],                          # already clean; keep as-is
    },
    "tim3_crc_icm": {
        "raw": ["TIM3_images", "CRC_ICM_clinical.xlsx", "CRC_ICM_images.zip", "labeling"],
        "extract_zip": {"archive": "CRC_ICM_images.zip", "target": "TIM3_images"},
        # phase1_diagnostic/, seg_compare/, *_contactsheet.jpg, *_crop_detail.jpg
        # -> _generated_outputs/
    },
}


def _matches(name: str, patterns: list[str]) -> bool:
    from fnmatch import fnmatch
    return any(fnmatch(name, p) for p in patterns)


def _already_consolidated(name: str) -> bool:
    return R.dataset_inputs(name).exists() and any(R.dataset_inputs(name).iterdir())


def plan(only: str | None = None) -> list[dict]:
    """Compute the list of moves without performing them."""
    actions = []
    for name, rec in R.datasets().items():
        if only and name != only:
            continue
        recipe = MIGRATION.get(name, {"raw": ["*"]})
        dst_inputs = R.dataset_inputs(name)
        if _already_consolidated(name):
            actions.append({"dataset": name, "op": "skip",
                            "reason": "already consolidated", "dst": str(dst_inputs)})
            continue
        src = R.resolve(name)
        if src is None:
            actions.append({"dataset": name, "op": "missing",
                            "reason": "not found in any legacy location"})
            continue
        stash = R.dataset_dir(name) / "_generated_outputs"
        if recipe.get("source_is_file"):
            actions.append({"dataset": name, "op": "move",
                            "src": str(src), "dst": str(dst_inputs / Path(src).name)})
            continue
        raw_pats = recipe.get("raw", ["*"])
        for child in sorted(Path(src).iterdir()):
            if child.name.startswith("."):
                continue
            if raw_pats == ["*"] or _matches(child.name, raw_pats):
                actions.append({"dataset": name, "op": "move",
                                "src": str(child), "dst": str(dst_inputs / child.name)})
            else:
                actions.append({"dataset": name, "op": "stash",
                                "src": str(child), "dst": str(stash / child.name)})
        ez = recipe.get("extract_zip")
        if ez and not (Path(src) / ez["target"]).exists():
            actions.append({"dataset": name, "op": "extract",
                            "src": str(Path(src) / ez["archive"]),
                            "dst": str(dst_inputs / ez["target"])})
    return actions


def _same_volume(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b.parent if not b.exists() else b).st_dev
    except FileNotFoundError:
        return os.stat(a).st_dev == os.stat(b.parents[0]).st_dev


def apply(only: str | None = None) -> dict:
    actions = plan(only)
    manifest = []
    for a in actions:
        if a["op"] in ("skip", "missing"):
            continue
        dst = Path(a["dst"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if a["op"] == "extract":
            with zipfile.ZipFile(a["src"]) as z:
                z.extractall(dst)
            manifest.append({**a, "done": True})
            continue
        src = Path(a["src"])
        if dst.exists():
            manifest.append({**a, "done": False, "reason": "dst exists"})
            continue
        shutil.move(str(src), str(dst))
        manifest.append({**a, "done": True})
    root = R.dataset_root()
    root.mkdir(parents=True, exist_ok=True)
    mpath = root / ".migration_manifest.json"
    prev = json.loads(mpath.read_text()) if mpath.exists() else []
    mpath.write_text(json.dumps(prev + manifest, indent=2))
    generate_readmes()
    return {"moves": sum(1 for m in manifest if m.get("done")), "manifest": str(mpath)}


# ── README generation ────────────────────────────────────────────────────────
def _tree(path: Path, prefix: str = "", depth: int = 2) -> list[str]:
    if depth < 0 or not path.exists():
        return []
    lines, entries = [], sorted([p for p in path.iterdir() if not p.name.startswith(".")])
    for i, p in enumerate(entries[:40]):
        conn = "└── " if i == len(entries) - 1 else "├── "
        lines.append(f"{prefix}{conn}{p.name}{'/' if p.is_dir() else ''}")
        if p.is_dir():
            lines += _tree(p, prefix + ("    " if i == len(entries) - 1 else "│   "), depth - 1)
    if len(entries) > 40:
        lines.append(f"{prefix}└── … ({len(entries) - 40} more)")
    return lines


def generate_readmes() -> None:
    from . import verify as V
    root = R.dataset_root()
    root.mkdir(parents=True, exist_ok=True)
    st = {r["name"]: r for r in V.status()}
    # per-dataset
    for name, rec in R.datasets().items():
        d = R.dataset_dir(name)
        if not d.exists():
            continue
        inputs = R.dataset_inputs(name)
        tree = "\n".join(_tree(inputs)) if inputs.exists() else "(inputs not yet consolidated)"
        s = st.get(name, {})
        (d / "README.md").write_text(f"""# {rec['title']}

{rec.get('what', '').strip()}

- **Source:** {rec.get('source_url', '?')}
- **License:** {rec.get('license', '?')}
- **Redistributable:** {"yes" if rec.get('redistributable') else "NO — do not upload/redistribute"}
- **Size:** {rec.get('size_hint', '?')}
- **Checksum ({rec.get('checksum_kind', 'manifest')}):** `{s.get('actual_checksum') or rec.get('checksum') or 'unpinned'}`
- **Used by validations:** {", ".join(rec.get('used_by', [])) or "—"}

## Citation
> {rec.get('citation', '').strip()}

## Expected structure
```
{rec.get('structure', '').strip()}
```

## Actual `inputs/` tree
```
inputs/
{tree}
```

Raw inputs live under `inputs/`. Any generated outputs stripped during
consolidation are under `_generated_outputs/`. Validation run outputs are written
to the repo's `validation_reports/` — never here.
""")
    # master
    rows = "\n".join(
        f"| `{n}` | {r['title']} | {r.get('dir', n)} | "
        f"{'✅' if st.get(n, {}).get('available') else '❌'} | "
        f"{'yes' if r.get('redistributable') else 'no'} |"
        for n, r in R.datasets().items())
    (root / "README.md").write_text(f"""# OASIS validation datasets

Canonical, consolidated home for every dataset the OASIS validation framework
uses. Raw **inputs** are separated from generated **outputs**; each dataset has
its own README with source, license, citation, and checksum.

**Root resolution** (`validation_data_dir`): `IHC_VALIDATION_DATA_DIR` env →
`~/.ihc_analyzer/setup.yaml:validation_data_dir` → `~/oasis_validation_datasets`.

This root is git-ignored — datasets are never committed. Restricted datasets
(e.g. HNSCC/TCIA) are documented here but never redistributed.

| name | dataset | folder | installed | redistributable |
|---|---|---|---|---|
{rows}

## Layout
```
validation_datasets/
  <DATASET>/
    inputs/              raw dataset (read-only)
    _generated_outputs/  outputs stripped out of the raw folder (if any)
    README.md            source / license / citation / checksum / structure
  datasets.yaml          machine-readable registry
  README.md              this file
```

## Commands
```
python -m validation.datasets.verify          # presence + checksum status
python -m validation.datasets.acquire          # dry-run consolidation plan
python -m validation.datasets.acquire --apply   # consolidate (move within volume)
python -m validation.datasets.verify --write     # pin checksums after consolidation
```
""")


def download(name: str) -> int:
    """Fetch an open dataset. Restricted datasets print instructions only."""
    rec = R.datasets().get(name)
    if not rec:
        print(f"unknown dataset: {name}"); return 1
    if not rec.get("redistributable"):
        print(f"{name} is NOT auto-downloadable ({rec.get('license')}).")
        print(f"Obtain it from: {rec.get('source_url')}")
        print(f"Then place it under: {R.dataset_inputs(name)}")
        return 2
    print(f"Automated download for '{name}' is not yet wired; source: "
          f"{rec.get('source_url')}\nPlace under: {R.dataset_inputs(name)}")
    return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Consolidate/verify validation datasets.")
    ap.add_argument("--apply", action="store_true", help="perform the consolidation")
    ap.add_argument("--readmes", action="store_true", help="(re)generate READMEs only")
    ap.add_argument("--download", metavar="NAME", help="fetch an open dataset")
    ap.add_argument("--only", metavar="NAME", help="limit to one dataset")
    args = ap.parse_args(argv)

    if args.download:
        return download(args.download)
    if args.readmes:
        generate_readmes()
        print(f"READMEs written under {R.dataset_root()}")
        return 0
    if args.apply:
        res = apply(args.only)
        print(f"Consolidated: {res['moves']} move(s). Manifest: {res['manifest']}")
        return 0
    # dry-run
    print(f"\nDRY-RUN — consolidation plan (root: {R.dataset_root()})\n")
    for a in plan(args.only):
        if a["op"] == "skip":
            print(f"  · {a['dataset']}: SKIP ({a['reason']})")
        elif a["op"] == "missing":
            print(f"  ✗ {a['dataset']}: MISSING ({a['reason']})")
        else:
            print(f"  {a['op'].upper():7} [{a['dataset']}] {a['src']}")
            print(f"          -> {a['dst']}")
    print("\n  Re-run with --apply to perform these moves.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
