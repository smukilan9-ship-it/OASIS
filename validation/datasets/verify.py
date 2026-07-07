"""
verify.py — dataset presence + integrity checks for the validation framework.

Checksum kinds (declared per-dataset in datasets.yaml):
  file      sha256 of the single key file (e.g. the CODEX CSV).
  manifest  sha256 of the STRUCTURE MANIFEST: sorted "relpath\\tsize" lines under
            the resolved inputs dir. Fast (no byte reads), catches missing/extra/
            resized files. Use --content for a full byte-level hash instead.

Usage:
  python -m validation.datasets.verify            # status table for every dataset
  python -m validation.datasets.verify --write     # compute + persist checksums
  python -m validation.datasets.verify --content    # manifest datasets hash bytes too
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from . import resolve as R

_REGISTRY_FILE = Path(__file__).resolve().parent / "datasets.yaml"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            yield p


def _manifest_hash(root: Path, content: bool = False) -> str:
    h = hashlib.sha256()
    for p in _iter_files(root):
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode())
        h.update(b"\t")
        h.update(str(p.stat().st_size).encode())
        if content:
            h.update(b"\t")
            h.update(_sha256_file(p).encode())
        h.update(b"\n")
    return h.hexdigest()


def compute_checksum(name: str, content: bool = False) -> str | None:
    """Compute the checksum for a resolved dataset, or None if unavailable."""
    rec = R.datasets().get(name, {})
    path = R.resolve(name)
    if path is None:
        return None
    kind = rec.get("checksum_kind", "manifest")
    if kind == "file":
        # resolve() returns the file itself for file-kind datasets.
        target = path if path.is_file() else (path / Path(rec.get("key_path", "")).name)
        return _sha256_file(target) if target.is_file() else None
    return _manifest_hash(path, content=content)


def status(content: bool = False) -> list[dict]:
    """Per-dataset status records for the CLI table and the Validation-tab API."""
    out = []
    for name, rec in R.datasets().items():
        path = R.resolve(name)
        available = path is not None
        expected = rec.get("checksum")
        actual = compute_checksum(name, content=content) if available else None
        if not available:
            state = "missing"
        elif expected is None:
            state = "present_unpinned"     # present but no checksum recorded yet
        elif actual == expected:
            state = "ok"
        else:
            state = "checksum_mismatch"
        out.append({
            "name": name,
            "title": rec.get("title", name),
            "dir": rec.get("dir", name),
            "available": available,
            "path": str(path) if path else None,
            "state": state,
            "expected_checksum": expected,
            "actual_checksum": actual,
            "redistributable": bool(rec.get("redistributable", False)),
            "size_hint": rec.get("size_hint"),
            "source_url": rec.get("source_url"),
        })
    return out


def write_checksums(content: bool = False) -> int:
    """Persist freshly computed checksums into datasets.yaml. Returns count written."""
    doc = yaml.safe_load(_REGISTRY_FILE.read_text()) or {}
    n = 0
    for name, rec in doc.get("datasets", {}).items():
        cs = compute_checksum(name, content=content)
        if cs is not None:
            rec["checksum"] = cs
            n += 1
    _REGISTRY_FILE.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    R._registry.cache_clear()
    return n


def _print_table(rows: list[dict]) -> None:
    icon = {"ok": "✓", "present_unpinned": "•", "missing": "✗",
            "checksum_mismatch": "⚠"}
    for r in rows:
        print(f"  {icon.get(r['state'], '?')} {r['name']:<16} {r['state']:<18} "
              f"{r['title']}")
        if r["available"]:
            print(f"      path: {r['path']}")
        elif not r["redistributable"]:
            print(f"      MISSING — restricted dataset, fetch from: {r['source_url']}")
        else:
            print(f"      MISSING — fetch from: {r['source_url']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify validation datasets.")
    ap.add_argument("--write", action="store_true", help="persist computed checksums")
    ap.add_argument("--content", action="store_true",
                    help="full byte-level hash for manifest datasets")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args(argv)

    if args.write:
        n = write_checksums(content=args.content)
        print(f"Wrote checksums for {n} available dataset(s) to datasets.yaml")
    rows = status(content=args.content)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"\nValidation datasets — root: {R.dataset_root()}\n")
        _print_table(rows)
        ok = sum(1 for r in rows if r["state"] in ("ok", "present_unpinned"))
        print(f"\n  {ok}/{len(rows)} available.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
