"""
run.py — headless CLI for the OASIS validation framework.

  python -m validation.run --list                 # list validations + status
  python -m validation.run <id> [<id> ...]         # run specific validation(s)
  python -m validation.run all                      # run everything runnable
  python -m validation.run all --tier instant       # only instant-tier validations
  python -m validation.run <id> --force              # run even if preflight fails

Same runner + report bundles as the desktop Validation tab, so a CLI run and a
UI run are byte-for-byte the same validation.
"""
from __future__ import annotations

import argparse
import sys

from . import registry, runner


def _list() -> int:
    for cat in registry.by_category():
        print(f"\n{cat['title']}")
        for v in cat["validations"]:
            pf = runner.preflight(v["id"])
            last = runner.last_report(v["id"])
            tag = "ready" if pf["ok"] else f"SKIP ({pf['reason']})"
            lastr = f"  last={last['status']}" if last else ""
            print(f"  {v['id']:<28} [{v['runtime_tier']:<7}] {tag}{lastr}")
    print()
    return 0


def _run(ids: list[str], force: bool) -> int:
    rc = 0
    for vid in ids:
        print(f"\n=== {vid} ===")
        rep = runner.run_validation(vid, on_line=lambda ln, lvl: print(f"  {ln}"),
                                    force=force)
        print(f"  -> {rep['status']}  metrics={rep.get('metrics')}")
        if rep["status"] == "FAIL":
            rc = 1
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run OASIS validations.")
    ap.add_argument("ids", nargs="*", help="validation id(s), or 'all'")
    ap.add_argument("--list", action="store_true", help="list validations + status")
    ap.add_argument("--tier", choices=["instant", "short", "long"], help="filter 'all' by tier")
    ap.add_argument("--force", action="store_true", help="run even if preflight fails")
    args = ap.parse_args(argv)

    if args.list or not args.ids:
        return _list()
    if args.ids == ["all"]:
        ids = [v["id"] for v in registry.VALIDATIONS
               if not args.tier or v["runtime_tier"] == args.tier]
    else:
        ids = args.ids
        unknown = [i for i in ids if registry.by_id(i) is None]
        if unknown:
            print(f"unknown validation(s): {', '.join(unknown)}")
            return 2
    return _run(ids, args.force)


if __name__ == "__main__":
    sys.exit(main())
