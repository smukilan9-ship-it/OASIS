"""
runner.py — execute a registered validation and emit a paper-grade report bundle.

Report bundle (per run):
  validation_reports/<id>/<timestamp>/
    report.json   status, metrics, expected, config, software+git, dataset checksums, timing
    run.log       captured stdout/stderr
    <plots>       any figures the validation wrote into the run dir

Status:
  PASS   exit code 0
  FAIL   non-zero exit code
  SKIP   preflight failed (missing dataset or external tool) — never a false FAIL

Metrics contract (zero-refactor): a validation MAY print one or more lines
  ##METRICS## {"f1": 0.81, ...}
which the runner merges into report["metrics"]. Absent that, metrics stay {} and
the human-readable output is preserved verbatim in run.log.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import registry
from .datasets import resolve as R
from .datasets import verify as V

REPO = Path(__file__).resolve().parents[1]
REPORTS_ROOT = REPO / "validation_reports"
_SETUP_FILE = Path.home() / ".ihc_analyzer" / "setup.yaml"
_METRICS_RE = re.compile(r"##METRICS##\s*(\{.*\})\s*$")


# ── provenance ────────────────────────────────────────────────────────────────
def _git_sha() -> dict:
    def _g(*args):
        try:
            return subprocess.run(["git", *args], cwd=str(REPO), capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"sha": _g("rev-parse", "HEAD"),
            "short": _g("rev-parse", "--short", "HEAD"),
            "dirty": bool(_g("status", "--porcelain"))}


def _lib_versions() -> dict:
    import importlib.metadata as m
    out = {"python": sys.version.split()[0]}
    for pkg in ("numpy", "scipy", "opencv-python", "shapely", "SimpleITK", "matplotlib"):
        try:
            out[pkg] = m.version(pkg)
        except Exception:
            out[pkg] = None
    return out


def _setup() -> dict:
    if _SETUP_FILE.exists():
        try:
            return yaml.safe_load(_SETUP_FILE.read_text()) or {}
        except Exception:
            return {}
    return {}


# ── preflight ─────────────────────────────────────────────────────────────────
def _dep_available(dep: str, setup: dict) -> bool:
    if dep == "qupath":
        p = os.path.expanduser(str(setup.get("qupath_binary", "")))
        return bool(p) and Path(p).exists()
    if dep == "instanseg":
        p = os.path.expanduser(str(setup.get("instanseg_model", "")))
        return bool(p) and Path(p).exists()
    if dep == "R":
        return shutil.which("Rscript") is not None
    return False


def preflight(vid: str) -> dict:
    """Return {ok, missing_datasets, missing_deps, reason}."""
    rec = registry.by_id(vid)
    if rec is None:
        return {"ok": False, "reason": f"unknown validation '{vid}'",
                "missing_datasets": [], "missing_deps": []}
    setup = _setup()
    missing_ds = [d for d in rec.get("datasets", []) if not R.is_available(d)]
    missing_dep = [d for d in rec.get("external_deps", []) if not _dep_available(d, setup)]
    ok = not missing_ds and not missing_dep
    reason = ""
    if missing_ds:
        reason = "missing dataset(s): " + ", ".join(missing_ds)
    if missing_dep:
        reason = (reason + "; " if reason else "") + "missing tool(s): " + ", ".join(missing_dep)
    return {"ok": ok, "missing_datasets": missing_ds, "missing_deps": missing_dep,
            "reason": reason}


# ── run ─────────────────────────────────────────────────────────────────────--
def _command(rec: dict) -> list[str]:
    r = rec["runner"]
    if r["kind"] == "pytest":
        return [sys.executable, "-m", "pytest", "-q", r["node"]]
    script = REPO / "validation" / r["script"]
    return [sys.executable, str(script), *r.get("argv", [])]


def run_validation(vid: str, on_line=None, force: bool = False) -> dict:
    """
    Execute a validation; stream each output line to on_line(line, level) if given;
    write the report bundle; return the report dict.
    `force` runs even if preflight fails (still recorded).
    """
    rec = registry.by_id(vid)
    if rec is None:
        raise ValueError(f"unknown validation '{vid}'")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = REPORTS_ROOT / vid / ts
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "run.log"

    pf = preflight(vid)
    report = {
        "id": vid, "title": rec["title"], "category": rec["category"],
        "claim": rec["claim"], "expected": rec["expected"],
        "timestamp_utc": ts, "status": None, "metrics": {},
        "preflight": pf,
        "software": {**_lib_versions(), "git": _git_sha()},
        "datasets": {d: (V.status() and next((s["actual_checksum"]
                     for s in V.status() if s["name"] == d), None))
                     for d in rec.get("datasets", [])},
        "command": None, "returncode": None, "duration_s": None,
        "log": "run.log", "plots": [],
    }

    if not pf["ok"] and not force:
        report["status"] = "SKIP"
        report["reason"] = pf["reason"]
        log_path.write_text(f"SKIP — {pf['reason']}\n")
        if on_line:
            on_line(f"SKIP — {pf['reason']}", "warn")
        _write(outdir, report)
        return report

    cmd = _command(rec)
    report["command"] = " ".join(cmd)
    metrics: dict = {}
    t0 = time.time()
    # Let plotting validations write figures into the run dir if they honour this.
    env = {**os.environ, "OASIS_REPORT_DIR": str(outdir)}
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                bufsize=1)
        for line in proc.stdout:
            clean = line.rstrip("\n")
            logf.write(line)
            m = _METRICS_RE.search(clean)
            if m:
                try:
                    metrics.update(json.loads(m.group(1)))
                except Exception:
                    pass
            if on_line:
                lvl = ("error" if re.search(r"FAIL|ERROR|Traceback", clean)
                       else "ok" if re.search(r"PASS|✓|complete", clean)
                       else "normal")
                on_line(clean, lvl)
        proc.wait()

    report["returncode"] = proc.returncode
    report["duration_s"] = round(time.time() - t0, 2)
    report["metrics"] = metrics
    report["status"] = "PASS" if proc.returncode == 0 else "FAIL"
    report["plots"] = sorted(p.name for p in outdir.glob("*.png"))
    _write(outdir, report)
    if on_line:
        on_line(f"{report['status']} ({report['duration_s']}s) — report: {outdir}",
                "ok" if report["status"] == "PASS" else "error")
    return report


def _write(outdir: Path, report: dict) -> None:
    (outdir / "report.json").write_text(json.dumps(report, indent=2))
    # config snapshot = the full registry record (claim/why/assumptions/limits/…)
    rec = registry.by_id(report["id"])
    (outdir / "config.yaml").write_text(yaml.safe_dump(rec, sort_keys=False, allow_unicode=True))
    _update_index(report, outdir)


def _update_index(report: dict, outdir: Path) -> None:
    idx = REPORTS_ROOT / "index.json"
    data = {}
    if idx.exists():
        try:
            data = json.loads(idx.read_text())
        except Exception:
            data = {}
    data[report["id"]] = {
        "status": report["status"], "timestamp_utc": report["timestamp_utc"],
        "duration_s": report["duration_s"], "metrics": report["metrics"],
        "dir": str(outdir.relative_to(REPO)),
    }
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    idx.write_text(json.dumps(data, indent=2))


def last_report(vid: str) -> dict | None:
    """Most recent report.json for a validation, or None."""
    d = REPORTS_ROOT / vid
    if not d.exists():
        return None
    runs = sorted([p for p in d.iterdir() if p.is_dir()])
    if not runs:
        return None
    rp = runs[-1] / "report.json"
    return json.loads(rp.read_text()) if rp.exists() else None
