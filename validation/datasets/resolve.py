"""
resolve.py — dataset path resolution for the validation framework.

Single rule for WHERE datasets live, honoured identically by the test suite, the
standalone validation scripts, the CLI runner, and the desktop Validation tab —
so nothing is hardcoded to ~/Desktop and the app stays reproducible when bundled.

Resolution order for the root `validation_data_dir`:
  1. env  IHC_VALIDATION_DATA_DIR
  2. ~/.ihc_analyzer/setup.yaml  ->  validation_data_dir
  3. default  <repo>/validation_datasets

A dataset resolves to  <root>/<dir>/inputs  when its `key_path` exists there.
For backward compatibility during migration, if the consolidated tree is absent
we fall back to the dataset's legacy locations (env IHC_DATA_<NAME>, the old
paths.yaml, and the `legacy_paths` in datasets.yaml).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
_REGISTRY_FILE = Path(__file__).resolve().parent / "datasets.yaml"
_SETUP_FILE = Path.home() / ".ihc_analyzer" / "setup.yaml"
_LEGACY_PATHS_YAML = REPO / "validation" / "datasets" / "paths.yaml"


@lru_cache(maxsize=1)
def _registry() -> dict:
    with open(_REGISTRY_FILE) as f:
        return yaml.safe_load(f) or {}


def datasets() -> dict:
    """name -> registry record."""
    return dict(_registry().get("datasets", {}))


def _setup_value(key: str):
    if not _SETUP_FILE.exists():
        return None
    try:
        data = yaml.safe_load(_SETUP_FILE.read_text()) or {}
        return data.get(key)
    except Exception:
        return None


def dataset_root() -> Path:
    """The configurable root that holds every consolidated dataset."""
    env = os.environ.get("IHC_VALIDATION_DATA_DIR")
    if env:
        return Path(env).expanduser()
    cfg = _setup_value("validation_data_dir")
    if cfg:
        return Path(str(cfg)).expanduser()
    # Default lives OUTSIDE the repo (home dir) so the project folder stays lean
    # and the location survives the app being bundled as a standalone binary.
    return Path.home() / "oasis_validation_datasets"


def dataset_dir(name: str) -> Path:
    """<root>/<dir> for a dataset (may not exist yet)."""
    rec = datasets().get(name, {})
    return dataset_root() / rec.get("dir", name)


def dataset_inputs(name: str) -> Path:
    """<root>/<dir>/inputs — the canonical raw-input location."""
    return dataset_dir(name) / "inputs"


def _legacy_candidates(name: str):
    """Yield legacy locations for back-compat during/after migration."""
    # 1. explicit env override (matches the old conftest contract)
    env = os.environ.get(f"IHC_DATA_{name.upper()}")
    if env:
        yield Path(env).expanduser()
    # 2. old paths.yaml
    if _LEGACY_PATHS_YAML.exists():
        try:
            y = yaml.safe_load(_LEGACY_PATHS_YAML.read_text()) or {}
            if y.get(name):
                yield Path(str(y[name])).expanduser()
        except Exception:
            pass
    # 3. legacy_paths baked into datasets.yaml
    for p in datasets().get(name, {}).get("legacy_paths", []) or []:
        yield Path(str(p)).expanduser()


def resolve(name: str) -> Path | None:
    """
    Return the path a validation should read for this dataset, or None if the
    dataset is not available anywhere.

    Prefers the consolidated  <root>/<dir>/inputs  tree (verified via key_path);
    falls back to legacy locations so nothing breaks mid-migration.
    """
    rec = datasets().get(name)
    if rec is None:
        return None
    inputs = dataset_inputs(name)
    key = rec.get("key_path", "")
    # key_path is relative to <dir>; strip a leading "inputs/" if present.
    key_rel = key[len("inputs/"):] if key.startswith("inputs/") else key
    # Single-file datasets (checksum_kind == "file", e.g. the CODEX CSV) resolve
    # to the FILE itself so consumers can open it directly; directory datasets
    # resolve to the inputs/ dir the consumer navigates from.
    is_file = rec.get("checksum_kind") == "file"
    if key_rel and (inputs / key_rel).exists():
        return (inputs / key_rel) if is_file else inputs
    if not key_rel and inputs.exists():
        return inputs
    for cand in _legacy_candidates(name):
        if cand.exists():
            return cand
    return None


def is_available(name: str) -> bool:
    return resolve(name) is not None
