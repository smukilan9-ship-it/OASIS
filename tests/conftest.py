"""
conftest.py — shared fixtures + dataset resolution for the IHC-analyzer test suite.

TIERS
  Tier 1 (unit)         — synthetic, known-answer; run everywhere, no data. (test_cross_k,
                          test_nulls, test_registration, test_calibration)
  Tier 1.5 (keystone)   — the serial-section degradation proof on LOCAL multiplex data
                          (test_degradation) — runs on the in-repo CODEX table.
  Tier 2 (integration)  — real datasets; SKIP automatically if the dataset folder is
                          absent (test_datasets_integration).
  Tier 3 (golden)       — pins the exact numbers quoted in the paper (test_golden_numbers).

DATASET LOCATIONS
  Each dataset has a canonical NAME (below). The suite looks for it at, in order:
    1. env var  IHC_DATA_<NAME_UPPER>   (e.g. IHC_DATA_DEEPLIIF=/mnt/drive/DeepLIIF_data)
    2. validation/datasets/paths.yaml   (key = name)
    3. the default path baked in DATASETS[name]["default"]
  See validation/datasets/README.md for the exact folder names to use on your Drive.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ── Dataset registry ─────────────────────────────────────────────────────────
# name -> {default path, human title, what it validates}
DATASETS = {
    "codex_crc": {
        "default": REPO / "validation" / "CRC_clusters_neighborhoods_markers.csv",
        "title": "Schürch et al. 2020 CRC CODEX single-cell table",
        "drive_name": "CRC_clusters_neighborhoods_markers.csv",
        "validates": "keystone degradation test (same-section multiplex → pseudo-serial)",
    },
    "cima_landmarks": {
        "default": REPO / "validation" / "public_landmarks",
        "title": "ANHIR/CIMA expert registration landmarks",
        "drive_name": "public_landmarks/",
        "validates": "registration TRE vs expert landmarks",
    },
    "deepliif": {
        "default": Path.home() / "Desktop" / "DeepLIIF_data",
        "title": "DeepLIIF (registered IHC+IF) dataset",
        "drive_name": "DeepLIIF_data/",
        "validates": "cell detection + classification vs IF-derived truth",
    },
    "hnscc": {
        "default": Path.home() / "PKG - HNSCC-mIF-mIHC-comparison_v2",
        "title": "HNSCC mIF/mIHC comparison v2",
        "drive_name": "PKG - HNSCC-mIF-mIHC-comparison_v2/",
        "validates": "membranous CD8 vs IF-derived truth",
    },
    "tim3_crc_icm": {
        "default": Path.home() / "Desktop" / "tim3 data",
        "title": "CRC-ICM TIM-3 hand-label harness",
        "drive_name": "tim3 data/",
        "validates": "TIM-3 membrane callability (hand-labelled)",
    },
}


def _paths_yaml():
    p = REPO / "validation" / "datasets" / "paths.yaml"
    if not p.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def dataset_path(name):
    """Resolve a dataset to a real path, or None if it isn't available.

    Delegates to the canonical validation.datasets.resolve layer (consolidated
    validation_data_dir tree, honouring env / setup.yaml), then falls back to the
    legacy conftest defaults so older drop-in locations still work.
    """
    try:
        from validation.datasets import resolve as _R
        p = _R.resolve(name)
        if p is not None:
            return Path(p)
    except Exception:
        pass
    env = os.environ.get(f"IHC_DATA_{name.upper()}")
    if env and Path(env).expanduser().exists():
        return Path(env).expanduser()
    y = _paths_yaml().get(name)
    if y and Path(y).expanduser().exists():
        return Path(y).expanduser()
    d = DATASETS.get(name, {}).get("default")
    if d and Path(d).exists():
        return Path(d)
    return None


def require_dataset(name):
    """Return the path or skip the test with a message naming the exact dataset."""
    p = dataset_path(name)
    if p is None:
        meta = DATASETS.get(name, {})
        pytest.skip(f"dataset '{name}' not found — put '{meta.get('drive_name', name)}' "
                    f"({meta.get('title','?')}) at {meta.get('default','?')} "
                    f"or set IHC_DATA_{name.upper()}")
    return p


# ── Synthetic point-pattern fixtures (deterministic) ─────────────────────────
@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def _poisson(n, w, h, rng):
    return np.column_stack([rng.uniform(0, w, n), rng.uniform(0, h, n)])


@pytest.fixture
def csr_pair(rng):
    """Two INDEPENDENT CSR populations in a 1000×1000 window (no association)."""
    w = h = 1000.0
    a = _poisson(300, w, h, rng)
    b = _poisson(300, w, h, rng)
    return dict(a=a, b=b, w=w, h=h, area=w * h)


@pytest.fixture
def attracted_pair(rng):
    """Population B clustered TIGHTLY around A (strong positive association)."""
    w = h = 1000.0
    a = _poisson(200, w, h, rng)
    idx = rng.integers(0, len(a), 300)
    b = a[idx] + rng.normal(0, 6.0, (300, 2))          # ~6 px around an A cell
    b = np.clip(b, 0, w)
    return dict(a=a, b=b, w=w, h=h, area=w * h)
