"""
Tier 2 — real-dataset integration checks. Each test SKIPS (naming the exact dataset
and where to put it) when the data is absent, so `pytest` is green on a bare checkout
and gains coverage as you attach datasets from your Drive.

These are light presence/schema checks. The HEAVY end-to-end validations (TRE vs
expert landmarks, membrane F1 vs IF truth, CODEX biological controls) live in the
standalone validation/validate_*.py scripts referenced in each test.
"""
import csv
from pathlib import Path

import pytest
from conftest import require_dataset, DATASETS


def test_codex_table_schema():
    """codex_crc — used by the keystone degradation test. Needs coords + markers."""
    path = require_dataset("codex_crc")
    hdr = next(csv.reader(open(path)))
    for col in ("X:X", "Y:Y", "spots",
                "CD8 - cytotoxic T cells:Cyc_3_ch_2", "PD-1 - checkpoint:Cyc_12_ch_4"):
        assert col in hdr, f"CODEX table missing column {col!r}"
    # heavy biological-control validation: validation/validate_real_data_production.py


def test_cima_landmarks_present():
    """cima_landmarks — expert landmarks for registration TRE.
    Heavy validation: validation/validate_anhir_landmarks.py"""
    path = require_dataset("cima_landmarks")
    assert (Path(path) / "annotations").exists() or any(Path(path).rglob("*.csv")), \
        "expected expert-landmark annotations under public_landmarks/"


def test_deepliif_present():
    """deepliif — IF-derived truth for cell detection/classification.
    Heavy validation: validation/deepliif_pipeline_validation.py"""
    path = require_dataset("deepliif")
    assert any(Path(path).iterdir()), "DeepLIIF_data is empty"


def test_hnscc_present():
    """hnscc — IF-derived truth for membranous CD8.
    Heavy validation: validation/validate_membrane_cd8_hnscc.py"""
    path = require_dataset("hnscc")
    assert (Path(path) / "mIF_Data").exists() or any(Path(path).iterdir()), \
        "HNSCC folder present but missing mIF_Data"


def test_dataset_registry_is_documented():
    """Every registered dataset must have a drive_name + title for the README guide."""
    for name, meta in DATASETS.items():
        assert meta.get("drive_name") and meta.get("title"), f"{name} under-documented"
