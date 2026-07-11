"""
Tier 3 — golden-number regression sentinels. These pin the DETERMINISTIC output of
the core pipelines on fixed synthetic inputs, so a code change that silently moves a
number fails loudly. (External paper numbers — DeepLIIF F1 0.81, membrane F1 0.76,
CIMA lung-lesion TRE 3.66 µm — get pinned in the integration tier once their datasets
are attached; they need heavy compute + the data.)
"""
import numpy as np
import pytest
from oasis.spatial import serial_registration as sr
from oasis.webui import calibration as C
from test_registration import _synthetic_pair, _similarity, _apply
from test_calibration import _synth_cells


def test_golden_register_similarity_dice():
    ref, mov = _synthetic_pair(angle=6.0, tx=18, ty=-10)
    reg = sr.register_similarity(ref, mov, pixel_size_um=0.75)
    assert reg["success"]
    assert reg["struct_dice"] == pytest.approx(0.95, abs=0.05)


def test_golden_propose_residual():
    ref, mov = _synthetic_pair()
    r = sr.propose_landmarks(ref, mov, pixel_size_um=0.75, max_points=8)
    Mfit = sr._fit_similarity_ls(np.array(r["mov_points"]), np.array(r["ref_points"]))
    med = float(np.median(np.linalg.norm(
        _apply(Mfit, np.array(r["mov_points"])) - np.array(r["ref_points"]), axis=1) * 0.75))
    assert med < 3.0


def test_golden_calibration_loo():
    cells = _synth_cells(seed=0)
    loo_f1, loo_auc, n = C._loo_f1_auc(cells, 99.0)
    assert loo_auc >= 0.99
    assert 0.90 <= loo_f1 <= 1.0


def test_golden_certified_tre_zero():
    rng = np.random.default_rng(2)
    mov = rng.uniform(50, 1950, (10, 2))
    ref = _apply(_similarity(6.0, 1.0, 30, -18), mov)
    res = sr.landmark_register_and_verify(ref, mov, 0.5, image_wh=(2000, 2000))
    assert res["verdict"] == "CERTIFIED"
    assert res["tre_median_um"] < 0.01          # exact points ⇒ ~0 TRE
