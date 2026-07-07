"""
Tier 1.5 — THE KEYSTONE: serial-section degradation proof.

Multiplex data measures two markers on the SAME physical section, so the true
cross-type verdict is known. We then DEGRADE it to imitate the serial-section case
(split the markers onto two 'sections' and inject a registration error the size of
our measured TRE) and require the pipeline to recover the same verdict. This is the
only way to validate the serial-section approximation, which by construction can
never be checked directly on the real CD8/TIM-3 data.

Runs on the in-repo CODEX table (Schürch et al. 2020 CRC); skips if absent. IMC
(Zenodo 3518284) is an optional independent drop-in — see the dataset README.
"""
import csv
import numpy as np
import pytest
import spatial_stats as ss
from conftest import require_dataset

R = np.arange(0, 101, 4.0)
CD8_COL = "CD8 - cytotoxic T cells:Cyc_3_ch_2"
PD1_COL = "PD-1 - checkpoint:Cyc_12_ch_4"


def _largest_spot(path):
    rows = list(csv.DictReader(open(path)))
    import collections
    spot = collections.Counter(r["spots"] for r in rows).most_common(1)[0][0]
    return [r for r in rows if r["spots"] == spot]


@pytest.fixture(scope="module")
def codex_field():
    path = require_dataset("codex_crc")
    sub = _largest_spot(path)
    X = np.array([float(r["X:X"]) for r in sub])
    Y = np.array([float(r["Y:Y"]) for r in sub])
    cd8 = np.array([float(r[CD8_COL]) for r in sub])
    pd1 = np.array([float(r[PD1_COL]) for r in sub])
    A = np.column_stack([X[cd8 > np.quantile(cd8, 0.80)], Y[cd8 > np.quantile(cd8, 0.80)]])
    B = np.column_stack([X[pd1 > np.quantile(pd1, 0.80)], Y[pd1 > np.quantile(pd1, 0.80)]])
    if len(A) < 40 or len(B) < 40:
        pytest.skip("chosen CODEX field has too few gated cells")
    area = (X.max() - X.min()) * (Y.max() - Y.min())
    center = np.array([X.mean(), Y.mean()])
    return dict(A=A, B=B, area=area, center=center)


def _verdict(a, b, area):
    o = ss.cross_k_all_nulls(a, b, R, area, 1.0, n_perm=299, seed=0)
    return o["robustness"]["verdict"]


def _degrade(pts, center, angle_deg, shift_px):
    th = np.radians(angle_deg)
    Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (Rot @ (pts - center).T).T + center + np.array([shift_px, -shift_px])


def test_real_verdict_survives_registration_error(codex_field):
    """The real CD8/PD-1 verdict must not FLIP under a realistic registration error."""
    A, B, area, c = (codex_field[k] for k in ("A", "B", "area", "center"))
    truth = _verdict(A, B, area)
    for ang, sh in [(1.0, 3.0), (2.0, 5.0), (3.0, 8.0)]:
        assert _verdict(A, _degrade(B, c, ang, sh), area) == truth, \
            f"verdict flipped under {ang}° / {sh}px registration error (truth={truth})"


def test_engineered_engagement_survives_degradation(codex_field):
    """Plant a genuinely ENGAGED partner on the real CD8 coordinates; the 'robust'
    verdict must survive the same degradation."""
    A, area, c = codex_field["A"], codex_field["area"], codex_field["center"]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(A), max(len(A), 120))
    B = A[idx] + rng.normal(0, 6.0, (len(idx), 2))         # tight around CD8 cells
    assert _verdict(A, B, area) == "robust"
    assert _verdict(A, _degrade(B, c, 2.0, 5.0), area) == "robust"


def test_engineered_independence_survives_degradation(codex_field):
    """Plant an INDEPENDENT partner (uniform over the field); must not become robust
    just because of the injected registration error."""
    A, area, c = codex_field["A"], codex_field["area"], codex_field["center"]
    rng = np.random.default_rng(1)
    xs = A[:, 0]; ys = A[:, 1]
    B = np.column_stack([rng.uniform(xs.min(), xs.max(), len(A)),
                         rng.uniform(ys.min(), ys.max(), len(A))])
    assert _verdict(A, B, area) != "robust"
    assert _verdict(A, _degrade(B, c, 2.0, 5.0), area) != "robust"
