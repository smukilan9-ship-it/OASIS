"""
warpfield.py — reapply the exported VALIS transforms to arbitrary point sets.

export_warps.py evaluated BOTH VALIS transforms on a regular grid over the moving
image. Because the grid is regular, the map is recovered exactly by bilinear
interpolation between grid nodes (checked against VALIS-warped held-out points in
check_warps.py). Nothing here re-registers anything.

Also provides the ARM-3 control: a magnitude-matched RANDOM smooth warp. It is built
by smoothing white noise on the SAME grid and rescaling so its displacement RMS equals
the real non-rigid residual's, then composed onto the rigid map. It therefore differs
from the real warp in ONE respect only — it does not know where the tissue is.
"""
import json
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
WARPS = os.path.join(HERE, "warps")


class PairWarp:
    """Rigid and non-rigid moving->fixed maps for one registered pair."""

    def __init__(self, npz_path):
        z = np.load(npz_path)
        meta = json.load(open(npz_path + ".json", encoding="utf-8"))
        self.meta = meta
        self.pair_id = meta["pair_id"]
        self.set = meta["set"]
        self.px_um = float(meta["px_um"])
        self.moving_wh = tuple(meta["moving_wh"])
        self.fixed_wh = tuple(meta["fixed_wh"])
        n = int(meta["grid_n"])

        src = z["src"].reshape(n, n, 2)
        self._gy = src[:, 0, 1]          # rows vary in y
        self._gx = src[0, :, 0]          # cols vary in x
        self._rigid = z["rigid"].reshape(n, n, 2)
        self._nonrigid = z["nonrigid"].reshape(n, n, 2)
        self._n = n
        self.hold_src = z["hold_src"]
        self.hold_rigid = z["hold_rigid"]
        self.hold_nonrigid = z["hold_nonrigid"]

        self._i_rigid = self._interp(self._rigid)
        self._i_nonrigid = self._interp(self._nonrigid)

        # real non-rigid residual = what micro-registration added on top of rigid
        d = self._nonrigid - self._rigid
        finite = np.isfinite(d).all(axis=-1)
        self.residual_rms_px = float(np.sqrt(np.nanmean(
            np.sum(np.where(finite[..., None], d, np.nan) ** 2, axis=-1))))

        # VALIS's rigid stage is an affine map, so it inverts exactly. Fit it and keep
        # the fit residual as proof (a non-affine "rigid" map would show up here).
        S = z["src"]
        Rg = z["rigid"]
        ok = np.isfinite(S).all(1) & np.isfinite(Rg).all(1)
        X = np.column_stack([S[ok], np.ones(ok.sum())])
        M, *_ = np.linalg.lstsq(X, Rg[ok], rcond=None)      # 3x2
        self._M = M
        pred = X @ M
        self.rigid_affine_resid_px = float(np.sqrt(np.mean(
            np.sum((pred - Rg[ok]) ** 2, axis=1))))
        A2 = M[:2, :].T                                     # 2x2 linear part
        self._Ainv = np.linalg.inv(A2)
        self._t = M[2, :]

    def inverse_rigid(self, xy):
        """Fixed-frame -> moving-frame through the (exactly invertible) rigid stage."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        return (xy - self._t) @ self._Ainv.T

    def _interp(self, field):
        return RegularGridInterpolator((self._gy, self._gx), field,
                                       method="linear", bounds_error=False,
                                       fill_value=None)   # extrapolate at the margin

    @staticmethod
    def _apply(interp, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.asarray(interp(np.column_stack([xy[:, 1], xy[:, 0]])), float)

    def rigid(self, xy):
        return self._apply(self._i_rigid, xy)

    def nonrigid(self, xy):
        return self._apply(self._i_nonrigid, xy)

    def random_smooth(self, xy, seed, smooth_nodes=6.0, target_rms_px=None):
        """ARM-3 control: rigid map + a smooth random displacement, tissue-blind.

        `target_rms_px` should be the RMS of the REAL micro-registration displacement
        AT THE SAME POINTS (see arms.py) — matching over the whole grid instead would
        be dominated by background, where the warp is unconstrained and wild.
        """
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((self._n, self._n, 2))
        for c in range(2):
            noise[..., c] = gaussian_filter(noise[..., c], smooth_nodes, mode="nearest")
        target = self.residual_rms_px if target_rms_px is None else float(target_rms_px)
        base = self._apply(self._interp(self._rigid), xy)
        disp = self._apply(self._interp(noise), xy)
        cur = np.sqrt(np.nanmean(np.sum(disp ** 2, axis=1)))
        if not np.isfinite(cur) or cur <= 0:
            return base
        return base + disp * (target / cur)


def load_all(warps_dir=WARPS, require_px_um=True):
    out = []
    idx_path = os.path.join(warps_dir, "index.json")
    if not os.path.exists(idx_path):
        return out
    for rec in json.load(open(idx_path, encoding="utf-8")):
        if not rec.get("ok"):
            continue
        if require_px_um and not rec.get("px_um"):
            continue
        p = os.path.join(warps_dir, rec["pair_id"] + ".npz")
        if os.path.exists(p) and os.path.exists(p + ".json"):
            try:
                out.append(PairWarp(p))
            except Exception:
                pass
    return out
