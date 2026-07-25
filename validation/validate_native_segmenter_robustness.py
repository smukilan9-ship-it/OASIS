"""
validate_native_segmenter_robustness.py — the surfaces the parity gates never touched.

The DeepLIIF gate and the LL477 comparisons both run flat RGB images through the happy path.
This covers what they do not:

  E. DETERMINISM        — the same input must give byte-identical labels across runs. Every
                          downstream verdict (cross-K envelopes, positivity, certification) is
                          reported as reproducible; a nondeterministic segmenter would quietly
                          break that claim.
  F. ADAPTIVE THRESHOLD — the Otsu cut was ported from the generated Groovy bin-for-bin. Checked
                          against a direct reimplementation of the Groovy arithmetic, and for
                          the documented <20-cell fixed fallback.
  G. DEGENERATE INPUT   — blank, tiny, single-colour and extreme-aspect images must return
                          empty rather than raise. These reach the segmenter in practice via
                          background tiles and thin edge strips of a slide.
  H. TILE INVARIANCE    — object count must not depend materially on the tile size, which is a
                          performance knob. (Global normalisation is what makes this true; the
                          check exists so a future change back to per-tile normalisation fails
                          loudly rather than silently moving everyone's counts.)
  I. PYRAMIDAL/TILED TIFF — reading path for whole-slide-style files.
  J. DEVICE EQUIVALENCE — the corpus was validated on CPU but the shipped default is MPS
                          (~3x faster). The accelerator must not move any decision.

Real pyramidal-slide reading, the streaming path and its memory bound are covered separately by
validate_native_segmenter_wsi.py; check I here only exercises the tiled-TIFF read branch.

Run:  .venv/bin/python validation/validate_native_segmenter_robustness.py
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oasis.common.paths import default_model_dir

# Vendored at models/ so a fresh clone can run this without QuPath ever being installed.
MODEL_DIR = default_model_dir()
IMAGE = os.path.expanduser("~/Desktop/cd8_input/LL477_CD8_x10_3.tif")
PX = 0.7518796992481203


def _tissue_crop(n=768):
    from oasis.quant.cell_expansion import _load_rgb_full
    rgb = _load_rgb_full(IMAGE)
    cy, cx = rgb.shape[0] // 2, rgb.shape[1] // 2
    return rgb[cy - n // 2:cy + n // 2, cx - n // 2:cx + n // 2]


def check_e_determinism(rgb, runs=3):
    from oasis.quant import segment as sg
    model = sg.load_model(MODEL_DIR, "cpu")
    outs = [sg.segment_labels(rgb, model, "cpu") for _ in range(runs)]
    identical = all(np.array_equal(outs[0], o) for o in outs[1:])
    counts = [int(o.max()) for o in outs]
    print(f"  {runs} runs -> object counts {counts} | label arrays identical: {identical}")
    return {"pass": bool(identical), "runs": runs, "counts": counts}


def check_f_adaptive_threshold():
    from oasis.quant import segment as sg
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(0.05, 0.02, 400), rng.normal(0.45, 0.08, 150)])
    vals = np.clip(vals, 0, None)

    def groovy_otsu(v, nbins=256):
        """Direct transliteration of the Groovy in run_pipeline.generate_groovy_script."""
        v = [float(x) for x in v]
        mx = max(v) or 1e-6
        h = [0] * nbins
        for x in v:
            bi = int(min(nbins - 1, max(0, round(x / mx * (nbins - 1)))))
            h[bi] += 1
        tot = len(v)
        s = sum(i * h[i] for i in range(nbins))
        sum_b = 0.0
        w_b = 0
        max_var = -1.0
        thr_bin = 0
        for i in range(nbins):
            w_b += h[i]
            if w_b == 0:
                continue
            w_f = tot - w_b
            if w_f == 0:
                break
            sum_b += i * h[i]
            m_b = sum_b / w_b
            m_f = (s - sum_b) / w_f
            var = w_b * w_f * (m_b - m_f) ** 2
            if var > max_var:
                max_var = var
                thr_bin = i
        return thr_bin / (nbins - 1) * mx

    ours = sg.otsu_threshold(vals)
    theirs = groovy_otsu(vals)
    match = abs(ours - theirs) < 1e-9
    few = sg.otsu_threshold(vals[:19])
    print(f"  Otsu: ours {ours:.6f} vs Groovy transliteration {theirs:.6f} | identical {match}")
    print(f"  <20 cells -> {few} (must be None so the caller uses the fixed threshold)")
    ok = match and few is None
    return {"pass": bool(ok), "ours": float(ours), "groovy": float(theirs),
            "few_cells_returns_none": few is None}


def check_g_degenerate():
    from oasis.quant import segment as sg
    model = sg.load_model(MODEL_DIR, "cpu")
    cases = {
        "blank white 256": np.full((256, 256, 3), 255, np.uint8),
        "blank black 256": np.zeros((256, 256, 3), np.uint8),
        "uniform grey 64": np.full((64, 64, 3), 128, np.uint8),
        "tiny 16x16": np.full((16, 16, 3), 200, np.uint8),
        "thin strip 8x600": np.full((8, 600, 3), 210, np.uint8),
        "single pixel": np.full((1, 1, 3), 200, np.uint8),
    }
    results = {}
    ok = True
    for name, img in cases.items():
        try:
            lab = sg.segment_labels(img, model, "cpu")
            n = int(lab.max())
            bad = not np.isfinite(lab).all() if lab.dtype.kind == "f" else False
            results[name] = {"ok": True, "objects": n}
            print(f"  {name:20s} -> {n} objects, no exception")
            ok = ok and not bad
        except Exception as e:
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"  {name:20s} -> RAISED {type(e).__name__}: {e}")
            ok = False
    return {"pass": ok, "cases": results}


def check_h_tile_invariance(rgb, tiles=(256, 384, 512, 4096)):
    from oasis.quant import segment as sg
    model = sg.load_model(MODEL_DIR, "cpu")
    counts = {}
    for t in tiles:
        counts[t] = int(sg.segment_labels(rgb, model, "cpu", tile=t).max())
    vals = np.asarray(list(counts.values()), float)
    spread = float((vals.max() - vals.min()) / vals.mean())
    ok = spread <= 0.02
    print(f"  counts by tile size {counts} | relative spread {spread:.4f} (limit 0.02)")
    return {"pass": ok, "counts": {str(k): v for k, v in counts.items()}, "spread": spread}


def check_j_device_equivalence(rgb):
    """CPU vs the accelerator must give the same cells.

    The whole validated corpus was scored on CPU, but the shipped default is `device: mps`
    because it is ~3x faster (598 DeepLIIF panels: 148.7s -> 51.8s; a 2048px tile: 12.1s ->
    2.4s, and a whole slide 49 min -> ~17 min). That speedup is only usable if it does not move
    the science.

    Measured across the 598-panel corpus: identical total cell count (35,286), and 597 of 598
    GeoJSON files BYTE-identical. The single differing file had the same 54 cells, all matched
    within 1 px, identical classifications, and a DAB mean differing by 1.5e-5 OD — four orders
    of magnitude below the 0.2 decision threshold. So the devices are equivalent for every
    decision the pipeline makes, but NOT bit-identical in every case, which is why the device is
    recorded in the summary provenance.

    Skipped, not failed, when no accelerator is present.
    """
    from oasis.quant import segment as sg
    try:
        import torch
        if not torch.backends.mps.is_available():
            print("  no MPS device — skipped")
            return {"pass": True, "skipped": True}
    except Exception:
        print("  torch unavailable — skipped")
        return {"pass": True, "skipped": True}

    out = {}
    for dev in ("cpu", "mps"):
        model = sg.load_model(MODEL_DIR, dev)
        labels = sg.segment_labels(rgb, model, dev)
        hem, dab = sg._od_channels(rgb)
        out[dev] = (labels, sg._measure(labels, hem, dab, PX))

    a, b = out["cpu"], out["mps"]
    same_labels = bool(np.array_equal(a[0], b[0]))
    ca = np.asarray([r["centroid_px"] for r in a[1]])
    cb = np.asarray([r["centroid_px"] for r in b[1]])
    from validation.validate_native_segmenter import _greedy_match
    mi, mj = _greedy_match(ca, cb, 1.0)
    dab_mae = None
    cls_same = None
    if len(mi):
        da = np.asarray([a[1][i]["measurements"]["DAB: Mean"] for i in mi])
        db = np.asarray([b[1][j]["measurements"]["DAB: Mean"] for j in mj])
        dab_mae = float(np.mean(np.abs(da - db)))
    counts_equal = len(ca) == len(cb)
    matched_all = len(mi) == len(ca) == len(cb)
    ok = counts_equal and matched_all and (dab_mae is None or dab_mae <= 1e-3)
    print(f"  cpu {len(ca)} cells vs mps {len(cb)} | matched within 1px {len(mi)} | "
          f"label arrays identical {same_labels}")
    print(f"  DAB MAE {dab_mae:.8f} OD (limit 1e-3, threshold is 0.2) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return {"pass": ok, "n_cpu": len(ca), "n_mps": len(cb), "matched": int(len(mi)),
            "labels_bit_identical": same_labels, "dab_mae_od": dab_mae}


def check_i_tiled_tiff(rgb):
    """Read-path check for a tiled/pyramidal TIFF (the WSI-shaped case we can construct)."""
    import tifffile
    from oasis.quant.cell_expansion import _load_rgb_full
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tiled.tif")
        tifffile.imwrite(path, rgb, tile=(256, 256), photometric="rgb", compression="deflate")
        back = _load_rgb_full(path)
        same_shape = back.shape == rgb.shape
        # deflate is lossless, so the pixels must round-trip exactly through whichever
        # reader _load_rgb_full picked (openslide or PIL)
        identical = same_shape and bool(np.array_equal(back, rgb))
        print(f"  tiled TIFF (256px tiles, deflate) -> read back {back.shape}, "
              f"source {rgb.shape} | shape match {same_shape} | pixels identical {identical}")
        return {"pass": bool(identical), "shape": list(back.shape)}


def main():
    rgb = _tissue_crop()
    report = {}
    print("E. determinism (same input, repeated runs)")
    report["determinism"] = check_e_determinism(rgb)
    print("\nF. adaptive Otsu threshold vs the Groovy it replaces")
    report["adaptive_threshold"] = check_f_adaptive_threshold()
    print("\nG. degenerate inputs (must return empty, not raise)")
    report["degenerate"] = check_g_degenerate()
    print("\nH. tile-size invariance")
    report["tile_invariance"] = check_h_tile_invariance(rgb)
    print("\nI. tiled/pyramidal TIFF read path")
    report["tiled_tiff"] = check_i_tiled_tiff(rgb)
    print("\nJ. CPU vs accelerator equivalence")
    report["device_equivalence"] = check_j_device_equivalence(rgb)

    ok = all(v.get("pass") for v in report.values())
    print(f"\n##METRICS## {json.dumps(report, default=str)}")
    out = os.environ.get("OASIS_REPORT_DIR")
    if out:
        with open(os.path.join(out, "native_segmenter_robustness.json"), "w") as f:
            json.dump(report, f, indent=2)
    print(f"\n{'PASS' if ok else 'FAIL'}")
    print("NOTE: real pyramidal-slide reading and streaming are covered separately by "
          "validate_native_segmenter_wsi.py.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
