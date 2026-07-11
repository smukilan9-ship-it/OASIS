"""
validate_entry_point_parity.py — prove the CLI and UI entry points compute the
SAME thing for a spatial-association run.

Before the correctness fixes, the two paths diverged:
  • pixel size   — the CLI read cfg["default_pixel_size"] (often 0.5) for the
                   Ripley's K metric, while the UI injected the resolved value;
  • cytoplasm    — the membrane-marker default (A off / B on) lived only in
                   webui/api.py, so a CLI run measured TIM-3 in the nucleus.

This test drives the SHARED resolution logic that both entry points now use and
asserts they agree:

  1. cytoplasm settings  — run_pipeline.cytoplasm_overrides_for_pair gives the
     same per-image flags for a bare CLI cfg and a UI-style cfg.
  2. pixel size          — the value used for the K-function equals the value the
     image's segmentation resolved (same get_pixel_size chain), and a CLI cfg and
     a UI cfg that describe the same calibration resolve to the same micron value;
     a genuine fall-through is reported as source "default_fallback".
  3. registration method — registration.compute_registration is entry-point
     independent and deterministic (same images -> same method + transform).

Exits non-zero if any expectation fails.
"""

import os
import sys
import tempfile
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_pipeline import cytoplasm_overrides_for_pair        # noqa: E402
from oasis.common.pixel_size_util import get_pixel_size, get_pixel_size_with_source  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Cytoplasm settings parity
# ──────────────────────────────────────────────────────────────────────────────

def test_cytoplasm_parity():
    print("\n" + "=" * 72)
    print("1. CYTOPLASM MEASUREMENT — CLI cfg vs UI cfg resolve identically")
    print("=" * 72)
    path_a = "/x/sample_CD8_x10.png"     # reference / image A
    path_b = "/x/sample_TIM3_x10.png"    # moving / image B
    a, b = os.path.basename(path_a), os.path.basename(path_b)

    # CLI: a bare config (no cytoplasm_overrides, no global flag)
    cli_cfg = {}
    cli = cytoplasm_overrides_for_pair(cli_cfg, path_a, path_b)

    # UI: webui/api.py passes explicit per-image flags built from the toggles
    # (safe defaults: both off; membrane remeasurement is explicit opt-in).
    ui_cfg = {"cytoplasm_overrides": {a: False, b: False},
              "use_cytoplasm_measurement": False}
    ui = cytoplasm_overrides_for_pair(ui_cfg, path_a, path_b)

    check("CLI and UI cytoplasm maps are identical", cli == ui, f"{cli} == {ui}")
    check("reference / image A defaults OFF (nuclear)", cli.get(a) is False)
    check("moving / image B defaults OFF (original QuPath)", cli.get(b) is False)

    # explicit per-image override must win over the role default
    over_cfg = {"cytoplasm_overrides": {b: True}}
    over = cytoplasm_overrides_for_pair(over_cfg, path_a, path_b)
    check("explicit membrane opt-in wins over role default", over.get(b) is True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Pixel-size parity
# ──────────────────────────────────────────────────────────────────────────────

def test_pixel_size_parity():
    print("\n" + "=" * 72)
    print("2. PIXEL SIZE — K-function µm/px equals segmentation µm/px, CLI == UI")
    print("=" * 72)
    # Same image A, 10x in the filename (-> 1.0 µm/px by the filename table).
    path_a = "/x/sample_CD8_x10.png"
    base_a = os.path.basename(path_a)

    # CLI cfg: no per-image override, not from UI -> filename parsing -> 1.0
    cli_cfg = {"default_pixel_size": 0.5}
    cli_val, cli_src = get_pixel_size_with_source(path_a, cli_cfg)

    # UI cfg: api.py injects the resolved session value as a per-image override
    # (here the same 1.0) and marks _pixel_size_from_ui.
    ui_cfg = {"default_pixel_size": 1.0, "_pixel_size_from_ui": True,
              "pixel_overrides": {base_a: 1.0}}
    ui_val, ui_src = get_pixel_size_with_source(path_a, ui_cfg)

    check("CLI and UI resolve the SAME micron value", cli_val == ui_val,
          f"CLI {cli_val} ({cli_src})  ==  UI {ui_val} ({ui_src})")
    check("CLI source is the filename magnification", cli_src == "filename")
    check("UI source is the per-image override", ui_src == "per_image_override")

    # The K-function value (get_pixel_size) must equal the segmentation value
    # (same function) for BOTH cfgs — this is the core of FIX 1.
    check("K-function px == segmentation px (CLI)",
          get_pixel_size(path_a, cli_cfg) == cli_val)
    check("K-function px == segmentation px (UI)",
          get_pixel_size(path_a, ui_cfg) == ui_val)

    # A genuine fall-through must be flagged so the spatial pipeline can warn.
    fall_cfg = {"default_pixel_size": 0.5}     # no token, no metadata, not from UI
    fall_val, fall_src = get_pixel_size_with_source("/x/slide001.png", fall_cfg)
    check("silent default fall-through is reported as 'default_fallback'",
          fall_src == "default_fallback" and fall_val == 0.5,
          f"{fall_val} ({fall_src})")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Registration-method parity (entry-point independence + determinism)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_pattern(rng, size=480, n_shapes=150, shift=(0, 0)):
    import cv2
    dx, dy = shift
    img = np.full((size, size, 3), 245, np.uint8)
    for _ in range(n_shapes):
        x = int(rng.integers(20, size - 20) + dx)
        y = int(rng.integers(20, size - 20) + dy)
        col = tuple(int(c) for c in rng.integers(30, 150, 3))
        k = rng.integers(0, 3)
        if k == 0:
            cv2.circle(img, (x, y), int(rng.integers(4, 10)), col, -1)
        elif k == 1:
            w, h = int(rng.integers(6, 16)), int(rng.integers(6, 16))
            cv2.rectangle(img, (x, y), (x + w, y + h), col, -1)
        else:
            cv2.line(img, (x, y), (x + int(rng.integers(-14, 14)),
                                   y + int(rng.integers(-14, 14))), col, 2)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def test_registration_parity():
    print("\n" + "=" * 72)
    print("3. REGISTRATION METHOD — entry-point independent + deterministic")
    print("=" * 72)
    from oasis.common.registration import compute_registration
    from PIL import Image

    tmp = tempfile.mkdtemp(prefix="parity_")
    try:
        base       = _draw_pattern(np.random.default_rng(7), shift=(0, 0))
        base_shift = _draw_pattern(np.random.default_rng(7), shift=(6, 4))
        p_ref = os.path.join(tmp, "ref.png")
        p_mov = os.path.join(tmp, "mov.png")
        Image.fromarray(base).save(p_ref)
        Image.fromarray(base_shift).save(p_mov)

        # Both entry points call the IDENTICAL function with the same images.
        reg_cli = compute_registration(p_ref, p_mov)
        reg_ui  = compute_registration(p_ref, p_mov)

        check("registration method identical across calls",
              reg_cli["method"] == reg_ui["method"],
              f"{reg_cli['method']} == {reg_ui['method']}")
        check("transform identical across calls (deterministic)",
              np.allclose(np.asarray(reg_cli["matrix"]),
                          np.asarray(reg_ui["matrix"]), atol=1e-4))
        check("a registrable pair did not fall back to identity",
              reg_cli["method"] != "identity", f"method={reg_cli['method']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_cytoplasm_parity()
    test_pixel_size_parity()
    test_registration_parity()

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if failures:
        print(f"  FAILED: {len(failures)} check(s): {failures}")
        print("  CLI and UI entry points do NOT agree — see failures above.")
        sys.exit(1)
    print("  ALL CHECKS PASS — CLI and UI entry points resolve pixel size,")
    print("  cytoplasm settings, and registration method identically.")
    sys.exit(0)
