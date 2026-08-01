#!/usr/bin/env python
"""
gpu_check.py — did this machine actually give us the GPU we are paying for?

THE FAILURE THIS EXISTS TO CATCH IS SILENT. A run that falls back to the CPU produces exactly
the same numbers as one that did not; the only symptom is the bill and the wall-clock. That
already happened once locally: `device: mps` on a CUDA box resolved to "cpu", so LoFTR used
the card (it detects its own) while InstanSeg did not, and nothing said so.

So before spending GPU-hours, print what each half of the pipeline will really use, and
measure one LoFTR pass so the speedup is a number rather than an assumption.

    python packaging/cloud/gpu_check.py            # report + timed LoFTR pass
    python packaging/cloud/gpu_check.py --no-bench # report only
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _line(k, v):
    print(f"  {k:<34} {v}")


def report():
    print("=" * 72)
    print("OASIS headless runner — hardware report")
    print("=" * 72)

    try:
        import torch
    except Exception as e:
        print(f"  torch unavailable: {e}")
        return False
    _line("torch", torch.__version__)
    cuda = bool(torch.cuda.is_available())
    _line("cuda available", cuda)
    if cuda:
        _line("cuda device", torch.cuda.get_device_name(0))
        _line("cuda capability", ".".join(map(str, torch.cuda.get_device_capability(0))))
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        _line("cuda memory", f"{total:.1f} GB")

    # The two consumers resolve their device independently and by different policies, which is
    # exactly how they came apart before. Print both, from the real code paths.
    from oasis.common.device import describe_device, resolve_device
    _line("InstanSeg (config 'auto')", describe_device("auto"))
    _line("InstanSeg (a stored 'mps')", describe_device("mps"))
    from oasis.spatial.loftr_matcher import _device as loftr_device
    _line("LoFTR", str(loftr_device()))

    if resolve_device("auto") == "cpu" or str(loftr_device()) == "cpu":
        print("\n  WARNING: at least one half of the pipeline is on the CPU. On a rented GPU")
        print("  that is money spent for nothing — check the NVIDIA driver and that the")
        print("  container was started with `--gpus all`.")
        return False
    return True


def bench():
    """One real LoFTR pass on synthetic tissue-like texture.

    Synthetic rather than a real slide so the check needs no data mounted, and so the number
    is comparable between machines. It measures the matcher, which is the part that dominates
    certification — not the whole pipeline.
    """
    import numpy as np
    from oasis.spatial.loftr_matcher import loftr_correspondences, clear_loftr_caches

    rng = np.random.default_rng(0)
    base = rng.integers(90, 200, size=(900, 900, 3), dtype=np.uint8)
    import cv2
    base = cv2.GaussianBlur(base, (0, 0), 2.0)          # tissue-scale structure, not white noise
    M = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, -4.0]])
    mov = cv2.warpAffine(base, M, (900, 900), borderValue=(255, 255, 255))

    clear_loftr_caches()
    t0 = time.time()
    c = loftr_correspondences(base, mov, pixel_size_um=1.0)
    dt = time.time() - t0
    print("\n" + "-" * 72)
    _line("LoFTR pass (900px, cold)", f"{dt:.1f} s   {c.get('n')} correspondences")
    print("  A T4 should be a few seconds here; tens of seconds means the CPU is doing it.")
    print("  Certification runs many of these, so this number multiplies.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bench", action="store_true")
    a = ap.parse_args()
    ok = report()
    print("\n  Reminder: this container reads only what you mount. Nothing is uploaded by it.")
    print("  Histology is patient data and WSI metadata often carries identifiers — public")
    print("  sets are free to run here; a real cohort is your institution's call.")
    if not a.no_bench:
        try:
            bench()
        except Exception as e:
            print(f"\n  benchmark skipped: {type(e).__name__}: {e}")
    sys.exit(0 if ok else 1)
