"""
validate_hyreco.py — validate the landmark-driven registration CERTIFICATION against
the public HyReCo dataset's expert ground-truth landmarks.

WHY HyReCo (the selected public dataset; see learn.md):
  • Contains CD8 (+ HE, Ki67, CD45) consecutive serial sections AND a re-stained
    (HE→PHH3) pair — the exact regime of our problem.
  • Ships 690 landmarks (11–19 per section) placed AND verified by two experienced
    researchers → an INDEPENDENT, annotator-checked ground truth (ANHIR-grade).
  • CC-BY-SA 4.0, public on IEEE DataPort.
  It has NO TIM-3 — no public CD8/TIM-3 paired set exists — so HyReCo validates the
  registration/certification METHOD on real serial/restained sections, not the
  CD8↔TIM-3 biology.

KEY POINT: our certification (`landmark_register_and_verify`) works on the landmark
COORDINATES alone — it does not need the multi-GB whole-slide images. So you only need
HyReCo's small landmark CSVs to run this. Expected, decisive outcomes:
  • re-stained pair (HE↔PHH3, ~0 z-gap): CERTIFIED, TRE ~sub-µm — proves we CERTIFY
    genuinely-corresponding pairs (the scope's "don't fail good pairs" requirement).
  • consecutive pairs (e.g. HE↔CD8): higher held-out TRE reflecting the z-gap floor —
    DEFORMED / NOT_CERTIFIABLE depending on tolerance, honestly.

USAGE
  Self-test (no data needed — proves the harness logic):
      python validation/validate_hyreco.py --selftest
  Real run (after downloading HyReCo landmark CSVs):
      python validation/validate_hyreco.py --csv-ref HE.csv --csv-mov CD8.csv \
             --pixel-size 0.22 --label "case29 HE↔CD8"
  CSVs use row-index correspondence (row i in ref ↔ row i in mov); X/Y columns are
  auto-detected (else the last two numeric columns). Landmarks must be in the pixel
  grid of the resolution whose µm/px you pass.
"""
import os, sys, csv, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oasis.spatial.serial_registration import landmark_register_and_verify  # noqa


def load_landmarks(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        return np.zeros((0, 2))
    header = rows[0]
    def col(name):
        for i, h in enumerate(header):
            if h.strip().lower() == name:
                return i
        return None
    xi, yi = col("x"), col("y")
    data = rows[1:]
    pts = []
    for r in data:
        if xi is not None and yi is not None:
            try:
                pts.append([float(r[xi]), float(r[yi])]); continue
            except (ValueError, IndexError):
                pass
        nums = [float(v) for v in r if _isnum(v)]
        if len(nums) >= 2:
            pts.append(nums[-2:])
    return np.array(pts, float) if pts else np.zeros((0, 2))


def _isnum(v):
    try:
        float(v); return True
    except ValueError:
        return False


def certify_from_landmarks(ref_pts, mov_pts, pixel_size_um, label, image_wh=None):
    """Half the (corresponding) landmarks fit the transform; the other half are a
    truly held-out, annotator-checked validation set — the independent-validation
    path, stronger than LOO."""
    n = min(len(ref_pts), len(mov_pts))
    ref, mov = ref_pts[:n], mov_pts[:n]
    idx = np.arange(n)
    fit = idx % 2 == 0          # even rows fit
    val = ~fit                  # odd rows validate (held-out)
    res = landmark_register_and_verify(
        ref[fit], mov[fit], pixel_size_um,
        val_ref_pts=ref[val], val_mov_pts=mov[val], image_wh=image_wh)
    print(f"\n{label}: {n} landmarks ({fit.sum()} fit / {val.sum()} held-out)")
    print(f"  validation : {res['validation']}")
    print(f"  est_scale  : {res['est_scale']}   fit-residual: {res['fit_residual_um']} µm")
    print(f"  held-out TRE: med {res['tre_median_um']}  p90 {res['tre_p90_um']}  "
          f"max {res['tre_max_um']} µm")
    print(f"  VERDICT    : {res['verdict']} — {res['reason']}")
    return res


def selftest():
    """Prove the harness + certification logic on synthetic 'ground truth' (no data)."""
    import math
    rng = np.random.default_rng(0)
    ref = rng.uniform([100, 100], [1800, 1800], size=(16, 2))
    th = math.radians(3.0)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    base = (R @ ref.T).T * 1.0 + np.array([40.0, -25.0])     # known similarity
    px = 0.22

    print("="*70 + "\nSELFTEST (synthetic ground truth, px=0.22 µm)\n" + "="*70)
    # (1) restained-like: ~0.3 px noise → sub-µm TRE → CERTIFIED
    mov1 = base + rng.normal(0, 0.3, base.shape)
    r1 = certify_from_landmarks(ref, mov1, px, "restained-like (tiny noise)", (1920, 1920))
    # (2) consecutive-like: ~12 px (~2.6 µm) correspondence scatter → borderline
    mov2 = base + rng.normal(0, 12.0, base.shape)
    r2 = certify_from_landmarks(ref, mov2, px, "consecutive-like (z-gap scatter)", (1920, 1920))
    # (3) mismatched: large incoherent scatter → NOT_CERTIFIABLE
    mov3 = base + rng.normal(0, 120.0, base.shape)
    r3 = certify_from_landmarks(ref, mov3, px, "mismatched (incoherent)", (1920, 1920))

    ok = (r1["verdict"] == "CERTIFIED"
          and r3["verdict"] in ("NOT_CERTIFIABLE", "DEFORMED"))
    print("\n" + "="*70)
    print(f"SELFTEST {'PASS' if ok else 'FAIL'} — harness certifies a clean pair and "
          f"rejects an incoherent one (real HyReCo run pending CSV download).")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv-ref"); ap.add_argument("--csv-mov")
    ap.add_argument("--pixel-size", type=float, default=0.22)
    ap.add_argument("--label", default="HyReCo pair")
    a = ap.parse_args()
    if a.selftest or not (a.csv_ref and a.csv_mov):
        if not a.selftest:
            print("No CSVs given — running --selftest. Provide --csv-ref/--csv-mov "
                  "with HyReCo landmark files for a real run.\n")
        return selftest()
    ref = load_landmarks(a.csv_ref); mov = load_landmarks(a.csv_mov)
    print(f"Loaded {len(ref)} ref / {len(mov)} mov landmarks from HyReCo CSVs")
    certify_from_landmarks(ref, mov, a.pixel_size, a.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
