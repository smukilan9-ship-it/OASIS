"""
report.py — turn the arm results into the numbers that decide whether there is a paper.

Arm 2 / Arm 3 are FALSE-POSITIVE rates: A and B were independent by construction, so
every rejection is an error. The reference is alpha.
Arm 1 is POWER: a real association was imposed, so rejection is correct.

The comparison that matters is PAIRED — same A, same B, same window, same null seed,
only the transform differs — so the headline is McNemar on discordant replicates, not
two independent proportions.

Run:  .venv/bin/python -m validation.nonrigid_bench.report
"""
import json
import math
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

OUT = os.path.join(_HERE, "results")
ALPHA = 0.05
NULLS = ["reweighted", "homogeneous", "dense_morphology"]
ARMS = ["rigid", "nonrigid", "random"]


def rejected(cell, null, crit="contact"):
    """Did this run CLAIM attraction?

    crit='contact' — the 10-20 µm band, i.e. the cell-scale colocalization claim.
                     This is the claim the paper is about.
    crit='global'  — the whole-curve DCLF test in the attraction direction.
    """
    if not isinstance(cell, dict) or cell.get("skipped"):
        return None
    d = cell.get(null)
    if not isinstance(d, dict):
        return None
    if crit == "contact":
        p, sig, dr = d.get("contact_p"), d.get("contact_sig"), d.get("contact_dir")
    elif crit == "coinfil":
        p, sig, dr = d.get("coinfil_p"), d.get("coinfil_sig"), d.get("coinfil_dir")
    else:
        p, sig, dr = d.get("p_dclf"), d.get("significant"), d.get("direction")
    if p is None:
        return None
    return bool(sig) and dr == "attraction"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b, c):
    """Exact two-sided binomial on the discordant pairs (b = only-arm1, c = only-arm2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tot = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tot / (2 ** n))


def load(arm):
    p = os.path.join(OUT, f"arm{arm}.json")
    return json.load(open(p)) if os.path.exists(p) else []


def table(rows, arm_no):
    present = [a for a in ARMS if any(a in r for r in rows)]
    print(f"\n{'='*78}\nARM {arm_no}   ({len(rows)} replicates, "
          f"{len(set(r['pair_id'] for r in rows))} pairs)\n{'='*78}")
    if rows:
        mm = [r.get("microreg_rms_um") for r in rows if r.get("microreg_rms_um")]
        if mm:
            mm.sort()
            print(f"micro-registration displacement at the analysed points: "
                  f"median {mm[len(mm)//2]:.1f} µm  "
                  f"(p10 {mm[len(mm)//10]:.1f}, p90 {mm[(9*len(mm))//10]:.1f})")
    label = "power" if arm_no == 1 else "false-positive rate"
    for crit in ("contact", "coinfil", "global"):
        print(f"\n  --- claim = {crit} "
              f"({'10-20 µm, the cell-scale claim' if crit=='contact' else '20-50 µm' if crit=='coinfil' else 'whole curve'}) ---")
        for null in NULLS:
            line = f"  null={null:18s} [{label}, alpha={ALPHA}]  "
            parts = []
            for a in present:
                k = sum(1 for r in rows if rejected(r.get(a), null, crit) is True)
                n = sum(1 for r in rows if rejected(r.get(a), null, crit) is not None)
                if n:
                    lo, hi = wilson(k, n)
                    parts.append(f"{a} {k}/{n}={k/n:.3f} [{lo:.2f},{hi:.2f}]")
                else:
                    parts.append(f"{a} n/a")
            print(line + "   ".join(parts))
            for a in present:
                if a == "rigid":
                    continue
                b = c = 0
                for r in rows:
                    x = rejected(r.get("rigid"), null, crit)
                    y = rejected(r.get(a), null, crit)
                    if x is None or y is None:
                        continue
                    if x and not y:
                        b += 1
                    elif y and not x:
                        c += 1
                if b or c:
                    print(f"{'':22s}    paired {a} vs rigid: +{c} gained / -{b} lost, "
                          f"McNemar p = {mcnemar(b, c):.4g}")


def outcome(cell, null, crit="contact"):
    """'attraction' | 'segregation' | 'none' | None(unusable)."""
    if not isinstance(cell, dict) or cell.get("skipped"):
        return None
    d = cell.get(null)
    if not isinstance(d, dict):
        return None
    if crit == "contact":
        p, sig, dr = d.get("contact_p"), d.get("contact_sig"), d.get("contact_dir")
    else:
        p, sig, dr = d.get("p_dclf"), d.get("significant"), d.get("direction")
    if p is None:
        return None
    return dr if sig else "none"


def signflips(rows, null="dense_morphology", crit="contact"):
    """ARM 1 is about a REAL attraction. Three ways to lose it, and they are not equal:
    losing significance is a power cost; being told SEGREGATION is a wrong answer."""
    print(f"\n  outcome of a TRUE attraction (null={null}, claim={crit}):")
    print(f"    {'arm':10s} {'attraction':>11s} {'none':>7s} {'SEGREGATION':>12s} "
          f"{'window destroyed':>17s}")
    for a in ARMS:
        if not any(a in r for r in rows):
            continue
        cnt = defaultdict(int)
        dead = 0
        for r in rows:
            if a not in r:
                continue
            if isinstance(r[a], dict) and r[a].get("skipped"):
                dead += 1
                continue
            o = outcome(r[a], null, crit)
            if o is None:
                continue
            cnt[o] += 1
        print(f"    {a:10s} {cnt['attraction']:11d} {cnt['none']:7d} "
              f"{cnt['segregation']:12d} {dead:17d}")


def dose_response(rows, null="dense_morphology", crit="contact"):
    """Does the damage scale with how far micro-registration actually moved things?
    If the effect is causal it must; if it is an artefact of something else it need not."""
    xs = [(r.get("microreg_rms_um") or 0.0, r) for r in rows if "nonrigid" in r]
    xs = [(m, r) for m, r in xs if m > 0]
    if len(xs) < 12:
        return
    xs.sort(key=lambda t: t[0])
    n = len(xs)
    print(f"\n  dose-response on micro-registration displacement "
          f"(null={null}, claim={crit}):")
    print(f"    {'displacement':>22s} {'n':>4s} {'rigid agrees':>13s} {'nonrigid agrees':>16s}")
    for lo, hi, lab in ((0, n // 3, "low third"), (n // 3, 2 * n // 3, "middle third"),
                        (2 * n // 3, n, "high third")):
        chunk = xs[lo:hi]
        if not chunk:
            continue
        rng = f"{chunk[0][0]:.0f}-{chunk[-1][0]:.0f} µm"
        kr = sum(1 for _, r in chunk if outcome(r.get("rigid"), null, crit) == "attraction")
        nr = sum(1 for _, r in chunk if outcome(r.get("nonrigid"), null, crit) == "attraction")
        dr = sum(1 for _, r in chunk
                 if isinstance(r.get("nonrigid"), dict) and r["nonrigid"].get("skipped"))
        print(f"    {lab:>11s} {rng:>10s} {len(chunk):4d} {kr:13d} "
              f"{nr:16d}   (+{dr} windows destroyed)")


def by_tissue(rows, null="dense_morphology"):
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        t = r["set"].rsplit("_", 1)[0]
        for a in ARMS:
            v = rejected(r.get(a), null)
            if v is None:
                continue
            agg[t][a][1] += 1
            agg[t][a][0] += int(v)
    if not agg:
        return
    print(f"\n  per tissue (null = {null}):")
    for t in sorted(agg):
        bits = []
        for a in ARMS:
            k, n = agg[t][a]
            if n:
                bits.append(f"{a} {k}/{n}")
        print(f"    {t:16s} " + "   ".join(bits))


def main():
    for arm in (2, 3, 1):
        rows = load(arm)
        if not rows:
            print(f"\nARM {arm}: no results yet")
            continue
        table(rows, arm)
        if arm == 1:
            for null in NULLS:
                signflips(rows, null)
        dose_response(rows)
        by_tissue(rows)


if __name__ == "__main__":
    main()
