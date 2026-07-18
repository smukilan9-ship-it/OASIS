"""
compare.py — merge ours_results.json + valis_results.json into REPORT.md.
RUN IN THE MAIN .venv (numpy only):  .venv/bin/python -m validation.valis_bench.compare

Produces:
  • aggregate rTRE per method (median-of-medians = ANHIR MMrTRE, plus mean & success rate),
    always alongside the identity baseline so improvement is visible;
  • the OASIS-similarity vs VALIS-rigid comparison (both distance-preserving, apples-to-apples)
    and VALIS-nonrigid as the forbidden-warp upper bound;
  • a non-circular GATE CALIBRATION table: bucket pairs by our gate's verdict and show the
    INDEPENDENT expert-landmark rTRE in each bucket (the gate never saw those landmarks),
    plus Spearman rank-corr between the gate's own TRE and the independent rTRE.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _med(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/d) if d else None


def agg(records, key):
    """median-of-median rTRE, mean, and success rate for a method key."""
    meds = []
    for r in records:
        v = r.get(key)
        if v and v.get("median") is not None:
            meds.append(v["median"])
    n_ok = len(meds)
    return {"MMrTRE": _med(meds), "mean_MrTRE": (float(np.mean(meds)) if meds else None),
            "n_ok": n_ok}


def main():
    ours = _load("ours_results.json")
    valis = _load("valis_results.json")
    lines = []
    P = lines.append
    P("# VALIS vs OASIS — serial-section registration on ANHIR/CIMA held-out landmarks\n")
    P("rTRE = target registration error relative to the fixed-image diagonal (ANHIR "
      "convention). Lower is better. **No method ever sees the expert landmarks** — they "
      "score registration driven purely by image pixels. MMrTRE = median over pairs of the "
      "per-pair median rTRE.\n")

    orecs = ours["results"] if ours else []
    vrecs = {r["pair_id"]: r for r in valis["results"]} if valis else {}
    n = len(orecs) or len(vrecs)
    sets = sorted({r["set"] for r in orecs} | {r["set"] for r in (valis["results"] if valis else [])})
    P(f"**Scope:** {n} directed pairs across {len(sets)} tissue set(s): {', '.join(sets)}.\n")

    # aggregate table
    P("## Aggregate accuracy\n")
    P("| method | transform | distance-preserving? | MMrTRE | mean MrTRE | pairs scored |")
    P("|---|---|---|---:|---:|---:|")
    if orecs:
        init = agg(orecs, "initial")
        P(f"| (no registration) | identity | — | {init['MMrTRE']:.4f} | {init['mean_MrTRE']:.4f} | {init['n_ok']} |")
        for key, tf, dp in [("ours_loftr", "similarity (LoFTR)", "yes"),
                            ("ours_auto", "similarity (structural)", "yes")]:
            a = agg(orecs, key)
            if a["MMrTRE"] is not None:
                P(f"| **OASIS** {key.split('_')[1]} | {tf} | {dp} | {a['MMrTRE']:.4f} | {a['mean_MrTRE']:.4f} | {a['n_ok']} |")
    if valis:
        vlist = valis["results"]
        for key, tf, dp in [("valis_rigid", "rigid", "yes"),
                            ("valis_nonrigid", "rigid+non-rigid", "NO (warp)")]:
            a = agg(vlist, key)
            if a["MMrTRE"] is not None:
                P(f"| VALIS {key.split('_')[1]} | {tf} | {dp} | {a['MMrTRE']:.4f} | {a['mean_MrTRE']:.4f} | {a['n_ok']} |")
    P("")

    # apples-to-apples note
    P("## Apples-to-apples: OASIS similarity vs VALIS-rigid (both cross-K-safe)\n")
    if orecs and valis:
        common = [r for r in orecs if r["pair_id"] in vrecs]
        do = [(r.get("ours_loftr") or {}).get("median") for r in common]
        dr = [(vrecs[r["pair_id"]].get("valis_rigid") or {}).get("median") for r in common]
        paired = [(a, b) for a, b in zip(do, dr) if a is not None and b is not None]
        if paired:
            oa = np.array([a for a, _ in paired]); ra = np.array([b for _, b in paired])
            P(f"On the {len(paired)} pairs where BOTH produced a transform: "
              f"OASIS-LoFTR median rTRE {np.median(oa):.4f} vs VALIS-rigid {np.median(ra):.4f}. "
              f"OASIS better on {int((oa<ra).sum())}/{len(paired)} pairs.\n")
        else:
            P("_No pairs where both produced a transform (see per-pair table / caveats)._\n")

    # gate calibration (non-circular)
    P("## Gate calibration (non-circular)\n")
    P("Our gate saw only the LoFTR correspondences; the rTRE below is on the independent "
      "expert landmarks. A trustworthy gate should show LOW rTRE for pass verdicts and HIGH "
      "for fail verdicts.\n")
    if orecs:
        buckets = {}
        gtre, indep = [], []
        for r in orecs:
            g = r.get("gate") or {}
            v = g.get("verdict") or "NONE"
            il = (r.get("ours_loftr") or {}).get("median")
            buckets.setdefault(v, []).append(il)
            if g.get("loo_tre_um") is not None and il is not None:
                gtre.append(g["loo_tre_um"]); indep.append(il)
        P("| gate verdict | pairs | median independent rTRE | pairs with a transform |")
        P("|---|---:|---:|---:|")
        for v, vals in sorted(buckets.items()):
            ok = [x for x in vals if x is not None]
            m = f"{np.median(ok):.4f}" if ok else "—"
            P(f"| {v} | {len(vals)} | {m} | {len(ok)} |")
        rho = _spearman(gtre, indep) if len(gtre) >= 3 else None
        P("")
        if rho is not None:
            P(f"Spearman(gate self-TRE, independent rTRE) = **{rho:.2f}** over {len(gtre)} pairs "
              f"(→ {'the gate tracks true error' if rho>0.5 else 'weak/ços — see caveats'}).\n")

    # per-pair table
    P("## Per-pair rTRE (median)\n")
    P("| pair | N | init | OASIS-LoFTR | OASIS-auto | VALIS-rigid | VALIS-nonrigid | gate |")
    P("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in orecs:
        vp = vrecs.get(r["pair_id"], {})
        def g(d, k="median"):
            return f"{d[k]:.4f}" if d and d.get(k) is not None else "—"
        P(f"| {r['moving_stain'][-10:]}→{r['fixed_stain'][-10:]} | {r['n_landmarks']} | "
          f"{g(r.get('initial'))} | {g(r.get('ours_loftr'))} | {g(r.get('ours_auto'))} | "
          f"{g(vp.get('valis_rigid'))} | {g(vp.get('valis_nonrigid'))} | "
          f"{(r.get('gate') or {}).get('verdict','—')} |")
    P("")

    out = os.path.join(HERE, "REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("wrote", out)
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
