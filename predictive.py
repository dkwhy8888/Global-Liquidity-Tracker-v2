"""
predictive.py - does the liquidity score PREDICT (lead) or FOLLOW (lag) equity-index returns?

Run:  python predictive.py        (prints a summary; build_artifact.py puts the full result
                                   on the dashboard's "Does it predict?" tab)

What is tested, for every index in config.INDICES against its region's score:
  FORWARD   Spearman rank correlation (IC) of the score at week t with the index return over
            the NEXT 4 / 13 / 26 / 52 weeks.      High IC = the score leads the market.
  BACKWARD  The same score against the PAST 4 / 13 / 26 / 52 weeks' return.
                                                  High IC = the score follows the market.
  PROFILE   4-week change in the score against the 4-week index return at lags of -24..+24
            weeks (cross-correlation): shows which one moves first.
  REGIMES   Average next-3-month return in each of the four regimes, and easy vs tight.
  TIMING    Hold the index when the impulse score is >= 0, else cash, with a 1-week delay.
  INPUTS    The same forward/backward test for every single input vs the region's main index.

Honesty rules built in:
  * POINT-IN-TIME data: every input is moved to the date it was really published
    (config.LAG_DAYS), so the test cannot use numbers that did not yet exist.
  * p-values come from a surrogate-data test: the score is replaced by random look-alikes
    with the same persistence (random Fourier phases) and the correlation is recomputed. This
    keeps the slow drift of the score and the overlap of multi-week returns, so it does not
    overstate significance. Checked on random series: about 5-7% flagged at the 5% level.
  * q-values (Benjamini-Hochberg false-discovery rate) correct for testing many indices at
    once. Headline claims use q, not raw p. Cells that pass only raw p are marked weak.
  * Everything is in-sample: the score design (inputs, signs, the impulse/conditions split)
    was chosen with this data in view. Treat results as evidence, not proof.
"""
import numpy as np
import pandas as pd

import config as C
import tracker as T

HORIZONS = (4, 13, 26, 52)
NPERM = 2000
SEED = 7
PROFILE_LAGS = list(range(-24, 25, 4))     # weeks; > 0: the score change comes first
MIN_WEEKS = 104                            # least joint history to report a result
HEADLINE_H = 13


# ----------------------------------------------------------------- data -----
def fetch_indices(start="2010-01-01"):
    """Weekly closes for every index in config.INDICES, plus the world equal-weight series."""
    tickers = sorted({t for lst in C.INDICES.values() for (_, _, t) in lst if t})
    raw = T.fetch_yf(tickers, start)
    out = {}
    for region, lst in C.INDICES.items():
        for key, label, t in lst:
            if t is None:
                continue
            s = raw.get(t)
            if s is None or len(s.dropna()) < MIN_WEEKS:
                print(f"  index skipped (no data): {label} [{t}]")
                continue
            out[key] = {"region": region, "label": label, "ticker": t, "close": T.to_weekly(s)}
    prim = [out[k]["close"] for k in C.PRIMARY_INDEX.values() if k in out and k != "acwi"]
    if len(prim) >= 4:
        lr = np.log(pd.concat(prim, axis=1)).diff()
        ew = lr.mean(axis=1).where(lr.notna().sum(axis=1) >= 4)
        out["world_ew"] = {"region": "GLOBAL", "ticker": None,
                           "label": dict((k, l) for k, l, _ in C.INDICES["GLOBAL"])["world_ew"],
                           "close": np.exp(ew.fillna(0).cumsum()).where(ew.notna())}
    return out


# ----------------------------------------------------------- statistics -----
def bh_qvalues(p):
    """Benjamini-Hochberg false-discovery-rate adjusted p-values (NaN-safe)."""
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def _z(a):
    a = np.asarray(a, dtype=float)
    sd = a.std()
    return (a - a.mean()) / sd if sd > 0 else a * 0.0


def surrogate_ranks(x, rng, nperm=NPERM):
    """Ranks (0..n-1, standardised) of `nperm` surrogate copies of x. Each surrogate has the
    same autocorrelation as x (random Fourier phases) but no link to any other series, so it
    is the right null for "a persistent score that has nothing to do with returns"."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    f = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, size=(nperm, len(f)))
    ph[:, 0] = 0.0
    if n % 2 == 0:
        ph[:, -1] = 0.0
    sur = np.fft.irfft(np.abs(f)[None, :] * np.exp(1j * ph), n=n, axis=1)
    rk = np.argsort(np.argsort(sur, axis=1), axis=1).astype(float)
    return (rk - (n - 1) / 2.0) / np.sqrt((n * n - 1) / 12.0), rk


def spearman_perm(x, y, h, rng, nperm=NPERM):
    """Spearman IC of x vs y, a two-sided surrogate-data p-value, the IC in each half of the
    sample, and the sample size. None if there is too little joint history."""
    df = pd.concat([x, y], axis=1).dropna()
    n = len(df)
    if n < MIN_WEEKS:
        return None
    rx, ry = _z(df.iloc[:, 0].rank().values), _z(df.iloc[:, 1].rank().values)
    ic = float((rx * ry).mean())
    zs, _ = surrogate_ranks(df.iloc[:, 0].values, rng, nperm)
    null = (zs * ry[None, :]).mean(axis=1)
    p = float((1 + np.sum(np.abs(null) >= abs(ic))) / (nperm + 1))
    half = n // 2
    h1 = float(df.iloc[:half, 0].rank().corr(df.iloc[:half, 1].rank())) if half >= 30 else np.nan
    h2 = float(df.iloc[half:, 0].rank().corr(df.iloc[half:, 1].rank())) if n - half >= 30 else np.nan
    return {"ic": ic, "p": p, "n": n, "h1": h1, "h2": h2,
            "start": df.index[0], "end": df.index[-1]}


def _r(v, nd):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


def returns(close, idx):
    lp = np.log(close.reindex(idx))
    fwd = {h: lp.shift(-h) - lp for h in HORIZONS}
    bwd = {h: lp - lp.shift(h) for h in HORIZONS}
    return lp, fwd, bwd


# ---------------------------------------------------------- the analyses ----
def profile(score, lp, step=4):
    """Cross-correlation of 4-week score changes with 4-week index returns."""
    d = score - score.shift(step)
    r = lp - lp.shift(step)
    out = []
    for L in PROFILE_LAGS:
        df = pd.concat([d, r.shift(-L)], axis=1).dropna()
        cs, ns = [], []
        for off in range(step):
            sub = df.iloc[off::step]
            if len(sub) >= 25:
                cs.append(sub.iloc[:, 0].rank().corr(sub.iloc[:, 1].rank()))
                ns.append(len(sub))
        out.append((L, float(np.mean(cs)) if cs else np.nan, int(np.median(ns)) if ns else 0))
    n = int(np.median([o[2] for o in out if o[2]] or [0]))
    return {"lags": [o[0] for o in out], "corr": [_r(o[1], 3) for o in out],
            "band": _r(2 / np.sqrt(n), 3) if n else None, "n": n}


def regime_table(score, mom, lp, rng):
    fwd13 = np.exp(lp.shift(-HEADLINE_H) - lp) - 1.0
    reg = pd.Series(index=score.index, dtype=object)
    ok = score.notna() & mom.notna()
    reg[ok] = [T.regime_label(a, b) for a, b in zip(score[ok], mom[ok])]
    df = pd.concat([reg.rename("reg"), score.rename("s"), fwd13.rename("f")], axis=1).dropna()
    if len(df) < MIN_WEEKS:
        return None
    rows = []
    for name in ["Easy & Improving", "Easy & Fading", "Tight & Improving", "Tight & Deteriorating"]:
        g = df[df.reg == name].f
        rows.append({"regime": name, "weeks": int(len(g)),
                     "mean": _r(g.mean() * 100, 1) if len(g) else None,
                     "pos": _r((g > 0).mean() * 100, 0) if len(g) else None})
    easy, tight = df[df.s >= 0].f, df[df.s < 0].f
    gap = p = None
    if len(easy) >= 26 and len(tight) >= 26:
        y = df.f.values
        n, ne = len(df), len(easy)
        gap = float(easy.mean() - tight.mean())
        _, rk = surrogate_ranks(df.s.values, rng)
        sh = (rk >= n - ne).astype(float)                 # same share of "easy" weeks as the real score
        null = (sh * y[None, :]).sum(axis=1) / ne - ((1 - sh) * y[None, :]).sum(axis=1) / (n - ne)
        p = float((1 + np.sum(np.abs(null) >= abs(gap))) / (len(null) + 1))
    return {"rows": rows, "all": _r(df.f.mean() * 100, 1), "easy": _r(easy.mean() * 100, 1) if len(easy) else None,
            "tight": _r(tight.mean() * 100, 1) if len(tight) else None,
            "gap": _r(gap * 100, 1) if gap is not None else None, "p": _r(p, 3),
            "easyWeeks": int(len(easy)), "tightWeeks": int(len(tight))}


def timing(score, close, idx):
    """Hold the index while score >= 0, else cash (0%); signal on Friday, trade the next week."""
    px = close.reindex(idx)
    r = px.pct_change()                                     # return earned over week t
    pos = (score.shift(1) >= 0).astype(float).where(score.shift(1).notna())
    df = pd.concat([r.rename("r"), pos.rename("pos")], axis=1).dropna()
    if len(df) < MIN_WEEKS:
        return None
    st = df.r * df.pos

    def stats(x):
        eq = (1 + x).cumprod()
        yrs = len(x) / 52.0
        vol = x.std() * np.sqrt(52)
        return {"cagr": _r(((eq.iloc[-1]) ** (1 / yrs) - 1) * 100, 1), "vol": _r(vol * 100, 1),
                "sharpe": _r((x.mean() * 52) / vol, 2) if vol > 0 else None,
                "mdd": _r((eq / eq.cummax() - 1).min() * 100, 0)}
    bh, sg = stats(df.r), stats(st)
    sg["invested"] = _r(df.pos.mean() * 100, 0)
    return {"start": df.index[0], "end": df.index[-1], "bh": bh, "st": sg,
            "bhCurve": (1 + df.r).cumprod(), "stCurve": (1 + st).cumprod()}


def lead_verdict(f13, f26):
    if f13 is None:
        return "short"
    stable = f13["h1"] == f13["h1"] and f13["h2"] == f13["h2"] and f13["h1"] > 0 and f13["h2"] > 0
    if f13["ic"] > 0 and f13["q"] is not None and f13["q"] < 0.10 and stable:
        return "leads"
    for f in (f13, f26):
        if f is not None and f["ic"] > 0 and f["p"] == f["p"] and f["p"] < 0.05:
            return "weak"
    if f13["ic"] < 0 and f13["p"] == f13["p"] and f13["p"] < 0.05:
        return "reverse"
    return "none"


def lag_verdict(b26):
    """Does the score follow the market? Same rule as the page headline: 6-month past return,
    corrected for testing many indices (q < 0.10)."""
    if b26 is None:
        return "short"
    q = b26.get("q")
    if q is not None and q == q and q < 0.10:
        return "follows" if b26["ic"] > 0 else "contrarian"
    return "none"


# ------------------------------------------------------------- main run -----
def analyze(ds, indices):
    """ds = point-in-time dataset (tracker.build_dataset(pit=True)); indices from fetch_indices."""
    rng = np.random.default_rng(SEED)
    idx = ds["composite"].index
    scores = {"impulse": ds["composite"], "conditions": ds["conditions"]}
    mom = ds["momentum"]

    # ---- per index: scorecard cells, profile, regimes, timing
    rows = {}
    for key, meta in indices.items():
        region = meta["region"]
        lp, fwd, bwd = returns(meta["close"], idx)
        rec = {"region": region, "label": meta["label"], "ticker": meta["ticker"],
               "first": lp.first_valid_index(), "last": lp.last_valid_index(), "cells": {}}
        for sname, sdf in scores.items():
            if region not in sdf.columns:
                continue
            s = sdf[region]
            cells = {"fwd": {}, "bwd": {}}
            for h in HORIZONS:
                cells["fwd"][h] = spearman_perm(s, fwd[h], h, rng)
                cells["bwd"][h] = spearman_perm(s, bwd[h], h, rng)
            rec["cells"][sname] = cells
            if sname == "impulse":
                rec["profile"] = profile(s, lp)
                rec["regimes"] = regime_table(s, mom[region], lp, rng)
                tm = timing(s, meta["close"], idx)
                rec["timing"] = tm
        rows[key] = rec

    # ---- FDR within each (score, direction, horizon) family across indices
    for sname in scores:
        for d in ("fwd", "bwd"):
            for h in HORIZONS:
                keys = [k for k, r in rows.items() if r["cells"].get(sname, {}).get(d, {}).get(h)]
                q = bh_qvalues([rows[k]["cells"][sname][d][h]["p"] for k in keys])
                for k, qv in zip(keys, q):
                    rows[k]["cells"][sname][d][h]["q"] = None if np.isnan(qv) else float(qv)
    for r in rows.values():
        for sname, cells in r["cells"].items():
            for d in ("fwd", "bwd"):
                for h in HORIZONS:
                    c = cells[d].get(h)
                    if c is not None:
                        c.setdefault("q", None)
        imp = r["cells"].get("impulse")
        if imp:
            r["leadVerdict"] = lead_verdict(imp["fwd"][13], imp["fwd"][26])
            r["lagVerdict"] = lag_verdict(imp["bwd"][26])
        con = r["cells"].get("conditions")
        if con:
            r["leadVerdictCond"] = lead_verdict(con["fwd"][13], con["fwd"][26])

    # ---- every single input vs the region's main index
    inputs = {}
    for region in C.REGIONS:
        key = C.PRIMARY_INDEX.get(region)
        if key not in indices:
            continue
        lp, fwd, bwd = returns(indices[key]["close"], idx)
        lst = []
        for k, m in ds["meta"].items():
            if m["region"] != region or k not in ds["compZ"].columns:
                continue
            s = ds["compZ"][k]
            rec = {"key": k, "bucket": m["bucket"],
                   "group": "impulse" if m["bucket"] in C.IMPULSE_BUCKETS else "conditions",
                   "f13": spearman_perm(s, fwd[13], 13, rng), "f26": spearman_perm(s, fwd[26], 26, rng),
                   "b13": spearman_perm(s, bwd[13], 13, rng)}
            if rec["f13"] is not None:
                lst.append(rec)
        inputs[region] = lst
    for name in ("f13", "f26", "b13"):
        flat = [(r, i) for r, lst in inputs.items() for i in lst if i[name] is not None]
        q = bh_qvalues([i[name]["p"] for _, i in flat])
        for (_, i), qv in zip(flat, q):
            i[name]["q"] = None if np.isnan(qv) else float(qv)
    return {"indices": rows, "inputs": inputs, "scores": list(scores)}


# ----------------------------------------------------------- JSON payload ---
def _cell(c):
    if c is None:
        return None
    return {"ic": _r(c["ic"], 3), "p": _r(c["p"], 3), "q": _r(c.get("q"), 3), "n": c["n"],
            "h1": _r(c["h1"], 2), "h2": _r(c["h2"], 2),
            "from": c["start"].strftime("%Y-%m"), "to": c["end"].strftime("%Y-%m")}


def payload(res):
    """JSON-ready dict for the dashboard tab."""
    out_idx = []
    for key, r in res["indices"].items():
        rec = {"key": key, "region": r["region"], "label": r["label"], "ticker": r["ticker"],
               "from": r["first"].strftime("%Y-%m-%d") if r["first"] is not None else None,
               "to": r["last"].strftime("%Y-%m-%d") if r["last"] is not None else None,
               "lead": r.get("leadVerdict"), "lag": r.get("lagVerdict"), "leadCond": r.get("leadVerdictCond"),
               "score": {}}
        for sname, cells in r["cells"].items():
            rec["score"][sname] = {d: {str(h): _cell(c) for h, c in cells[d].items()} for d in ("fwd", "bwd")}
        rec["profile"] = r.get("profile")
        rec["regimes"] = r.get("regimes")
        tm = r.get("timing")
        if tm:
            sd = tm["bhCurve"].index.strftime("%Y-%m-%d")
            rec["timing"] = {"start": sd[0], "end": sd[-1], "bh": tm["bh"], "st": tm["st"],
                             "bhCurve": [round(float(v), 4) for v in tm["bhCurve"].values],
                             "stCurve": [round(float(v), 4) for v in tm["stCurve"].values]}
        out_idx.append(rec)

    inputs = {}
    for region, lst in res["inputs"].items():
        inputs[region] = [{"key": i["key"], "group": i["group"], "f13": _cell(i["f13"]), "f26": _cell(i["f26"]),
                           "b13": _cell(i["b13"])} for i in lst]

    # headline numbers, so the page can state them without recomputing
    def fam(sname, d, h):
        cells = [r["cells"][sname][d][h] for r in res["indices"].values()
                 if r["cells"].get(sname, {}).get(d, {}).get(h)]
        ics = [c["ic"] for c in cells]
        return {"n": len(cells), "pos": int(sum(i > 0 for i in ics)),
                "sig": int(sum(1 for c in cells if c["p"] == c["p"] and c["p"] < 0.05)),
                "sigPos": int(sum(1 for c in cells if c["p"] == c["p"] and c["p"] < 0.05 and c["ic"] > 0)),
                "fdr": int(sum(1 for c in cells if c.get("q") is not None and c["q"] < 0.10)),
                "median": _r(float(np.median(ics)), 3) if ics else None}
    head = {sname: {d: {str(h): fam(sname, d, h) for h in HORIZONS} for d in ("fwd", "bwd")} for sname in res["scores"]}
    cells_total = sum(head[s][d][str(h)]["n"] for s in head for d in ("fwd", "bwd") for h in HORIZONS)
    cells_sig = sum(head[s][d][str(h)]["sig"] for s in head for d in ("fwd", "bwd") for h in HORIZONS)
    tms = [r["timing"] for r in res["indices"].values() if r.get("timing")]
    timing_head = {"n": len(tms),
                   "sharpeBetter": int(sum(1 for t in tms if (t["st"]["sharpe"] or 0) > (t["bh"]["sharpe"] or 0))),
                   "ddBetter": int(sum(1 for t in tms if (t["st"]["mdd"] or 0) > (t["bh"]["mdd"] or 0))),
                   "cagrBetter": int(sum(1 for t in tms if (t["st"]["cagr"] or 0) > (t["bh"]["cagr"] or 0)))}
    gaps = [r["regimes"] for r in res["indices"].values() if r.get("regimes") and r["regimes"]["gap"] is not None]
    regime_head = {"n": len(gaps), "easyBeatsTight": int(sum(1 for g in gaps if g["gap"] > 0)),
                   "sig": int(sum(1 for g in gaps if g["p"] is not None and g["p"] < 0.05 and g["gap"] > 0))}
    return {"horizons": list(HORIZONS), "profileLags": PROFILE_LAGS, "nperm": NPERM,
            "indices": out_idx, "inputs": inputs, "headline": head,
            "cellsTotal": cells_total, "cellsSig": cells_sig, "timing": timing_head, "regime": regime_head,
            "lags": C.LAG_DAYS}


if __name__ == "__main__":
    ds = T.build_dataset()
    ix = fetch_indices()
    res = analyze(ds["pit"], ix)
    pl = payload(res)
    pd.set_option("display.width", 220)
    print(f"\n{'index':34s} {'region':6s} lead(13w IC, p, q)            lag(13w IC, p)   verdicts")
    for r in pl["indices"]:
        f = (r["score"].get("impulse") or {}).get("fwd", {}).get("13")
        b = (r["score"].get("impulse") or {}).get("bwd", {}).get("13")
        if f:
            print(f"{r['label'][:34]:34s} {r['region']:6s} {f['ic']:+.2f} p={f['p']:.3f} q={f['q'] if f['q'] is not None else float('nan'):.3f}   "
                  f"{b['ic']:+.2f} p={b['p']:.3f}   {r['lead']}/{r['lag']}")
    print(f"\nheadline: {pl['cellsSig']} of {pl['cellsTotal']} cells have p<0.05 (chance alone: ~{0.05 * pl['cellsTotal']:.0f})")
    print("timing:", pl["timing"], " regimes:", pl["regime"])
