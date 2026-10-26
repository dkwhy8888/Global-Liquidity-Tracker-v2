"""
stress_multi.py - the sell-off warning test across several markets and a longer history.

stress.py used one market (S&P 500) and 6 sell-offs. This adds power in two ways:

  DAILY   the US daily pulse signals against sell-offs in six markets (S&P 500, Euro Stoxx 50,
          Nikkei 225, Hang Seng, KOSPI, Straits Times). US stress travels, so a US warning is a
          fair warning for the others. Market-stress (VIX, MOVE) reaches back to 2007 and so
          adds 2008-2011. Plumbing and the full pulse need net-liquidity history: 2013 onward.
  WEEKLY  each region's own liquidity scores (point-in-time) against that region's own index.

Rules are those of stress.py, fixed in advance: fire when the signal <= -1, only while the
market is within 5% of its running peak; event = a fall >= 10% within the next 63 trading days
(13 weeks for the weekly test) or >= 5% within 21 days; signals used 2 business days late
(weekly: 1 week late).

Pooling: the statistic is the average, over markets, of (sell-off rate after a fire minus the
rate on other calm days). The null shifts the signal in time against ALL markets at once, so
the strong co-movement of sell-offs across markets is kept and one global crash is not counted
six times. One-sided p. Benjamini-Hochberg q-values across the 10 pooled tests.
Run: python stress_multi.py
"""
import numpy as np
import pandas as pd

import config as C
import predictive as PR
import pulse as P
import stress as ST
import tracker as T

MARKETS = {"US": ("^GSPC", "S&P 500"), "EU": ("^STOXX50E", "Euro Stoxx 50"), "JP": ("^N225", "Nikkei 225"),
           "HK": ("^HSI", "Hang Seng"), "KR": ("^KS11", "KOSPI"), "SG": ("^STI", "Straits Times")}
START_LONG = "2007-06-01"
START_MID = ST.START
H_LONG, D_LONG = ST.H_LONG, ST.DEPTH_LONG
H_SHORT, D_SHORT = ST.H_SHORT, ST.DEPTH_SHORT
W_H, W_DEPTH = 13, -0.10                      # weekly test: 13 weeks, 10%
NSHIFT = 4000
MIN_SHIFT_D, MIN_SHIFT_W = 252, 52
SEED = 23
MERGE_DAYS = 90                               # peaks within this many days count as one global episode


# ------------------------------------------------------------- pooled test ----
def pooled_test(fire, ev, mask, rng, min_shift, nshift=NSHIFT, min_fire=10, names=None):
    """fire, ev, mask: (T x M) arrays. Returns per-market and pooled hit/base/diff and one-sided p.
    The null rolls `fire` in time (all markets together) against fixed events."""
    fire = np.asarray(fire, bool)
    ev = np.asarray(ev, float)
    mk = np.asarray(mask, bool)
    n, m = fire.shape
    evm = ev * mk
    n_mask = mk.sum(0)
    ev_tot = evm.sum(0)

    def stat(f):
        fm = f & mk
        n1 = fm.sum(0)
        s1 = (evm * f).sum(0)
        n0 = n_mask - n1
        with np.errstate(invalid="ignore", divide="ignore"):
            d = s1 / n1 - (ev_tot - s1) / n0
        return np.where((n1 > 0) & (n0 > 0), d, 0.0), n1
    d_obs, n1_obs = stat(fire)
    ok = (n1_obs >= min_fire) & (n_mask - n1_obs >= min_fire)
    if ok.sum() == 0:
        return None
    ks = rng.integers(min_shift, n - min_shift, size=nshift)
    null = np.empty((nshift, m))
    for i, k in enumerate(ks):
        null[i] = stat(np.roll(fire, int(k), axis=0))[0]
    pooled_obs = float(d_obs[ok].mean())
    pooled_null = null[:, ok].mean(1)
    rows = []
    for j in range(m):
        base = float(evm[:, j].sum() / n_mask[j]) if n_mask[j] else None
        hit = float((evm[:, j] * fire[:, j]).sum() / n1_obs[j]) if n1_obs[j] else None
        p = float((1 + np.sum(null[:, j] >= d_obs[j])) / (nshift + 1)) if ok[j] else None
        rows.append({"market": names[j] if names else j, "nFire": int(n1_obs[j]), "n": int(n_mask[j]), "hit": hit, "base": base,
                     "diff": float(d_obs[j]), "p": p, "used": bool(ok[j])})
    used = [r for r in rows if r["used"]]
    return {"pooled": {"hit": float(np.mean([r["hit"] for r in used])), "base": float(np.mean([r["base"] for r in used])),
                       "diff": pooled_obs, "p": float((1 + np.sum(pooled_null >= pooled_obs)) / (nshift + 1)),
                       "markets": len(used)}, "rows": rows}


# ----------------------------------------------------------------- daily ------
def daily_data():
    tk = [v[0] for v in MARKETS.values()]
    px = T.fetch_yf(tk, "2005-01-01")
    end = max(s.index[-1] for s in px.values())
    grid = pd.bdate_range("2005-01-03", end)
    cols = {}
    for m, (t, _) in MARKETS.items():
        cols[m] = px[t].dropna().reindex(grid).ffill(limit=5)
    return pd.DataFrame(cols)


def market_frames(P_):
    """Per market: drawdown, calm, event flags and 'known' flags on the common business-day grid."""
    out = {}
    for m in P_.columns:
        s = P_[m].dropna()
        dd = (s / s.cummax() - 1.0).reindex(P_.index)
        fl, fs = ST.fwd_drawdown(s, H_LONG).reindex(P_.index), ST.fwd_drawdown(s, H_SHORT).reindex(P_.index)
        out[m] = {"calm": dd >= ST.CALM, "long": fl <= D_LONG, "short": fs <= D_SHORT,
                  "kl": fl.notna(), "ks": fs.notna(), "dd": dd, "px": s}
    return out


def effective_episodes(dates):
    d = sorted(dates)
    n, last = 0, None
    for t in d:
        if last is None or (t - last).days > MERGE_DAYS:
            n += 1
        last = t
    return n


def daily_family(pl, P_, F, subset, start, rng, thresholds=(ST.THRESH,)):
    grid = P_.index
    score = ST.sub_score(pl, ST.SUBSETS[subset])
    sig = ST.lagged(score, grid)
    insample = pd.Series(grid >= pd.Timestamp(start), index=grid)
    out = {}
    for ek, kk in (("long", "kl"), ("short", "ks")):
        fire = np.column_stack([(sig <= ST.THRESH).values for _ in P_.columns])
        ev = np.column_stack([F[m][ek].values for m in P_.columns])
        mask = np.column_stack([(F[m]["calm"] & F[m][kk] & sig.notna() & insample).values for m in P_.columns])
        out[ek] = pooled_test(fire, ev, mask, rng, MIN_SHIFT_D, names=list(P_.columns))
    return sig, out


def episodes_table(P_, F, sig, start):
    """Per market: sell-offs >=10% since `start`; warned in time; signal episodes from calm and hits."""
    rows, peaks = [], []
    grid = P_.index
    for m in P_.columns:
        s = F[m]["px"]
        corr = [c for c in ST.corrections(s) if c["peak"] >= pd.Timestamp(start)]
        warned = 0
        for c in corr:
            pk = grid.get_loc(c["peak"]) if c["peak"] in grid else grid.searchsorted(c["peak"])
            win = grid[max(pk - 20, 0): grid.searchsorted(c["hit"]) + 1]
            f = [t for t in win if sig.get(t) == sig.get(t) and sig[t] <= ST.THRESH and F[m]["dd"][t] >= ST.CALM]
            if f and s.get(grid[min(grid.searchsorted(c["trough"]), len(grid) - 1)], np.nan) / s.asof(f[0]) - 1 <= -0.05:
                warned += 1
            peaks.append(c["peak"])
        fire = ((sig <= ST.THRESH) & F[m]["calm"] & F[m]["kl"] & (grid >= pd.Timestamp(start))).fillna(False)
        starts = ST.signal_episodes(fire)
        hits = sum(1 for t in starts if bool(F[m]["long"].get(t, False)))
        rows.append({"market": m, "label": MARKETS[m][1], "selloffs": len(corr), "warned": warned,
                     "alarms": len(starts), "followed": int(hits)})
    return rows, effective_episodes(peaks), len(peaks)


# ---------------------------------------------------------------- weekly ------
def weekly_family(ds_pit, rng):
    comp, cond = ds_pit["composite"], ds_pit["conditions"]
    idx = comp.index
    regs = [r for r in C.REGIONS if r != "GLOBAL" and r in comp.columns]
    ix = PR.fetch_indices()
    close = {}
    for r in regs:
        k = C.PRIMARY_INDEX.get(r)
        if k in ix:
            close[r] = ix[k]["close"].resample("W-FRI").last().reindex(idx)
    regs = [r for r in regs if r in close]
    ev, calm, known, peaks = {}, {}, {}, []
    for r in regs:
        s = close[r].dropna()
        dd = (s / s.cummax() - 1.0).reindex(idx)
        f = ST.fwd_drawdown(s, W_H).reindex(idx)
        ev[r], calm[r], known[r] = (f <= W_DEPTH), (dd >= ST.CALM), f.notna()
    out, rows_by = {}, {}
    for sk, sdf in (("impulse", comp), ("conditions", cond)):
        sig = {r: sdf[r].shift(1) for r in regs}
        fire = np.column_stack([(sig[r] <= ST.THRESH).values for r in regs])
        e = np.column_stack([ev[r].values for r in regs])
        mask = np.column_stack([(calm[r] & known[r] & sig[r].notna()).values for r in regs])
        res = pooled_test(fire, e, mask, rng, MIN_SHIFT_W, names=regs)
        out[sk] = res
        # sell-offs and warnings per region (weekly)
        tab = []
        for r in regs:
            s = close[r].dropna()
            corr = [c for c in ST.corrections(s) if sig[r].first_valid_index() is not None and c["peak"] >= sig[r].first_valid_index()]
            warned = 0
            for c in corr:
                lo = max(idx.searchsorted(c["peak"]) - 4, 0)
                win = idx[lo: idx.searchsorted(c["hit"]) + 1]
                f = [t for t in win if sig[r].get(t) == sig[r].get(t) and sig[r][t] <= ST.THRESH and (s.asof(t) / s[:t].max() - 1) >= ST.CALM]
                if f and s.asof(c["trough"]) / s.asof(f[0]) - 1 <= -0.05:
                    warned += 1
                if sk == "impulse":
                    peaks.append(c["peak"])
            fire_r = ((sig[r] <= ST.THRESH) & calm[r] & known[r]).fillna(False)
            starts, last = [], None
            for t in fire_r.index[fire_r.values]:
                pos = fire_r.index.get_loc(t)
                if last is None or pos - last > 4:
                    starts.append(t)
                last = pos
            hits = sum(1 for t in starts if bool(ev[r].get(t, False)))
            tab.append({"market": r, "selloffs": len(corr), "warned": warned, "alarms": len(starts), "followed": int(hits)})
        rows_by[sk] = tab
    return out, rows_by, effective_episodes(peaks), len(peaks)


# No power simulation here: a planted warning before the global episodes is diluted across markets
# whose own sell-offs fall at other times, so its p-values do not say how much a real signal would show.
# The honest measure of added evidence is the number of DISTINCT global episodes (see effective_episodes).


def robustness(pl, P_, F, starts=("2007-06-01", "2010-01-01")):
    """Same market-stress test, started after the 2007-09 crisis: is it one episode?"""
    out = []
    for st in starts:
        _, res = daily_family(pl, P_, F, "market", st, np.random.default_rng(SEED + 9))
        r = res["long"]
        if r:
            p = r["pooled"]
            out.append({"start": st, "markets": p["markets"], "hit": p["hit"], "base": p["base"], "p": p["p"]})
    return out


# ----------------------------------------------------------------- driver ------
def run(pl, ds_pit):
    rng = np.random.default_rng(SEED)
    P_ = daily_data()
    F = market_frames(P_)
    tests, tables = [], {}

    def add(name, family, signal, event, res, start):
        if res:
            tests.append({"id": name, "family": family, "signal": signal, "event": event, "start": start,
                          **{k: res["pooled"][k] for k in ("hit", "base", "diff", "p", "markets")}, "rows": res["rows"]})
    # long daily test: market stress from 2007
    sig_l, res_l = daily_family(pl, P_, F, "market", START_LONG, rng)
    for ek, r in res_l.items():
        add(f"D-long-market-{ek}", "daily-long", "market", ek, r, START_LONG)
    tables["daily-long"] = episodes_table(P_, F, sig_l, START_LONG)
    # 2013+ daily tests, all three signals
    for sk in ("plumbing", "market", "full"):
        sig_m, res_m = daily_family(pl, P_, F, sk, START_MID, rng)
        for ek, r in res_m.items():
            add(f"D-mid-{sk}-{ek}", "daily-mid", sk, ek, r, START_MID)
        tables[f"daily-mid-{sk}"] = episodes_table(P_, F, sig_m, START_MID)
    # weekly regional
    wres, wtab, weff, wn = weekly_family(ds_pit, rng)
    for sk, r in wres.items():
        add(f"W-{sk}", "weekly", sk, "long", r, None)
    q = PR.bh_qvalues([t["p"] for t in tests])
    for t, qv in zip(tests, q):
        t["q"] = float(qv)
    return {"tests": tests, "tables": tables, "weekly": wtab, "weeklyEff": (weff, wn), "grid": P_, "F": F,
            "robust": robustness(pl, P_, F)}


def _r(v, nd=3):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


def payload(res):
    tests = []
    for t in res["tests"]:
        rows = []
        for r in t["rows"]:
            rows.append({"market": r["market"], "nFire": r["nFire"], "n": r["n"], "used": r["used"],
                         "hit": _r(r["hit"] * 100, 1) if r["hit"] is not None else None,
                         "base": _r(r["base"] * 100, 1) if r["base"] is not None else None,
                         "lift": _r(r["hit"] / r["base"], 2) if r["hit"] is not None and r["base"] else None,
                         "p": _r(r["p"])})
        tests.append({"id": t["id"], "family": t["family"], "signal": t["signal"], "event": t["event"], "start": t["start"],
                      "hit": _r(t["hit"] * 100, 1), "base": _r(t["base"] * 100, 1),
                      "lift": _r(t["hit"] / t["base"], 2) if t["base"] else None, "p": _r(t["p"]), "q": _r(t["q"]),
                      "markets": t["markets"], "rows": rows})
    tab = {k: {"rows": rows, "effective": eff, "peaks": n} for k, (rows, eff, n) in res["tables"].items()}
    rob = [{"start": r["start"], "markets": r["markets"], "hit": _r(r["hit"] * 100, 1), "base": _r(r["base"] * 100, 1),
            "lift": _r(r["hit"] / r["base"], 2), "p": _r(r["p"])} for r in res["robust"]]
    return {"markets": {m: v[1] for m, v in MARKETS.items()}, "tests": tests, "tables": tab, "weekly": res["weekly"],
            "weeklyEffective": res["weeklyEff"][0], "weeklyPeaks": res["weeklyEff"][1],
            "robust": rob, "startLong": START_LONG, "startMid": START_MID, "nshift": NSHIFT}


if __name__ == "__main__":
    pl = P.build_pulse(start="2005-01-01")
    ds = T.build_dataset()
    res = run(pl, ds["pit"])
    pay = payload(res)
    print(f"\n{'test':26s} {'mkts':>4s} {'hit':>6s} {'base':>6s} {'ratio':>6s} {'p':>6s} {'q':>6s}")
    for t in pay["tests"]:
        print(f"{t['id']:26s} {t['markets']:4d} {t['hit']:6.1f} {t['base']:6.1f} {t['lift']!s:>6s} {t['p']:6.3f} {t['q']:6.3f}")
    for k, v in pay["tables"].items():
        print(f"\n{k}: sell-offs {v['peaks']} (global episodes {v['effective']})")
        for r in v["rows"]:
            print("  ", r)
    print("\nweekly effective episodes:", pay["weeklyEffective"], "of", pay["weeklyPeaks"])
    for k, v in pay["weekly"].items():
        print("weekly", k, v)
    print("\nper-market (daily-long, market stress, long event):")
    for t in pay["tests"]:
        if t["id"] == "D-long-market-long":
            for r in t["rows"]:
                print("  ", r)
    print("\nweekly per-region impulse:")
    for t in pay["tests"]:
        if t["id"] == "W-impulse":
            for r in t["rows"]:
                print("  ", r)
    print("\nrobustness (market stress, long):", pay["robust"])
