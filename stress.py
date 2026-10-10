"""
stress.py - does the DAILY pulse warn before equity sell-offs?

predictive.py asks whether the weekly scores predict AVERAGE returns (they do not). This asks a
different question: when the pulse turns tight, is a large fall in the S&P 500 more likely?

Design (fixed before looking at results):
  signals   three scores built from the pulse inputs' signed z-scores (+ = easier):
              plumbing = net liquidity (daily) + SOFR - IORB        (liquidity-specific)
              market   = VIX + MOVE                                  (market stress)
              full     = every pulse input, as shipped
            HY/IG spreads only exist from late 2023 on FRED, so they cannot enter a long test.
  timing    every signal is used 2 business days after its date (weekly Fed data are dated
            Wednesday and published Thursday; FRED daily series arrive a day late).
  fire      signal <= -1 (one standard deviation tighter than its own 3-year norm).
  from calm only count a fire while the S&P 500 is within 5% of its running peak. A fire in the
            middle of a sell-off is the market already talking, not a warning.
  events    S&P 500 falls >= 10% within the next 63 trading days (and >= 5% within 21).
  null      circular shifts of the signal against the events: keeps the clustering of both
            series, so the p-value is not flattered by overlapping days. One-sided (tight -> more
            sell-offs). Benjamini-Hochberg q-values across the 6 tests.
  episodes  every peak-to-trough fall >= 10% since the sample start, and whether each signal
            fired from calm before the market had fallen 5%.
Run: python stress.py
"""
import numpy as np
import pandas as pd

import config as C
import predictive as PR
import pulse as P
import tracker as T

LAG_BDAYS = 2
THRESH = -1.0
CALM = -0.05
H_LONG, DEPTH_LONG = 63, -0.10
H_SHORT, DEPTH_SHORT = 21, -0.05
EP_DEPTH = -0.10
GAP = 21                 # trading days of quiet that separate two distinct signal episodes
NSHIFT = 5000
MIN_SHIFT = 252
SEED = 11
START = "2013-11-01"     # first date every pulse input needed here has a 3-year z-score history

SUBSETS = {
    "plumbing": ["net_liq_daily", "sofr_iorb"],
    "market": ["vix", "move"],
    "full": None,
}
LABELS = {"plumbing": "Funding plumbing (net liquidity, SOFR − IORB)",
          "market": "Market stress (VIX, MOVE)",
          "full": "Full daily pulse"}


# ------------------------------------------------------------------ data ----
def fetch_spx(start="2005-01-01"):
    return T.fetch_yf(["^GSPC"], start)["^GSPC"].dropna()


def sub_score(pl, keys):
    """Weighted mean of the available signed z-scores (same rule as the pulse itself)."""
    z = pl["z"]
    cols = [k for k in (keys or list(z.columns)) if k in z.columns]
    w = np.array([pl["meta"][k]["weight"] for k in cols], dtype=float)
    v = z[cols].values
    m = ~np.isnan(v)
    num = np.where(m, v * w, 0.0).sum(axis=1)
    den = np.where(m, w, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.Series(np.where(den > 0, num / den, np.nan), index=z.index)


def lagged(score, idx, lag=LAG_BDAYS):
    """Signal known on each trading day idx: the value from `lag` business days earlier."""
    s = score.dropna().resample("B").last().ffill(limit=C.PULSE_STALE_DAYS).shift(lag)
    return s.reindex(idx, method="ffill", limit=C.PULSE_STALE_DAYS + lag)


# ---------------------------------------------------------------- events ----
def fwd_drawdown(px, h):
    """Worst close-to-close fall over the next h trading days (<= 0). NaN where unknown."""
    r = px[::-1].rolling(h, min_periods=h).min()[::-1].shift(-1)
    return np.minimum(r / px - 1.0, 0.0)


def corrections(px, depth=EP_DEPTH):
    """Peak-to-trough falls >= |depth|: one per cycle between two all-time highs."""
    out, peak_i, peak = [], 0, px.iloc[0]
    i, n = 0, len(px)
    while i < n:
        v = px.iloc[i]
        if v >= peak:
            peak, peak_i = v, i
            i += 1
            continue
        if v / peak - 1 <= depth:                       # a correction is under way
            j = i
            while j < n and px.iloc[j] < peak:
                j += 1
            seg = px.iloc[peak_i:j]
            tr = seg.idxmin()
            out.append({"peak": px.index[peak_i], "hit": px.index[i], "trough": tr,
                        "depth": float(seg.min() / peak - 1), "end": px.index[min(j, n - 1)]})
            i = j
            continue
        i += 1
    return out


def signal_episodes(fire):
    """Start dates of distinct runs of fires, merging runs closer than GAP trading days."""
    starts, last = [], None
    for k, t in enumerate(fire.index[fire.values]):
        pos = fire.index.get_loc(t)
        if last is None or pos - last > GAP:
            starts.append(t)
        last = pos
    return starts


# ------------------------------------------------------------- the test ----
def hit_rate_test(fire, event, rng, nshift=NSHIFT):
    """fire, event: boolean arrays over the same calm-only days. Returns hit rate, base rate,
    and a one-sided circular-shift p-value for (hit rate - rate on the other days)."""
    fire = np.asarray(fire, dtype=bool)
    event = np.asarray(event, dtype=float)
    n, nf = len(fire), int(fire.sum())
    if nf < 10 or nf > n - 10 or n < 3 * MIN_SHIFT:
        return None
    ev_tot = event.sum()

    def stat(f):
        k = f.sum()
        return event[f].sum() / k - (ev_tot - event[f].sum()) / (n - k)
    obs = stat(fire)
    ks = rng.integers(MIN_SHIFT, n - MIN_SHIFT, size=nshift)
    null = np.array([stat(np.roll(fire, int(k))) for k in ks])
    p = float((1 + np.sum(null >= obs)) / (nshift + 1))
    return {"n": n, "nFire": nf, "hit": float(event[fire].mean()), "base": float(event.mean()),
            "diff": float(obs), "p": p}


def analyze(pl, px, thresh=THRESH):
    rng = np.random.default_rng(SEED)
    idx = px.index                                    # full history: the running peak must start in 2005
    dd = px / px.cummax() - 1.0
    calm = (dd >= CALM)
    ev = {"long": fwd_drawdown(px, H_LONG) <= DEPTH_LONG, "short": fwd_drawdown(px, H_SHORT) <= DEPTH_SHORT}
    known = {"long": fwd_drawdown(px, H_LONG).notna(), "short": fwd_drawdown(px, H_SHORT).notna()}
    scores = {k: sub_score(pl, keys) for k, keys in SUBSETS.items()}
    sigs = {k: lagged(s, idx) for k, s in scores.items()}
    in_sample = pd.Series(idx >= pd.Timestamp(START), index=idx)

    corr = [c for c in corrections(px) if c["peak"] >= pd.Timestamp(START)]
    tests = []
    for sk, sig in sigs.items():
        fire_all = (sig <= thresh)
        ok = sig.notna() & in_sample & calm
        for ek in ("long", "short"):
            m = ok & known[ek]
            r = hit_rate_test(fire_all[m].values, ev[ek][m].values, rng)
            if r:
                r.update({"signal": sk, "event": ek})
                tests.append(r)
    q = PR.bh_qvalues([t["p"] for t in tests])
    for t, qv in zip(tests, q):
        t["q"] = float(qv)

    # ---- episodes: did each signal fire from calm before the market had fallen 5%?
    episodes = []
    for c in corr:
        pk = idx.get_loc(c["peak"])
        row = {"peak": c["peak"], "trough": c["trough"], "depth": c["depth"],
               "days": int(idx.get_loc(c["trough"]) - pk), "signals": {}}
        win = idx[max(pk - 20, 0): idx.get_loc(c["hit"]) + 1]
        for sk, sig in sigs.items():
            f = [t for t in win if sig.get(t) is not None and sig.get(t) == sig.get(t) and sig[t] <= thresh and dd[t] >= CALM]
            at_peak = sig.get(c["peak"])
            if f:
                t0 = f[0]
                row["signals"][sk] = {"fired": True, "date": t0, "lead": int(idx.get_loc(t0) - pk),
                                      "fallSoFar": float(px[t0] / px[c["peak"]] - 1 if t0 >= c["peak"] else 0.0),
                                      "fallLeft": float(px[c["trough"]] / px[t0] - 1),
                                      "atPeak": None if at_peak != at_peak else float(at_peak)}
            else:
                row["signals"][sk] = {"fired": False, "atPeak": None if at_peak is None or at_peak != at_peak else float(at_peak)}
        episodes.append(row)

    # ---- false alarms: signal episodes from calm, and what followed
    alarms = {}
    for sk, sig in sigs.items():
        fire = ((sig <= thresh) & calm & in_sample & known["long"]).fillna(False)
        starts = signal_episodes(fire)
        hits = sum(1 for t in starts if bool(ev["long"].get(t, False)))
        alarms[sk] = {"episodes": len(starts), "followed": int(hits)}
    return {"tests": tests, "episodes": episodes, "alarms": alarms, "scores": scores, "sigs": sigs,
            "px": px, "dd": dd, "corr": corr, "start": pd.Timestamp(START), "thresh": thresh,
            "ev": ev["long"], "calm": calm, "known": known["long"]}


def power_check(res, trials=20, nshift=1000):
    """How big a warning could this test see? Plant a fire 5-30 days before a share of the real
    sell-offs, add random noise fires, and count how often the test returns p < 0.05."""
    rng = np.random.default_rng(SEED + 1)
    px, idx = res["px"], res["px"].index
    m = (res["calm"] & res["known"] & (idx >= res["start"])).values
    ev = res["ev"].values[m]
    out = {}
    for recall in (1.0, 0.5, 1 / 3):
        ps = []
        for r in range(trials):
            f = pd.Series(False, index=idx)
            for c in res["corr"]:
                if rng.random() <= recall:
                    pk = idx.get_loc(c["peak"])
                    f.iloc[max(pk - 30, 0):pk - 5] = True
            for k in np.where(rng.random(len(idx)) < 0.004)[0]:
                f.iloc[k:k + 8] = True
            o = hit_rate_test(f.values[m], ev, np.random.default_rng(r), nshift=nshift)
            if o:
                ps.append(o["p"])
        out[f"{recall:.2f}"] = {"share": round(float(np.mean(np.array(ps) < 0.05)), 2), "medianP": round(float(np.median(ps)), 3)}
    return out


def sensitivity(pl, px, cuts=(-0.5, -1.5)):
    rows = []
    for th in cuts:
        r = analyze(pl, px, thresh=th)
        for t in r["tests"]:
            if t["event"] == "long":
                a = r["alarms"][t["signal"]]
                rows.append({"thresh": th, "signal": t["signal"], "lift": round(t["hit"] / t["base"], 2),
                             "p": round(t["p"], 3), "episodes": a["episodes"], "followed": a["followed"]})
    return rows


# --------------------------------------------------------------- payload ----
def _d(t):
    return None if t is None else pd.Timestamp(t).strftime("%Y-%m-%d")


def payload(res, sens=None, power=None):
    px, corr = res["px"], res["corr"]
    wk = {k: s[s.index >= res["start"]].resample("W-FRI").last().dropna() for k, s in res["scores"].items()}
    dates = wk["full"].index
    chart = {"dates": [_d(d) for d in dates],
             **{k: [round(float(v), 2) for v in wk[k].reindex(dates).ffill().fillna(0).values] for k in wk},
             "spx": [round(float(v), 1) for v in px.resample("W-FRI").last().reindex(dates).ffill().values],
             "areas": [{"from": _d(c["peak"]), "to": _d(c["trough"]), "depth": round(c["depth"] * 100, 1)} for c in corr]}
    eps = []
    for e in res["episodes"]:
        sg = {}
        for k, v in e["signals"].items():
            sg[k] = {"fired": v["fired"], "atPeak": None if v.get("atPeak") is None else round(v["atPeak"], 2)}
            if v["fired"]:
                sg[k].update({"date": _d(v["date"]), "lead": v["lead"], "fallSoFar": round(v["fallSoFar"] * 100, 1),
                              "fallLeft": round(v["fallLeft"] * 100, 1)})
        eps.append({"peak": _d(e["peak"]), "trough": _d(e["trough"]), "depth": round(e["depth"] * 100, 1),
                    "days": e["days"], "signals": sg})
    tests = [{"signal": t["signal"], "event": t["event"], "nFire": t["nFire"], "n": t["n"],
              "hit": round(t["hit"] * 100, 1), "base": round(t["base"] * 100, 1),
              "lift": round(t["hit"] / t["base"], 2) if t["base"] > 0 else None,
              "p": round(t["p"], 3), "q": round(t["q"], 3)} for t in res["tests"]]
    warned = {k: sum(1 for e in res["episodes"] if e["signals"][k]["fired"] and e["signals"][k]["fallLeft"] <= -0.05)
              for k in SUBSETS}
    return {"labels": LABELS, "from": _d(res["start"]), "to": _d(px.index[-1]), "lag": LAG_BDAYS,
            "thresh": THRESH, "nshift": NSHIFT, "chart": chart, "episodes": eps, "tests": tests,
            "alarms": res["alarms"], "warned": warned, "sens": sens or [], "power": power or {}, "nEpisodes": len(res["episodes"]),
            "now": {k: (None if res["sigs"][k].dropna().empty else round(float(res["scores"][k].dropna().iloc[-1]), 2))
                    for k in SUBSETS}}


if __name__ == "__main__":
    pl = P.build_pulse(start="2005-01-01")
    px = fetch_spx()
    res = analyze(pl, px)
    pay = payload(res, sensitivity(pl, px), power_check(res))
    print(f"\nsample {pay['from']} to {pay['to']}, S&P 500 falls >= 10%: {pay['nEpisodes']}")
    print("\nsignal fires from calm (z <= -1):  P(>=10% fall in 63d)  vs  base   lift   p      q")
    for t in pay["tests"]:
        print(f"  {t['signal']:9s} {t['event']:5s} fire-days {t['nFire']:4d}/{t['n']}   {t['hit']:5.1f}%  vs {t['base']:5.1f}%   {t['lift']}   p={t['p']:.3f}  q={t['q']:.3f}")
    print("\nalarms (distinct signal episodes from calm):", pay["alarms"])
    print("corrections warned in time (fired from calm, >=5% fall still to come):", pay["warned"], "of", pay["nEpisodes"])
    for e in pay["episodes"]:
        s = "  ".join(f"{k}:{v['lead']:+d}d" if v["fired"] else f"{k}:no" for k, v in e["signals"].items())
        print(f"  peak {e['peak']}  fall {e['depth']:6.1f}%  {e['days']:3d}d   {s}")
    print("\nother cut-offs (10% fall within 63d):")
    for r in pay["sens"]:
        print("  ", r)
    print("power (share of runs with p<0.05 by share of sell-offs the planted warning catches):", pay["power"])
