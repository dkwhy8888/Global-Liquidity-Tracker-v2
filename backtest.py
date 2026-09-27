"""
backtest.py - does each liquidity signal lead equity returns?

For every region, each signed z input (+ = easier) and the region composite are
tested against that region's equity index return over the next 4 / 13 / 26 weeks.
GLOBAL inputs are tested against an equal-weight average of the regional indices.

Method (kept simple and honest):
- IC = Spearman rank correlation of signal_t with the forward log return.
- Non-overlapping samples: the series is sampled every h weeks; this is repeated for
  each of the h start offsets and the ICs are averaged, so every week is used but no
  single IC double-counts overlapping returns. t = IC * sqrt(n-2) / sqrt(1-IC^2), n per
  offset (a conservative count).
- Stability: IC in the first and second half of each signal's own history.
- Easy-vs-tight: average forward 13-week return when the signal is > 0 vs < 0.

Run:  python backtest.py            (writes backtest_results.csv, prints the summary)
"""
import numpy as np
import pandas as pd

import config as C

HORIZONS = (4, 13, 26)
MIN_OBS = 150          # weeks of joint history needed to report a signal


def _ic(sig, fwd, h):
    """Mean non-overlapping Spearman IC over the h start offsets, and a t-stat."""
    df = pd.concat([sig, fwd], axis=1).dropna()
    if len(df) < MIN_OBS:
        return np.nan, np.nan, 0
    ics, ns = [], []
    for k in range(h):
        sub = df.iloc[k::h]
        if len(sub) >= 12:
            ics.append(sub.iloc[:, 0].rank().corr(sub.iloc[:, 1].rank()))
            ns.append(len(sub))
    if not ics:
        return np.nan, np.nan, 0
    ic, n = float(np.nanmean(ics)), int(np.mean(ns))
    t = ic * np.sqrt(max(n - 2, 1)) / np.sqrt(max(1 - ic ** 2, 1e-9))
    return ic, t, n


def _halves(sig, fwd, h):
    df = pd.concat([sig, fwd], axis=1).dropna()
    if len(df) < MIN_OBS:
        return np.nan, np.nan
    mid = len(df) // 2
    a = _ic(df.iloc[:mid, 0], df.iloc[:mid, 1], h)[0] if mid >= MIN_OBS // 2 else np.nan
    b = _ic(df.iloc[mid:, 0], df.iloc[mid:, 1], h)[0] if len(df) - mid >= MIN_OBS // 2 else np.nan
    return a, b


def _halves_ic(sig, fwd, h):
    # _ic enforces MIN_OBS on the whole frame; halves need a looser floor
    global MIN_OBS
    keep, MIN_OBS = MIN_OBS, MIN_OBS // 2
    try:
        return _halves(sig, fwd, h)
    finally:
        MIN_OBS = keep


def forward_returns(eq, h):
    lr = np.log(eq.ffill(limit=2))
    return lr.shift(-h) - lr


def run(ds):
    eq = ds["equities"]
    comp = ds["composite"]
    compZ = ds["compZ"]
    meta = ds["meta"]
    fwd = {h: forward_returns(eq, h) for h in HORIZONS}
    world = {h: fwd[h].mean(axis=1, skipna=True).where(fwd[h].notna().sum(axis=1) >= 4) for h in HORIZONS}

    rows = []
    def add(region, key, label, sig, target_name, targets):
        rec = {"region": region, "signal": key, "label": label, "target": target_name}
        for h in HORIZONS:
            ic, t, n = _ic(sig, targets[h], h)
            rec[f"ic_{h}w"], rec[f"t_{h}w"], rec[f"n_{h}w"] = ic, t, n
        a, b = _halves_ic(sig, targets[13], 13)
        rec["ic13_first_half"], rec["ic13_second_half"] = a, b
        df = pd.concat([sig, targets[13]], axis=1).dropna()
        rec["ret13_when_easy"] = df[df.iloc[:, 0] > 0].iloc[:, 1].mean() * 100 if len(df) else np.nan
        rec["ret13_when_tight"] = df[df.iloc[:, 0] < 0].iloc[:, 1].mean() * 100 if len(df) else np.nan
        rec["start"] = df.index[0].strftime("%Y-%m") if len(df) else None
        rows.append(rec)

    for r in C.REGIONS:
        if r == "GLOBAL":
            targets, tname = world, "World (equal-weight)"
        elif r in eq.columns:
            targets, tname = {h: fwd[h][r] for h in HORIZONS}, r
        else:
            continue
        if r in comp.columns:
            add(r, "COMPOSITE", "Composite score", comp[r], tname, targets)
        for k, m in meta.items():
            if m["region"] == r and k in compZ.columns:
                add(r, k, k, compZ[k], tname, targets)
    res = pd.DataFrame(rows)
    res["verdict"] = res.apply(verdict, axis=1)
    return res


def verdict(r):
    ic, t = r["ic_13w"], r["t_13w"]
    a, b = r["ic13_first_half"], r["ic13_second_half"]
    if pd.isna(ic):
        return "too little history"
    stable = pd.notna(a) and pd.notna(b) and np.sign(a) == np.sign(b) == np.sign(ic)
    if ic >= 0.10 and t >= 2 and stable:
        return "useful"
    if ic >= 0.05 and stable:
        return "weak but consistent"
    if ic <= -0.10 and t <= -2 and stable:
        return "works in reverse"
    return "no reliable link"


if __name__ == "__main__":
    import pickle, sys
    if len(sys.argv) > 1:
        ds = pickle.load(open(sys.argv[1], "rb"))
    else:
        import tracker as T
        ds = T.build_dataset()
    res = run(ds)
    res.to_csv("backtest_results.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    cols = ["region", "signal", "ic_4w", "ic_13w", "t_13w", "ic_26w", "ic13_first_half",
            "ic13_second_half", "ret13_when_easy", "ret13_when_tight", "start", "verdict"]
    print(res[cols].round(2).to_string(index=False))
