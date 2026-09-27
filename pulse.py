"""
pulse.py - the DAILY pulse: a fast read on US funding and market stress that
sits beside the weekly regional composites (it never feeds them).

Inputs (config.PULSE): a daily estimate of US net liquidity, the SOFR - IORB
funding spread, credit spreads, the 10y real yield, VIX, MOVE and the dollar.
Each is z-scored on a rolling business-day window and signed so + = easier,
then blended into one pulse score.

Daily net liquidity = Fed total assets (WALCL, weekly, carried forward)
                    - TGA (daily, US Treasury Daily Treasury Statement;
                           falls back to the weekly FRED WDTGAL if the Treasury
                           API is unreachable)
                    - RRP (RRPONTSYD, daily).   All in $bn.

Run:  python pulse.py      (prints the latest pulse table)
"""
import json
import urllib.request

import numpy as np
import pandas as pd

import config as C
import tracker as T

DTS_URL = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/"
           "accounting/dts/operating_cash_balance")


# ------------------------------------------------------------------ fetch ---
def fetch_tga_daily(start="2021-10-01"):
    """Daily TGA closing balance in $bn from the Daily Treasury Statement.
    Returns an empty Series if the API cannot be reached."""
    q = (f"?filter=record_date:gte:{start}"
         "&fields=record_date,account_type,open_today_bal,close_today_bal"
         "&sort=record_date&page%5Bsize%5D=10000")
    try:
        with urllib.request.urlopen(DTS_URL + q, timeout=30) as r:
            rows = json.load(r)["data"]
    except Exception as e:
        print(f"  DTS  FAIL  daily TGA ({e.__class__.__name__}: {e}) - using weekly WDTGAL")
        return pd.Series(dtype=float)
    out = {}
    for row in rows:
        if "closing balance" not in row.get("account_type", "").lower():
            continue
        # the closing-balance row carries its value in open_today_bal on newer
        # statements and in close_today_bal on older ones; take whichever is filled
        for col in ("close_today_bal", "open_today_bal"):
            v = row.get(col)
            if v not in (None, "", "null"):
                try:
                    out[pd.Timestamp(row["record_date"])] = float(v) / 1000.0   # $mn -> $bn
                    break
                except ValueError:
                    pass
    s = pd.Series(out).sort_index()
    print(f"  DTS  ok    daily TGA          {len(s)} obs")
    return s


def _bdays(s):
    """Align to a business-day grid; carry forward at most PULSE_STALE_DAYS."""
    return s.resample("B").last().ffill(limit=C.PULSE_STALE_DAYS)


def derived_daily(fred):
    out, notes = {}, {}
    # SOFR - IORB (IOER before Jul-2021, when IORB replaced it)
    try:
        floor = pd.concat([fred["IOER"].dropna(), fred["IORB"].dropna()]).sort_index()
        floor = floor[~floor.index.duplicated(keep="last")]
        sofr = fred["SOFR"].dropna()
        df = pd.concat([sofr, floor.reindex(sofr.index, method="ffill")], axis=1).dropna()
        out["sofr_iorb"] = df.iloc[:, 0] - df.iloc[:, 1]
    except Exception as e:
        print("  pulse sofr_iorb FAILED:", e)
    # daily net liquidity ($bn)
    try:
        assets = fred["WALCL"] / 1000.0                     # $mn -> $bn, weekly (Wed)
        rrp = fred["RRPONTSYD"]                             # already $bn, daily
        tga_w = fred["WDTGAL"] / 1000.0                     # $mn -> $bn, weekly (Wed)
        tga_d = fetch_tga_daily()
        if len(tga_d):
            tga = pd.concat([tga_w[tga_w.index < tga_d.index[0]], tga_d]).sort_index()
            notes["tga"] = "daily"
        else:
            tga = tga_w
            notes["tga"] = "weekly"
        idx = pd.bdate_range(max(assets.index[0], tga.index[0], rrp.index[0]),
                             max(assets.index[-1], tga.index[-1], rrp.index[-1]))
        a = assets.reindex(idx.union(assets.index)).ffill().reindex(idx)
        t = tga.reindex(idx.union(tga.index)).ffill().reindex(idx)
        r = rrp.reindex(idx.union(rrp.index)).ffill(limit=C.PULSE_STALE_DAYS).reindex(idx)
        out["net_liq_daily"] = (a - t - r).dropna()
        notes["tga_last"] = tga.index[-1].strftime("%Y-%m-%d")
        notes["assets_last"] = assets.index[-1].strftime("%Y-%m-%d")
    except Exception as e:
        print("  pulse net_liq_daily FAILED:", e)
    return out, notes


def zscore_daily(s):
    w = C.PULSE_Z_DAYS
    m = s.rolling(w, min_periods=w // 3).mean()
    sd = s.rolling(w, min_periods=w // 3).std()
    return ((s - m) / sd).clip(-3, 3)


# --------------------------------------------------------------- assemble ----
def build_pulse(start="2016-01-01", fred=None, mkt=None):
    """Return a dict: raw (display units), signed z per input, pulse score, meta."""
    fred_ids = sorted({p["id"] for p in C.PULSE if p["source"] == "fred"}
                      | {"SOFR", "IORB", "IOER", "WALCL", "WDTGAL", "RRPONTSYD"})
    yf_ids = sorted({p["id"] for p in C.PULSE if p["source"] == "yf"})
    fred = dict(fred or {})
    mkt = dict(mkt or {})
    need_f = [i for i in fred_ids if i not in fred]
    need_y = [i for i in yf_ids if i not in mkt]
    if need_f:
        print("Fetching FRED (pulse) ..."); fred.update(T.fetch_fred(need_f, start))
    if need_y:
        print("Fetching market (pulse) ..."); mkt.update(T.fetch_yf(need_y, start))
    deriv, notes = derived_daily(fred)

    cutoff = pd.Timestamp.today().normalize()
    raw, z, meta = {}, {}, {}
    for p in C.PULSE:
        src = {"fred": fred, "yf": mkt, "derived": deriv}[p["source"]]
        s = src.get(p["id"])
        if s is None or len(s.dropna()) < 250:
            print(f"  skip pulse (no/insufficient data): {p['key']}")
            continue
        s = s.dropna()
        s = s[s.index <= cutoff] * p["scale"]
        s.index = pd.DatetimeIndex(s.index).normalize()
        s = s[~s.index.duplicated(keep="last")]
        raw[p["key"]] = s
        scored = s.diff(20) if p["score"] == "chg20" else s
        z[p["key"]] = zscore_daily(_bdays(scored).dropna()) * p["sign"]
        meta[p["key"]] = p

    zdf = pd.DataFrame(z)
    zdf = zdf[zdf.index <= cutoff].ffill(limit=C.PULSE_STALE_DAYS)
    if zdf.empty:
        return None
    w = np.array([meta[k]["weight"] for k in zdf.columns], dtype=float)
    vals = zdf.values
    mask = ~np.isnan(vals)
    num = np.nansum(np.where(mask, vals * w, 0.0), axis=1)
    den = np.where(mask, w, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = pd.Series(np.where(den > 0, num / den, np.nan), index=zdf.index).dropna()
    return dict(raw=raw, z=zdf, score=score, meta=meta, notes=notes)


def attribution(pl, top=2):
    """Latest signed z per input (easing / tightening) and the biggest 5-day movers."""
    zl = pl["z"].loc[pl["score"].index[-1]].dropna()
    ch5 = (pl["z"].iloc[-1] - pl["z"].iloc[-6]).dropna() if len(pl["z"]) > 5 else pd.Series(dtype=float)
    ease = [(k, v) for k, v in zl.sort_values(ascending=False).items() if v > 0.25][:top]
    tight = [(k, v) for k, v in zl.sort_values().items() if v < -0.25][:top]
    movers = [(k, v) for k, v in ch5.reindex(ch5.abs().sort_values(ascending=False).index).items()
              if abs(v) >= 0.3][:top]
    return dict(easing=ease, tightening=tight, movers=movers)


if __name__ == "__main__":
    pl = build_pulse()
    if pl is None:
        raise SystemExit("no pulse data")
    last = pl["score"].index[-1]
    print(f"\nDaily pulse {last.date()}: {pl['score'].iloc[-1]:+.2f}  "
          f"(5d change {pl['score'].iloc[-1] - pl['score'].iloc[-6]:+.2f})  TGA source: {pl['notes'].get('tga')}")
    for k, s in pl["raw"].items():
        print(f"  {k:14s} {s.index[-1].date()}  {s.iloc[-1]:>10.2f}  z {pl['z'][k].iloc[-1]:+.2f}")
