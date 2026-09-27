"""
tracker.py - fetch, compute, and export the multi-region liquidity tracker.

Run:   python tracker.py        ->  writes liquidity_tracker.xlsx + prints the dashboard table
Needs: env var FRED_API_KEY    ->  free key at https://fred.stlouisfed.org/docs/api/api_key.html

Pipeline:  fetch (FRED + yfinance + manual CSV)
        -> transform (yoy for stocks, level for prices/rates/spreads)
        -> z-score (rolling 3y) and apply sign so + = easier liquidity
        -> weighted region composite -> regime -> lead/lag corr vs equity index
"""
import os
import sys
import numpy as np
import pandas as pd

import config as C
import market_fallback as MF


# ------------------------------------------------------------------ fetch ---
def _fred_client():
    """Return a FRED client, or None if no key is set (dashboard then runs
    on yfinance + manual data only, instead of hard-exiting)."""
    try:
        from fredapi import Fred
    except ImportError:
        sys.exit("Missing dependency: pip install fredapi")
    key = os.environ.get(C.FRED_API_KEY_ENV)
    if not key:
        print(f"  WARNING: {C.FRED_API_KEY_ENV} not set - skipping all FRED series "
              f"(macro/monetary layer). Set {C.FRED_API_KEY_ENV} to enable. "
              f"(Windows CMD:  set {C.FRED_API_KEY_ENV}=your_key)")
        return None
    return Fred(api_key=key)


def fetch_fred(ids, start="2010-01-01"):
    fred = _fred_client()
    out = {}
    if fred is None:
        return out
    for sid in ids:
        try:
            s = fred.get_series(sid, observation_start=start).dropna()
            out[sid] = s
            print(f"  FRED ok    {sid:18s} {len(s)} obs")
        except Exception as e:
            print(f"  FRED FAIL  {sid:18s} {e}")
    return out


def fetch_yf(tickers, start="2010-01-01"):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Missing dependency: pip install yfinance")
    out = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, progress=False, auto_adjust=False)
            s = df["Close"].dropna()
            if isinstance(s, pd.DataFrame):       # multiindex guard (newer yfinance)
                s = s.iloc[:, 0]
            if len(s):
                out[t] = s
                print(f"  YF   ok    {t:12s} {len(s)} obs")
            else:
                print(f"  YF   EMPTY {t:12s}")
        except Exception as e:
            print(f"  YF   FAIL  {t:12s} {e}")
        # yfinance gaps (e.g. 000300.SS): use the TradingView-sourced fallback file
        if len(out.get(t, ())) < 30:
            fb = MF.load(t)
            if fb is not None and len(fb) >= 30:
                out[t] = fb[fb.index >= pd.Timestamp(start)]
                print(f"  TV   ok    {t:12s} {len(out[t])} obs (market_fallback, to {fb.index[-1].date()})")
    return out


def fetch_manual(path="manual_inputs.csv"):
    if not os.path.exists(path):
        print("  manual_inputs.csv not found - manual (HK/SG/KR/CN) series will be blank")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    print(f"  manual ok  {df.shape[1]} cols, {len(df)} rows")
    return df


# -------------------------------------------------------------- transforms ---
def to_weekly(s):
    return s.resample(C.RESAMPLE).last().ffill()


def yoy(s):
    s = to_weekly(s)
    return (s / s.shift(52) - 1.0) * 100.0


def chg13(s):
    s = to_weekly(s)
    return s - s.shift(C.MOM_WEEKS)


def apply_transform(s, how):
    if how == "yoy":
        return yoy(s)
    if how == "chg13":
        return chg13(s)
    return to_weekly(s)


def zscore(s, win=None):
    win = win or C.Z_WINDOW
    m = s.rolling(win, min_periods=win // 3).mean()
    sd = s.rolling(win, min_periods=win // 3).std()
    return ((s - m) / sd).clip(-3, 3)


# ----------------------------------------------------------------- derived ---
def derived_series(fred, mkt):
    """Series that are computed, not pulled directly."""
    out = {}
    # Fed net liquidity = Fed assets - TGA - RRP   (all converted to USD millions)
    try:
        walcl = to_weekly(fred["WALCL"])              # USD mn
        tga   = to_weekly(fred["WTREGEN"])            # USD mn
        rrp   = to_weekly(fred["RRPONTSYD"]) * 1000.0 # USD bn -> mn
        out["fed_net_liq"] = pd.concat([walcl, tga, rrp], axis=1).ffill().dropna() \
                               .apply(lambda r: r.iloc[0] - r.iloc[1] - r.iloc[2], axis=1)
    except Exception as e:
        print("  derived fed_net_liq FAILED:", e)
    # G3 central-bank assets in USD mn = Fed + ECB*EURUSD + BOJ(100MnYen)*100/USDJPY
    try:
        fed = to_weekly(fred["WALCL"])
        eur = to_weekly(mkt["EURUSD=X"])
        ecb = to_weekly(fred["ECBASSETSW"]) * eur
        jpy = to_weekly(mkt["USDJPY=X"])
        boj = to_weekly(fred["JPNASSETS"]) * 100.0 / jpy
        out["g3_assets"] = pd.concat([fed, ecb, boj], axis=1).ffill().dropna().sum(axis=1)
    except Exception as e:
        print("  derived g3_assets FAILED (skips ECB/BOJ if ids missing):", e)
    return out


def fed_balance_sheet(fred):
    """US Fed balance-sheet components in $bn, aligned weekly.
    Net Liquidity = Fed Total Assets − TGA − RRP.  Units confirmed from FRED:
    WALCL/WTREGEN/WRESBAL are $ millions (÷1,000 → $bn); RRPONTSYD is already $bn."""
    try:
        assets   = to_weekly(fred["WALCL"])    / 1000.0     # $mn -> $bn
        tga      = to_weekly(fred["WTREGEN"])  / 1000.0     # $mn -> $bn
        reserves = to_weekly(fred["WRESBAL"])  / 1000.0     # $mn -> $bn
        rrp      = to_weekly(fred["RRPONTSYD"])             # already $bn
        df = pd.concat({"Fed Total Assets": assets, "Reserve Balances": reserves,
                        "TGA": tga, "RRP": rrp}, axis=1).ffill().dropna()
        df["Net Liquidity"] = df["Fed Total Assets"] - df["TGA"] - df["RRP"]
        return df[["Net Liquidity", "Fed Total Assets", "Reserve Balances", "TGA", "RRP"]]
    except Exception as e:
        print("  fed_balance_sheet FAILED (needs WALCL/WTREGEN/WRESBAL/RRPONTSYD):", e)
        return pd.DataFrame()


# --------------------------------------------------------------- assemble ----
def _series_for(m, fred, mkt, man, deriv):
    src = m["source"]
    if src == "fred":
        return fred.get(m["id"])
    if src == "yf":
        return mkt.get(m["id"])
    if src == "derived":
        return deriv.get(m["id"])
    if src == "manual" and not man.empty and m["id"] in man.columns:
        return man[m["id"]].dropna()
    return None


def build_dataset(start="2010-01-01"):
    fred_ids = sorted({m["id"] for m in C.METRICS if m["source"] == "fred"}
                      | {"WALCL", "WTREGEN", "RRPONTSYD", "WRESBAL", "ECBASSETSW", "JPNASSETS"})
    yf_ids   = sorted({m["id"] for m in C.METRICS if m["source"] == "yf"}
                      | {"EURUSD=X", "USDJPY=X"})

    print("Fetching FRED ...");    fred = fetch_fred(fred_ids, start)
    print("Fetching market ...");  mkt  = fetch_yf(yf_ids, start)
    print("Fetching manual ...");  man  = fetch_manual()
    deriv = derived_series(fred, mkt)
    fed_bs = fed_balance_sheet(fred)

    comp_z, raw_w, equities, meta = {}, {}, {}, {}
    for m in C.METRICS:
        s = _series_for(m, fred, mkt, man, deriv)
        if s is None or len(s.dropna()) < 30:
            print(f"  skip (no/insufficient data): {m['key']}")
            continue
        s = s.dropna()
        raw_w[m["key"]] = to_weekly(s)
        if m["bucket"] == "equity_index":
            equities[m["region"]] = to_weekly(s)
            continue
        comp_z[m["key"]] = zscore(apply_transform(s, m["transform"])) * m["sign"]
        meta[m["key"]] = m

    # Drop the still-open week: FX quotes that tick over the weekend open a W-FRI bin
    # dated in the future, where the FRED series have no value yet.
    cutoff = pd.Timestamp.today().normalize()
    compZ = pd.DataFrame(comp_z)
    compZ = compZ[compZ.index <= cutoff]
    # Carry each component's latest reading forward until its next release (monthly
    # series publish with a lag), capped at STALE_WEEKS so discontinued series drop
    # out of the composite instead of freezing at an old value.
    compZ = compZ.ffill(limit=C.STALE_WEEKS)
    raw_df = pd.DataFrame(raw_w)
    raw_df = raw_df[raw_df.index <= cutoff]
    eq_df = pd.DataFrame(equities)
    eq_df = eq_df[eq_df.index <= cutoff]
    if not fed_bs.empty:
        fed_bs = fed_bs[fed_bs.index <= cutoff]

    # region composite = weighted mean of available signed-z components
    region_comp = {}
    for r in C.REGIONS:
        cols = [k for k, mm in meta.items() if mm["region"] == r]
        if not cols:
            continue
        w = np.array([meta[k]["weight"] for k in cols], dtype=float)
        vals = compZ[cols].values
        mask = ~np.isnan(vals)
        num = np.nansum(np.where(mask, vals * w, 0.0), axis=1)
        den = np.where(mask, w, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            region_comp[r] = pd.Series(np.where(den > 0, num / den, np.nan), index=compZ.index)

    comp = pd.DataFrame(region_comp)
    mom = comp - comp.shift(C.MOM_WEEKS)
    return dict(compZ=compZ, rawW=raw_df, equities=eq_df, composite=comp, momentum=mom,
                meta=meta, fed_bs=fed_bs)


# ----------------------------------------------------------- attribution -----
def region_attribution(ds, region, top=2):
    """Data-driven 'why': rank a region's signed-z components by their contribution
    to the composite level (z × weight / Σweight) and to its MOM_WEEKS momentum
    (Δ signed-z over the window, same weighting). Returns the top easing/tightening
    drivers of the *level* and the top improving/fading drivers of the *change*.
    Everything is computed from the data — no editorial narrative."""
    meta, compZ = ds["meta"], ds["compZ"]
    keys = [k for k in meta if meta[k]["region"] == region and k in compZ.columns]
    latest = {k: compZ[k].iloc[-1] for k in keys if pd.notna(compZ[k].iloc[-1])}
    if not latest:
        return None
    w = {k: meta[k]["weight"] for k in latest}
    wsum = sum(w.values()) or 1.0
    contrib = {k: latest[k] * w[k] / wsum for k in latest}          # impact on level
    dz = {}
    for k in latest:
        s = compZ[k].dropna()
        if len(s) > C.MOM_WEEKS:
            dz[k] = (s.iloc[-1] - s.iloc[-1 - C.MOM_WEEKS]) * w[k] / wsum
    by_lvl = sorted(contrib.items(), key=lambda x: x[1])
    by_mom = sorted(dz.items(), key=lambda x: x[1])
    return {
        "easing":     [(k, latest[k], c) for k, c in reversed(by_lvl) if c > 0][:top],
        "tightening": [(k, latest[k], c) for k, c in by_lvl if c < 0][:top],
        "improving":  [(k, d) for k, d in reversed(by_mom) if d > 0][:top],
        "fading":     [(k, d) for k, d in by_mom if d < 0][:top],
    }


# ------------------------------------------------------------ regime/report --
def regime_label(level, momentum):
    if pd.isna(level) or pd.isna(momentum):
        return "n/a"
    easy, impr = level >= 0, momentum >= 0
    if easy and impr:           return "Easy & Improving"
    if easy and not impr:       return "Easy & Fading"
    if (not easy) and impr:     return "Tight & Improving"
    return "Tight & Deteriorating"


def summary_table(ds):
    comp, mom, eq = ds["composite"], ds["momentum"], ds["equities"]
    rows = []
    for r in comp.columns:
        lvl, mo = comp[r].iloc[-1], mom[r].iloc[-1]
        corr = np.nan
        if r in eq.columns:                       # composite_t vs index fwd return
            idx = eq[r].reindex(comp.index).ffill()
            fwd = idx.shift(-C.LEAD_WEEKS) / idx - 1.0
            j = pd.concat([comp[r], fwd], axis=1).dropna()
            if len(j) > 52:
                corr = j.iloc[:, 0].corr(j.iloc[:, 1])
        rows.append({
            "region": r,
            "liquidity_z": round(float(lvl), 2) if pd.notna(lvl) else np.nan,
            f"mom_{C.MOM_WEEKS}w": round(float(mo), 2) if pd.notna(mo) else np.nan,
            "regime": regime_label(lvl, mo),
            f"corr_lead{C.LEAD_WEEKS}w": round(float(corr), 2) if pd.notna(corr) else np.nan,
        })
    return pd.DataFrame(rows).set_index("region")


def write_excel(ds, path="liquidity_tracker.xlsx"):
    summ = summary_table(ds)
    with pd.ExcelWriter(path, engine="xlsxwriter") as xl:
        summ.to_excel(xl, sheet_name="Dashboard")
        ds["composite"].dropna(how="all").to_excel(xl, sheet_name="Composites")
        ds["compZ"].dropna(how="all").to_excel(xl, sheet_name="Component_Z")
        ds["rawW"].dropna(how="all").to_excel(xl, sheet_name="Data")
        ws = xl.sheets["Dashboard"]
        ws.conditional_format(1, 1, len(summ), 1, {
            "type": "3_color_scale",
            "min_color": "#F8696B", "mid_color": "#FFEB84", "max_color": "#63BE7B",
            "min_type": "num", "min_value": -2,
            "mid_type": "num", "mid_value": 0,
            "max_type": "num", "max_value": 2})
        ws.set_column(0, 0, 10)
        ws.set_column(1, 4, 17)
    print(f"\nwrote {path}")
    return path


if __name__ == "__main__":
    ds = build_dataset()
    write_excel(ds)
    print("\n=== Current liquidity dashboard ===")
    print(summary_table(ds).to_string())
