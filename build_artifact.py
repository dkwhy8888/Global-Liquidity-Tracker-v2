"""
build_artifact.py - build the Claude-artifact version of the dashboard.

Runs the same engine as the Streamlit app (tracker.py + narrative.py), packs the
results into JSON and injects them into artifact/template.html, writing
artifact/liquidity_tracker.html. Publish that file to refresh the artifact.

Run:  python build_artifact.py        (needs FRED_API_KEY)
"""
import json
import math
import os
from datetime import datetime

import pandas as pd

import config as C
import backtest as B
import narrative as N
import pulse as P
import tracker as T

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "artifact", "template.html")
OUT = os.path.join(HERE, "artifact", "liquidity_tracker.html")

REGION_NAMES = {"GLOBAL": "Global", "US": "United States", "EU": "Euro area",
                "JP": "Japan", "HK": "Hong Kong", "SG": "Singapore",
                "KR": "South Korea", "CN": "China"}
EQUITY_NAMES = {"US": "S&P 500", "EU": "Euro Stoxx 50", "JP": "Nikkei 225",
                "HK": "Hang Seng", "SG": "Straits Times Index", "KR": "KOSPI",
                "CN": "CSI 300"}
LABELS = {
    "fed_net_liquidity": "Fed net liquidity", "g3_cb_assets_usd": "G3 central-bank assets",
    "g4_cb_assets_usd": "G4 central-bank assets (USD)", "global_m2_usd": "Global M2 (USD)",
    "credit_baa": "Baa corporate spread",
    "us_hy_oas": "US high-yield spread", "us_ig_oas": "US investment-grade spread",
    "nfci": "Chicago Fed financial conditions", "vix": "VIX (equity volatility)",
    "move": "MOVE (bond volatility)", "broad_usd": "Broad US dollar index",
    "cn_credit_impulse": "China credit impulse", "us_m2_yoy": "US M2 money supply",
    "us_fed_funds": "Fed funds rate", "us_2s10s": "US 2s10s yield curve",
    "ecb_assets": "ECB balance sheet", "eu_m3_yoy": "Euro-area M3", "eurusd": "EUR/USD",
    "boj_assets": "BOJ balance sheet", "jp_m2_yoy": "Japan M2", "usdjpy": "USD/JPY",
    "hk_agg_balance": "HK Aggregate Balance", "hibor_3m": "3-month HIBOR", "usdhkd": "USD/HKD",
    "sora": "SORA", "sg_m2_yoy": "Singapore M2", "usdsgd": "USD/SGD",
    "kr_base_rate": "BOK base rate", "kr_m2_yoy": "Korea M2", "usdkrw": "USD/KRW",
    "cn_m2_yoy": "China M2", "cn_tsf_yoy": "China total social financing",
    "cn_7d_repo": "China 7-day repo", "cn_rrr": "China reserve requirement", "usdcny": "USD/CNY",
    "us_bank_credit": "US bank credit", "us_reserves_gdp": "US reserves / GDP",
    "us_sofr_iorb": "SOFR − IORB", "hk_m2_yoy": "Hong Kong M2", "kr_3m_rate": "Korea 3m interbank rate",
    "cn_pboc_assets": "PBOC balance sheet", "cn_loan_growth": "China loan growth",
    "cn_3m_rate": "China 3m interbank rate",
}
# unit of the series the model actually scores; yoy transforms are always "% YoY"
LEVEL_UNITS = {"credit_baa": "%", "us_hy_oas": "%", "us_ig_oas": "%", "nfci": "index", "vix": "pts",
               "move": "pts", "broad_usd": "index", "us_fed_funds": "%", "us_2s10s": "pp",
               "hk_agg_balance": "HK$bn", "hibor_3m": "%", "sora": "%", "kr_base_rate": "%",
               "cn_7d_repo": "%", "cn_rrr": "%", "cn_credit_impulse": "% GDP",
               "sg_m2_yoy": "% YoY", "kr_m2_yoy": "% YoY", "cn_m2_yoy": "% YoY",
               "cn_tsf_yoy": "% YoY", "us_reserves_gdp": "% GDP", "us_sofr_iorb": "bp",
               "kr_3m_rate": "%", "cn_3m_rate": "%", "cn_loan_growth": "% YoY"}   # FX pairs: no unit
SOURCE = {"fred": "FRED", "yf": "Yahoo", "tv": "TradingView", "manual": "Manual", "derived": "Derived"}
PULSE_LABELS = {"net_liq_daily": "US net liquidity (daily est.)", "sofr_iorb": "SOFR − IORB",
                "hy_oas": "US high-yield spread", "ig_oas": "US investment-grade spread",
                "real_10y": "US 10-year real yield", "vix": "VIX", "move": "MOVE",
                "dxy": "US dollar index (DXY)"}
PULSE_IDS = {"net_liq_daily": "WALCL − TGA − RRPONTSYD", "sofr_iorb": "SOFR − IORB"}
DERIVED_IDS = {"fed_net_liq": "WALCL − WTREGEN − RRPONTSYD",
               "g3_assets": "WALCL + ECBASSETSW + JPNASSETS",
               "g4_assets": "G3 + PBOC (CNCBBS)",
               "global_m2": "M2SL + EUM3 + JPM2 + CNM2",
               "reserves_gdp": "WRESBAL / GDP", "sofr_iorb_w": "SOFR − IORB"}


def _num(v, nd):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def _arr(s, idx, nd):
    return [_num(v, nd) for v in s.reindex(idx).values]


def _corr_word(c):
    if c is None:
        return None
    a = abs(c)
    return "negligible" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong"


def pulse_payload():
    """Daily pulse block for the page, or None if it could not be built."""
    try:
        pl = P.build_pulse()
    except Exception as e:
        print("  pulse FAILED:", e)
        return None
    if pl is None or len(pl["score"]) < 10:
        return None
    score = pl["score"]
    last = score.index[-1]
    cut = last - pd.Timedelta(days=C.PULSE_SHOW_DAYS)
    idx = score.index[score.index >= cut]
    chg5 = float(score.iloc[-1] - score.iloc[-6])
    att = P.attribution(pl)
    items = []
    for k, m in pl["meta"].items():
        raw = pl["raw"][k]
        rb = raw.resample("B").last().ffill(limit=C.PULSE_STALE_DAYS)
        zs = pl["z"][k]
        def chg(n):
            return _num(raw.iloc[-1] - raw.iloc[-1 - n], 2) if len(raw) > n else None
        items.append({
            "key": k, "label": PULSE_LABELS.get(k, k),
            "plain": N.PULSE_PHRASES.get(k, (k,))[0],
            "id": PULSE_IDS.get(k, m["id"]),
            "source": SOURCE.get(m["source"], m["source"]),
            "unit": m["unit"], "sign": m["sign"], "weight": m["weight"], "score": m["score"],
            "lastObs": raw.index[-1].strftime("%Y-%m-%d"),
            "last": _num(raw.iloc[-1], 2), "d1": chg(1), "d5": chg(5),
            "z": _num(zs.iloc[-1], 2) if len(zs) else None,
            "raw": _arr(rb, idx, 2), "zs": _arr(zs, idx, 2),
        })
    return {
        "asOf": last.strftime("%Y-%m-%d"),
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "score": _arr(score, idx, 3),
        "now": _num(score.iloc[-1], 2), "chg5": _num(chg5, 2),
        "narrative": N.pulse_narrative(float(score.iloc[-1]), chg5, att),
        "tga": pl["notes"].get("tga"), "tgaLast": pl["notes"].get("tga_last"),
        "zDays": C.PULSE_Z_DAYS,
        "items": items,
    }


def changes_payload(ds):
    """Regime changes: this week vs last week, and vs 4 weeks ago."""
    comp, mom = ds["composite"].dropna(how="all"), ds["momentum"]
    out = []
    for r in [c for c in C.REGIONS if c in comp.columns]:
        def reg(i):
            if len(comp) < abs(i):
                return "n/a"
            return T.regime_label(comp[r].iloc[i], mom[r].reindex(comp.index).iloc[i])
        now, wk, mo = reg(-1), reg(-2), reg(-5)
        if now != wk:
            out.append({"region": r, "name": REGION_NAMES.get(r, r), "from": wk, "to": now, "when": "this week"})
        elif now != mo:
            out.append({"region": r, "name": REGION_NAMES.get(r, r), "from": mo, "to": now, "when": "in the last 4 weeks"})
    return out


def backtest_payload(ds):
    try:
        res = B.run(ds)
    except Exception as e:
        print("  backtest FAILED:", e)
        return None
    rows = []
    for _, r in res.iterrows():
        rows.append({
            "region": r["region"], "key": r["signal"],
            "label": "Composite score" if r["signal"] == "COMPOSITE" else LABELS.get(r["signal"], r["signal"]),
            "ic4": _num(r["ic_4w"], 2), "ic13": _num(r["ic_13w"], 2), "ic26": _num(r["ic_26w"], 2),
            "t13": _num(r["t_13w"], 1), "h1": _num(r["ic13_first_half"], 2), "h2": _num(r["ic13_second_half"], 2),
            "retEasy": _num(r["ret13_when_easy"], 1), "retTight": _num(r["ret13_when_tight"], 1),
            "start": r["start"], "verdict": r["verdict"], "target": r["target"],
        })
    return {"rows": rows, "horizons": list(B.HORIZONS)}


def build():
    ds = T.build_dataset()
    summ = T.summary_table(ds)
    comp = ds["composite"].dropna(how="all")
    idx = comp.index
    as_of = idx[-1]
    mom_col, corr_col = f"mom_{C.MOM_WEEKS}w", f"corr_lead{C.LEAD_WEEKS}w"
    regions = [r for r in C.REGIONS if r in comp.columns]

    summary = []
    for r in regions:
        row = summ.loc[r]
        att = T.region_attribution(ds, r, top=3)
        rkeys = [k for k in ds["meta"] if ds["meta"][k]["region"] == r]
        fx_only = bool(rkeys) and {ds["meta"][k]["bucket"] for k in rkeys} <= {"fx"}
        c = _num(row[corr_col], 2)
        summary.append({
            "region": r, "name": REGION_NAMES.get(r, r),
            "z": _num(row["liquidity_z"], 2), "mom": _num(row[mom_col], 2),
            "regime": row["regime"], "corr": c, "corrWord": _corr_word(c),
            "equity": EQUITY_NAMES.get(r) if r in ds["equities"].columns else None,
            "fxOnly": fx_only, "nInputs": len(rkeys),
            "narrative": N.region_narrative(r, row["regime"], row["liquidity_z"], row[mom_col],
                                            att, mom_weeks=C.MOM_WEEKS, fx_only=fx_only),
        })

    comps = []
    for k, m in ds["meta"].items():
        raw = ds["rawW"][k].dropna()
        inp = T.apply_transform(raw, m["transform"]).dropna()
        zser = ds["compZ"][k]
        last_obs = raw.index[-1] if len(raw) else None
        weeks_old = (as_of - last_obs).days // 7 if last_obs is not None else None
        status = ("live" if weeks_old is not None and weeks_old <= 2 else
                  "lagged" if weeks_old is not None and weeks_old <= C.STALE_WEEKS else
                  "discontinued")
        z_now = zser.loc[:as_of].dropna()
        chg = inp.iloc[-1] - inp.iloc[-1 - C.MOM_WEEKS] if len(inp) > C.MOM_WEEKS else None
        comps.append({
            "key": k, "region": m["region"], "label": LABELS.get(k, k),
            "plain": N.PHRASES.get(k, (k,))[0],
            "source": SOURCE.get(m["source"], m["source"]),
            "id": DERIVED_IDS.get(m["id"], m["id"]),
            "transform": m["transform"],
            "unit": "% YoY" if m["transform"] == "yoy" else LEVEL_UNITS.get(k, ""),
            "sign": m["sign"], "weight": m["weight"],
            "lastObs": last_obs.strftime("%Y-%m-%d") if last_obs is not None else None,
            "status": status,
            "input": _num(inp.iloc[-1], 3) if len(inp) else None,
            "inputChg": _num(chg, 3),
            "zLast": _num(z_now.iloc[-1], 2) if len(z_now) and z_now.index[-1] == as_of else None,
            "z": _arr(zser, idx, 3),
            "inp": _arr(inp, idx, 3),
        })

    eq = ds["equities"]
    equities = {r: _arr(eq[r].ffill(limit=2), idx, 2) for r in regions if r in eq.columns}

    fed = None
    fb = ds["fed_bs"]
    if fb is not None and not fb.empty:
        fb = fb[fb.index >= idx[0]]
        fed = {"dates": [d.strftime("%Y-%m-%d") for d in fb.index],
               "series": {c: [_num(v, 1) for v in fb[c].values] for c in fb.columns}}

    payload = {
        "asOf": as_of.strftime("%Y-%m-%d"),
        "builtAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "params": {"zWindow": C.Z_WINDOW, "momWeeks": C.MOM_WEEKS,
                   "leadWeeks": C.LEAD_WEEKS, "staleWeeks": C.STALE_WEEKS},
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "regions": regions,
        "summary": summary,
        "composite": {r: _arr(comp[r], idx, 3) for r in regions},
        "equities": equities,
        "components": comps,
        "fed": fed,
        "pulse": pulse_payload(),
        "changes": changes_payload(ds),
        "backtest": backtest_payload(ds),
    }
    for c in payload["changes"]:
        print(f"CHANGE: {c['name']} moved from {c['from']} to {c['to']} {c['when']}")
    if not payload["changes"]:
        print("CHANGE: no regime changes in the last 4 weeks")
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    if html.count("@@DATA@@") != 1:
        raise SystemExit("template.html must contain the @@DATA@@ placeholder exactly once")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html.replace("@@DATA@@", blob))
    print(f"\nwrote {OUT}  ({len(blob) / 1024:,.0f} KB of data, as of {payload['asOf']})")
    return payload


if __name__ == "__main__":
    build()
