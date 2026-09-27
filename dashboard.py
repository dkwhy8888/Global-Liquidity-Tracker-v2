"""
dashboard.py - live, click-to-update Global Liquidity dashboard (charts + numbers).

Run:  streamlit run dashboard.py   (or double-click run_dashboard.bat)

Adds, on top of the charts:
- Section 3 "Current levels & changes": a snapshot table per region in the
  Value / Week / YoY style - pick which change periods to show, abs or %.
- Section 4 "Full data table": every weekly value, any date range you choose,
  for raw values / z-scores / composite scores, with CSV download.
"""
import os
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tracker as T
import config as C
import narrative as N
import pulse as P

st.set_page_config(page_title="Global Liquidity Tracker", layout="wide")

COLOR = {"Easy & Improving": "#1a9850", "Easy & Fading": "#91cf60",
         "Tight & Improving": "#fee08b", "Tight & Deteriorating": "#d73027", "n/a": "#999999"}

# RdYlGn cell shading (matplotlib) — − = tighter/red, + = easier/green
RYG = "RdYlGn"

# key -> region / bucket (covers equity indices too, not just composite inputs)
KEY_REGION = {m["key"]: m["region"] for m in C.METRICS}
HORIZONS = {"1W": 1, "4W (1M)": 4, "13W (3M)": 13, "26W (6M)": 26, "52W (1Y)": 52}

# ----------------------------------------------------- data + click refresh ---
if "refresh" not in st.session_state:
    st.session_state.refresh = 0


@st.cache_data(show_spinner=False)
def load(refresh_token):
    ds = T.build_dataset()
    return ds, T.summary_table(ds), datetime.now()


@st.cache_data(show_spinner=False)
def load_pulse(refresh_token):
    try:
        return P.build_pulse()
    except Exception as e:
        print("pulse FAILED:", e)
        return None


left, right = st.columns([1, 5])
with left:
    if st.button("🔄  Update now", type="primary", width="stretch"):
        st.session_state.refresh += 1
        st.cache_data.clear()
with right:
    st.title("Global Liquidity Tracker")

with st.spinner("Fetching latest H.4.1 + market data ..."):
    ds, summ, fetched_at = load(st.session_state.refresh)

st.caption(f"Data fetched: **{fetched_at:%Y-%m-%d %H:%M:%S}** (local) · "
           "press **Update now** to pull the freshest live data.")

if not os.environ.get(C.FRED_API_KEY_ENV):
    st.warning(
        "**FRED_API_KEY not set** — the macro/monetary layer (Fed/ECB/BOJ balance "
        "sheets, M2, credit spreads, net liquidity) is currently skipped, so regions "
        "are showing FX/equity/manual inputs only. Get a free key at "
        "https://fred.stlouisfed.org/docs/api/api_key.html , set it "
        "(`set FRED_API_KEY=your_key`), and press **Update now**.")

with st.expander("ℹ️  How to read this — why z-scores?"):
    st.markdown(
        "Most metrics here are shown as a **z-score**: how many standard deviations the "
        "series sits from its own **rolling 3-year average**. Two reasons:\n\n"
        "1. **Comparability & combinability.** Raw inputs live on wildly different scales — "
        "Fed assets in $trn, policy rates in %, credit spreads in bps, VIX in index points. "
        "You can't average those directly. Converting each to a z-score puts them on one "
        "common axis so they can be blended into a single regional **composite**.\n"
        "2. **Context vs. history.** z = +1.5 means *‘unusually easy vs the last 3 years’*; "
        "z = 0 means *‘about normal’*. That's the regime read at a glance.\n\n"
        "z-scores are **not more accurate** — they're a *normalisation* that trades actual "
        "units for comparability. The true levels are always one click away: the **‘Raw value "
        "(actual units)’** toggle in §3, the **$bn** Fed balance-sheet section, and the "
        "**‘Raw values’** view in the full data table. Sign is applied so **+ always = easier "
        "liquidity**, and z is clipped to ±3.")

LOOKBACK = {"1Y": 52, "3Y": 156, "5Y": 260, "Max": None}
win_choice = st.radio("Chart window", list(LOOKBACK.keys()), index=1, horizontal=True)
WEEKS = LOOKBACK[win_choice]


def clip(s):
    return s if WEEKS is None else s.tail(WEEKS)


def chg(s, n, pct):
    s = s.dropna()
    if len(s) <= n:
        return np.nan
    return (s.iloc[-1] / s.iloc[-1 - n] - 1) * 100 if pct else s.iloc[-1] - s.iloc[-1 - n]


def snapshot_table(region, horizons, pct):
    keys = [k for k in ds["rawW"].columns if KEY_REGION.get(k) == region]
    rows = {}
    for k in keys:
        s = ds["rawW"][k].dropna()
        if s.empty:
            continue
        row = {"Latest": s.iloc[-1]}
        for label in horizons:
            row[("%Δ " if pct else "Δ ") + label] = chg(s, HORIZONS[label], pct)
        zser = ds["compZ"][k].dropna() if k in ds["compZ"].columns else pd.Series(dtype=float)
        row["Latest z"] = zser.iloc[-1] if not zser.empty else np.nan
        rows[k] = row
    return pd.DataFrame(rows).T


# ----------------------------------------------------------- 1 · regime row ---
st.subheader("1 · Current regime")
cols = st.columns(len(summ))
for c, (r, row) in zip(cols, summ.iterrows()):
    c.markdown(f"**{r}**")
    c.markdown(
        f"<div style='background:{COLOR.get(row['regime'], '#999')};padding:8px;"
        f"border-radius:6px;color:white;font-size:12px;text-align:center'>"
        f"{row['regime']}<br><b>z = {row['liquidity_z']}</b></div>",
        unsafe_allow_html=True)
mom_col = f"mom_{C.MOM_WEEKS}w"
st.dataframe(
    summ.style.format("{:.2f}", na_rep="—", subset=[c for c in summ.columns if c != "regime"])
        .background_gradient(cmap=RYG, subset=["liquidity_z"], vmin=-2, vmax=2)
        .background_gradient(cmap=RYG, subset=[mom_col], vmin=-1.5, vmax=1.5),
    width="stretch")


with st.expander("📝  What's going on — plain-English read on each region",
                 expanded=True):
    for rr, row in summ.iterrows():
        att = T.region_attribution(ds, rr, top=3)
        rkeys = [k for k in ds["meta"] if ds["meta"][k]["region"] == rr]
        fx_only = bool(rkeys) and {ds["meta"][k]["bucket"] for k in rkeys} <= {"fx"}
        text = N.region_narrative(rr, row["regime"], row["liquidity_z"], row[mom_col],
                                  att, mom_weeks=C.MOM_WEEKS, fx_only=fx_only)
        st.markdown(f"**{N.FLAG.get(rr, '')} {rr} — {row['regime']}**  \n{text}")
        st.markdown("")
    st.caption("Plain-English interpretation built from the same component data as the tables "
               "below — each phrase is a fixed reading of one metric, switched on by whether "
               "it's currently an easing or tightening force. Edit the wording in narrative.py.")

# ----------------------------------- 2 · combined scores, all regions ---------
st.subheader("2 · Liquidity score overlay — combined or by country")
all_regions = list(ds["composite"].columns)
pick = st.multiselect("Regions to plot  (default = all; deselect to isolate one country)",
                      all_regions, default=all_regions)
fig = go.Figure()
for r in (pick or all_regions):
    s = clip(ds["composite"][r].dropna())
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name=r, mode="lines"))
fig.add_hline(y=0, line_dash="dot", line_color="#bbbbbb")
fig.update_layout(height=440, yaxis_title="liquidity z  (+ easier / − tighter)",
                  legend=dict(orientation="h"), margin=dict(t=20))
st.plotly_chart(fig, width="stretch")

# --------------------------- 3 · region deep-dive: chart + numbers + panels ---
st.subheader("3 · Region deep-dive")
r = st.selectbox("Region", list(ds["composite"].columns), index=0)

# 3a · combined composite vs equity index
full_idx = ds["composite"][r].dropna().index
comp = clip(ds["composite"][r].dropna())
f1 = go.Figure()
f1.add_trace(go.Scatter(x=comp.index, y=comp.values, name=f"{r} composite (combined)",
                        line=dict(color="#2c7fb8", width=3)))
if r in ds["equities"].columns:
    idx = clip(ds["equities"][r].reindex(full_idx).ffill())
    f1.add_trace(go.Scatter(x=idx.index, y=idx.values, name=f"{r} equity index",
                            yaxis="y2", line=dict(color="#aaaaaa", dash="dot")))
f1.add_hline(y=0, line_dash="dot", line_color="#bbbbbb")
f1.update_layout(height=360, margin=dict(t=20), legend=dict(orientation="h"),
                 yaxis=dict(title="composite z"),
                 yaxis2=dict(title="equity index", overlaying="y", side="right"),
                 title=f"{r}: combined liquidity score vs equity index")
st.plotly_chart(f1, width="stretch")

# 3b · snapshot numbers table (Value / Week / YoY style)
st.markdown(f"**{r} — current levels & changes**")
cc1, cc2 = st.columns([3, 1])
horizons = cc1.multiselect("Change periods", list(HORIZONS.keys()),
                           default=["1W", "52W (1Y)"])
pct = cc2.radio("Change as", ["Absolute", "%"], horizontal=True) == "%"
snap = snapshot_table(r, horizons, pct)
if snap.empty:
    st.info("No metric data loaded for this region yet.")
else:
    st.dataframe(snap.style.format("{:,.2f}", na_rep="—")
                 .background_gradient(cmap=RYG, subset=["Latest z"], vmin=-2, vmax=2),
                 width="stretch")
    st.caption("Latest = current level in the metric's native units "
               "(FRED dollar series are in $ millions — divide by 1,000 for $bn). "
               "Δ columns = change over each period; **Latest z** = signed liquidity "
               "score (+ = easier). Once your US H.4.1 sheet is wired in, the US rows "
               "show clean $bn straight from your file.")

# 3c · individual component time series (small multiples)
keys = [k for k in ds["meta"] if ds["meta"][k]["region"] == r]
if keys:
    ncol = 2
    nrow = math.ceil(len(keys) / ncol)
    titles = [f"{k}  ({'↑=easier' if ds['meta'][k]['sign'] == 1 else '↓=easier'})"
              for k in keys]
    view = st.radio("Individual-metric chart view",
                    ["Z-score (comparable, signed)", "Raw value (actual units)"],
                    horizontal=True)
    use_z = view.startswith("Z")
    sub = make_subplots(rows=nrow, cols=ncol, subplot_titles=titles,
                        vertical_spacing=0.14, horizontal_spacing=0.08)
    for i, k in enumerate(keys):
        rr, cc = i // ncol + 1, i % ncol + 1
        s = clip((ds["compZ"][k] if use_z else ds["rawW"][k]).dropna())
        sub.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                 line=dict(color="#2c7fb8" if use_z else "#41806b")),
                      row=rr, col=cc)
        if use_z:
            sub.add_hline(y=0, line_dash="dot", line_color="#dddddd", row=rr, col=cc)
    sub.update_layout(height=235 * nrow, showlegend=False, margin=dict(t=50))
    sub.update_annotations(font_size=12)
    st.plotly_chart(sub, width="stretch")

# ------------------------------------------ daily pulse (US funding + stress) ---
st.subheader("⚡ Daily pulse — US funding and market stress")
pl = load_pulse(st.session_state.refresh)
if pl is None or len(pl["score"]) < 10:
    st.info("The daily pulse needs FRED and Yahoo Finance data. Press **Update now**.")
else:
    sc = pl["score"]
    chg5 = float(sc.iloc[-1] - sc.iloc[-6])
    c1, c2, c3 = st.columns(3)
    c1.metric("Pulse score", f"{sc.iloc[-1]:+.2f}", f"{chg5:+.2f} over 5 days")
    c2.metric("Latest day", f"{sc.index[-1]:%d %b %Y}")
    c3.metric("TGA source", "daily (Treasury)" if pl["notes"].get("tga") == "daily" else "weekly (FRED)")
    st.markdown(N.pulse_narrative(float(sc.iloc[-1]), chg5, P.attribution(pl)))
    recent = sc[sc.index >= sc.index[-1] - pd.Timedelta(days=C.PULSE_SHOW_DAYS)]
    fig = go.Figure(go.Scatter(x=recent.index, y=recent.values, mode="lines", name="Daily pulse"))
    fig.add_hline(y=0, line_dash="dot", line_color="#999999")
    fig.update_layout(height=300, margin=dict(t=20, b=20), yaxis_title="z (+ = easier)")
    st.plotly_chart(fig, width="stretch")
    rows = []
    for k, m in pl["meta"].items():
        r = pl["raw"][k]
        rows.append({"Input": N.PULSE_PHRASES.get(k, (k,))[0], "Unit": m["unit"],
                     "Latest": r.iloc[-1], "1 day": r.iloc[-1] - r.iloc[-2],
                     "5 days": r.iloc[-1] - r.iloc[-6], "Signal (z)": pl["z"][k].iloc[-1],
                     "Last print": f"{r.index[-1]:%Y-%m-%d}"})
    tbl = pd.DataFrame(rows).set_index("Input")
    st.dataframe(tbl.style.format({"Latest": "{:,.2f}", "1 day": "{:+,.2f}", "5 days": "{:+,.2f}",
                                   "Signal (z)": "{:+.2f}"})
                 .background_gradient(cmap=RYG, subset=["Signal (z)"], vmin=-3, vmax=3),
                 width="stretch")
    st.caption("Separate from the weekly scores. Signal compares each input with its own "
               "3-year daily history, signed so + = easier. Net liquidity is scored on its "
               "20-day change.")

# ----------------------- 4 · US Fed balance sheet (net liquidity components) ---
st.subheader("4 · US Fed balance sheet — net liquidity & its components ($bn)")
fed = ds.get("fed_bs")
if fed is None or fed.empty:
    st.info("Needs FRED (WALCL / WTREGEN / WRESBAL / RRPONTSYD). "
            "Set FRED_API_KEY and press **Update now**.")
else:
    order = list(fed.columns)   # Net Liquidity, Fed Total Assets, Reserve Balances, TGA, RRP

    # 4a · snapshot: latest $bn + changes
    frows = {}
    for col in order:
        s = fed[col].dropna()
        frows[col] = {"Latest ($bn)": s.iloc[-1], "Δ 1W": chg(s, 1, False),
                      "Δ 4W": chg(s, 4, False), "Δ 13W": chg(s, 13, False),
                      "Δ 52W": chg(s, 52, False)}
    fed_tbl = pd.DataFrame(frows).T[["Latest ($bn)", "Δ 1W", "Δ 4W", "Δ 13W", "Δ 52W"]]
    st.dataframe(fed_tbl.style.format("{:,.1f}", na_rep="—"), width="stretch")

    # 4b · weekly driver waterfall: ΔAssets − ΔTGA − ΔRRP = ΔNet liquidity
    wf_h = st.selectbox("Driver waterfall — horizon", list(HORIZONS.keys()), index=0,
                        key="fed_wf_h")
    n = HORIZONS[wf_h]
    if len(fed) > n:
        last, prev = fed.iloc[-1], fed.iloc[-1 - n]
        dA = last["Fed Total Assets"] - prev["Fed Total Assets"]
        dT = last["TGA"] - prev["TGA"]
        dR = last["RRP"] - prev["RRP"]
        dN = last["Net Liquidity"] - prev["Net Liquidity"]
        wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Δ Assets", "− Δ TGA", "− Δ RRP", "= Δ Net Liq"],
            y=[dA, -dT, -dR, dN],
            text=[f"{v:+,.0f}" for v in [dA, -dT, -dR, dN]], textposition="outside",
            connector=dict(line=dict(color="#cccccc")),
            increasing=dict(marker=dict(color="#1a9850")),
            decreasing=dict(marker=dict(color="#d73027")),
            totals=dict(marker=dict(color="#2c7fb8"))))
        wf.update_layout(height=340, margin=dict(t=30), yaxis_title="$bn",
                         title=f"Net-liquidity drivers, last {wf_h}")
        st.plotly_chart(wf, width="stretch")
        st.caption(f"Over the last {wf_h}, net liquidity moved **{dN:+,.0f} $bn** = "
                   f"Fed assets {dA:+,.0f} − ΔTGA {dT:+,.0f} − ΔRRP {dR:+,.0f}. "
                   "TGA and RRP *drain* liquidity when they rise, so they enter with a minus.")

    # 4c · component small-multiples ($bn)
    nrow2 = math.ceil(len(order) / 2)
    fsub = make_subplots(rows=nrow2, cols=2, subplot_titles=order,
                         vertical_spacing=0.13, horizontal_spacing=0.09)
    for i, col in enumerate(order):
        rr2, cc2 = i // 2 + 1, i % 2 + 1
        s = clip(fed[col].dropna())
        fsub.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                  line=dict(color="#41806b")), row=rr2, col=cc2)
    fsub.update_layout(height=235 * nrow2, showlegend=False, margin=dict(t=50))
    fsub.update_annotations(font_size=12)
    st.plotly_chart(fsub, width="stretch")
    st.caption("All in **$bn**. Net Liquidity = Fed Total Assets − TGA − RRP. With RRP "
               "now ~drained (~$6bn vs $2.5trn peak), **Reserve Balances are the marginal "
               "liquidity absorber** — watch them for tightening.")

# ------------------------------- 5 · full data table, any period, download ----
st.subheader("5 · Full data table — any period, downloadable")
tables = {"Raw values": ds["rawW"],
          "Z-scores (signed, + = easier)": ds["compZ"],
          "Composite scores (final)": ds["composite"]}
which = st.selectbox("Show", list(tables.keys()))
df = tables[which].dropna(how="all").sort_index()
if not df.empty:
    dmin, dmax = df.index.min().date(), df.index.max().date()
    default_start = max(dmin, (df.index.max() - pd.Timedelta(weeks=52)).date())
    c1, c2 = st.columns(2)
    start = c1.date_input("From", value=default_start, min_value=dmin, max_value=dmax)
    end = c2.date_input("To", value=dmax, min_value=dmin, max_value=dmax)
    out = df.loc[str(start):str(end)].sort_index(ascending=False)   # newest first
    styler = out.style.format("{:,.2f}", na_rep="—")
    # heatmap the signed tables (z / composite); skip raw (mixed units) and huge frames
    if which != "Raw values" and out.size <= 12000:
        styler = styler.background_gradient(cmap=RYG, vmin=-3, vmax=3, axis=None)
    elif which != "Raw values":
        st.caption("Heatmap shading skipped for this many cells — narrow the date "
                   "range to shade the z-scores / composites.")
    st.dataframe(styler, width="stretch", height=440)
    st.download_button("⬇  Download CSV", out.to_csv().encode(),
                       file_name=f"liquidity_{which.split()[0].lower()}_{start}_{end}.csv",
                       mime="text/csv")
    st.caption("Read any two dates off this table for a custom-period change, "
               "or download the full range.")

# ------------------------------------------- 6 · regime quadrant map -----------
st.subheader("6 · Regime map (level vs momentum)")
lvl, mo = ds["composite"].iloc[-1], ds["momentum"].iloc[-1]
q = go.Figure()
q.add_trace(go.Scatter(x=lvl.values, y=mo.values, mode="markers+text",
                       text=lvl.index, textposition="top center",
                       marker=dict(size=13, color="#2c7fb8")))
q.add_hline(y=0, line_dash="dot"); q.add_vline(x=0, line_dash="dot")
q.update_layout(xaxis_title="liquidity level (z)",
                yaxis_title=f"{C.MOM_WEEKS}-week momentum", height=440, margin=dict(t=20))
st.plotly_chart(q, width="stretch")
st.caption("Top-right = easy & improving (risk-on). Bottom-left = tight & deteriorating (risk-off).")
