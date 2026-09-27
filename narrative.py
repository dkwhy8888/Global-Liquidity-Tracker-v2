"""
narrative.py - turn the data-driven driver attribution into plain-English
interpretation for a non-economist. No invented facts: every clause is a fixed,
human-written reading of ONE metric, switched on only by whether that metric is
currently an easing (+) or tightening (-) force in the composite, and by the
region's regime/momentum. Edit the wording here freely - it changes the prose
without touching any of the maths.
"""

FLAG = {"GLOBAL": "🌍", "US": "🇺🇸", "EU": "🇪🇺", "JP": "🇯🇵",
        "HK": "🇭🇰", "SG": "🇸🇬", "KR": "🇰🇷", "CN": "🇨🇳"}

REGIME_OPEN = {
    "Easy & Improving":      "Liquidity is **easier than normal and still improving** — a supportive, risk-on backdrop.",
    "Easy & Fading":         "Liquidity is **easier than normal, but the tailwind is fading** — still supportive, just losing momentum.",
    "Tight & Improving":     "Liquidity is **tighter than normal, but beginning to improve** — a headwind that's starting to ease.",
    "Tight & Deteriorating": "Liquidity is **tighter than normal and still deteriorating** — a risk-off, headwind backdrop.",
    "n/a":                   "Not enough data is loaded to classify this region yet.",
}

# key -> (plain name, reading when it's an EASING force (+), reading when TIGHTENING (-))
PHRASES = {
 "fed_net_liquidity": ("Fed net liquidity",
    "the net cash the Fed is pumping into markets is expanding",
    "the net cash the Fed is pumping into markets is shrinking"),
 "g3_cb_assets_usd": ("global central-bank balance sheets",
    "the major central banks (Fed, ECB, BOJ) are collectively growing their balance sheets",
    "the major central banks (Fed, ECB, BOJ) are collectively shrinking their balance sheets (QT)"),
 "g4_cb_assets_usd": ("the big four central banks' balance sheets",
    "the Fed, ECB, BOJ and PBOC together are growing their balance sheets in dollar terms",
    "the Fed, ECB, BOJ and PBOC together are shrinking their balance sheets in dollar terms"),
 "global_m2_usd": ("global money supply",
    "broad money in the US, euro area, Japan and China is growing faster than usual in dollar terms",
    "broad money in the US, euro area, Japan and China is growing slower than usual in dollar terms"),
 "credit_baa": ("corporate credit spreads",
    "corporate borrowing spreads are narrow, so credit is easy to get",
    "corporate borrowing spreads are wide, a sign of credit stress"),
 "us_hy_oas": ("high-yield credit spreads",
    "junk-bond risk premiums are tight, so markets are relaxed about credit risk",
    "junk-bond risk premiums have widened, a sign of credit stress"),
 "us_ig_oas": ("investment-grade spreads",
    "blue-chip credit spreads are calm",
    "blue-chip credit spreads have widened, hinting at caution"),
 "nfci": ("US financial conditions",
    "the Chicago Fed's broad financial-conditions gauge reads loose",
    "the Chicago Fed's broad financial-conditions gauge reads tight"),
 "vix": ("equity volatility (VIX)",
    "stock-market volatility is low, reflecting calm risk appetite",
    "stock-market volatility is elevated, a sign of nervous markets"),
 "move": ("bond volatility (MOVE)",
    "bond-market volatility is subdued",
    "bond-market volatility is elevated"),
 "broad_usd": ("the US dollar",
    "the dollar is on the softer side, which loosens global funding conditions",
    "the dollar is strong, which tightens global funding conditions"),
 "cn_credit_impulse": ("China's credit impulse",
    "China is pushing fresh credit into its economy",
    "China's credit creation is pulling back"),
 "us_m2_yoy": ("US money supply (M2)",
    "the US money supply is growing again year-on-year",
    "the US money supply is flat-to-shrinking year-on-year"),
 "us_fed_funds": ("the Fed's policy rate",
    "the Fed's policy rate now sits below its 3-year norm, so rate cuts are loosening policy",
    "the Fed's policy rate is still high versus recent years, keeping policy restrictive"),
 "us_2s10s": ("the yield curve",
    "the yield curve has steepened, which usually points to easier conditions ahead",
    "the yield curve is flat or inverted, a classic tight-policy signal"),
 "ecb_assets": ("the ECB's balance sheet",
    "the ECB is expanding its balance sheet",
    "the ECB is shrinking its balance sheet"),
 "eu_m3_yoy": ("euro-area money supply (M3)",
    "euro-area money supply is growing",
    "euro-area money supply is weak"),
 "eurusd": ("the euro",
    "a firmer euro against the dollar",
    "a weaker euro against the dollar"),
 "boj_assets": ("the BOJ's balance sheet",
    "the Bank of Japan is still expanding its balance sheet",
    "the Bank of Japan is trimming its balance sheet"),
 "jp_m2_yoy": ("Japan's money supply",
    "Japan's money supply is growing",
    "Japan's money supply is soft"),
 "usdjpy": ("the yen",
    "a weaker yen, which flatters Japanese exporters and reflects easy BOJ policy",
    "a stronger yen, a drag on exporters and a sign of less BOJ easing"),
 "hk_agg_balance": ("HK interbank cash (Aggregate Balance)",
    "Hong Kong's banking-system cash pile is ample",
    "Hong Kong's banking-system cash pile is thin"),
 "hibor_3m": ("HK interbank rates (HIBOR)",
    "Hong Kong interbank borrowing rates are low",
    "Hong Kong interbank borrowing rates are high"),
 "usdhkd": ("the HK dollar peg",
    "the HK dollar is on the strong side of its band, where the HKMA tends to add liquidity",
    "the HK dollar is pinned to the weak side of its band, where the HKMA drains liquidity"),
 "sora": ("Singapore's funding rate (SORA)",
    "Singapore's funding rate is low",
    "Singapore's funding rate is high"),
 "sg_m2_yoy": ("Singapore's money supply",
    "Singapore's money supply is growing",
    "Singapore's money supply is soft"),
 "usdsgd": ("the Singapore dollar",
    "a firm Singapore dollar",
    "a weak Singapore dollar"),
 "kr_base_rate": ("the Bank of Korea's policy rate",
    "Korea's policy rate is low or being cut",
    "Korea's policy rate is high"),
 "kr_m2_yoy": ("Korea's money supply",
    "Korea's money supply is growing",
    "Korea's money supply is soft"),
 "usdkrw": ("the Korean won",
    "a firmer won, which supports foreign inflows",
    "an unusually weak won, which raises the risk of capital outflows and forces tighter conditions"),
 "cn_m2_yoy": ("China's money supply",
    "China's money supply is growing",
    "China's money supply is soft"),
 "cn_tsf_yoy": ("China's total credit (TSF)",
    "China's total credit creation is accelerating",
    "China's total credit creation is slowing"),
 "cn_7d_repo": ("China's interbank rate",
    "China's short-term funding rate is low",
    "China's short-term funding rate is high"),
 "cn_rrr": ("China's reserve-requirement ratio",
    "the PBOC has cut banks' reserve requirements, freeing up lending capacity",
    "bank reserve requirements are high, restraining lending"),
 "us_bank_credit": ("US bank lending",
    "US banks are growing their loan and securities books faster than usual",
    "US bank credit growth is slow, so banks are adding little new money"),
 "us_reserves_gdp": ("US bank reserves",
    "banks hold ample reserves at the Fed relative to the size of the economy",
    "bank reserves at the Fed are thin relative to the economy, so the system has less cushion"),
 "us_sofr_iorb": ("overnight funding (SOFR − IORB)",
    "overnight funding is cheap versus the rate the Fed pays banks, so cash is plentiful",
    "overnight funding costs are pressing up against the Fed's rate, a sign reserves are getting scarce"),
 "hk_m2_yoy": ("Hong Kong's money supply",
    "Hong Kong's money supply is growing",
    "Hong Kong's money supply is soft"),
 "kr_3m_rate": ("Korea's interbank rate",
    "Korean short-term interest rates are low versus recent years",
    "Korean short-term interest rates are high versus recent years"),
 "cn_pboc_assets": ("the PBOC's balance sheet",
    "the People's Bank of China is expanding its balance sheet",
    "the People's Bank of China is shrinking its balance sheet"),
 "cn_loan_growth": ("Chinese bank lending",
    "Chinese bank loan growth is strong versus recent years",
    "Chinese bank loan growth is weak versus recent years, a sign of soft credit demand"),
 "cn_3m_rate": ("China's interbank rate",
    "China's short-term funding rate is low versus recent years",
    "China's short-term funding rate is high versus recent years"),
 "usdcny": ("the yuan",
    "a firmer yuan",
    "a weaker yuan"),
}


def _join(items):
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "; ".join(items[:-1]) + "; and " + items[-1]   # "A; B; and C"


def _names(names):
    names = [n for n in names if n]
    if not names:
        return ""
    return " and ".join(names[:2])


def region_narrative(region, regime, level, mom, att, mom_weeks=13, fx_only=False):
    """Return a short plain-English paragraph interpreting the region's regime."""
    out = [REGIME_OPEN.get(regime, "")]
    if att:
        ease = [PHRASES[k][1] for k, z, c in att.get("easing", []) if k in PHRASES]
        tight = [PHRASES[k][2] for k, z, c in att.get("tightening", []) if k in PHRASES]
        if ease:
            out.append("**What's helping:** " + _join(ease) + ".")
        if tight:
            out.append("**What's weighing:** " + _join(tight) + ".")
        impr = _names([PHRASES[k][0] for k, d in att.get("improving", []) if k in PHRASES])
        fade = _names([PHRASES[k][0] for k, d in att.get("fading", []) if k in PHRASES])
        if regime.endswith("Improving") and impr:
            out.append(f"The pickup over the past ~{mom_weeks} weeks is led by "
                       f"{impr} moving in an easier direction.")
        elif (regime.endswith("Fading") or regime.endswith("Deteriorating")) and fade:
            out.append(f"The loss of momentum over the past ~{mom_weeks} weeks is mostly "
                       f"{fade} drifting tighter.")
    if fx_only:
        out.append("_(This region currently reflects exchange-rate moves only — its "
                   "policy-rate and money-supply inputs aren't loaded yet.)_")
    return " ".join(p for p in out if p)


# ------------------------------------------------------------- daily pulse ---
# key -> (plain name, reading when EASING (+), reading when TIGHTENING (-))
PULSE_PHRASES = {
 "net_liq_daily": ("US net liquidity (daily estimate)",
    "net cash in the US system has risen over the past month",
    "net cash in the US system has fallen over the past month, often because the Treasury is refilling its cash account"),
 "sofr_iorb": ("overnight funding (SOFR − IORB)",
    "overnight funding is cheap versus the rate the Fed pays banks, so cash is plentiful",
    "overnight funding costs are pressing up against the Fed's rate, an early sign that bank reserves are getting scarce"),
 "hy_oas": ("high-yield credit spreads",
    "junk-bond spreads are tight, so lenders are relaxed about credit risk",
    "junk-bond spreads are wide, a sign of credit stress"),
 "ig_oas": ("investment-grade spreads",
    "blue-chip credit spreads are calm",
    "blue-chip credit spreads have widened"),
 "real_10y": ("the 10-year real yield",
    "real (inflation-adjusted) yields are low, which supports asset prices",
    "real (inflation-adjusted) yields are high, which raises the bar for risk assets"),
 "vix": ("equity volatility (VIX)",
    "stock-market volatility is low",
    "stock-market volatility is elevated"),
 "move": ("bond volatility (MOVE)",
    "bond-market volatility is subdued",
    "bond-market volatility is elevated"),
 "dxy": ("the US dollar (DXY)",
    "the dollar is on the soft side, which eases global funding",
    "the dollar is strong, which tightens global funding"),
}


def pulse_narrative(score, chg5, att):
    """Plain-English read of the daily pulse from fixed phrases."""
    lvl = "easier than normal" if score >= 0 else "tighter than normal"
    if abs(chg5) < 0.1:
        drift = "and little changed over the past week"
    elif chg5 > 0:
        drift = "and it moved easier over the past week"
    else:
        drift = "and it moved tighter over the past week"
    out = [f"Day-to-day funding and market conditions are **{lvl}**, {drift}."]
    ease = [PULSE_PHRASES[k][1] for k, _ in att.get("easing", []) if k in PULSE_PHRASES]
    tight = [PULSE_PHRASES[k][2] for k, _ in att.get("tightening", []) if k in PULSE_PHRASES]
    if ease:
        out.append("**Helping:** " + _join(ease) + ".")
    if tight:
        out.append("**Weighing:** " + _join(tight) + ".")
    mv = att.get("movers", [])
    if mv:
        parts = [f"{PULSE_PHRASES[k][0]} moved {'easier' if d > 0 else 'tighter'}"
                 for k, d in mv if k in PULSE_PHRASES]
        if parts:
            out.append("**Biggest moves this week:** " + ", and ".join(parts) + ".")
    return " ".join(out)
