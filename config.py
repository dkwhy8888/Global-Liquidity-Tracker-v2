"""
config.py - Liquidity Tracker configuration.
Edit THIS file to add/remove metrics, flip signs, change weights or lookbacks.

Each METRIC entry:
  key       : unique column header
  region    : US | EU | JP | HK | SG | KR | CN | GLOBAL
  bucket    : cb_balance | money_credit | rates | fx | stress | equity_index
  source    : fred | yf | tv | manual | derived
  id        : FRED series id | yfinance ticker | econ_cache code (TradingView) | manual_inputs.csv column | derived key
  transform : level | yoy | chg13   (how to condition the series before z-scoring)
  sign      : +1 if higher = EASIER liquidity,  -1 if higher = TIGHTER
  weight    : relative weight inside its region composite

Notes
-----
- bucket "equity_index" rows are TARGETS, not inputs: excluded from the
  liquidity composite, used only for the lead/lag overlay vs the composite.
- transform "yoy" is for STOCK/quantity series (balance sheets, M2) so the
  composite reads the liquidity *impulse*, not the secular uptrend in levels.
- "# verify" = confirm the FRED id / yf ticker resolves on your first run.
"""

FRED_API_KEY_ENV = "FRED_API_KEY"

Z_WINDOW  = 156       # rolling weeks for z-scoring (~3y)
MOM_WEEKS = 13        # momentum / change horizon
LEAD_WEEKS = 8        # assumed lead of liquidity over equities (overlay corr)
RESAMPLE  = "W-FRI"   # weekly grid everything is aligned to
STALE_WEEKS = 13      # carry a component's last reading forward at most this long (release lags);
                      # anything staler (discontinued series) drops out of the composite

METRICS = [
    # ---------------- GLOBAL overlay (the master layer) ----------------
    {"key": "fed_net_liquidity", "region": "GLOBAL", "bucket": "cb_balance",   "source": "derived", "id": "fed_net_liq",       "transform": "yoy",   "sign": +1, "weight": 2.0},
    {"key": "g3_cb_assets_usd",  "region": "GLOBAL", "bucket": "cb_balance",   "source": "derived", "id": "g3_assets",         "transform": "yoy",   "sign": +1, "weight": 1.5},
    {"key": "us_hy_oas",         "region": "GLOBAL", "bucket": "stress",       "source": "fred",    "id": "BAMLH0A0HYM2",      "transform": "level", "sign": -1, "weight": 1.0},
    {"key": "us_ig_oas",         "region": "GLOBAL", "bucket": "stress",       "source": "fred",    "id": "BAMLC0A0CM",        "transform": "level", "sign": -1, "weight": 0.5},
    {"key": "nfci",              "region": "GLOBAL", "bucket": "stress",       "source": "fred",    "id": "NFCI",              "transform": "level", "sign": -1, "weight": 1.0},  # +NFCI = tighter
    {"key": "vix",               "region": "GLOBAL", "bucket": "stress",       "source": "yf",      "id": "^VIX",              "transform": "level", "sign": -1, "weight": 0.75},
    {"key": "move",              "region": "GLOBAL", "bucket": "stress",       "source": "yf",      "id": "^MOVE",             "transform": "level", "sign": -1, "weight": 0.75},
    {"key": "broad_usd",         "region": "GLOBAL", "bucket": "fx",           "source": "fred",    "id": "DTWEXBGS",          "transform": "level", "sign": -1, "weight": 1.0},  # strong USD = global tightening
    {"key": "cn_credit_impulse", "region": "GLOBAL", "bucket": "money_credit", "source": "manual",  "id": "cn_credit_impulse", "transform": "level", "sign": +1, "weight": 1.5},

    # ---------------- US ----------------
    {"key": "us_m2_yoy",   "region": "US", "bucket": "money_credit", "source": "fred", "id": "M2SL",   "transform": "yoy",   "sign": +1, "weight": 1.0},
    {"key": "us_fed_funds","region": "US", "bucket": "rates",        "source": "fred", "id": "DFF",    "transform": "level", "sign": -1, "weight": 1.0},
    {"key": "us_2s10s",    "region": "US", "bucket": "rates",        "source": "fred", "id": "T10Y2Y", "transform": "level", "sign": +1, "weight": 0.5},  # steeper ~ easier (small weight)
    {"key": "us_bank_credit",  "region": "US", "bucket": "money_credit", "source": "fred",    "id": "TOTBKCR",      "transform": "yoy",   "sign": +1, "weight": 1.0},  # H.8 bank credit, weekly
    {"key": "us_reserves_gdp", "region": "US", "bucket": "cb_balance",   "source": "derived", "id": "reserves_gdp", "transform": "level", "sign": +1, "weight": 1.0},  # WRESBAL / nominal GDP, %
    {"key": "us_sofr_iorb",    "region": "US", "bucket": "rates",        "source": "derived", "id": "sofr_iorb_w",  "transform": "level", "sign": -1, "weight": 0.75}, # bp; above IORB = reserves scarce
    {"key": "sp500",       "region": "US", "bucket": "equity_index", "source": "yf",   "id": "^GSPC",  "transform": "level", "sign": +1, "weight": 0.0},

    # ---------------- Europe ----------------
    {"key": "ecb_assets", "region": "EU", "bucket": "cb_balance",   "source": "fred", "id": "ECBASSETSW",       "transform": "yoy",   "sign": +1, "weight": 1.5},  # verify
    {"key": "eu_m3_yoy",  "region": "EU", "bucket": "money_credit", "source": "tv",   "id": "EUM3",             "transform": "yoy",   "sign": +1, "weight": 1.0},  # TradingView (FRED copy died Nov-2023)
    {"key": "eurusd",     "region": "EU", "bucket": "fx",           "source": "yf",   "id": "EURUSD=X",         "transform": "level", "sign": +1, "weight": 0.5},
    {"key": "estoxx50",   "region": "EU", "bucket": "equity_index", "source": "yf",   "id": "^STOXX50E",        "transform": "level", "sign": +1, "weight": 0.0},

    # ---------------- Japan ----------------
    {"key": "boj_assets","region": "JP", "bucket": "cb_balance",   "source": "fred", "id": "JPNASSETS",      "transform": "yoy",   "sign": +1, "weight": 1.5},  # verify units (100 Mn Yen)
    {"key": "jp_m2_yoy", "region": "JP", "bucket": "money_credit", "source": "tv",   "id": "JPM2",           "transform": "yoy",   "sign": +1, "weight": 1.0},  # TradingView (FRED copy died Feb-2017)
    {"key": "usdjpy",    "region": "JP", "bucket": "fx",           "source": "yf",   "id": "USDJPY=X",       "transform": "level", "sign": +1, "weight": 0.5},  # SIGN CHOICE: weak JPY (USDJPY up) = easy BOJ + exporter tailwind -> +1. Flip to -1 to read it as USD-funding stress.
    {"key": "nikkei",    "region": "JP", "bucket": "equity_index", "source": "yf",   "id": "^N225",          "transform": "level", "sign": +1, "weight": 0.0},

    # ---------------- Hong Kong ----------------
    {"key": "hk_agg_balance","region": "HK", "bucket": "cb_balance",   "source": "manual", "id": "hk_agg_balance", "transform": "level", "sign": +1, "weight": 1.5},  # HKMA API in v2
    {"key": "hibor_3m",      "region": "HK", "bucket": "rates",        "source": "tv",     "id": "HKINBR",         "transform": "level", "sign": -1, "weight": 1.0},
    {"key": "hk_m2_yoy",     "region": "HK", "bucket": "money_credit", "source": "tv",     "id": "HKM2",           "transform": "yoy",   "sign": +1, "weight": 1.0},
    {"key": "usdhkd",        "region": "HK", "bucket": "fx",           "source": "yf",     "id": "USDHKD=X",       "transform": "level", "sign": -1, "weight": 0.5},  # toward 7.85 (weak side) = HKMA drains AB = tighter
    {"key": "hsi",           "region": "HK", "bucket": "equity_index", "source": "yf",     "id": "^HSI",           "transform": "level", "sign": +1, "weight": 0.0},

    # ---------------- Singapore ----------------
    {"key": "sora",      "region": "SG", "bucket": "rates",        "source": "manual", "id": "sora",      "transform": "level", "sign": -1, "weight": 1.0},  # MAS API in v2
    {"key": "sg_m2_yoy", "region": "SG", "bucket": "money_credit", "source": "tv",     "id": "SGM2",      "transform": "yoy",   "sign": +1, "weight": 1.0},
    {"key": "usdsgd",    "region": "SG", "bucket": "fx",           "source": "yf",     "id": "USDSGD=X",  "transform": "level", "sign": -1, "weight": 0.5},
    {"key": "sti",       "region": "SG", "bucket": "equity_index", "source": "yf",     "id": "^STI",      "transform": "level", "sign": +1, "weight": 0.0},

    # ---------------- Korea ----------------
    {"key": "kr_3m_rate",  "region": "KR", "bucket": "rates",        "source": "fred",   "id": "IR3TIB01KRM156N", "transform": "level", "sign": -1, "weight": 1.0},  # 3m interbank (OECD via FRED)
    {"key": "kr_m2_yoy",   "region": "KR", "bucket": "money_credit", "source": "tv",     "id": "KRM2",         "transform": "yoy",   "sign": +1, "weight": 1.0},  # break-adjusted Feb-2026, see econ_cache.BREAKS
    {"key": "usdkrw",      "region": "KR", "bucket": "fx",           "source": "yf",     "id": "USDKRW=X",     "transform": "level", "sign": -1, "weight": 0.75},  # weak KRW = foreign outflow risk
    {"key": "kospi",       "region": "KR", "bucket": "equity_index", "source": "yf",     "id": "^KS11",        "transform": "level", "sign": +1, "weight": 0.0},

    # ---------------- China ----------------
    {"key": "cn_m2_yoy",  "region": "CN", "bucket": "money_credit", "source": "tv",     "id": "CNM2",       "transform": "yoy",   "sign": +1, "weight": 1.0},
    {"key": "cn_pboc_assets", "region": "CN", "bucket": "cb_balance", "source": "tv", "id": "CNCBBS",     "transform": "yoy",   "sign": +1, "weight": 1.0},
    {"key": "cn_loan_growth", "region": "CN", "bucket": "money_credit", "source": "tv", "id": "CNLG",     "transform": "level", "sign": +1, "weight": 1.0},  # already % YoY
    {"key": "cn_tsf_yoy", "region": "CN", "bucket": "money_credit", "source": "manual", "id": "cn_tsf_yoy", "transform": "level", "sign": +1, "weight": 1.5},
    {"key": "cn_3m_rate", "region": "CN", "bucket": "rates",        "source": "fred",   "id": "IR3TIB01CNM156N", "transform": "level", "sign": -1, "weight": 1.0},  # 3m interbank (OECD via FRED)
    {"key": "cn_rrr",     "region": "CN", "bucket": "rates",        "source": "manual", "id": "cn_rrr",     "transform": "level", "sign": -1, "weight": 1.0},  # higher RRR = tighter
    {"key": "usdcny",     "region": "CN", "bucket": "fx",           "source": "yf",     "id": "USDCNY=X",   "transform": "level", "sign": -1, "weight": 0.5},
    {"key": "csi300",     "region": "CN", "bucket": "equity_index", "source": "yf",     "id": "000300.SS",  "transform": "level", "sign": +1, "weight": 0.0},
]

REGIONS = ["GLOBAL", "US", "EU", "JP", "HK", "SG", "KR", "CN"]

# ---------------------------------------------------------------------------
# DAILY PULSE - a separate, faster read on US funding and market stress.
# It does NOT feed the weekly regional composites above; it sits beside them.
#   score     : level | chg20  (chg20 = change over 20 business days, the impulse)
#   scale     : multiply the raw series by this for display (e.g. % -> bp)
# ---------------------------------------------------------------------------
PULSE_Z_DAYS    = 756    # rolling business days for the pulse z-score (~3y)
PULSE_SHOW_DAYS = 380    # calendar days of history shipped to the charts
PULSE_STALE_DAYS = 5     # carry a daily series forward at most this many business days

PULSE = [
    {"key": "net_liq_daily", "source": "derived", "id": "net_liq_daily", "unit": "$bn", "scale": 1,   "score": "chg20", "sign": +1, "weight": 2.0},
    {"key": "sofr_iorb",     "source": "derived", "id": "sofr_iorb",     "unit": "bp",  "scale": 100, "score": "level", "sign": -1, "weight": 1.5},
    {"key": "hy_oas",        "source": "fred",    "id": "BAMLH0A0HYM2",  "unit": "bp",  "scale": 100, "score": "level", "sign": -1, "weight": 1.0},
    {"key": "ig_oas",        "source": "fred",    "id": "BAMLC0A0CM",    "unit": "bp",  "scale": 100, "score": "level", "sign": -1, "weight": 0.5},
    {"key": "real_10y",      "source": "fred",    "id": "DFII10",        "unit": "%",   "scale": 1,   "score": "level", "sign": -1, "weight": 1.0},
    {"key": "vix",           "source": "yf",      "id": "^VIX",          "unit": "pts", "scale": 1,   "score": "level", "sign": -1, "weight": 0.75},
    {"key": "move",          "source": "yf",      "id": "^MOVE",         "unit": "pts", "scale": 1,   "score": "level", "sign": -1, "weight": 0.75},
    {"key": "dxy",           "source": "yf",      "id": "DX-Y.NYB",      "unit": "index", "scale": 1, "score": "level", "sign": -1, "weight": 1.0},
]
