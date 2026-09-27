# Global Liquidity Tracker (v1)

Tracks liquidity conditions across **US, Europe, Japan, Hong Kong, Singapore, Korea, China**
plus a **Global overlay**, compresses each region into a single z-scored composite
(+ = easier liquidity), classifies a regime, and measures the composite's lead/lag
correlation to the regional equity index. Outputs an Excel workbook and an interactive
Streamlit dashboard.

## What v1 covers
- **All 7 regions, market layer** (FX vs USD, equity index, vol) -> yfinance, no key needed.
- **US / Europe / Japan monetary layer** (Fed net liquidity, M2, ECB & BOJ assets) -> FRED.
- **Global overlay**: Fed net liquidity (WALCL-TGA-RRP), G3 CB assets in USD, US HY/IG OAS,
  Chicago Fed NFCI, VIX, MOVE, broad USD, China credit impulse.
- **HK / SG / KR / CN monetary inputs** -> manual_inputs.csv for now (Asia central-bank
  APIs are the v2 upgrade: HKMA Aggregate Balance, BOK ECOS, MAS/data.gov.sg, ECB SDW).

## Setup (Windows CMD)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
set FRED_API_KEY=your_free_key_here
```
Get a free FRED key (30 seconds): https://fred.stlouisfed.org/docs/api/api_key.html

Copy the manual template so the China/Asia inputs load:
```
copy manual_inputs_template.csv manual_inputs.csv
```

## Run
```
python tracker.py            ->  writes liquidity_tracker.xlsx + prints the dashboard table
streamlit run dashboard.py   ->  opens the interactive dashboard in your browser
```

## How to read the output
- **liquidity_z** — the region composite, in standard deviations vs its own 3y history.
  Positive = liquidity easier than normal; negative = tighter. This is the headline.
- **mom_13w** — 13-week change in the composite. The *direction* matters as much as the level.
- **regime** — the quadrant: Easy & Improving (best for risk) / Easy & Fading /
  Tight & Improving / Tight & Deteriorating (worst for risk).
- **corr_lead8w** — correlation of the composite to the equity index 8 weeks forward.
  High positive = liquidity has been a useful leading read for that market.
- Excel tabs: **Dashboard** (heatmap), **Composites** (weekly series per region),
  **Component_Z** (every signed-z input), **Data** (raw weekly values).

## Manual inputs (manual_inputs.csv)
One row per observation date; forward-filled weekly. Update monthly. Columns and sources:
| Column | Series | Where to get it |
|---|---|---|
| cn_credit_impulse | China credit impulse (12m d TSF flow / GDP, %) | Trading Economics / compute from PBOC TSF |
| cn_m2_yoy, cn_tsf_yoy | China M2 & TSF stock, YoY % | PBOC / NBS monthly release |
| cn_7d_repo | PBOC 7-day reverse repo rate, % | PBOC |
| cn_rrr | Major-bank reserve requirement ratio, % | PBOC |
| hk_agg_balance | HKMA Aggregate Balance, HKD bn | HKMA (Monetary Base statistics) |
| hibor_3m | 3-month HIBOR, % | HKAB / HKMA |
| sora | Singapore Overnight Rate Average, % | MAS |
| sg_m2_yoy | Singapore M2, YoY % | MAS / SingStat |
| kr_base_rate | BOK base rate, % | Bank of Korea |
| kr_m2_yoy | Korea M2, YoY % | Bank of Korea (ECOS) |

The single example row in the template (dated 2025-12-31) is **illustrative placeholder
values only** — replace with real data before relying on the HK/SG/KR/CN composites.

## Tuning (config.py)
Every metric has a `sign` (+1 easier / -1 tighter) and `weight`. Edit freely. The one
genuinely debatable sign is **USD/JPY** — v1 treats a weaker yen as *easier* for Japanese
equities (easy BOJ + exporter tailwind, sign +1). Flip it to -1 if you'd rather read
USD/JPY as USD-funding stress. `Z_WINDOW`, `MOM_WEEKS`, and `LEAD_WEEKS` are the other knobs.

## Honest limits
Liquidity explains a meaningful share of *medium-term* equity returns and tends to *lead*
at multi-week-to-month horizons, but it is **not** a short-term timing signal and the
relationship breaks (e.g. 2022: ample reserves, yet QT + a rate shock crushed equities).
Use this as regime/context that frames fundamentals and technicals — not as a standalone
buy/sell trigger.

## v2 roadmap
- Replace manual HK/SG/KR inputs with live APIs (HKMA, BOK ECOS, MAS, ECB SDW).
- Auto-source China TSF + compute the credit impulse.
- Add PBOC to the global CB-assets aggregate; add JPY/EUR cross-currency basis.
- Schedule a daily/weekly refresh via Claude Cowork / Dispatch.
