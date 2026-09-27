# CLAUDE.md — Global Liquidity Tracker

Context handoff for continuing this project in Claude Code (local or on the web). It
orients an agent on what the project is, the rules it must preserve, and what to build
next. Read it before changing anything.

## What this is
A click-to-update dashboard tracking liquidity conditions across 7 markets (US, Europe,
Japan, Hong Kong, Singapore, Korea, China) plus a Global overlay. Each region is
compressed into one z-scored composite (**+ = easier liquidity**), assigned a regime,
explained in plain English, and shown against its equity index with a lead/lag
correlation. Purpose: a macro/liquidity lens to sit alongside fundamentals and technicals
when judging *why* markets are moving. Owner: investment professional — **not an
economist**, so every signal must also be interpreted in plain English.

## Stack
Python · FRED API (`fredapi`) · `yfinance` · pandas/numpy · Streamlit + Plotly ·
matplotlib (Styler gradients) · xlsxwriter. Artifact front end: static HTML + ECharts
5.5.0 from cdnjs. Free FRED key required, read from env var `FRED_API_KEY`.

## Two front ends, one engine
| Front end | How it runs | Refresh |
|---|---|---|
| **Claude artifact** (owner's main view) | https://claude.ai/artifact/M93qEeNEJFiv8nreLXVBfa — private page on claude.ai | Scheduled Routine "Liquidity tracker daily rebuild": Tue–Sat 07:46 Kuala Lumpur. Each run clones or pulls this public repo (github.com/dkwhy8888/Global-Liquidity-Tracker-v2), so it works in a fresh routine session with no repo attached (a routine session cannot attach a private repo). The old private repo Global-Liquidity-Tracker is a frozen backup; develop here. The TradingView connector is attached to the routine in the claude.ai Routines UI. Manual: `python build_artifact.py`, then republish `artifact/liquidity_tracker.html` to that URL (pass it as `url`; read the artifact first). Never create a second artifact. |
| **Streamlit** (local) | `run_dashboard.bat` on the owner's Windows PC → http://localhost:8501 | "Update now" button in the app |

Both import the same `tracker.py` + `narrative.py`, so a methodology fix lands in both.

## Files
- **config.py** — the metric registry (the knobs). Every metric's `region`, `bucket`,
  `source` (fred/yf/manual/derived), `id`, `transform` (level/yoy/chg13), `sign`,
  `weight`; plus `Z_WINDOW`, `MOM_WEEKS`, `LEAD_WEEKS`, `RESAMPLE`, `STALE_WEEKS`.
- **tracker.py** — fetch (FRED + yfinance + manual CSV) → derive (Fed net liquidity, G3 CB
  assets, `fed_balance_sheet()` in $bn) → transform → z-score → weighted region
  composites → regime → lead/lag corr → `liquidity_tracker.xlsx`. Also
  `region_attribution()` (ranks each region's drivers). Run: `python tracker.py`.
- **narrative.py** — plain-English layer. `PHRASES[key] = (plain name, easing reading,
  tightening reading)` for every non-target metric; `region_narrative()` stitches them into
  a paragraph from the attribution. Phrases are fixed, human-written readings switched on
  only by the sign in the data — never generate macro narrative or numbers from model
  knowledge. Owner may edit the wording here.
- **dashboard.py** — Streamlit app. Sections: z-score explainer; (1) regime tiles, summary
  and plain-English reads; (2) all-region overlay with a region multiselect; (3) region
  deep-dive (composite vs equity, snapshot table, component small-multiples, z/raw toggle);
  (4) US Fed balance sheet in $bn (tiles, driver waterfall, small-multiples); (5) full data
  table (raw / z / composite, CSV download); (6) regime quadrant.
- **pulse.py** — the **daily pulse**: US net liquidity (daily estimate), SOFR − IORB, HY/IG
  spreads, 10y real yield (`DFII10`), VIX, MOVE, DXY. Daily business-day grid, rolling
  `PULSE_Z_DAYS` z-score, signed + = easier, weighted into one pulse score. Separate from the
  weekly composites — never feeds them. Settings in `config.PULSE*`; phrases in
  `narrative.PULSE_PHRASES`. Daily TGA comes from the US Treasury Daily Treasury Statement API
  (`api.fiscaldata.treasury.gov`); if unreachable it falls back to weekly FRED `WDTGAL`.
- **market_fallback.py** + **market_fallback/** — price history for tickers yfinance can't
  supply (today `000300.SS` CSI 300, sourced from TradingView `SSE:000300` weekly bars).
  `tracker.fetch_yf()` uses the file when yfinance returns <30 rows. Refresh:
  TradingView `get_ohlcv` → `python market_fallback.py merge 000300.SS <bars.json>`.
- **build_artifact.py** — runs the engine and injects JSON into `artifact/template.html`,
  writing `artifact/liquidity_tracker.html` (build output, gitignored).
- **artifact/template.html** — the artifact's design and JS (edit layout here, never the
  built file). Must keep exactly one `@@DATA@@` placeholder.
- **manual_inputs.csv** — HK/SG/KR/CN monetary inputs not yet auto-fetched (template:
  `manual_inputs_template.csv`). One placeholder row today, so those inputs are skipped
  (<30 obs) and HK/SG/KR/CN composites are **FX-only**.
- **run_dashboard.bat** — owner's Windows launcher: creates venv on first run, prompts for
  the FRED key only if the env var is empty, opens the browser, runs Streamlit headless.
- **requirements.txt**, **.gitignore**, **.gitattributes** (`.bat` kept CRLF).

## 🔐 Secrets
`FRED_API_KEY` comes from the environment (a persisted Windows user variable locally; an
environment variable in cloud sessions). Never print, hardcode or commit it. The local
`.claude/` folder contains it in plain text, so `.claude/` is gitignored — keep it that way.

## Methodology — invariants, do not break
- **Transforms**: stock/quantity series (balance sheets, M2, net liquidity) → **YoY %**
  before z-scoring, so the composite reads the liquidity *impulse*, not the secular
  uptrend in levels. Prices / rates / spreads / vol → **level**.
- **Z-score**: rolling `Z_WINDOW` weeks (default 156 ≈ 3y), clipped ±3.
- **Sign**: +1 if higher = easier, −1 if tighter; applied *after* z-scoring so + always
  means easier.
- **Composite**: weighted mean of available signed-z components per region.
- **Regime**: quadrant from composite level (≥0 easy) × `MOM_WEEKS` momentum (≥0 improving).
- **Lead/lag**: correlation of composite_t vs equity index return over the next
  `LEAD_WEEKS`.
- All series resampled to weekly (W-FRI); monthly series forward-filled. The still-open
  week (bin dated after today) is dropped, and each component's signed z is carried
  forward at most `STALE_WEEKS` (13) past its last print, so release lags don't knock a
  series out of the latest composite but discontinued series do drop out.
- Known dead FRED ids (resolve but stopped updating): `MABMM301EZM189S` EU M3 (last Nov
  2023), `MYAGM2JPM189S` JP M2 (last Feb 2017). Replace via ECB SDW / BOJ (task 4).
- `equity_index` metrics are TARGETS (weight 0): excluded from the composite, used only
  for the overlay/correlation.

### The one debatable sign
`usdjpy` is set **+1** (weak yen = easy BOJ + exporter tailwind = easier for JP equities).
Flip to −1 to read it as USD-funding stress. Owner's call — confirm before changing.

## Units (verified against FRED)
`WALCL`, `WTREGEN`, `WRESBAL` are **$ millions** (÷1,000 → $bn); `RRPONTSYD` is already
**$bn**; `ECBASSETSW` is € millions; `JPNASSETS` is **100 million yen**. The owner's own
US H.4.1 sheet is in **$bn**. Display $bn wherever a human reads it.

## Conventions (owner's standing rules)
- **Windows scripts are CMD batch only**: no `$` variables, no `#` comments, hardcoded
  URLs. Not PowerShell, not bash. (Python itself is cross-platform; cloud sandboxes are
  Linux and can't run the `.bat`.)
- **Data sourcing priority**: company/market data goes Quartr → TradingView → Yahoo/Google
  Finance; never training data or article snippets for live values. For this tracker the
  live layers are FRED (macro/monetary) and yfinance (FX/indices/vol).
- Communication: emoji section headers, tables over prose, plain English, minimal
  citations. Ask before changing a sign or weight, or committing proprietary files.

## Open tasks (roughly prioritized)
1. ~~Cloud + scheduled refresh~~ **Done** (Routine, Tue–Sat 07:46 KL). Daily TGA needs
   `api.fiscaldata.treasury.gov` allowed in the cloud environment's network settings; until
   then the pulse uses weekly TGA.
2. **Wire the US H.4.1 sheet as the US module** (highest value). Owner maintains
   `US Liquidity Chart.xlsx` locally (newest ~657KB copy), updated weekly, 129 columns.
   Build an adapter reading **Net Liquidity / Fed Assets / RRP / TGA / Reserve Balances**
   ($bn) as the US inputs. Needs from owner: the 5 column letters, and a decision on
   whether the file (or a sample) may be committed.
3. **US reserve-scarcity gauges.** (SOFR − IORB is in the daily pulse; not yet in the weekly
   US composite.) RRP has drained to ~$1bn (Sep 2026, vs $2.5trn peak), so
   reserves are the marginal absorber. Add **SOFR − IORB spread** (FRED `SOFR`, `IORB`) and
   **reserves/GDP** to flag tightness before it reaches net liquidity.
4. **Replace manual and dead inputs with live APIs.** HKMA (Aggregate Balance, HIBOR), BOK
   ECOS (base rate, M2), MAS / data.gov.sg (SORA, M2), ECB SDW (EU M3), BOJ (JP M2).
5. **Auto-source China.** TSF + credit impulse (12m ΔTSF flow / GDP); LPR, 7-day reverse
   repo, RRR. CSI 300 now comes from TradingView via `market_fallback/` (yfinance `000300.SS` returns 1 row).
6. **Extend the global aggregate.** Add PBOC to G3 → G4 CB assets; add JPY/EUR
   cross-currency basis as a USD-funding-stress signal.
7. **Extend the driver waterfall beyond the US** where data allows (US done).
8. **Tests.** A `pytest` suite on synthetic data for the invariants above.

## Known gotchas (already solved — keep solved)
- Missing FRED key must not `sys.exit`: FRED is skipped and both front ends say so.
- Streamlit's first-run **email prompt** blocks the server → launch with
  `--server.headless true` (in run_dashboard.bat). Use `width="stretch"`, not the
  deprecated `use_container_width`.
- `Styler.background_gradient` requires **matplotlib** (in requirements). Streamlit uses
  `RdYlGn` shading on the summary, snapshot table and the z/composite data-table heatmap
  (raw view unshaded; frames >12k cells skip shading).
- Weekend FX quotes open a future-dated W-FRI bin → dropped (see invariants), else the US
  composite reads NaN.
- Artifact page: scripts only from the CDN allowlist (cdnjs etc.), data inlined as JSON.
  ECharts piecewise `visualMap` crashes if every piece is open-ended → keep finite bounds.
  The waterfall needs `stackStrategy: 'all'` for floating bars below zero. No dual-axis
  charts (composite and equity index are separate charts).
- Source files contain non-ASCII (emoji, −); open them with `encoding="utf-8"`.
