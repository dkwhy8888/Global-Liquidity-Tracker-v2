"""
econ_cache.py - monthly macro series from the TradingView connector (ECONOMICS:*).

The cloud network cannot reach the Asian central-bank, ECB or BOJ APIs, and FRED's
copies of those money-supply series stopped in 2017-2023. TradingView carries them,
but only through the Claude connector, not from Python. So a Claude session pulls
the latest points and merges them into econ_cache/<CODE>.csv; tracker.py reads the
files (metric source "tv").

Refresh one series (TradingView get_economic_data JSON saved to a file):
    python econ_cache.py merge CNM2 result.json
List the codes the tracker uses:
    python econ_cache.py codes
Date to fetch from (3 months before the last cached point, so revisions are picked up):
    python econ_cache.py since CNM2
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "econ_cache")

# code -> divisor applied to TradingView's raw value (stored units in brackets)
SCALE = {
    "CNM2": 1e9,     # CNY bn
    "CNCBBS": 1e9,   # CNY bn, PBOC total assets
    "CNLG": 1,       # % YoY, loan growth
    "HKM2": 1e9,     # HKD bn
    "HKINBR": 1,     # %, HIBOR
    "SGM2": 1e9,     # SGD bn
    "KRM2": 1e12,    # KRW trn
    "EUM3": 1e9,     # EUR bn
    "JPM2": 1e12,    # JPY trn
    "CNCRR": 1,      # %, reserve requirement ratio
}

# Level breaks (a one-month jump that is a definition change, not money leaving the
# system). Earlier values are rescaled so the break month grows at the median pace
# of the prior 12 months. KRM2 fell 10.3% in Feb-2026 on TradingView's feed.
BREAKS = {"KRM2": ["2026-02-01"]}


def path_for(code):
    return os.path.join(DIR, f"{code}.csv")


def load(code, adjust=True):
    p = path_for(code)
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, parse_dates=["date"]).set_index("date")["value"].sort_index().dropna()
    if adjust:
        for b in BREAKS.get(code, []):
            b = pd.Timestamp(b)
            if b not in s.index or s.index.get_loc(b) < 13:
                continue
            i = s.index.get_loc(b)
            normal = s.iloc[i - 12:i].pct_change().median()
            factor = (s.iloc[i] / s.iloc[i - 1]) / (1 + normal)
            s.iloc[:i] = s.iloc[:i] * factor
    return s


def merge(code, json_path):
    if code not in SCALE:
        sys.exit(f"unknown code {code}; add it to SCALE first")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("series", data) if isinstance(data, dict) else data
    new = pd.Series({pd.Timestamp(r["date"]): float(r["value"]) / SCALE[code]
                     for r in rows if r.get("value") is not None})
    if new.empty:
        sys.exit(f"{code}: no points in {json_path}")
    old = load(code, adjust=False)
    s = new if old is None else pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    os.makedirs(DIR, exist_ok=True)
    s.rename("value").rename_axis("date").to_csv(path_for(code))
    print(f"{code}: {len(s)} rows, last {s.index[-1].date()} = {s.iloc[-1]:,.4g}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "merge":
        merge(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2 and sys.argv[1] == "codes":
        print(" ".join(SCALE))
    elif len(sys.argv) == 3 and sys.argv[1] == "since":
        s = load(sys.argv[2], adjust=False)
        start = pd.Timestamp("2018-01-01") if s is None else s.index[-1] - pd.DateOffset(months=3)
        print(start.strftime("%Y-%m-%d"))
    else:
        sys.exit(__doc__)
