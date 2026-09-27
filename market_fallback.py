"""
market_fallback.py - price history for tickers that yfinance cannot supply.

Some tickers fail on yfinance (e.g. 000300.SS, CSI 300, returns 1 row). For those,
a Claude session pulls bars from the TradingView connector and merges them into
market_fallback/<yf ticker>.csv (columns: date, close). tracker.fetch_yf() uses
that file when yfinance returns fewer than 30 rows.

Merge new bars (TradingView get_ohlcv JSON saved to a file):
    python market_fallback.py merge 000300.SS bars.json
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "market_fallback")

# yfinance ticker -> TradingView symbol used to refresh it
TV_SYMBOLS = {"000300.SS": "SSE:000300"}


def path_for(ticker):
    return os.path.join(DIR, f"{ticker}.csv")


def load(ticker):
    p = path_for(ticker)
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, parse_dates=["date"]).set_index("date")["close"].sort_index()
    return s.dropna()


def merge(ticker, json_path):
    with open(json_path, encoding="utf-8") as f:
        bars = json.load(f)["bars"]
    new = pd.Series({pd.Timestamp(b["t"], unit="s").normalize(): float(b["c"]) for b in bars})
    old = load(ticker)
    s = new if old is None else pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    os.makedirs(DIR, exist_ok=True)
    s.rename("close").rename_axis("date").to_csv(path_for(ticker), float_format="%.4f")
    print(f"{ticker}: {len(s)} rows, {s.index[0].date()} to {s.index[-1].date()}, last close {s.iloc[-1]:.2f}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "merge":
        merge(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
