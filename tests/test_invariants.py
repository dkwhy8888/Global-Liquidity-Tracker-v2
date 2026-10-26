"""Invariants from CLAUDE.md, checked on synthetic data (no network).  Run: pytest -q"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C            # noqa: E402
import econ_cache as EC       # noqa: E402
import predictive as PR       # noqa: E402
import stress as ST           # noqa: E402
import stress_multi as SM     # noqa: E402
import tracker as T           # noqa: E402

rng = np.random.default_rng(0)


def weekly(values, start="2015-01-02"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=C.RESAMPLE), dtype=float)


def test_zscore_clipped_and_centred():
    s = weekly(rng.normal(0, 1, 600))
    s.iloc[-1] = 50                                   # extreme outlier
    z = T.zscore(s)
    assert z.max() <= 3 and z.min() >= -3
    assert z.iloc[-1] == 3
    assert z.iloc[: C.Z_WINDOW // 3 - 1].isna().all()  # needs min_periods of history


def test_yoy_is_52_week_percent_change():
    s = weekly(np.full(120, 100.0))
    s.iloc[60:] = 110.0
    y = T.yoy(s)
    assert y.iloc[60 + 51] == pytest.approx(10.0)
    assert y.iloc[60 + 52] == pytest.approx(0.0)


def test_monthly_series_forward_filled_to_weekly():
    m = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3, freq="MS"))
    w = T.to_weekly(m)
    assert w.index.freqstr.startswith("W-FRI")
    assert w.notna().all() and w.iloc[-1] == 3.0


def test_sign_applied_after_zscore():
    s = weekly(np.linspace(0, 10, 300))
    z = T.zscore(s)
    assert (T.zscore(s) * -1).iloc[-1] == pytest.approx(-z.iloc[-1])
    assert z.iloc[-1] > 0     # rising series -> positive z; a -1 sign makes it "tighter"


def test_regime_quadrants():
    assert T.regime_label(0.0, 0.0) == "Easy & Improving"
    assert T.regime_label(0.5, -0.1) == "Easy & Fading"
    assert T.regime_label(-0.5, 0.2) == "Tight & Improving"
    assert T.regime_label(-0.5, -0.2) == "Tight & Deteriorating"
    assert T.regime_label(np.nan, 1) == "n/a"


def _composite(vals, weights):
    """Same rule as tracker.build_dataset: weighted mean of available components."""
    v = np.array(vals, dtype=float)
    w = np.array(weights, dtype=float)
    mask = ~np.isnan(v)
    return (v[mask] * w[mask]).sum() / w[mask].sum()


def test_composite_ignores_missing_components():
    assert _composite([1.0, np.nan, -1.0], [2.0, 5.0, 1.0]) == pytest.approx((2 - 1) / 3)


def test_equity_targets_have_zero_weight():
    for m in C.METRICS:
        if m["bucket"] == "equity_index":
            assert m["weight"] == 0


def test_every_input_has_a_plain_english_phrase():
    import narrative as N
    for m in C.METRICS:
        if m["bucket"] != "equity_index":
            assert m["key"] in N.PHRASES, m["key"]
    for p in C.PULSE:
        assert p["key"] in N.PULSE_PHRASES, p["key"]


def test_stock_series_use_yoy():
    for m in C.METRICS:
        # hk_agg_balance is a level by design: it swings with HKMA peg operations, no uptrend
        if m["bucket"] in ("cb_balance",) and m["source"] != "derived" and m["key"] != "hk_agg_balance":
            assert m["transform"] == "yoy", m["key"]


def test_econ_cache_break_adjustment(tmp_path, monkeypatch):
    monkeypatch.setattr(EC, "DIR", str(tmp_path))
    idx = pd.date_range("2024-01-01", periods=24, freq="MS")
    vals = 100 * 1.01 ** np.arange(24)
    vals[18:] *= 0.9                                   # 10% definition break
    pd.Series(vals, index=idx).rename("value").rename_axis("date").to_csv(tmp_path / "XX.csv")
    monkeypatch.setitem(EC.BREAKS, "XX", [idx[18].strftime("%Y-%m-%d")])
    s = EC.load("XX")
    assert s.iloc[18] / s.iloc[17] - 1 == pytest.approx(0.01, abs=1e-9)   # break removed
    assert s.iloc[-1] == pytest.approx(vals[-1])                           # latest untouched


def test_econ_cache_merge_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(EC, "DIR", str(tmp_path))
    monkeypatch.setitem(EC.SCALE, "YY", 1e9)
    j = tmp_path / "yy.json"
    j.write_text(json.dumps({"series": [{"date": "2026-01-01", "value": 2e12}, {"date": "2026-02-01", "value": 3e12}]}))
    EC.merge("YY", str(j)); EC.merge("YY", str(j))
    s = EC.load("YY")
    assert list(s.values) == [2000.0, 3000.0]


def test_shift_available_moves_dates_forward():
    m = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
    out = T.shift_available(m, 45)
    assert list(out.index) == [pd.Timestamp("2024-02-15"), pd.Timestamp("2024-03-17")]
    assert list(m.index)[0] == pd.Timestamp("2024-01-01")        # input untouched
    assert T.shift_available(m, 0) is m


def test_every_lagged_series_exists_in_the_registry():
    ids = {m["id"] for m in C.METRICS} | {p["id"] for p in C.PULSE} | {"SOFR", "GDP"}   # SOFR, GDP feed derived series
    for k in C.LAG_DAYS:
        assert k in ids, k
    for region, items in C.INDICES.items():
        assert C.PRIMARY_INDEX[region] in [k for k, _, _ in items], region


def test_bh_qvalues():
    q = PR.bh_qvalues([0.001, 0.04, 0.5, np.nan])
    assert q[0] == pytest.approx(0.003) and q[1] == pytest.approx(0.06)
    assert np.isnan(q[3]) and q[2] == pytest.approx(0.5)
    assert (q[:3] >= np.array([0.001, 0.04, 0.5])).all()


def test_spearman_perm_detects_a_real_lead_and_ignores_noise():
    n = 600
    r = np.random.default_rng(1)
    sig = weekly(np.cumsum(r.normal(0, 1, n)) * 0.2 + r.normal(0, 1, n))
    fut = sig.shift(-13) * 0.0 + weekly(r.normal(0, 1, n))
    real = fut + 0.6 * sig                                     # future return carries the signal
    hit = PR.spearman_perm(sig, real, 13, np.random.default_rng(2), nperm=500)
    assert hit["ic"] > 0.3 and hit["p"] < 0.01
    noise = PR.spearman_perm(sig, weekly(r.normal(0, 1, n)), 13, np.random.default_rng(3), nperm=500)
    assert abs(noise["ic"]) < 0.15 and noise["p"] > 0.01
    assert PR.spearman_perm(sig.iloc[:50], real.iloc[:50], 13, np.random.default_rng(4)) is None   # too short


def test_surrogate_p_values_not_too_lenient_for_persistent_series():
    """Two unrelated random walks: the false-positive rate at 5% must stay near 5%."""
    r = np.random.default_rng(11)
    hits, trials = 0, 60
    for k in range(trials):
        x = weekly(np.cumsum(r.normal(0, 1, 300)))
        y = weekly(np.cumsum(r.normal(0, 1, 300)))
        res = PR.spearman_perm(x, y.diff().fillna(0), 13, np.random.default_rng(k), nperm=199)
        hits += res["p"] < 0.05
    assert hits / trials < 0.15


def test_verdicts():
    good = {"ic": 0.2, "p": 0.01, "q": 0.03, "h1": 0.1, "h2": 0.2}
    assert PR.lead_verdict(good, None) == "leads"
    assert PR.lead_verdict(dict(good, h1=-0.1), None) == "weak"           # unstable halves
    assert PR.lead_verdict(dict(good, q=0.4), None) == "weak"
    assert PR.lead_verdict(dict(good, ic=-0.2), None) == "reverse"
    assert PR.lead_verdict(dict(good, ic=0.01, p=0.9, q=0.9), None) == "none"
    assert PR.lead_verdict(None, None) == "short"
    assert PR.lag_verdict({"ic": 0.2, "p": 0.01, "q": 0.05}) == "follows"
    assert PR.lag_verdict({"ic": -0.2, "p": 0.01, "q": 0.05}) == "contrarian"
    assert PR.lag_verdict({"ic": 0.2, "p": 0.04, "q": 0.30}) == "none"        # passes 5% only, not after correction
    assert PR.lag_verdict(None) == "short"


def test_timing_rule_uses_last_weeks_signal():
    idx = pd.date_range("2015-01-02", periods=200, freq=C.RESAMPLE)
    score = pd.Series(np.where(np.arange(200) % 2 == 0, 1.0, -1.0), index=idx)
    close = pd.Series(100 * np.exp(np.cumsum(np.where(np.arange(200) % 2 == 1, 0.02, -0.02))), index=idx)
    t = PR.timing(score, close, idx)                 # score >= 0 on even weeks -> held next (odd) week, which rises
    assert t["st"]["cagr"] > 50 and t["bh"]["cagr"] < 5


def test_every_input_is_in_exactly_one_score():
    for m in C.METRICS:
        if m["bucket"] == "equity_index":
            continue
        assert (m["bucket"] in C.IMPULSE_BUCKETS) != (m["bucket"] in C.CONDITIONS_BUCKETS), m["key"]


def test_group_composite_uses_only_its_buckets():
    idx = pd.date_range("2024-01-05", periods=3, freq=C.RESAMPLE)
    meta = {"a": {"region": "US", "bucket": "money_credit", "weight": 1.0},
            "b": {"region": "US", "bucket": "stress", "weight": 1.0}}
    z = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [-2.0, -2.0, -2.0]}, index=idx)
    assert T.group_composite(meta, z, C.IMPULSE_BUCKETS)["US"].iloc[-1] == 1.0
    assert T.group_composite(meta, z, C.CONDITIONS_BUCKETS)["US"].iloc[-1] == -2.0
    assert T.group_composite(meta, z, None)["US"].iloc[-1] == -0.5


# ------------------------------------------------------------------ stress.py ----
def _px(n=1500, seed=3):
    r = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(r.normal(0.0004, 0.01, n))),
                     index=pd.bdate_range("2013-01-01", periods=n))


def test_fwd_drawdown_looks_only_forward():
    px = pd.Series([100.0, 90.0, 95.0, 80.0, 100.0], index=pd.bdate_range("2020-01-01", periods=5))
    d = ST.fwd_drawdown(px, 2)
    assert d.iloc[0] == pytest.approx(-0.10)            # next 2 closes: 90, 95
    assert d.iloc[1] == pytest.approx(80 / 90 - 1)      # next 2 closes: 95, 80
    assert d.iloc[3] == 0.0 or np.isnan(d.iloc[3])      # 100 afterwards: no fall (or unknown)
    assert d.iloc[-1] != d.iloc[-1]                     # no future left: unknown, not zero


def test_corrections_finds_peak_to_trough():
    idx = pd.bdate_range("2020-01-01", periods=10)
    px = pd.Series([100, 105, 110, 100, 95, 98, 100, 105, 111, 112], index=idx, dtype=float)
    c = ST.corrections(px, depth=-0.10)
    assert len(c) == 1
    assert c[0]["peak"] == idx[2] and c[0]["trough"] == idx[4]
    assert c[0]["depth"] == pytest.approx(95 / 110 - 1)


def test_lagged_signal_is_not_known_early():
    idx = pd.bdate_range("2020-01-01", periods=30)
    s = pd.Series(np.arange(30.0), index=idx)
    out = ST.lagged(s, idx, lag=2)
    assert out.iloc[10] == 8.0 and out.iloc[:2].isna().all()


def test_stress_test_detects_a_planted_warning_and_ignores_noise():
    r = np.random.default_rng(0)
    n = 2500
    ev = np.zeros(n, bool)
    for k in range(100, n - 100, 220):
        ev[k - 40:k] = True                               # "a fall within the next 40 days" days, clustered
    fire = np.zeros(n, bool)
    for k in range(100, n - 100, 220):
        fire[k - 35:k - 20] = True                        # the signal fires inside each window
    hit = ST.hit_rate_test(fire, ev, np.random.default_rng(1), nshift=500)
    assert hit["p"] < 0.05 and hit["hit"] > hit["base"]
    ps = []
    for k in range(25):                                   # unrelated fires: false-positive rate stays small
        f = np.zeros(n, bool)
        for s0 in r.integers(0, n - 30, 10):
            f[s0:s0 + 15] = True
        ps.append(ST.hit_rate_test(f, ev, np.random.default_rng(k), nshift=300)["p"])
    assert np.mean(np.array(ps) < 0.05) < 0.2


def test_sub_score_uses_only_its_keys():
    idx = pd.bdate_range("2020-01-01", periods=3)
    pl = {"z": pd.DataFrame({"vix": [1.0] * 3, "move": [3.0] * 3, "net_liq_daily": [-2.0] * 3}, index=idx),
          "meta": {"vix": {"weight": 1.0}, "move": {"weight": 1.0}, "net_liq_daily": {"weight": 2.0}}}
    assert ST.sub_score(pl, ["vix", "move"]).iloc[-1] == pytest.approx(2.0)
    assert ST.sub_score(pl, ["net_liq_daily"]).iloc[-1] == pytest.approx(-2.0)
    assert ST.sub_score(pl, None).iloc[-1] == pytest.approx((1 + 3 - 4) / 4)


def test_pooled_stress_test_detects_planted_warning_in_every_market_and_ignores_noise():
    n, m = 2600, 4
    ev = np.zeros((n, m), bool)
    fire = np.zeros((n, m), bool)
    for k in range(120, n - 120, 230):
        for j in range(m):
            ev[k - 40:k, j] = True
            fire[k - 35:k - 20, j] = True
    mask = np.ones((n, m), bool)
    hit = SM.pooled_test(fire, ev, mask, np.random.default_rng(1), 100, nshift=400, names=list("abcd"))
    assert hit["pooled"]["p"] < 0.05 and hit["pooled"]["hit"] > hit["pooled"]["base"]
    assert [r["market"] for r in hit["rows"]] == list("abcd")
    r = np.random.default_rng(2)
    ps = []
    for k in range(25):
        f = np.zeros((n, m), bool)
        for s0 in r.integers(0, n - 30, 10):
            f[s0:s0 + 15, :] = True
        ps.append(SM.pooled_test(f, ev, mask, np.random.default_rng(k), 100, nshift=300)["pooled"]["p"])
    assert np.mean(np.array(ps) < 0.05) < 0.2


def test_effective_episodes_merges_clustered_peaks():
    d = pd.to_datetime(["2008-01-01", "2008-02-15", "2008-03-30", "2009-06-01", "2009-07-01"])
    assert SM.effective_episodes(d) == 2
    assert SM.effective_episodes([]) == 0
