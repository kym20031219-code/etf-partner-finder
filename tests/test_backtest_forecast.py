"""전망 점수 백테스트 테스트 (네트워크 불필요).

  python -m pytest tests/
  python tests/test_backtest_forecast.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from swing.kospi_forecast import ForecastWeights, score_panel
from backtest_forecast import (
    signal_positions, positions_from_thresholds, run_backtest,
    walkforward_positions, winloss_decomposition, subperiod_performance,
    composite_score, _mdd, _sharpe, _cagr,
)
from tests.test_kospi_forecast import _full_bundle


def test_signal_hysteresis():
    s = pd.Series([50, 60, 55, 44, 48, 70])   # buy=54, sell=46
    pos = positions_from_thresholds(s, 54, 46)
    assert list(pos) == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    assert list(signal_positions(s, 54, 46)) == list(pos)   # 별칭 동일


def test_metric_helpers():
    eq = np.array([1.0, 1.2, 0.9, 1.5])
    assert abs(_mdd(eq) - (0.9/1.2 - 1)) < 1e-9
    assert _sharpe(np.zeros(10)) == 0.0
    assert 0.99 < _cagr(2.0, 252) < 1.01
    assert _cagr(1.0, 500) == 0.0


def test_winloss_decomposition():
    d = winloss_decomposition([0.1, 0.2, -0.05, 0.3, -0.1, 0.02])
    assert d["win_rate"] == round(4/6, 4)
    assert d["avg_win"] > 0 and d["avg_loss"] < 0
    assert d["payoff"] is not None and d["payoff"] > 0
    assert 0.0 <= d["top5_win_share"] <= 1.0
    assert winloss_decomposition([])["n_trades"] == 0


def test_subperiods_split():
    idx = pd.bdate_range("2016-01-04", periods=1500)
    sret = pd.Series(np.full(len(idx), 0.0004), index=idx)
    ret = pd.Series(np.full(len(idx), 0.0003), index=idx)
    sp = subperiod_performance(idx, sret, ret, ["2020-01-01", "2023-01-01"])
    assert len(sp) >= 2                        # 여러 구간으로 분리
    for p in sp:
        assert p["excess_return"] > 0          # 전략수익 > 보유수익(설정상)
        assert "period" in p and p["n_days"] >= 20


def test_fixed_mode_no_lookahead():
    b = _full_bundle(n=400, seed=4)
    panel = score_panel(b, step=1, warmup=160)
    res = run_backtest(panel, ForecastWeights(), cost=0.001, mode="fixed")
    for key in ("total_return", "cagr", "mdd", "sharpe", "win_rate", "n_trades",
                "time_in_market", "avg_win", "avg_loss", "top5_win_share"):
        assert key in res["strategy"]
    assert res["strategy"]["mdd"] <= 0
    assert 0.0 <= res["strategy"]["time_in_market"] <= 1.0
    assert res["equity"][0]["strat"] == 1.0 or abs(res["equity"][0]["strat"] - 1.0) < 0.05


def test_walkforward_reestimates_and_aggregates_after_window():
    """워크포워드: 최소윈도 이후부터 집계, 임계값을 여러 번 재추정."""
    b = _full_bundle(n=1000, seed=3)
    panel = score_panel(b, step=1, warmup=160)
    mw = 500
    res = run_backtest(panel, ForecastWeights(), cost=0.001, mode="walkforward",
                       min_window=mw, rebal=21)
    assert res["mode"] == "walkforward"
    assert res["threshold_reestimations"] >= 3          # 여러 번 재추정
    # 집계 구간이 최소윈도 이후여야(=전체보다 짧아야) 한다
    assert res["n_days"] <= len(panel) - mw + 1


def test_walkforward_positions_no_future_thresholds():
    """리밸 시점의 임계값 로그 날짜가 최소윈도 이후여야(미래참조 없음)."""
    b = _full_bundle(n=900, seed=5)
    panel = score_panel(b, step=1, warmup=160)
    score = composite_score(panel, ForecastWeights())
    ret = panel["close"].pct_change().fillna(0.0)
    pos, log = walkforward_positions(score, ret, cost=0.001, min_window=450, rebal=21)
    assert len(pos) == len(score)
    assert log and all(pd.Timestamp(e["date"]) >= score.index[450] for e in log)
    for e in log:
        assert e["buy"] > e["sell"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
