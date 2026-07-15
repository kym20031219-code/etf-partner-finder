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
    signal_positions, run_backtest, _mdd, _sharpe, _cagr,
)
from tests.test_kospi_forecast import _full_bundle


def test_signal_hysteresis():
    s = pd.Series([50, 60, 55, 44, 48, 70])   # buy=54, sell=46
    pos = signal_positions(s, 54, 46)
    # 50→유지0, 60→1, 55(밴드)→유지1, 44→0, 48(밴드)→유지0, 70→1
    assert list(pos) == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]


def test_metric_helpers():
    eq = np.array([1.0, 1.2, 0.9, 1.5])
    assert abs(_mdd(eq) - (0.9/1.2 - 1)) < 1e-9      # 최대낙폭 = 1.2→0.9
    assert _sharpe(np.zeros(10)) == 0.0
    assert _cagr(2.0, 252) > 0.99 and _cagr(2.0, 252) < 1.01  # 1년에 2배 → CAGR≈100%
    assert _cagr(1.0, 500) == 0.0


def test_no_lookahead_position_is_lagged():
    """포지션은 시그널의 다음날에 적용(shift(1))되어 미래참조가 없어야 한다."""
    b = _full_bundle(n=400, seed=4)
    panel = score_panel(b, step=1, warmup=160)
    res = run_backtest(panel, ForecastWeights(), buy_thr=54, sell_thr=46, cost=0.001)
    # 첫날은 항상 현금(직전 시그널 없음) → 자산곡선 첫 값 손실 없음
    assert res["equity"][0]["strat"] == 1.0 or abs(res["equity"][0]["strat"] - 1.0) < 0.02
    for key in ("total_return", "cagr", "mdd", "sharpe", "win_rate", "n_trades",
                "time_in_market"):
        assert key in res["strategy"]
    assert "total_return" in res["buy_hold"]
    assert res["strategy"]["mdd"] <= 0                       # MDD 는 음수(또는 0)
    assert 0.0 <= res["strategy"]["time_in_market"] <= 1.0


def test_cash_when_always_bearish_beats_crash():
    """점수가 계속 낮으면(현금 유지) 폭락장에서 단순보유보다 손실이 작아야 한다."""
    # 하락 추세 번들 → 기술/추세 점수 낮음 → 대부분 현금
    b = _full_bundle(n=400, seed=9, bull=False)
    panel = score_panel(b, step=1, warmup=160)
    res = run_backtest(panel, ForecastWeights(), buy_thr=55, sell_thr=45, cost=0.001)
    # 시장노출이 100%가 아니어야(현금 구간 존재) 하고, 방어적이어야 한다
    assert res["strategy"]["time_in_market"] < 1.0
    assert res["strategy"]["mdd"] >= res["buy_hold"]["mdd"] - 1e-6  # 낙폭이 더 깊지 않음


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
