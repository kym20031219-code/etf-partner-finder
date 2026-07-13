"""분봉 특징·실시간 신호 무결성 테스트 (네트워크/KIS 불필요)."""
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from overnight.strategy import ClosingParams
from intraday.features import DailyContext, compute_features
from intraday.synthetic import make_minute_day


def _ctx(code, prev, p, up=True, seed=0):
    """완만한(노이즈 있는) 상승/하락 추세의 과거 일봉으로 기준선 생성.

    단조 증가면 RSI 가 100 에 붙어 과열필터에 걸리므로, 현실적인 잔변동을 준다.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.001 if up else -0.001, 0.005, 60)
    closes = prev * np.cumprod(1 + rets)
    daily = pd.DataFrame({
        "Open": closes, "High": closes * 1.008,
        "Low": closes * 0.992, "Close": closes,
        "Volume": [1_000_000] * 60,
    }, index=pd.bdate_range("2026-04-01", periods=60))
    return DailyContext.from_daily(code, daily, p)


def test_day_ret_matches_snapshot():
    p = ClosingParams()
    ctx = _ctx("T", 10000, p)
    m = make_minute_day(prev_close=ctx.prev_close, day_ret=0.05, seed=1)
    f = compute_features(m, ctx, p, snapshot=time(15, 18))
    px = float(m[m.index.time <= time(15, 18)]["Close"].iloc[-1])
    assert abs(f["day_ret"] - (px / ctx.prev_close - 1) * 100) < 0.02


def test_snapshot_truncates_future_bars():
    """스냅샷 이후 분봉은 반영되지 않아야 한다 (미래참조 없음)."""
    p = ClosingParams()
    ctx = _ctx("T", 10000, p)
    m = make_minute_day(prev_close=ctx.prev_close, day_ret=0.05, seed=2)
    early = compute_features(m, ctx, p, snapshot=time(10, 0))["price"]
    late = compute_features(m, ctx, p, snapshot=time(15, 18))["price"]
    assert early != late


def test_strong_day_triggers_signal():
    """큰 상승 + 거래량 급증 + 신고가 → 신호 발생, 국면 off 면 꺼짐."""
    p = ClosingParams()
    ctx = _ctx("T", 10000, p, up=True, seed=1)
    m = make_minute_day(prev_close=ctx.prev_close, day_ret=0.08, late_strength=0.7,
                        base_min_vol=9000, seed=3)
    f = compute_features(m, ctx, p, snapshot=time(15, 18), regime_on=True)
    assert f["enough"] and f["signal"], f
    f_off = compute_features(m, ctx, p, snapshot=time(15, 18), regime_on=False)
    assert not f_off["signal"]


def test_weak_day_no_signal():
    p = ClosingParams()
    ctx = _ctx("T", 10000, p)
    m = make_minute_day(prev_close=ctx.prev_close, day_ret=0.004, base_min_vol=1000, seed=4)
    f = compute_features(m, ctx, p, snapshot=time(15, 18))
    assert not f["signal"]


def test_vwap_between_low_and_high():
    p = ClosingParams()
    ctx = _ctx("T", 10000, p)
    m = make_minute_day(prev_close=ctx.prev_close, day_ret=0.03, seed=5)
    d = m[m.index.time <= time(15, 18)]
    f = compute_features(m, ctx, p, snapshot=time(15, 18))
    vwap = f["price"] / (1 + f["vwap_pos"] / 100)
    assert float(d["Low"].min()) <= vwap <= float(d["High"].max())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
