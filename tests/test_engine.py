"""엔진 기본 무결성 테스트 (네트워크 불필요).

  python -m pytest tests/            # pytest 있으면
  python tests/test_engine.py        # 그냥 실행해도 통과 시 OK
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swing import data as datamod
from swing.engine import extract_trades
from swing.strategy import PullbackParams, generate_signals, rsi
import pandas as pd


def test_rsi_bounds():
    s = pd.Series(range(1, 100)).astype(float)
    r = rsi(s, 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def _first_seed_with_trades(p, days=800):
    """매매가 하나라도 나오는 시드를 찾아 (df, trades) 반환."""
    for seed in range(50):
        df = datamod.make_synthetic("T", seed=seed, days=days)
        trades = extract_trades("T", df, p)
        if trades:
            return df, trades
    raise AssertionError("어떤 시드에서도 매매가 나오지 않음")


def test_no_lookahead_entry_is_next_open():
    """진입가는 반드시 '신호 다음 봉'의 시가여야 한다."""
    p = PullbackParams()
    df, trades = _first_seed_with_trades(p)
    d = generate_signals(df, p)
    for t in trades:
        sig_rows = d.index.get_indexer([t.entry_date])
        ei = sig_rows[0]
        assert ei >= 1
        # 진입 전날(ei-1)에 신호가 있어야 하고, 진입가 == 당일 시가
        assert bool(d["entry"].iloc[ei - 1]) is True
        assert abs(t.entry_price - float(d["Open"].iloc[ei])) < 1e-6


def test_exit_within_max_hold():
    df = datamod.make_synthetic("T", seed=3, days=600)
    p = PullbackParams()
    for t in extract_trades("T", df, p):
        # eod 청산 제외하고는 보유일이 max_hold 를 넘지 않아야 한다
        if t.reason != "eod":
            assert t.bars_held <= p.max_hold


def test_returns_include_costs():
    """target 청산이면 수익률은 목표% - 비용 근처여야 한다."""
    from swing.engine import ROUND_TRIP_COST
    df = datamod.make_synthetic("T", seed=1, days=600)
    p = PullbackParams()
    for t in extract_trades("T", df, p):
        if t.reason == "target":
            assert abs(t.ret - (p.target_pct - ROUND_TRIP_COST)) < 0.02


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
