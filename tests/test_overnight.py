"""종가매매(오버나이트) 엔진·전략·스터디 무결성 테스트 (네트워크 불필요).

  python tests/test_overnight.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from overnight import data as odata
from overnight.engine import extract_trades
from overnight.strategy import ClosingParams, generate_signals
from overnight.study import gap_feature_study

# 엔진/스터디 '기계적 동작'을 검증하려는 테스트다. 운영 기본 파라미터(실데이터
# 최적화값)는 조건이 까다로워 합성 표본에선 신호가 드무므로, 여기서는 신호가
# 충분히 나오는 완화된 파라미터를 고정으로 쓴다.
TP = ClosingParams(up_min=0.02, close_pos_min=0.6, vol_mult=1.5,
                   breakout_lookback=20, rsi_high=80.0)


def test_entry_is_close_exit_is_next_open():
    """진입가는 신호일 '종가', 청산가는 '다음날 시가'여야 한다 (미래참조 없음)."""
    p = TP
    uni = odata.overnight_universe(n=10, days=600)
    found = 0
    for code, df in uni.items():
        d = generate_signals(df, p)
        for t in extract_trades(code, df, p):
            i = d.index.get_indexer([t.entry_date])[0]
            assert bool(d["entry"].iloc[i]) is True          # 신호일에 매수
            assert abs(t.entry_price - float(d["Close"].iloc[i])) < 1e-6   # 종가 매수
            assert abs(t.exit_price - float(d["Open"].iloc[i + 1])) < 1e-6  # 다음 시가 매도
            assert t.bars_held == 1
            found += 1
    assert found > 0, "합성 데이터에서 매매가 하나도 나오지 않음"


def test_returns_include_cost():
    """수익률은 (다음시가/종가 − 1 − 비용) 과 정확히 일치해야 한다."""
    p = TP
    uni = odata.overnight_universe(n=8, days=500)
    for code, df in uni.items():
        for t in extract_trades(code, df, p):
            gross = t.exit_price / t.entry_price - 1
            assert abs(t.ret - (gross - p.cost)) < 1e-9


def test_signal_lifts_gap_up_rate():
    """규칙 통과 시 갭상승 확률이 기준선(base rate)보다 높아야 한다 (엣지 존재 확인)."""
    p = TP
    uni = odata.overnight_universe(n=30, days=900)
    study = gap_feature_study(uni, p)
    assert study["n_days"] > 1000
    sig = study["signal"]
    assert sig["n_signals"] > 10
    assert sig["gap_up_rate"] > study["base_gap_up_rate"]   # lift > 1
    assert sig["lift"] > 1.0


def test_no_signal_without_breakout():
    """돌파가 없으면(약한 마감) 신호가 과도하게 남발되지 않는다."""
    p = TP
    uni = odata.overnight_universe(n=15, days=700)
    total_days = total_sig = 0
    for _, df in uni.items():
        d = generate_signals(df, p)
        total_days += len(d)
        total_sig += int(d["entry"].sum())
    # 종가매매 신호는 희소해야 한다 (전체의 5% 미만)
    assert total_sig / total_days < 0.05


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
