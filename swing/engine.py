"""백테스트 엔진.

1) extract_trades: 한 종목의 신호로부터 개별 매매(진입/청산)를 뽑아낸다.
   - 진입: 신호 발생 다음 봉 '시가'로 체결 (미래 참조 방지)
   - 청산 우선순위(보수적): 손절 → 익절 → 추세이탈 → 시간초과
2) simulate_portfolio: 여러 종목 매매를 동시 보유 한도 안에서 자본에 태워
   일별 자산곡선을 만든다 → MDD/CAGR 계산 가능.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy import PullbackParams, generate_signals


@dataclass
class Trade:
    code: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    ret: float          # 수수료/세금 반영 후 수익률
    bars_held: int
    reason: str         # stop / target / trend_break / time / eod


# 국내 현실 반영: 왕복 비용 대략치 (수수료 + 매도 세금 ~0.20% + 슬리피지)
ROUND_TRIP_COST = 0.0035


def extract_trades(
    code: str, df: pd.DataFrame, p: PullbackParams, regime: pd.Series | None = None
) -> list[Trade]:
    d = generate_signals(df, p, regime=regime)
    o = d["Open"].to_numpy(float)
    h = d["High"].to_numpy(float)
    low = d["Low"].to_numpy(float)
    c = d["Close"].to_numpy(float)
    ma_m = d["ma_m"].to_numpy(float)
    atr_arr = d["atr"].to_numpy(float)
    entry_sig = d["entry"].to_numpy(bool)
    dates = d.index

    trades: list[Trade] = []
    i = 0
    n = len(d)
    while i < n - 1:
        if not entry_sig[i]:
            i += 1
            continue

        # 다음 봉 시가로 진입
        ei = i + 1
        entry_price = o[ei]
        if not np.isfinite(entry_price) or entry_price <= 0:
            i += 1
            continue
        # 손절/익절 가격: 고정 % 또는 진입시점 ATR 배수
        entry_atr = atr_arr[ei] if np.isfinite(atr_arr[ei]) else entry_price * p.stop_pct
        if p.use_atr_exits:
            stop_px = entry_price - p.atr_stop_mult * entry_atr
            target_px = entry_price + p.atr_target_mult * entry_atr
        else:
            stop_px = entry_price * (1 - p.stop_pct)
            target_px = entry_price * (1 + p.target_pct)

        exit_i = None
        exit_px = None
        reason = None
        peak_close = entry_price  # 트레일링 기준: 보유중 최고 종가
        for j in range(ei, min(ei + p.max_hold + 1, n)):
            # 손절 (갭하락이면 시가로 체결)
            if low[j] <= stop_px:
                exit_px = min(o[j], stop_px) if o[j] <= stop_px else stop_px
                exit_i, reason = j, "stop"
                break
            # 익절 (갭상승이면 시가로 체결)
            if h[j] >= target_px:
                exit_px = max(o[j], target_px) if o[j] >= target_px else target_px
                exit_i, reason = j, "target"
                break
            # 트레일링 스톱 (선택): 최고종가 대비 trail_atr_mult*ATR 만큼 밀리면 청산
            if p.trail_atr_mult > 0:
                peak_close = max(peak_close, c[j])
                trail_px = peak_close - p.trail_atr_mult * entry_atr
                if c[j] <= trail_px and c[j] > entry_price:  # 이익 구간에서만 추적청산
                    exit_px, exit_i, reason = c[j], j, "trail"
                    break
            # 추세이탈: 종가가 20일선 -버퍼 아래로 마감
            if c[j] < ma_m[j] * (1 - p.ma_mid_break):
                exit_px, exit_i, reason = c[j], j, "trend_break"
                break
            # 시간초과
            if j - ei >= p.max_hold:
                exit_px, exit_i, reason = c[j], j, "time"
                break
        if exit_i is None:  # 데이터 끝까지 미청산
            exit_i, exit_px, reason = n - 1, c[n - 1], "eod"

        gross = exit_px / entry_price - 1
        net = gross - ROUND_TRIP_COST
        trades.append(
            Trade(
                code=code,
                entry_date=dates[ei],
                entry_price=float(entry_price),
                exit_date=dates[exit_i],
                exit_price=float(exit_px),
                ret=float(net),
                bars_held=int(exit_i - ei),
                reason=reason,
            )
        )
        i = exit_i + 1  # 청산 후부터 재탐색
    return trades


def simulate_portfolio(
    trades: list[Trade],
    starting_cash: float = 10_000_000,
    max_positions: int = 5,
) -> pd.DataFrame:
    """동시 보유 한도(max_positions)로 자본을 배분하는 간이 포트폴리오 시뮬.

    - 신규 진입 시 (현금 / 남은 슬롯) 만큼 매수
    - 청산일에 손익 실현
    반환: 일별 equity DataFrame (index=date, columns=[equity]).
    """
    if not trades:
        return pd.DataFrame(columns=["equity"])

    trades = sorted(trades, key=lambda t: t.entry_date)
    all_dates = pd.DatetimeIndex(
        sorted({d for t in trades for d in (t.entry_date, t.exit_date)})
    )

    cash = starting_cash
    open_pos: list[tuple[Trade, float]] = []  # (trade, invested_amount)
    # 날짜별 이벤트 모음
    entries_by_date: dict[pd.Timestamp, list[Trade]] = {}
    exits_by_date: dict[pd.Timestamp, list[Trade]] = {}
    for t in trades:
        entries_by_date.setdefault(t.entry_date, []).append(t)
        exits_by_date.setdefault(t.exit_date, []).append(t)

    equity_curve = []
    for day in all_dates:
        # 먼저 청산 처리 (당일 진입/청산 동시면 청산 우선 → 슬롯 회수)
        for t in exits_by_date.get(day, []):
            for k, (ot, amt) in enumerate(open_pos):
                if ot is t:
                    cash += amt * (1 + t.ret)
                    open_pos.pop(k)
                    break
        # 신규 진입
        for t in entries_by_date.get(day, []):
            if len(open_pos) >= max_positions:
                continue
            slots_left = max_positions - len(open_pos)
            invest = cash / slots_left
            cash -= invest
            open_pos.append((t, invest))
        # 장부가치 = 현금 + 미청산 포지션 원금(간이: 평가손익 미반영, 실현 기준 곡선)
        held = sum(amt for _, amt in open_pos)
        equity_curve.append((day, cash + held))

    eq = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    return eq
