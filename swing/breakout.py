"""신고가 돌파(추세추종) 전략.

눌림목(평균회귀)과 반대 접근:
- **진입**: N일 신고가를 종가로 돌파 + 장기추세 위 + 거래량 확인 → 다음 봉 시가 매수
- **청산**: 고정 익절 없음(추세 지속 시 계속 보유). 아래 중 먼저 닿는 것으로 청산
    1. 초기 손절: 진입가 − k×ATR
    2. 샹들리에 트레일링: 보유중 최고가 − m×ATR (이익을 따라 올라감)
    3. 돌파 이탈: 종가가 exit_lookback 일 최저가 아래로 마감
    4. 시간 초과

미래참조 없음(신호 다음 봉 시가 체결), 왕복 비용 반영.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import ROUND_TRIP_COST, Trade
from .strategy import atr, sma


@dataclass
class BreakoutParams:
    entry_lookback: int = 120     # 돌파 기준: 직전 120일 최고가 갱신
    ma_trend: int = 120           # 종가가 이 이평선 위 (장기 상승)
    vol_ma: int = 20
    vol_mult: float = 1.3         # 돌파일 거래량 ≥ 20일 평균 × 1.3
    atr_period: int = 14
    init_stop_atr: float = 2.5    # 초기 손절 = 진입가 − 2.5×ATR
    trail_atr: float = 3.5        # 샹들리에 트레일 = 최고가 − 3.5×ATR
    exit_lookback: int = 20       # 종가가 20일 최저가 이탈 시 청산
    max_hold: int = 80            # 최대 보유 거래일 (추세추종이라 넉넉히)


def _indicators(df: pd.DataFrame, p: BreakoutParams) -> pd.DataFrame:
    d = df.copy()
    d["atr"] = atr(d, p.atr_period)
    d["ma_t"] = sma(d["Close"], p.ma_trend)
    d["vol_ma"] = sma(d["Volume"], p.vol_ma)
    d["prior_high"] = d["High"].rolling(p.entry_lookback).max().shift(1)
    d["exit_low"] = d["Low"].rolling(p.exit_lookback).min().shift(1)
    return d


def extract_trades_breakout(
    code: str, df: pd.DataFrame, p: BreakoutParams, regime: pd.Series | None = None
) -> list[Trade]:
    d = _indicators(df, p)
    o = d["Open"].to_numpy(float)
    h = d["High"].to_numpy(float)
    low = d["Low"].to_numpy(float)
    c = d["Close"].to_numpy(float)
    atr_a = d["atr"].to_numpy(float)
    exit_low = d["exit_low"].to_numpy(float)
    dates = d.index

    entry_sig = (
        (d["Close"] > d["prior_high"])            # 신고가 돌파
        & (d["Close"] > d["ma_t"])                # 장기추세 위
        & (d["Volume"] >= d["vol_ma"] * p.vol_mult)  # 거래량 확인
    ).fillna(False)
    if regime is not None:
        on = regime.reindex(d.index).ffill().fillna(False).astype(bool)
        entry_sig = entry_sig & on
    entry_sig = entry_sig.to_numpy(bool)

    trades: list[Trade] = []
    n = len(d)
    i = 0
    while i < n - 1:
        if not entry_sig[i]:
            i += 1
            continue
        ei = i + 1
        entry_price = o[ei]
        entry_atr = atr_a[ei] if np.isfinite(atr_a[ei]) else entry_price * 0.03
        if not np.isfinite(entry_price) or entry_price <= 0 or entry_atr <= 0:
            i += 1
            continue
        init_stop = entry_price - p.init_stop_atr * entry_atr
        highest = h[ei]

        exit_i = exit_px = reason = None
        for j in range(ei, min(ei + p.max_hold + 1, n)):
            highest = max(highest, h[j])
            stop = max(init_stop, highest - p.trail_atr * entry_atr)
            if low[j] <= stop:                       # 손절/트레일 (갭이면 시가)
                exit_px = min(o[j], stop) if o[j] <= stop else stop
                exit_i, reason = j, ("stop" if stop == init_stop else "trail")
                break
            if np.isfinite(exit_low[j]) and c[j] < exit_low[j]:  # 돌파 이탈
                exit_px, exit_i, reason = c[j], j, "channel"
                break
            if j - ei >= p.max_hold:
                exit_px, exit_i, reason = c[j], j, "time"
                break
        if exit_i is None:
            exit_i, exit_px, reason = n - 1, c[n - 1], "eod"

        net = (exit_px / entry_price - 1) - ROUND_TRIP_COST
        trades.append(
            Trade(code=code, entry_date=dates[ei], entry_price=float(entry_price),
                  exit_date=dates[exit_i], exit_price=float(exit_px), ret=float(net),
                  bars_held=int(exit_i - ei), reason=reason)
        )
        i = exit_i + 1
    return trades
