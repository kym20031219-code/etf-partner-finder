"""오버나이트(종가매매) 백테스트 엔진.

한 종목의 종가 매수 신호로부터 **1박(overnight) 매매**를 뽑는다.
  - 진입: 신호가 뜬 날 '종가'로 매수  (장 마감 직전 15:20 체결 가정)
  - 청산: 바로 다음날 '시가'로 매도    (갭 실현)
  - 수익률 = 다음시가/당일종가 − 1 − 왕복비용

눌림목 엔진(swing.engine)과 같은 Trade 규격을 써서 성과 지표(swing.metrics)와
포트폴리오 시뮬(swing.engine.simulate_portfolio)을 그대로 재사용한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing.engine import Trade
from .strategy import ClosingParams, generate_signals


def extract_trades(
    code: str, df: pd.DataFrame, p: ClosingParams, regime: pd.Series | None = None
) -> list[Trade]:
    d = generate_signals(df, p, regime=regime)
    o = d["Open"].to_numpy(float)
    c = d["Close"].to_numpy(float)
    entry_sig = d["entry"].to_numpy(bool)
    dates = d.index

    trades: list[Trade] = []
    n = len(d)
    for i in range(n - 1):          # 마지막 봉은 다음 시가가 없어 제외
        if not entry_sig[i]:
            continue
        entry_price = c[i] * (1 + p.entry_slippage)  # 15:20 매수(≈종가+슬리피지)
        exit_price = o[i + 1]                          # 다음날 시가로 매도
        if not (np.isfinite(entry_price) and np.isfinite(exit_price)) or entry_price <= 0:
            continue
        gross = exit_price / entry_price - 1
        net = gross - p.cost
        trades.append(
            Trade(
                code=code,
                entry_date=dates[i],
                entry_price=float(entry_price),
                exit_date=dates[i + 1],
                exit_price=float(exit_price),
                ret=float(net),
                bars_held=1,
                reason="overnight",
            )
        )
    return trades
