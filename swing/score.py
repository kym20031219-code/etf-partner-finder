"""눌림목 스윙 후보 **스코어링**.

`strategy.py` 의 이진(매수/관심) 신호를 확장해, 상승추세 종목을 **0~100점**으로
줄세워 '오늘 눈여겨볼 순위'를 만든다. 실제 매매 주문은 포함하지 않으며 참고용
추천 점수일 뿐이다.

총점(total) 은 네 개의 하위 점수를 가중 평균한 값이다:
  - trend    : 정배열·장기선 우상향 등 **추세 강도**
  - pullback : 종가가 **20일선에 근접**한 정도(눌림 자리일수록 높음)
  - momentum : RSI 가 **반등 여력이 있는 구간**(과열·과매도 아님)인지
  - volume   : 최근 거래량이 20일 평균 대비 **실린 정도**

가중치·기준값은 아래 `ScoreParams` 에서 조정한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from swing.strategy import PullbackParams, generate_signals


def _clip01(x: float) -> float:
    """0~1 로 자르기 (NaN 은 0)."""
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, x))


@dataclass
class ScoreParams:
    # 하위 점수 가중치 (합이 1이 되도록)
    w_trend: float = 0.35
    w_pullback: float = 0.30
    w_momentum: float = 0.20
    w_volume: float = 0.15
    # 기준값 (각 하위 점수의 정규화 스케일)
    trend_gap_cap: float = 0.15     # 종가가 60일선보다 이만큼(+15%) 위면 만점
    align_gap_cap: float = 0.05     # 20일선이 60일선보다 이만큼(+5%) 위면 만점
    slope_cap: float = 0.05         # 60일선 20일 상승률이 이만큼(+5%)이면 만점
    pullback_band: float = 0.06     # 종가가 20일선 ±6% 밖이면 눌림 점수 0
    rsi_ideal: float = 52.0         # 반등 여력 이상적 RSI
    rsi_span: float = 22.0          # 이상값에서 이만큼 벗어나면 0점
    vol_full: float = 1.5           # 거래량이 20일 평균의 1.5배면 만점


def _components(last: pd.Series, p: ScoreParams) -> dict[str, float]:
    """가장 최근 봉 한 줄에서 하위 점수(0~100)들을 계산."""
    close = float(last["Close"])
    ma20 = float(last["ma_m"])
    ma60 = float(last["ma_l"])
    ma60_prev = float(last["ma_l_prev"])
    rsi_v = float(last["rsi"])
    vol = float(last["Volume"])
    vol_ma = float(last["vol_ma"])

    # 추세: 60일선 위 여유 + 정배열 간격 + 60일선 기울기
    above = _clip01((close / ma60 - 1) / p.trend_gap_cap)
    align = _clip01((ma20 / ma60 - 1) / p.align_gap_cap)
    slope = _clip01((ma60 / ma60_prev - 1) / p.slope_cap)
    trend = 100 * (0.4 * above + 0.3 * align + 0.3 * slope)

    # 눌림: 20일선에 가까울수록 만점, ±band 밖이면 0
    dist = abs(close / ma20 - 1)
    pullback = 100 * _clip01(1 - dist / p.pullback_band)

    # 모멘텀: RSI 가 이상 구간에 가까울수록 만점 (과열·과매도에서 0)
    momentum = 100 * _clip01(1 - abs(rsi_v - p.rsi_ideal) / p.rsi_span)

    # 거래량: 20일 평균 대비 실린 정도
    ratio = vol / vol_ma if vol_ma > 0 else 0.0
    volume = 100 * _clip01(ratio / p.vol_full)

    total = (
        p.w_trend * trend
        + p.w_pullback * pullback
        + p.w_momentum * momentum
        + p.w_volume * volume
    )
    return {
        "total": round(total, 1),
        "trend": round(trend, 1),
        "pullback": round(pullback, 1),
        "momentum": round(momentum, 1),
        "volume": round(volume, 1),
        "close": round(close, 2),
        "ma20": round(ma20, 2),
        "rsi": round(rsi_v, 1),
        "dist_pct": round((close / ma20 - 1) * 100, 1),
    }


def score_universe(
    universe: dict[str, pd.DataFrame],
    pull: PullbackParams | None = None,
    p: ScoreParams | None = None,
    names: dict[str, str] | None = None,
    top: int | None = None,
    regime: pd.Series | None = None,
) -> list[dict]:
    """상승추세 종목을 점수화해 총점 내림차순으로 정렬해 반환.

    - 상승추세(종가>60일선·정배열·60일선 우상향)가 아닌 종목은 제외한다.
      (눌림목 전략의 전제이므로, 점수 매김 대상이 아니다)
    - 매수 신호가 확정된 봉이면 `signal="buy"`, 아니면 `signal="watch"`.
    - `top` 이 주어지면 상위 N 개만 남긴다.
    """
    pull = pull or PullbackParams()
    p = p or ScoreParams()
    names = names or {}

    rows: list[dict] = []
    for code, df in universe.items():
        if len(df) < pull.ma_long + pull.trend_rise_lookback:
            continue
        d = generate_signals(df, pull, regime=regime)
        last = d.iloc[-1]
        close = float(last["Close"])
        ma20 = float(last["ma_m"])
        ma60 = float(last["ma_l"])
        ma60_prev = float(last["ma_l_prev"])

        # 상승추세만 후보로
        uptrend = (close > ma60) and (ma20 > ma60) and (ma60 > ma60_prev)
        if not uptrend:
            continue

        comp = _components(last, p)
        rows.append(
            {
                "code": code,
                "name": names.get(code, code),
                "date": str(d.index[-1].date()),
                "signal": "buy" if bool(last["entry"]) else "watch",
                "stop": round(close * (1 - pull.stop_pct), 2),
                "target": round(close * (1 + pull.target_pct), 2),
                **comp,
            }
        )

    rows.sort(key=lambda r: r["total"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    if top is not None:
        rows = rows[:top]
    return rows
