"""눌림목 매수(pullback) 스윙 전략.

핵심 아이디어: **상승 추세(정배열)** 인 종목이 20일 이동평균선까지 **눌렸다가**
그 자리에서 **반등**하는 첫 신호를 매수 후보로 잡는다.

모든 신호는 종가 확정 후 판단하며(미래 참조 없음), 실제 진입은 다음 봉 시가로
체결한다고 가정한다(engine.py 참고).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 보조지표
# ---------------------------------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


# ---------------------------------------------------------------------------
# 파라미터
# ---------------------------------------------------------------------------
@dataclass
class PullbackParams:
    # 이동평균
    ma_short: int = 5
    ma_mid: int = 20      # 눌림 기준선
    ma_long: int = 60     # 추세 기준선
    # 추세 필터
    trend_rise_lookback: int = 20   # ma_long 이 이 기간 전보다 높아야(상승 추세)
    # 눌림 조건
    pullback_touch: float = 0.015   # 당일 저가가 20일선 +1.5% 이내로 근접해야 (실데이터 튜닝값)
    runup_lookback: int = 20        # 직전 고점 확인 구간
    runup_min: float = 0.08         # 직전 고점이 현재가보다 8%+ 높았어야(먼저 올랐던 자리)
    # 반등 트리거
    rsi_period: int = 14
    rsi_low: float = 40.0
    rsi_high: float = 68.0
    vol_ma: int = 20
    vol_mult: float = 0.8           # 반등일 거래량 >= 20일 평균 * 이 값
    # 청산 (기본: 고정 %) — 실데이터 튜닝(2020~2026, KOSPI100)에서 검증구간 최선 조합
    stop_pct: float = 0.07          # 손절 -7%
    target_pct: float = 0.15        # 익절 +15%
    max_hold: int = 20              # 최대 보유 거래일
    ma_mid_break: float = 0.03      # 종가가 20일선 -3% 아래로 마감하면 추세이탈 청산
    # 청산 (선택: ATR 기반 — 튜닝 레버, 기본 비활성)
    atr_period: int = 14
    use_atr_exits: bool = False     # True 면 손절/익절을 진입시점 ATR 배수로 산정
    atr_stop_mult: float = 2.0      # 손절 = 진입가 - atr_stop_mult * ATR
    atr_target_mult: float = 3.0    # 익절 = 진입가 + atr_target_mult * ATR
    trail_atr_mult: float = 0.0     # >0 이면 (보유중 최고종가 - trail_atr_mult*ATR) 이탈 시 청산


# ---------------------------------------------------------------------------
# 지표 부착 + 신호 생성
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame, p: PullbackParams) -> pd.DataFrame:
    out = df.copy()
    out["ma_s"] = sma(out["Close"], p.ma_short)
    out["ma_m"] = sma(out["Close"], p.ma_mid)
    out["ma_l"] = sma(out["Close"], p.ma_long)
    out["rsi"] = rsi(out["Close"], p.rsi_period)
    out["atr"] = atr(out, p.atr_period)
    out["vol_ma"] = sma(out["Volume"], p.vol_ma)
    out["prior_high"] = out["High"].rolling(p.runup_lookback).max().shift(1)
    out["ma_l_prev"] = out["ma_l"].shift(p.trend_rise_lookback)
    return out


def generate_signals(df: pd.DataFrame, p: PullbackParams) -> pd.DataFrame:
    """각 봉에 대해 진입 신호(entry) 컬럼을 붙여 반환."""
    d = add_indicators(df, p)

    uptrend = (
        (d["Close"] > d["ma_l"])          # 장기선 위 = 추세 유지
        & (d["ma_m"] > d["ma_l"])         # 정배열(20>60)
        & (d["ma_l"] > d["ma_l_prev"])    # 장기선 우상향
    )
    pullback = (
        (d["Low"] <= d["ma_m"] * (1 + p.pullback_touch))   # 20일선까지 눌림
        & (d["Low"] >= d["ma_l"])                          # 단, 60일선은 안 깨짐
        & (d["prior_high"] >= d["Close"] * (1 + p.runup_min))  # 먼저 올랐던 자리
    )
    bounce = (
        (d["Close"] > d["ma_m"])          # 종가가 20일선 위로 회복
        & (d["Close"] > d["Open"])        # 양봉
        & (d["rsi"].between(p.rsi_low, p.rsi_high))
        & (d["Volume"] >= d["vol_ma"] * p.vol_mult)
    )

    d["entry"] = (uptrend & pullback & bounce).fillna(False)
    return d


def latest_candidates(universe: dict[str, pd.DataFrame], p: PullbackParams) -> list[dict]:
    """각 종목의 '가장 최근 봉'에서 진입 신호가 뜬 종목만 추려 반환.

    → 향후 텔레그램 알림 / 웹 대시보드에 그대로 넘길 수 있는 형태.
    """
    hits = []
    for code, df in universe.items():
        if len(df) < p.ma_long + p.trend_rise_lookback:
            continue
        d = generate_signals(df, p)
        last = d.iloc[-1]
        if bool(last["entry"]):
            hits.append(
                {
                    "code": code,
                    "date": str(d.index[-1].date()),
                    "close": round(float(last["Close"]), 2),
                    "ma20": round(float(last["ma_m"]), 2),
                    "rsi": round(float(last["rsi"]), 1),
                    "stop": round(float(last["Close"]) * (1 - p.stop_pct), 2),
                    "target": round(float(last["Close"]) * (1 + p.target_pct), 2),
                }
            )
    return hits


# 관심 관찰(watch) 판정 밴드: 종가가 20일선의 이 배수 범위에 있으면 '근접'
WATCH_BAND = (0.96, 1.06)


def current_picks(
    universe: dict[str, pd.DataFrame],
    p: PullbackParams,
    watch_band: tuple[float, float] = WATCH_BAND,
) -> dict[str, list[dict]]:
    """대시보드용 2단계 추천.

    - buy   : 가장 최근 봉에서 매수 신호가 확정된 종목 (강한 신호, 드묾)
    - watch : 상승추세이면서 20일선에 근접한 '지켜볼 자리' (아직 반등 미확정)

    매수 신호가 뜨면 watch 에서는 제외한다(중복 방지).
    """
    buy, watch = [], []
    for code, df in universe.items():
        if len(df) < p.ma_long + p.trend_rise_lookback:
            continue
        d = generate_signals(df, p)
        last = d.iloc[-1]
        close = float(last["Close"])
        ma20 = float(last["ma_m"])
        ma60 = float(last["ma_l"])
        ma60_prev = float(last["ma_l_prev"])
        rsi_v = float(last["rsi"])
        date = str(d.index[-1].date())
        levels = {
            "stop": round(close * (1 - p.stop_pct), 2),
            "target": round(close * (1 + p.target_pct), 2),
        }

        if bool(last["entry"]):
            buy.append(
                {
                    "code": code, "date": date, "close": round(close, 2),
                    "ma20": round(ma20, 2), "rsi": round(rsi_v, 1),
                    "dist_pct": round((close / ma20 - 1) * 100, 1),
                    "tier": "buy", **levels,
                }
            )
            continue

        # 관심: 상승추세(정배열·장기선 우상향) + 20일선 근접 + 과열 아님
        uptrend = (close > ma60) and (ma20 > ma60) and (ma60 > ma60_prev)
        near_ma20 = watch_band[0] * ma20 <= close <= watch_band[1] * ma20
        not_hot = rsi_v < p.rsi_high
        if uptrend and near_ma20 and not_hot:
            watch.append(
                {
                    "code": code, "date": date, "close": round(close, 2),
                    "ma20": round(ma20, 2), "rsi": round(rsi_v, 1),
                    "dist_pct": round((close / ma20 - 1) * 100, 1),
                    "tier": "watch", **levels,
                }
            )

    # 20일선에 가까운 순으로 정렬 (근접도 = |dist_pct|)
    watch.sort(key=lambda x: abs(x["dist_pct"]))
    return {"buy": buy, "watch": watch}
