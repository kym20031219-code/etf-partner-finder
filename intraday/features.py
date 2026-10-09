"""분봉 → 오전/오후/막판 특징 + 실시간 종가매매 신호.

핵심: 장중(예: 15:15) 스냅샷만으로 종가매매 규칙을 판정한다. 규칙의 임계값은
`overnight.strategy.ClosingParams`(실데이터 검증값)를 그대로 쓰되, '오늘의 종가'
대신 '현재가'를, '당일 거래량' 대신 '현재까지 누적거래량'을 사용한다.

기준선(전일 종가·20/5일선·60일 신고가·20일 평균거래량·RSI)은 **장 시작 전에 과거
일봉으로 미리 계산**해 `DailyContext` 로 넘긴다(장중에 안 변하므로).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from overnight.strategy import ClosingParams
from swing.strategy import rsi

# 세션 경계 (KRX 연속매매 09:00~15:20, 이후 종가 동시호가 15:20~15:30)
MORNING_END = time(12, 0)
DEFAULT_SNAPSHOT = time(15, 18)   # 15:20 매수 직전 판단 시각


@dataclass
class DailyContext:
    """장 시작 전 과거 일봉으로 계산해 두는 종목별 기준선."""
    code: str
    prev_close: float           # 전일 종가
    last_closes: list[float]    # 과거 일봉 종가들(최신이 뒤) — 최소 60개 권장
    prior_high_60: float        # 직전 60일 고가(오늘 제외)
    vol_ma20: float             # 20일 평균 거래량(일)
    name: str = ""

    @classmethod
    def from_daily(cls, code: str, daily: pd.DataFrame, p: ClosingParams, name: str = "") -> "DailyContext":
        """어제까지의 일봉 df(Open/High/Low/Close/Volume)로 기준선 생성."""
        closes = daily["Close"].astype(float)
        return cls(
            code=code,
            prev_close=float(closes.iloc[-1]),
            last_closes=[float(x) for x in closes.iloc[-max(p.ma_mid, 60):]],
            prior_high_60=float(daily["High"].iloc[-p.breakout_lookback:].max()),
            vol_ma20=float(daily["Volume"].iloc[-p.vol_ma:].mean()),
            name=name,
        )


def _sma_live(prior: list[float], price_now: float, n: int) -> float:
    """오늘 종가를 price_now 로 가정한 n일 단순이동평균(장중 근사)."""
    series = np.array(prior[-(n - 1):] + [price_now]) if n > 1 else np.array([price_now])
    return float(series.mean())


def compute_features(
    minute: pd.DataFrame,
    ctx: DailyContext,
    p: ClosingParams,
    snapshot: time = DEFAULT_SNAPSHOT,
    regime_on: bool = True,
) -> dict:
    """오늘치 분봉(DatetimeIndex, Open/High/Low/Close/Volume)과 기준선으로 특징+신호 산출.

    snapshot 이후의 분봉은 '아직 안 온 것'으로 보고 잘라낸다(=실시간 재현).
    """
    df = minute[minute.index.time <= snapshot]
    if len(df) < 5:
        return {"code": ctx.code, "enough": False}

    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    price_now = float(close.iloc[-1])
    day_high = float(df["High"].max())
    day_low = float(df["Low"].min())
    cum_vol = float(vol.sum())

    # 전일 대비 / 고가권 마감 정도
    day_ret = price_now / ctx.prev_close - 1
    rng = max(day_high - day_low, 1e-9)
    close_pos = float(np.clip((price_now - day_low) / rng, 0, 1))

    # VWAP(당일 거래량가중평균가) 대비 위치 — 위에 있을수록 강함
    vwap = float((close * vol).sum() / max(cum_vol, 1e-9))
    vwap_pos = price_now / vwap - 1

    # 오전/오후 분해
    morn = df[df.index.time <= MORNING_END]
    aft = df[df.index.time > MORNING_END]
    morning_close = float(morn["Close"].iloc[-1]) if len(morn) else ctx.prev_close
    morning_ret = morning_close / ctx.prev_close - 1
    morning_vol = float(morn["Volume"].sum())
    vol_share_morning = morning_vol / max(cum_vol, 1e-9)
    afternoon_ret = price_now / morning_close - 1 if morning_close else 0.0

    # 막판 30분 강도 + 고가 발생 시점(늦게 고가 = 강세)
    last30 = df[df.index >= df.index[-1] - pd.Timedelta(minutes=30)]
    last30_ret = price_now / float(last30["Close"].iloc[0]) - 1 if len(last30) > 1 else 0.0
    high_pos_idx = int(np.argmax(df["High"].to_numpy()))
    high_time_frac = high_pos_idx / max(len(df) - 1, 1)      # 0(장초반)~1(막판)
    up_bars_share = float((close.diff() > 0).mean())

    # 장중 근사 이동평균·RSI (오늘 종가 = 현재가로 가정)
    ma5 = _sma_live(ctx.last_closes, price_now, p.ma_short)
    ma20 = _sma_live(ctx.last_closes, price_now, p.ma_mid)
    rsi_live = float(rsi(pd.Series(ctx.last_closes + [price_now]), p.rsi_period).iloc[-1])

    # 거래량 배수: 현재까지 누적 vs 20일 평균(일). 15:18이면 하루의 ~97%가 들어와 근사 성립.
    vol_ratio = cum_vol / max(ctx.vol_ma20, 1e-9)
    breakout = price_now >= ctx.prior_high_60

    # ---- 종가매매 신호 (실데이터 검증 규칙을 실시간 스냅샷으로 판정) ----
    signal = bool(
        (day_ret >= p.up_min)
        and (price_now > float(df["Open"].iloc[0]))       # 시가 대비 양봉
        and (close_pos >= p.close_pos_min)
        and (vol_ratio >= p.vol_mult)
        and breakout
        and (ma5 > ma20) and (price_now > ma5)
        and (rsi_live < p.rsi_high)
        and regime_on                                     # 지수 상승국면일 때만
    )

    return {
        "code": ctx.code, "name": ctx.name, "enough": True,
        "snapshot": snapshot.strftime("%H:%M"),
        "price": round(price_now, 2),
        "day_ret": round(day_ret * 100, 2),
        "close_pos": round(close_pos * 100),
        "vol_ratio": round(vol_ratio, 2),
        "rsi": round(rsi_live, 1),
        "breakout": breakout,
        # 오전/오후/막판 특징 (검증 대상: 진짜 예측력은 과거 분봉 백테스트로 확인 필요)
        "morning_ret": round(morning_ret * 100, 2),
        "afternoon_ret": round(afternoon_ret * 100, 2),
        "vol_share_morning": round(vol_share_morning * 100),
        "vwap_pos": round(vwap_pos * 100, 2),
        "last30_ret": round(last30_ret * 100, 2),
        "high_time_frac": round(high_time_frac, 2),
        "up_bars_share": round(up_bars_share * 100),
        "signal": signal,
    }
