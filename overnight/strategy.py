"""종가매매법(오버나이트) 매수 전략.

핵심 아이디어: 장 마감 직전(≈15:20)에 **강하게 마감한** 종목을 매수해서 **다음날
아침 시가**(갭상승)에 판다. "갭상승은 전날의 강한 마감이 예고한다"는 가설을
규칙으로 옮긴 것.

전날(=매수 당일)이 보여야 하는 특징(모두 종가 확정 후 판단 → 미래 참조 없음):
  1. 강한 양봉      : 당일 상승률이 up_min 이상, 종가 > 시가
  2. 고가권 마감     : 종가가 당일 (고-저) 범위의 상단(close_pos_min 이상)에서 마감
  3. 거래량 급증     : 당일 거래량 ≥ 20일 평균 × vol_mult
  4. 돌파           : 종가가 최근 breakout_lookback 일 고가를 상향 돌파
  5. 단기 상승 정렬  : 5일선 > 20일선, 종가 > 5일선
  6. 과열 아님       : RSI < rsi_high (상한가 따라잡기식 과열 배제)

체결 가정(engine.py): 신호가 뜬 날 '종가'로 매수 → 다음날 '시가'로 매도.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# 지표는 눌림목 전략과 공유 (중복 구현 방지)
from swing.strategy import sma, rsi, atr  # noqa: F401


@dataclass
class ClosingParams:
    # 이동평균
    ma_short: int = 5
    ma_mid: int = 20
    # 강한 마감(양봉) 조건
    up_min: float = 0.02          # 당일 상승률(전일종가 대비) ≥ +2%
    close_pos_min: float = 0.60   # 종가가 당일 (고-저) 범위 상단 60% 이상에서 마감
    # 거래량
    vol_ma: int = 20
    vol_mult: float = 1.5         # 당일 거래량 ≥ 20일 평균 × 1.5
    # 돌파
    breakout_lookback: int = 20   # 최근 N일 고가 대비 돌파 여부
    # 과열 필터
    rsi_period: int = 14
    rsi_high: float = 80.0        # RSI 이 값 이상이면 과열로 보고 제외
    # 스터디용: 다음날 시가가 이만큼 이상 뜨면 '갭상승'으로 라벨링
    gap_threshold: float = 0.01   # +1%
    # 오버나이트 왕복 비용(매수수수료+매도수수료+거래세+슬리피지 대략치)
    cost: float = 0.0025
    # 매수 슬리피지: 실제 매수는 15:20 동시호가 직전이라 공식 종가(15:30)와 다를 수
    # 있다. 이 값(양수)만큼 종가보다 비싸게 산다고 보수적으로 가정한다.
    entry_slippage: float = 0.0


# ---------------------------------------------------------------------------
# 특징(feature) 부착
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame, p: ClosingParams) -> pd.DataFrame:
    """각 봉에 종가매매 판단에 쓰는 특징 컬럼을 붙인다."""
    out = df.copy()
    prev_close = out["Close"].shift(1)

    out["ma_s"] = sma(out["Close"], p.ma_short)
    out["ma_m"] = sma(out["Close"], p.ma_mid)
    out["rsi"] = rsi(out["Close"], p.rsi_period)
    out["vol_ma"] = sma(out["Volume"], p.vol_ma)

    # 당일 상승률(전일 종가 대비)
    out["day_ret"] = out["Close"] / prev_close - 1
    # 고가권 마감 정도: 0(저가 마감) ~ 1(고가 마감)
    rng = (out["High"] - out["Low"]).replace(0, np.nan)
    out["close_pos"] = ((out["Close"] - out["Low"]) / rng).clip(0, 1).fillna(0.5)
    # 거래량 배수(20일 평균 대비)
    out["vol_ratio"] = out["Volume"] / out["vol_ma"]
    # 돌파: 직전 breakout_lookback 일 고가(오늘 제외) 상향 돌파
    out["prior_high"] = out["High"].rolling(p.breakout_lookback).max().shift(1)
    out["breakout"] = out["Close"] >= out["prior_high"]

    # 라벨(스터디용): 다음날 시가 갭 = 다음 시가 / 오늘 종가 - 1
    out["next_open"] = out["Open"].shift(-1)
    out["gap"] = out["next_open"] / out["Close"] - 1
    out["gap_up"] = out["gap"] >= p.gap_threshold
    return out


# ---------------------------------------------------------------------------
# 매수 신호
# ---------------------------------------------------------------------------
def generate_signals(df: pd.DataFrame, p: ClosingParams) -> pd.DataFrame:
    """각 봉에 대해 종가 매수 신호(entry) 컬럼을 붙여 반환.

    entry[i] == True → i일 종가로 매수하고 (i+1)일 시가로 매도.
    """
    d = add_features(df, p)

    strong_close = (
        (d["day_ret"] >= p.up_min)        # 강한 상승
        & (d["Close"] > d["Open"])        # 양봉
        & (d["close_pos"] >= p.close_pos_min)  # 고가권 마감
    )
    volume = d["vol_ratio"] >= p.vol_mult
    trend = (d["ma_s"] > d["ma_m"]) & (d["Close"] > d["ma_s"])
    not_hot = d["rsi"] < p.rsi_high

    entry = (strong_close & volume & d["breakout"] & trend & not_hot).fillna(False)
    d["entry"] = entry
    return d


def _pick_row(code: str, d: pd.DataFrame, p: ClosingParams) -> dict:
    """가장 최근 봉을 후보 dict 로 변환 (대시보드/알림 공용 형태)."""
    last = d.iloc[-1]
    close = float(last["Close"])
    return {
        "code": code,
        "date": str(d.index[-1].date()),
        "close": round(close, 2),
        "day_ret": round(float(last["day_ret"]) * 100, 2),   # 당일 상승률 %
        "close_pos": round(float(last["close_pos"]) * 100),  # 고가권 마감 정도 %
        "vol_ratio": round(float(last["vol_ratio"]), 2),     # 거래량 배수
        "rsi": round(float(last["rsi"]), 1),
        # 참고용 예상 손익(전일 시가 갭 통계와 무관한 규칙 표시)
        "target_hint": round(close * (1 + p.gap_threshold), 2),
    }


def latest_picks(universe: dict[str, pd.DataFrame], p: ClosingParams) -> list[dict]:
    """각 종목의 '가장 최근 봉'에서 종가 매수 신호가 뜬 종목만 추려 반환.

    → 장 마감 직전 스캔 결과. 그대로 알림/웹 대시보드로 넘긴다.
    """
    hits = []
    for code, df in universe.items():
        if len(df) < p.ma_mid + p.breakout_lookback + 5:
            continue
        d = generate_signals(df, p)
        if bool(d.iloc[-1]["entry"]):
            hits.append(_pick_row(code, d, p))
    # 강한 신호부터: 거래량 배수 × 당일 상승률 로 대략 정렬
    hits.sort(key=lambda x: x["vol_ratio"] * max(x["day_ret"], 0), reverse=True)
    return hits
