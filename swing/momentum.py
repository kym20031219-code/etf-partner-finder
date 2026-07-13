"""모멘텀 신고가 돌파 스윙 전략 (며칠~수주간 강한 상승 포착).

설계 목표(사용자 요구):
  - **며칠간 강한 상승**을 노린다 → N일 신고가 돌파 + 강한 추세/거래량에서만 진입
  - **손실은 짧게** → 진입 즉시 ATR 배수의 초기 손절 (작게)
  - **수익은 길게** → 고정 익절 없음. 샹들리에(최고가−ATR 배수) 트레일링으로 추세를 끝까지 탄다

`swing/breakout.py` 의 아이디어를 이어받되, **모든 진입 조건·청산 배수·랭킹 점수
가중치를 파라미터화**해 백테스트로 최적화할 수 있게 만들었다. 미래참조 없음(신호
다음 봉 시가 체결), 왕복 비용 반영(engine.ROUND_TRIP_COST).

⚠️ 이 모듈은 종목 '추천 점수'와 백테스트만 계산한다. 실제 매수/매도 주문·증권사
   계좌 접근은 포함하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import ROUND_TRIP_COST, Trade
from .strategy import atr, rsi, sma


def _clip01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# 파라미터
# ---------------------------------------------------------------------------
@dataclass
class MomentumParams:
    # --- 진입 (돌파) ---
    entry_lookback: int = 55      # 직전 N일 신고가를 종가로 돌파하면 진입
    ma_trend: int = 120           # 종가가 이 이평선 위 (장기 상승 추세)
    trend_slope_lb: int = 20      # ma_trend 가 이 기간 전보다 높아야(우상향)
    mom_lookback: int = 20        # 모멘텀(ROC) 계산 기간
    mom_min: float = 0.0          # 진입 시 최소 ROC (0 이상=상승 중)
    vol_ma: int = 20
    vol_mult: float = 1.3         # 돌파일 거래량 ≥ 20일 평균 × 이 값
    rsi_period: int = 14
    rsi_min: float = 50.0         # 진입 시 최소 RSI (약한 종목 배제)
    rsi_max: float = 85.0         # 과열(블로우오프) 배제 상한
    # --- 청산 (손실 짧게 / 수익 길게) ---
    atr_period: int = 14
    init_stop_atr: float = 2.0    # 초기 손절 = 진입가 − 2.0×ATR  (작게 = 손실 짧게)
    trail_atr: float = 3.5        # 트레일 = 보유중 최고가 − 3.5×ATR (넓게 = 수익 길게)
    exit_lookback: int = 20       # 종가가 N일 최저가 아래로 마감하면 청산(추세 붕괴)
    max_hold: int = 60            # 최대 보유 거래일 (추세추종이라 넉넉히)


@dataclass
class ScoreWeights:
    """랭킹 점수 가중치 + 정규화 스케일. 백테스트로 최적화하는 대상."""
    w_trend: float = 0.25
    w_breakout: float = 0.25
    w_momentum: float = 0.25
    w_volume: float = 0.10
    w_rsi: float = 0.15
    # 정규화 스케일(기준값)
    trend_above_cap: float = 0.20   # 종가가 추세선보다 +20% 위면 만점
    trend_slope_cap: float = 0.06   # 추세선 기울기(N일) +6%면 만점
    breakout_band: float = 0.10     # 종가가 N일고점 −10% 이하로 떨어지면 0점
    mom_cap: float = 0.20           # ROC +20%면 만점
    vol_cap: float = 2.0            # 거래량이 20일 평균의 2배면 만점
    rsi_ideal: float = 62.0         # 모멘텀에 이상적인 RSI
    rsi_span: float = 25.0          # 이상값에서 이만큼 벗어나면 0점

    def normalized(self) -> "ScoreWeights":
        """가중치 합이 1이 되도록 정규화한 사본."""
        s = self.w_trend + self.w_breakout + self.w_momentum + self.w_volume + self.w_rsi
        if s <= 0:
            return ScoreWeights()
        return ScoreWeights(
            w_trend=self.w_trend / s, w_breakout=self.w_breakout / s,
            w_momentum=self.w_momentum / s, w_volume=self.w_volume / s,
            w_rsi=self.w_rsi / s,
            trend_above_cap=self.trend_above_cap, trend_slope_cap=self.trend_slope_cap,
            breakout_band=self.breakout_band, mom_cap=self.mom_cap,
            vol_cap=self.vol_cap, rsi_ideal=self.rsi_ideal, rsi_span=self.rsi_span,
        )


# ---------------------------------------------------------------------------
# 지표 + 하위 점수
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame, p: MomentumParams) -> pd.DataFrame:
    d = df.copy()
    d["atr"] = atr(d, p.atr_period)
    d["ma_t"] = sma(d["Close"], p.ma_trend)
    d["ma_t_prev"] = d["ma_t"].shift(p.trend_slope_lb)
    d["vol_ma"] = sma(d["Volume"], p.vol_ma)
    d["rsi"] = rsi(d["Close"], p.rsi_period)
    d["roc"] = d["Close"] / d["Close"].shift(p.mom_lookback) - 1
    d["prior_high"] = d["High"].rolling(p.entry_lookback).max().shift(1)
    d["exit_low"] = d["Low"].rolling(p.exit_lookback).min().shift(1)
    return d


def _row_scores(row: pd.Series, w: ScoreWeights) -> dict[str, float]:
    """지표가 부착된 한 봉에서 0~100 하위 점수 + 총점 계산."""
    close = float(row["Close"])
    ma_t = float(row["ma_t"])
    ma_t_prev = float(row["ma_t_prev"])
    prior_high = float(row["prior_high"])
    roc = float(row["roc"])
    vol = float(row["Volume"])
    vol_ma = float(row["vol_ma"])
    rsi_v = float(row["rsi"])

    # 추세: 추세선 위 여유 + 추세선 기울기
    above = _clip01((close / ma_t - 1) / w.trend_above_cap) if ma_t > 0 else 0.0
    slope = _clip01((ma_t / ma_t_prev - 1) / w.trend_slope_cap) if ma_t_prev > 0 else 0.0
    trend = 100 * (0.5 * above + 0.5 * slope)

    # 돌파: 종가가 N일 고점에 근접/돌파할수록 만점
    gap = (close / prior_high - 1) if prior_high > 0 else -1.0  # +면 돌파
    breakout = 100 * _clip01((gap + w.breakout_band) / w.breakout_band)

    # 모멘텀: 최근 ROC
    momentum = 100 * _clip01(roc / w.mom_cap)

    # 거래량: 20일 평균 대비 실린 정도
    ratio = vol / vol_ma if vol_ma > 0 else 0.0
    volume = 100 * _clip01(ratio / w.vol_cap)

    # RSI: 강하되 과열은 아닌 구간
    rsi_s = 100 * _clip01(1 - abs(rsi_v - w.rsi_ideal) / w.rsi_span)

    total = (
        w.w_trend * trend + w.w_breakout * breakout + w.w_momentum * momentum
        + w.w_volume * volume + w.w_rsi * rsi_s
    )
    return {
        "total": round(total, 1), "trend": round(trend, 1),
        "breakout": round(breakout, 1), "momentum": round(momentum, 1),
        "volume": round(volume, 1), "rsi_score": round(rsi_s, 1),
        "close": round(close, 2), "rsi": round(rsi_v, 1),
        "roc_pct": round(roc * 100, 1),
        "high_gap_pct": round(gap * 100, 1),
    }


# ---------------------------------------------------------------------------
# 백테스트 매매 추출
# ---------------------------------------------------------------------------
def _entry_signal(d: pd.DataFrame, p: MomentumParams, regime: pd.Series | None) -> np.ndarray:
    sig = (
        (d["Close"] > d["prior_high"])                 # N일 신고가 돌파
        & (d["Close"] > d["ma_t"])                     # 장기추세 위
        & (d["ma_t"] > d["ma_t_prev"])                 # 추세선 우상향
        & (d["roc"] >= p.mom_min)                      # 모멘텀(+)
        & (d["Volume"] >= d["vol_ma"] * p.vol_mult)    # 거래량 확인
        & (d["rsi"] >= p.rsi_min) & (d["rsi"] <= p.rsi_max)
    ).fillna(False)
    if regime is not None:
        on = regime.reindex(d.index).ffill().fillna(False).astype(bool)
        sig = sig & on
    return sig.to_numpy(bool)


def extract_trades_momentum(
    code: str, df: pd.DataFrame, p: MomentumParams, regime: pd.Series | None = None
) -> list[Trade]:
    """돌파 진입 → ATR 초기손절 + 샹들리에 트레일 → 매매 리스트."""
    d = add_features(df, p)
    o = d["Open"].to_numpy(float)
    h = d["High"].to_numpy(float)
    low = d["Low"].to_numpy(float)
    c = d["Close"].to_numpy(float)
    atr_a = d["atr"].to_numpy(float)
    exit_low = d["exit_low"].to_numpy(float)
    dates = d.index
    entry_sig = _entry_signal(d, p, regime)

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
            if low[j] <= stop:                               # 손절/트레일 (갭이면 시가)
                exit_px = min(o[j], stop) if o[j] <= stop else stop
                exit_i, reason = j, ("stop" if stop <= init_stop else "trail")
                break
            if np.isfinite(exit_low[j]) and c[j] < exit_low[j]:  # 추세 붕괴(채널 이탈)
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


def trades_with_features(
    code: str, df: pd.DataFrame, p: MomentumParams, w: ScoreWeights,
    regime: pd.Series | None = None,
) -> list[tuple[dict, float]]:
    """각 진입 신호 봉의 하위 점수와 그 매매의 실현 수익률을 짝지어 반환.

    → 랭킹 점수 가중치가 '실제 성과'를 얼마나 잘 예측하는지 검증하는 데 쓴다.
    """
    d = add_features(df, p)
    entry_sig = _entry_signal(d, p, regime)
    trades = extract_trades_momentum(code, df, p, regime=regime)
    ret_by_entry = {t.entry_date: t.ret for t in trades}

    out: list[tuple[dict, float]] = []
    for i in range(len(d) - 1):
        if not entry_sig[i]:
            continue
        entry_date = d.index[i + 1]  # 다음 봉 진입
        if entry_date not in ret_by_entry:
            continue  # 겹치는 신호는 엔진이 건너뜀
        row = d.iloc[i]
        if not np.isfinite(row.get("prior_high", np.nan)):
            continue
        out.append((_row_scores(row, w), ret_by_entry[entry_date]))
    return out


# ---------------------------------------------------------------------------
# 대시보드/알림용: 최신 봉 랭킹
# ---------------------------------------------------------------------------
def rank_universe(
    universe: dict[str, pd.DataFrame],
    p: MomentumParams | None = None,
    w: ScoreWeights | None = None,
    names: dict[str, str] | None = None,
    top: int | None = None,
    regime: pd.Series | None = None,
) -> list[dict]:
    """각 종목의 '가장 최근 봉'을 점수화해 총점 내림차순으로 반환.

    - 장기추세 위(종가>추세선·추세선 우상향)인 종목만 대상.
    - 최신 봉이 돌파 진입 조건까지 충족하면 signal="buy", 아니면 "watch".
    """
    p = p or MomentumParams()
    w = (w or ScoreWeights()).normalized()
    names = names or {}

    rows: list[dict] = []
    for code, df in universe.items():
        if len(df) < p.ma_trend + p.trend_slope_lb:
            continue
        d = add_features(df, p)
        last = d.iloc[-1]
        if not (np.isfinite(last["ma_t"]) and np.isfinite(last["prior_high"])):
            continue
        close = float(last["Close"])
        ma_t = float(last["ma_t"])
        ma_t_prev = float(last["ma_t_prev"])
        if not (close > ma_t and ma_t > ma_t_prev):   # 장기 상승추세만
            continue

        sc = _row_scores(last, w)
        is_buy = bool(_entry_signal(d, p, regime)[-1])
        entry_atr = float(last["atr"]) if np.isfinite(last["atr"]) else close * 0.03
        rows.append(
            {
                "code": code, "name": names.get(code, code),
                "date": str(d.index[-1].date()),
                "signal": "buy" if is_buy else "watch",
                # 참고용 리스크 관리 레벨 (초기 손절만; 익절은 트레일링이라 미고정)
                "stop": round(close - p.init_stop_atr * entry_atr, 2),
                "trail_hint": round(close - p.trail_atr * entry_atr, 2),
                **sc,
            }
        )

    rows.sort(key=lambda r: r["total"], reverse=True)
    for idx, r in enumerate(rows, 1):
        r["rank"] = idx
    if top is not None:
        rows = rows[:top]
    return rows
