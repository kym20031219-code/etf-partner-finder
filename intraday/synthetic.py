"""오프라인 검증용 합성 분봉 생성 (KIS 없이 로직 테스트).

하루(09:00~15:20, 1분봉 380개)를 목표 일간수익률·강도에 맞춰 생성한다.
"""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd


def make_minute_day(
    prev_close: float = 10_000.0,
    day_ret: float = 0.06,      # 목표 당일 상승률
    late_strength: float = 0.5, # 0(장초반 상승)~1(막판 상승) 편향
    base_min_vol: int = 3_000,
    seed: int | None = 0,
    date: str = "2026-07-13",
) -> pd.DataFrame:
    """1분봉 하루치 OHLCV 생성 (DatetimeIndex)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{date} 09:00", f"{date} 15:20", freq="1min")
    n = len(idx)

    # 하루 로그수익을 시간에 따라 배분: late_strength 클수록 뒤쪽에 몰림
    w = np.linspace(0.2, 1.8, n) ** (0.5 + late_strength * 2)
    w = w / w.sum()
    total_log = np.log1p(day_ret)
    drift = total_log * w
    noise = rng.normal(0, 0.0003, n)
    logret = drift + noise
    cum = np.cumsum(logret)
    # 종료점을 목표 수익률에 정확히 고정(노이즈로 인한 표류 보정) — 테스트 재현성
    cum = cum + (total_log - cum[-1]) * (np.arange(1, n + 1) / n)
    close = prev_close * np.exp(cum)

    prev = np.concatenate([[prev_close], close[:-1]])
    open_ = prev
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0006, n)))
    # 거래량: 변동 큰 분봉·장 초반/막판에 증가
    move = np.abs(logret)
    shape = 1 + 3 * (w / w.max())
    vol = (base_min_vol * shape * (1 + 40 * move) * rng.uniform(0.7, 1.3, n)).astype(int)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )
