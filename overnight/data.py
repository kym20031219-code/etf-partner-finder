"""오버나이트 검증용 합성 데이터.

swing.data.make_synthetic 로 만든 현실적인 일봉에, **강하게 마감한 날 다음 시가가
살짝 더 뜨는** 약한(그리고 노이즈가 큰) 갭 편향을 주입한다. 실제 시장의 '종가매매
엣지'가 존재한다면 얼마나 미약하고 노이즈에 묻히는지를 흉내 내기 위함이다.

⚠️ 어디까지나 엔진·분석 로직 검증용 합성 데이터다. 여기서 나오는 승률·기대수익률은
**실제 시장 성과가 아니다.** 진짜 숫자는 `--source real` 로 회원님 PC나 GitHub
Actions(네트워크 열린 환경)에서 돌려야 나온다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing.data import make_synthetic


def _inject_gap_bias(df: pd.DataFrame, strength: float, noise: float, seed: int) -> pd.DataFrame:
    """강한 마감(양봉·고가권·거래량) 다음날 시가에 완만한 갭 편향을 더한다."""
    rng = np.random.default_rng(seed + 777)
    out = df.copy()
    close = out["Close"].to_numpy(float)
    high = out["High"].to_numpy(float)
    low = out["Low"].to_numpy(float)
    openp = out["Open"].to_numpy(float)
    vol = out["Volume"].to_numpy(float)

    prev_close = np.concatenate([[close[0]], close[:-1]])
    day_ret = close / prev_close - 1
    rng_hl = np.maximum(high - low, 1e-9)
    close_pos = np.clip((close - low) / rng_hl, 0, 1)
    vol_ma = pd.Series(vol).rolling(20).mean().to_numpy()
    vol_ratio = np.divide(vol, vol_ma, out=np.ones_like(vol), where=vol_ma > 0)

    # 마감 강도 점수 (0 근처~) : 상승률·고가권·거래량이 모두 좋을 때 커진다
    score = (
        np.clip(day_ret / 0.03, 0, 2)          # +3%면 1.0
        * np.clip((close_pos - 0.5) / 0.5, 0, 1)
        * np.clip((vol_ratio - 1) / 1.0, 0, 1.5)
    )
    # 다음날 시가에 (엣지 + 큰 노이즈) 를 반영
    gap = strength * score + rng.normal(0, noise, len(close))
    gap = np.concatenate([[0.0], gap[:-1]])    # score[i] 는 open[i+1] 에 작용

    new_open = close * np.concatenate([[1.0], np.ones(len(close) - 1)])  # placeholder
    new_open = prev_close * (1 + gap)          # 갭은 전일 종가 기준
    new_open[0] = openp[0]
    # 시가는 당일 고저 사이로 정렬 (지나친 붕괴 방지)
    new_open = np.clip(new_open, low * 0.98, high * 1.02)

    out["Open"] = new_open
    # 시가가 바뀌면 고/저도 시가를 포함하도록 살짝 넓힌다
    out["High"] = np.maximum(out["High"], out["Open"])
    out["Low"] = np.minimum(out["Low"], out["Open"])
    return out


def overnight_universe(
    n: int = 30,
    days: int = 750,
    strength: float = 0.011,   # 엣지: 마감 강도 1.0당 시가 +1.1%
    noise: float = 0.010,      # 익일 갭 노이즈 표준편차 1.0%
) -> dict[str, pd.DataFrame]:
    """검증용 합성 종목 묶음 (완만한 종가매매 엣지 주입)."""
    uni = {}
    for i in range(n):
        code = f"SYN{i:03d}"
        base = make_synthetic(code, days=days, seed=i)
        uni[code] = _inject_gap_bias(base, strength=strength, noise=noise, seed=i)
    return uni
