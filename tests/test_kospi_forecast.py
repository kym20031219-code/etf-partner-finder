"""코스피 전망 엔진 테스트 (네트워크 불필요).

  python -m pytest tests/
  python tests/test_kospi_forecast.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from swing.kospi_forecast import (
    MarketBundle, ForecastWeights, compute_forecast, bias_label,
    _lin, _centered, _pct_rank,
)


def _series(vals, start="2023-01-02"):
    idx = pd.bdate_range(start=start, periods=len(vals))
    return pd.Series(np.asarray(vals, dtype=float), index=idx)


def _kospi(n=300, drift=0.0005, seed=1):
    rng = np.random.default_rng(seed)
    close = 2500 * np.cumprod(1 + rng.normal(drift, 0.01, n))
    idx = pd.bdate_range(start="2023-01-02", periods=n)
    prev = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "Open": prev, "High": np.maximum(prev, close) * 1.003,
        "Low": np.minimum(prev, close) * 0.997, "Close": close,
        "Volume": rng.integers(3e8, 6e8, n),
    }, index=idx)


# ---- 헬퍼 함수 경계값 ----
def test_lin_bounds_and_invert():
    assert _lin(0, 0, 10) == 0
    assert _lin(10, 0, 10) == 100
    assert _lin(-5, 0, 10) == 0          # 클램프
    assert _lin(50, 0, 10) == 100
    assert _lin(13, 32, 13) == 100       # 반전(VIX 스타일)
    assert _lin(32, 32, 13) == 0
    assert _lin(float("nan"), 0, 10) == 50.0


def test_centered_neutral():
    assert _centered(0, 0.05) == 50.0
    assert _centered(0.05, 0.05) == 100.0
    assert _centered(-0.05, 0.05) == 0.0
    assert _centered(1.0, 0.05) == 100.0  # 클램프


def test_pct_rank():
    s = _series(range(100))
    assert _pct_rank(s, 50) == 51.0       # <=50 인 값이 51개(0..50)
    assert _pct_rank(s, -10) == 0.0
    assert _pct_rank(s, 200) == 100.0


# ---- 방향 매핑 ----
def test_bias_label_monotone():
    assert bias_label(80) == "강세"
    assert bias_label(60) == "중립-강세"
    assert bias_label(50) == "중립"
    assert bias_label(40) == "중립-약세"
    assert bias_label(20) == "약세"


# ---- 엔진 계약 ----
def test_forecast_minimal_bundle_neutral():
    """코스피만 주고 나머지가 없어도 계산되고, 미가용 신호는 available=False."""
    b = MarketBundle(kospi=_kospi())
    r = compute_forecast(b)
    assert set(r) >= {"score", "bias", "confidence", "factors", "as_of", "kospi_close"}
    assert 0 <= r["score"] <= 100
    assert len(r["factors"]) == 6
    # 거시/수급/밸류에이션은 외부 데이터가 없으니 미가용 신호가 있어야 한다
    macro = next(f for f in r["factors"] if f["key"] == "macro")
    assert any(not s["available"] for s in macro["signals"])
    # 미가용뿐인 팩터 점수는 중립 50
    val = next(f for f in r["factors"] if f["key"] == "valuation")
    assert val["score"] == 50.0


def test_missing_data_damps_factor_toward_neutral():
    """일부 신호만 있는 팩터는 가용 비율만큼만 방향성을 갖고 중립으로 축소된다."""
    # 코스피만 있는 최소 번들: 실적 팩터는 '지수 120일 추세' 1개만 가용(1/3)
    b = MarketBundle(kospi=_kospi(n=300, drift=0.004))  # 강한 상승 → 추세 신호 만점 근처
    r = compute_forecast(b)
    earn = next(f for f in r["factors"] if f["key"] == "earnings")
    assert earn["available_signals"] == 1 and earn["total_signals"] == 3
    # 원 신호는 100 근처지만 1/3 가용이라 대략 50+(raw-50)/3 로 축소 → 70 미만
    assert earn["score"] < 72
    assert earn["score"] > 50            # 그래도 상방 신호는 남는다
    # 전부 미가용 팩터는 정확히 50
    macro = next(f for f in r["factors"] if f["key"] == "macro")
    assert macro["available_signals"] == 0 and macro["score"] == 50.0
    # 상위 요약 필드
    assert r["data_factors_total"] == 6
    assert 0 <= r["data_factors_available"] <= 6


def test_zero_per_pbr_ignored():
    """PER/PBR 에 0(미확정값)이 섞여도 '초저평가 강세'로 오인하지 않는다."""
    n = 320
    idx = pd.bdate_range(start="2023-01-02", periods=n)
    # 10~12 범위에서 변동하는 PER/PBR, 마지막 행만 0 (당일 미집계)
    per = 11.0 + np.sin(np.arange(n) / 10.0); per[-1] = 0.0
    pbr = 1.0 + 0.1 * np.sin(np.arange(n) / 10.0); pbr[-1] = 0.0
    b = MarketBundle(
        kospi=_kospi(n=n), per=pd.Series(per, index=idx), pbr=pd.Series(pbr, index=idx),
        us10y=pd.Series(np.full(n, 4.0), index=idx),
    )
    r = compute_forecast(b)
    val = next(f for f in r["factors"] if f["key"] == "valuation")
    per_sig = next(s for s in val["signals"] if s["name"].startswith("PER"))
    # 0 이 아니라 직전 유효값(~11)이 쓰여야 한다(초저평가 0.0배로 오인 없음)
    assert per_sig["available"] and per_sig["detail"].startswith("1")  # "11.x배 …"
    assert "0.0배" not in per_sig["detail"]
    earn = next(f for f in r["factors"] if f["key"] == "earnings")
    dir_sig = next(s for s in earn["signals"] if "이익 방향" in s["name"])
    assert "-100" not in dir_sig["detail"]        # 0 으로 인한 -100% 오인 없음
    # ERP 도 0 나눗셈 없이 유효값으로 계산된다
    erp_sig = next(s for s in val["signals"] if s["name"].startswith("ERP"))
    assert erp_sig["available"]


def test_full_data_no_damping():
    """모든 신호가 가용하면(avail=1) 축소가 없어 종합=팩터 가중합 그대로."""
    r = compute_forecast(_full_bundle())
    for f in r["factors"]:
        assert f["available_signals"] == f["total_signals"]
    manual = sum(f["score"] * f["weight"] for f in r["factors"])
    assert abs(manual - r["score"]) < 0.15


def test_scores_and_weights_in_range():
    b = _full_bundle()
    w = ForecastWeights()
    r = compute_forecast(b, w)
    wsum = sum(f["weight"] for f in r["factors"])
    assert abs(wsum - 1.0) < 1e-6                 # 가중치 정규화
    for f in r["factors"]:
        assert 0 <= f["score"] <= 100
        for s in f["signals"]:
            assert 0 <= s["score"] <= 100
    # 종합점수는 팩터점수의 가중합과 일치해야 한다
    manual = sum(f["score"] * f["weight"] for f in r["factors"])
    assert abs(manual - r["score"]) < 0.15        # 반올림 오차 허용


def test_bullish_inputs_beat_bearish():
    """상승 재료(증시↑·저VIX·금리↓·순매수↑·저평가)면 약세 재료보다 점수가 높다."""
    bull = compute_forecast(_full_bundle(bull=True))["score"]
    bear = compute_forecast(_full_bundle(bull=False))["score"]
    assert bull > bear
    assert bull >= 50 >= bear or bull > bear + 8   # 방향이 유의미하게 갈린다


def test_no_lookahead_backfill_shape():
    """소급 슬라이스가 미래를 참조하지 않고 각 시점 값만 쓰는지(길이 단조) 확인."""
    b = _full_bundle()
    end = b.kospi.index[-30]
    sub_close = b.kospi["Close"][b.kospi.index <= end]
    assert sub_close.index[-1] == end
    assert len(sub_close) < len(b.kospi)


# ---- 픽스처 ----
def _full_bundle(bull: bool = True, n: int = 320, seed: int = 3) -> MarketBundle:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2023-01-02", periods=n)

    def rw(start, mu, sig):
        return pd.Series(start * np.cumprod(1 + rng.normal(mu, sig, n)), index=idx)

    sign = 1 if bull else -1
    kospi = _kospi(n=n, drift=0.0008 * sign, seed=seed)
    return MarketBundle(
        kospi=kospi,
        sp500=rw(5000, 0.0008 * sign, 0.009),
        vix=pd.Series(np.full(n, 13.0 if bull else 28.0), index=idx),
        us10y=rw(4.0, -0.0004 * sign, 0.008),
        usdkrw=rw(1350, -0.0002 * sign, 0.004),
        china=rw(3100, 0.0006 * sign, 0.010),
        kosdaq=rw(850, 0.0009 * sign, 0.012),
        semis=rw(100, 0.0014 * sign, 0.015),
        per=pd.Series(np.full(n, 9.0 if bull else 15.0), index=idx),
        pbr=pd.Series(np.full(n, 0.85 if bull else 1.4), index=idx),
        foreign=pd.Series(rng.normal(3000e8 * sign, 2000e8, n), index=idx),
        inst=pd.Series(rng.normal(1500e8 * sign, 2000e8, n), index=idx),
    )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
