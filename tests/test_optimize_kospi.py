"""코스피 가중치 최적화기 테스트 (네트워크 불필요).

  python -m pytest tests/
  python tests/test_optimize_kospi.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from swing.kospi_forecast import ForecastWeights, composite_from, factor_scores
from optimize_kospi import (
    FACTORS, rank_ic, hit_rate, build_panel, attach_forward, optimize_horizon,
)

# 오프라인 합성 번들 재사용
from tests.test_kospi_forecast import _full_bundle


def test_rank_ic_perfect_and_zero():
    x = np.arange(50, dtype=float)
    assert rank_ic(x, x) > 0.999           # 완전 단조 → IC≈1
    assert rank_ic(x, -x) < -0.999          # 완전 역단조 → IC≈-1
    rng = np.random.default_rng(0)
    ic = rank_ic(rng.normal(size=400), rng.normal(size=400))
    assert abs(ic) < 0.2                    # 무상관 → 0 근처


def test_hit_rate_bounds():
    scores = np.array([60, 40, 55, 45, 50])
    fwd = np.array([0.02, -0.01, 0.03, 0.01, -0.02])
    hr = hit_rate(scores, fwd)
    assert 0.0 <= hr <= 1.0


def test_composite_helpers_match_engine():
    b = _full_bundle(n=200)
    sc = factor_scores(b)
    assert set(sc) == set(FACTORS)
    # 균등 가중이면 종합=평균
    eq = ForecastWeights(1, 1, 1, 1, 1, 1)
    assert abs(composite_from(sc, eq) - np.mean(list(sc.values()))) < 1e-6


def test_panel_and_forward_no_lookahead():
    b = _full_bundle(n=320, seed=5)
    panel = build_panel(b, step=4, warmup=160)
    assert len(panel) > 10
    assert set(FACTORS).issubset(panel.columns)
    h = 20
    ph = attach_forward(panel, b.kospi["Close"], h)
    # 미래수익 라벨이 붙은 마지막 시점은 전체 마지막보다 최소 h 이전이어야(미래참조 없음)
    assert ph.index[-1] <= b.kospi.index[-1 - h]
    # fwd 값이 실제 t→t+h 수익률과 일치하는지 한 점 검증
    d0 = ph.index[0]
    i0 = b.kospi.index.get_loc(d0)
    expect = b.kospi["Close"].iloc[i0 + h] / b.kospi["Close"].iloc[i0] - 1
    assert abs(ph["fwd"].iloc[0] - expect) < 1e-9


def test_optimize_respects_cap_and_normalizes():
    b = _full_bundle(n=500, seed=2)
    panel = build_panel(b, step=2, warmup=160)
    ph = attach_forward(panel, b.kospi["Close"], 40)
    res = optimize_horizon(ph, samples=300, train_frac=0.65, seed=1,
                           max_weight=0.45, shrink=0.35)
    w = res["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-6          # 정규화
    # shrink(기본 prior 혼합) 덕에 어떤 팩터도 0 이나 몰빵이 아니어야 한다
    assert all(v > 0 for v in w.values())
    assert max(w.values()) < 0.6
    # 지표가 모두 존재
    for key in ("ic_train", "ic_val", "hit_val", "baseline_ic_val"):
        assert key in res


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
