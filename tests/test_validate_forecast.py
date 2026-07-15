"""전망 유효성 검증 모듈 테스트 (네트워크 불필요)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from swing.kospi_forecast import score_panel
from validate_forecast import (
    forward_return, factor_ic, score_vs_forward, _pearson, _rank_ic, _tstat,
)
from tests.test_kospi_forecast import _full_bundle


def test_corr_helpers():
    x = np.arange(60, dtype=float)
    assert _pearson(x, x) > 0.999
    assert _rank_ic(x, -x) < -0.999
    assert abs(_pearson(x, np.ones(60))) < 1e-9   # 상수 → 0
    assert _tstat(0.0, 100) == 0.0
    assert _tstat(0.3, 200) > 2                    # 유의


def test_forward_return_alignment():
    idx = pd.bdate_range("2023-01-02", periods=100)
    close = pd.Series(np.linspace(100, 200, 100), index=idx)
    fwd = forward_return(close, idx, 20)
    # t 시점 fwd = close[t+20]/close[t]-1, 끝 20개는 NaN
    assert np.isnan(fwd.iloc[-1])
    i = 10
    assert abs(fwd.iloc[i] - (close.iloc[i+20]/close.iloc[i]-1)) < 1e-9


def test_factor_ic_structure():
    b = _full_bundle(n=400, seed=2)
    panel = score_panel(b, step=1, warmup=160)
    fwd = forward_return(b.kospi["Close"], panel.index, 20)
    rows = factor_ic(panel, fwd)
    keys = {r["key"] for r in rows}
    assert "_composite" in keys and len({"macro","korea","earnings","flows",
                                         "valuation","technical"} & keys) == 6
    assert rows[0]["key"] == "_composite"          # 종합점수가 맨 위
    for r in rows:
        assert -1 <= r["ic"] <= 1 and -1 <= r["rank_ic"] <= 1
        assert isinstance(r["significant"], bool)
        # 유의하지 않은 개별 팩터만 제거후보
        if r["key"] != "_composite":
            assert r["removal_candidate"] == (not r["significant"])
        else:
            assert r["removal_candidate"] is False


def test_score_vs_forward_buckets():
    b = _full_bundle(n=500, seed=6)
    panel = score_panel(b, step=1, warmup=160)
    fwd = forward_return(b.kospi["Close"], panel.index, 20)
    svf = score_vs_forward(panel, fwd)
    assert svf["bucket_table"] and svf["scatter"]
    tot = sum(r["n"] for r in svf["bucket_table"])
    assert tot == svf["n"]                          # 구간 표본 합 == 전체
    for r in svf["bucket_table"]:
        assert 0.0 <= r["up_ratio"] <= 1.0
    for p in svf["scatter"]:
        assert 0 <= p["score"] <= 100


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
