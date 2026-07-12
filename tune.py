#!/usr/bin/env python3
"""파라미터 그리드 서치 (과최적화 방지용 학습/검증 분리 포함).

핵심: 데이터 기간을 앞(학습) / 뒤(검증) 로 나눠, **학습 구간에서 좋았던 파라미터가
검증 구간에서도 유지되는지**를 함께 본다. 학습에서만 좋고 검증에서 무너지는 조합은
과최적화이므로 버린다.

사용:
  # 실데이터 (네트워크 열린 환경)
  python tune.py --source real --market KOSPI --top 100 --start 2019-01-01

  # 오프라인 검증
  python tune.py --source synthetic --n 60 --days 1000

출력:
  - 콘솔에 검증 성적 상위 파라미터 순위
  - tune_results.csv (전체 조합 성적)
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from datetime import date

import pandas as pd

from swing import data as datamod
from swing.engine import extract_trades
from swing.metrics import trade_stats
from swing.strategy import PullbackParams

# 탐색할 격자 (실전에선 너무 넓히지 말 것 — 조합이 곱으로 늘어난다)
GRID = {
    "stop_pct": [0.04, 0.05, 0.07],
    "target_pct": [0.08, 0.10, 0.15],
    "rsi_low": [35, 40, 45],
    "pullback_touch": [0.015, 0.02, 0.03],
    "max_hold": [10, 15, 20],
}

MIN_TRADES = 20  # 이보다 표본이 적은 조합은 신뢰 불가로 제외


def score(stats: dict) -> float:
    """표본 수로 보정한 기대값. 매매가 적으면 점수를 깎아 우연을 배제."""
    if stats.get("trades", 0) < MIN_TRADES:
        return float("-inf")
    import math
    return stats["expectancy"] * math.sqrt(stats["trades"])


def run_universe(universe: dict, p: PullbackParams) -> dict:
    trades = []
    for code, df in universe.items():
        trades.extend(extract_trades(code, df, p))
    return trade_stats(trades)


def split_universe(universe: dict, ratio: float = 0.6) -> tuple[dict, dict]:
    """각 종목 시계열을 앞(학습)/뒤(검증)로 분할."""
    train, test = {}, {}
    for code, df in universe.items():
        k = int(len(df) * ratio)
        if k > 150 and len(df) - k > 150:
            train[code] = df.iloc[:k]
            test[code] = df.iloc[k:]
    return train, test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "synthetic"], default="synthetic")
    ap.add_argument("--market", default="KOSPI")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--days", type=int, default=1000)
    ap.add_argument("--top-show", type=int, default=12)
    ap.add_argument("--out-dir", default=".", help="결과 저장 폴더 (results 등)")
    args = ap.parse_args()

    if args.source == "real":
        codes = datamod.fetch_universe(args.market, args.top)
        universe = {}
        for code in codes:
            try:
                df = datamod.fetch_ohlcv(code, args.start, args.end)
                if len(df) > 400:
                    universe[code] = df
            except Exception:  # noqa: BLE001
                pass
    else:
        universe = datamod.synthetic_universe(n=args.n, days=args.days)

    train, test = split_universe(universe)
    print(f"종목 {len(universe)}개 · 학습 {len(train)} / 검증 {len(test)}")

    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"파라미터 조합 {len(combos)}개 평가 중...\n")

    rows = []
    base = PullbackParams()
    for combo in combos:
        overrides = dict(zip(keys, combo))
        p = replace(base, **overrides)
        tr = run_universe(train, p)
        te = run_universe(test, p)
        rows.append(
            {
                **overrides,
                "train_trades": tr.get("trades", 0),
                "train_winrate": round(tr.get("win_rate", 0) * 100, 1),
                "train_expect": round(tr.get("expectancy", 0) * 100, 2),
                "test_trades": te.get("trades", 0),
                "test_winrate": round(te.get("win_rate", 0) * 100, 1),
                "test_expect": round(te.get("expectancy", 0) * 100, 2),
                "test_payoff": round(te.get("payoff", 0), 2) if te.get("trades") else 0,
                "train_score": score(tr),
                "test_score": score(te),
            }
        )

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out_dir, "tune_results.csv"), index=False, encoding="utf-8-sig")

    # 학습에서 상위였던 조합을 '검증 성적' 순으로 정렬 → 강건한 조합이 위로
    robust = res[res["train_score"] > float("-inf")].copy()
    robust = robust.sort_values(["test_score", "train_score"], ascending=False)

    # 상위 조합을 JSON 으로도 저장 (워크플로우에서 안정적으로 읽기 위함)
    import json as _json
    top_cols = keys + ["train_trades", "train_winrate", "train_expect",
                       "test_trades", "test_winrate", "test_expect", "test_payoff"]
    top_records = robust[top_cols].head(15).to_dict(orient="records") if not robust.empty else []
    with open(os.path.join(args.out_dir, "tune_top.json"), "w", encoding="utf-8") as f:
        _json.dump(
            {"grid": GRID, "min_trades": MIN_TRADES,
             "universe_size": len(universe), "train": len(train), "test": len(test),
             "top": top_records},
            f, ensure_ascii=False, indent=2,
        )

    show_cols = keys + ["train_trades", "train_expect", "test_trades", "test_winrate", "test_expect", "test_payoff"]
    print("=" * 90)
    print("  검증 구간 성적 상위 파라미터 (과최적화 배제: 학습·검증 모두 표본 충분)")
    print("=" * 90)
    if robust.empty:
        print("  표본이 충분한 조합이 없습니다. 종목/기간을 늘리거나 조건을 완화하세요.")
    else:
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(robust[show_cols].head(args.top_show).to_string(index=False))
    print("\n[저장] tune_results.csv (전체 조합)")
    print("※ 검증(test)에서도 기대값이 (+)이고 학습과 크게 어긋나지 않는 조합을 고르세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
