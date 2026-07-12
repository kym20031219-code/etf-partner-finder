#!/usr/bin/env python3
"""돌파+국면 전략 강건성 검증.

1) 학습/검증 분리: 전체 기간을 앞 60%(학습) / 뒤 40%(검증)로 나눠, 검증 구간에서도
   매매당 기대값이 (+) 인지 본다. (매매를 진입일 기준으로 분할 → 워밍업 손실 없음)
2) 벤치마크: 같은 기간 KOSPI 지수 buy&hold 수익률과 비교. 전략이 '더 큰 위험을 지고도
   지수를 못 이기는' 것이면 의미 없다.

  python validate_breakout.py --market KOSPI --top 100 --start 2020-01-01 \
      --report-json results/breakout_validation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from swing import data as datamod
from swing.breakout import BreakoutParams, extract_trades_breakout
from swing.engine import simulate_portfolio
from swing.metrics import trade_stats, equity_stats
from swing.strategy import market_regime


def cagr(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) < 2:
        return {}
    total = s.iloc[-1] / s.iloc[0] - 1
    years = (s.index[-1] - s.index[0]).days / 365.25
    return {
        "total_return": float(total),
        "cagr": float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0,
        "years": round(years, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="KOSPI")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--split", type=float, default=0.6, help="학습 구간 비율")
    ap.add_argument("--regime-ma", type=int, default=120)
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    p = BreakoutParams()

    # 데이터
    codes = datamod.fetch_universe(args.market, args.top)
    universe = {}
    for code in codes:
        try:
            df = datamod.fetch_ohlcv(code, args.start, args.end)
            if len(df) > 200:
                universe[code] = df
        except Exception:  # noqa: BLE001
            pass
    idx = datamod.fetch_index("KS11", args.start, args.end)
    regime = market_regime(idx["Close"], args.regime_ma)
    print(f"종목 {len(universe)}개 · risk-on 비중 {regime.mean()*100:.0f}%", flush=True)

    # 분할 기준일 (지수 타임라인의 split 지점)
    split_date = idx.index[int(len(idx) * args.split)]
    print(f"학습/검증 분할일: {split_date.date()}", flush=True)

    # 전체 매매 → 진입일 기준 학습/검증 분할
    all_trades, train_trades, test_trades = [], [], []
    for code, df in universe.items():
        for t in extract_trades_breakout(code, df, p, regime=regime):
            all_trades.append(t)
            (train_trades if t.entry_date < split_date else test_trades).append(t)

    train_stats = trade_stats(train_trades)
    test_stats = trade_stats(test_trades)

    # 전략 검증구간 포트폴리오 성과
    test_eq = simulate_portfolio(test_trades)
    test_port = equity_stats(test_eq)

    # 벤치마크: KOSPI buy&hold (전체 / 검증구간)
    bench_full = cagr(idx["Close"])
    bench_test = cagr(idx["Close"].loc[split_date:])

    out = {
        "strategy": "breakout+regime",
        "split_date": str(split_date.date()),
        "params": p.__dict__,
        "train": {k: train_stats.get(k) for k in ("trades", "win_rate", "expectancy", "payoff")},
        "test": {k: test_stats.get(k) for k in ("trades", "win_rate", "expectancy", "payoff")},
        "strategy_test_portfolio": test_port,
        "benchmark_kospi_full": bench_full,
        "benchmark_kospi_test": bench_test,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # 판정 (파이썬 bool 로 캐스팅 — numpy.bool_ 은 JSON 직렬화 불가)
    te = float(test_stats.get("expectancy", 0) or 0)
    strat_cagr = float(test_port.get("cagr", -1) or -1)
    bench_cagr = float(bench_test.get("cagr", 0) or 0)
    edge_holds = bool(te > 0)
    beats_bench = bool(strat_cagr > bench_cagr)
    print("\n=== 판정 ===")
    print(f"검증구간 기대값 (+)?      {'예' if edge_holds else '아니오'} ({te*100:+.2f}%/건)")
    print(f"검증구간 지수 초과수익?   {'예' if beats_bench else '아니오'} "
          f"(전략 CAGR {strat_cagr*100:+.1f}% vs KOSPI CAGR {bench_cagr*100:+.1f}%)")

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        out["verdict"] = {"edge_holds_oos": edge_holds, "beats_benchmark_oos": beats_bench}
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[저장] {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
