#!/usr/bin/env python3
"""신고가 돌파(추세추종) 전략 백테스트.

  python run_breakout.py --source real --market KOSPI --top 100 --start 2020-01-01 \
      --report-json results/breakout.json
  python run_breakout.py --source synthetic --n 40 --days 900   # 오프라인 검증
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from swing import data as datamod
from swing.breakout import BreakoutParams, extract_trades_breakout
from swing.engine import simulate_portfolio
from swing.metrics import trade_stats, equity_stats, format_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "synthetic"], default="synthetic")
    ap.add_argument("--market", default="KOSPI")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--days", type=int, default=900)
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--cash", type=float, default=10_000_000)
    ap.add_argument("--regime", action="store_true")
    ap.add_argument("--regime-ma", type=int, default=120)
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    p = BreakoutParams()
    if args.source == "real":
        codes = datamod.fetch_universe(args.market, args.top)
        universe = {}
        for code in codes:
            try:
                df = datamod.fetch_ohlcv(code, args.start, args.end)
                if len(df) > 200:
                    universe[code] = df
            except Exception:  # noqa: BLE001
                pass
    else:
        universe = datamod.synthetic_universe(n=args.n, days=args.days)
    if not universe:
        print("데이터가 비었습니다.", file=sys.stderr)
        return 1

    regime = None
    if args.regime and args.source == "real":
        from swing.strategy import market_regime
        idx = datamod.fetch_index("KS11", args.start, args.end)
        regime = market_regime(idx["Close"], args.regime_ma)
        print(f"[국면] risk-on 비중 {regime.mean()*100:.0f}%", flush=True)

    trades = []
    for code, df in universe.items():
        trades.extend(extract_trades_breakout(code, df, p, regime=regime))

    tstats = trade_stats(trades)
    eq = simulate_portfolio(trades, args.cash, args.max_positions)
    estats = equity_stats(eq)
    note = f"신고가 돌파 · 종목 {len(universe)}개 · 소스={args.source}"
    print("\n" + format_report(tstats, estats, note))

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "strategy": "breakout",
                    "meta": {"source": args.source, "market": args.market, "top": args.top,
                             "start": args.start, "end": args.end,
                             "universe_size": len(universe),
                             "regime_filter": bool(regime is not None)},
                    "params": asdict(p),
                    "trade_stats": tstats,
                    "equity_stats": estats,
                },
                f, ensure_ascii=False, indent=2,
            )
        print(f"[저장] {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
