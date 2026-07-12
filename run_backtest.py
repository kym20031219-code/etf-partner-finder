#!/usr/bin/env python3
"""눌림목 스윙 전략 백테스트 실행기.

사용법:
  # 실제 국내 데이터 (네트워크 필요: 회원님 PC / GitHub Actions)
  python run_backtest.py --source real --market KOSPI --top 100 --start 2020-01-01

  # 오프라인 검증 (합성 데이터, 네트워크 불필요)
  python run_backtest.py --source synthetic --n 30

결과:
  - 콘솔에 성과 리포트 출력
  - trades.csv            : 개별 매매 내역
  - signals_latest.json   : 가장 최근 봉에서 신호 뜬 '오늘의 후보' (알림/웹 연동용)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from swing import data as datamod
from swing.engine import extract_trades, simulate_portfolio
from swing.metrics import trade_stats, equity_stats, format_report
from swing.strategy import PullbackParams, latest_candidates


def load_universe(args) -> dict[str, pd.DataFrame]:
    if args.source == "real":
        print(f"[데이터] {args.market} 시총 상위 {args.top}종목 목록 조회...", flush=True)
        codes = datamod.fetch_universe(args.market, args.top)
        uni: dict[str, pd.DataFrame] = {}
        for i, code in enumerate(codes, 1):
            try:
                df = datamod.fetch_ohlcv(code, args.start, args.end)
                if len(df) > 150:
                    uni[code] = df
                print(f"  ({i}/{len(codes)}) {code}  {len(df)}봉", flush=True)
            except Exception as e:  # 개별 종목 실패는 건너뜀
                print(f"  ({i}/{len(codes)}) {code}  실패: {e}", flush=True)
        return uni
    # synthetic
    print(f"[데이터] 합성 종목 {args.n}개 생성 (오프라인 검증)", flush=True)
    return datamod.synthetic_universe(n=args.n, days=args.days)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "synthetic"], default="synthetic")
    ap.add_argument("--market", default="KOSPI")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--n", type=int, default=30, help="합성 종목 수")
    ap.add_argument("--days", type=int, default=750, help="합성 봉 수")
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--cash", type=float, default=10_000_000)
    ap.add_argument("--report-json", default=None, help="성과 지표를 JSON 으로 저장할 경로")
    ap.add_argument("--regime", action="store_true", help="KOSPI 지수 상승국면에서만 매수")
    ap.add_argument("--regime-ma", type=int, default=120)
    args = ap.parse_args()

    p = PullbackParams()
    universe = load_universe(args)
    if not universe:
        print("데이터가 비었습니다.", file=sys.stderr)
        return 1

    # 시장 국면 필터 (실데이터만 지원)
    regime = None
    if args.regime and args.source == "real":
        from swing.strategy import market_regime
        idx = datamod.fetch_index("KS11", args.start, args.end)
        regime = market_regime(idx["Close"], args.regime_ma)
        on_ratio = float(regime.reindex(idx.index).mean())
        print(f"[국면] KOSPI {args.regime_ma}일선 기준 risk-on 비중 {on_ratio*100:.0f}%", flush=True)
    elif args.regime:
        print("[국면] 합성 데이터는 지수가 없어 국면 필터를 건너뜁니다.", flush=True)

    all_trades = []
    for code, df in universe.items():
        all_trades.extend(extract_trades(code, df, p, regime=regime))

    tstats = trade_stats(all_trades)
    eq = simulate_portfolio(all_trades, args.cash, args.max_positions)
    estats = equity_stats(eq)
    note = f"종목 {len(universe)}개 · 소스={args.source} · 동시보유≤{args.max_positions}"
    print("\n" + format_report(tstats, estats, note))

    # 성과 지표 JSON 저장 (워크플로우에서 결과를 안정적으로 읽기 위함)
    if args.report_json:
        from dataclasses import asdict
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        # exit_reasons 의 dict 는 그대로 직렬화 가능
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {
                        "source": args.source, "market": args.market, "top": args.top,
                        "start": args.start, "end": args.end,
                        "universe_size": len(universe),
                        "max_positions": args.max_positions,
                        "regime_filter": bool(regime is not None),
                        "regime_ma": args.regime_ma if regime is not None else None,
                    },
                    "params": asdict(p),
                    "trade_stats": tstats,
                    "equity_stats": estats,
                },
                f, ensure_ascii=False, indent=2,
            )
        print(f"[저장] {args.report_json}")

    # 개별 매매 저장
    if all_trades:
        rows = [
            {
                "code": t.code,
                "entry_date": str(t.entry_date.date()),
                "entry_price": round(t.entry_price, 2),
                "exit_date": str(t.exit_date.date()),
                "exit_price": round(t.exit_price, 2),
                "ret_pct": round(t.ret * 100, 2),
                "bars_held": t.bars_held,
                "reason": t.reason,
            }
            for t in all_trades
        ]
        pd.DataFrame(rows).to_csv("trades.csv", index=False, encoding="utf-8-sig")
        print(f"\n[저장] trades.csv ({len(rows)}건)")

    # 오늘의 후보 (알림/웹 연동용)
    candidates = latest_candidates(universe, p)
    out = {
        "generated_at": str(pd.Timestamp.now()),
        "source": args.source,
        "params": p.__dict__,
        "candidates": candidates,
    }
    with open("signals_latest.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[저장] signals_latest.json — 최근 봉 신호 종목 {len(candidates)}개")
    if candidates:
        for c in candidates:
            print(f"    · {c['code']}  종가 {c['close']}  손절 {c['stop']}  목표 {c['target']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
