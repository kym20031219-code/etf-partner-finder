#!/usr/bin/env python3
"""종가매매법(오버나이트) 백테스트 + 갭상승 특징 분석 + 오늘의 후보 스캔.

사용법:
  # 실제 국내 데이터 (네트워크 필요: 회원님 PC / GitHub Actions)
  python run_overnight.py --source real --market KOSPI --top 200 --start 2020-01-01

  # 오프라인 검증 (합성 데이터, 네트워크 불필요)
  python run_overnight.py --source synthetic --n 40 --days 900

결과:
  - 콘솔: 갭상승 특징 분석 + 백테스트 성과(승률·기대수익률 등)
  - results/overnight_study.json  : 분석·성과 지표
  - state/overnight_latest.json   : 장 마감 직전 '오늘의 종가매매 후보' (웹/알림용)
  - overnight_trades.csv          : 개별 매매 내역
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from swing import data as datamod
from swing.engine import simulate_portfolio
from swing.metrics import trade_stats, equity_stats, format_report
from overnight import data as odata
from overnight.engine import extract_trades
from overnight.strategy import ClosingParams, latest_picks
from overnight.study import gap_feature_study, format_study

RESULTS = Path("results")
STATE = Path("state")


def load_universe(args) -> tuple[dict, dict]:
    """(universe, names) 반환."""
    if args.source == "real":
        print(f"[데이터] {args.market} 시총 상위 {args.top}종목 조회...", flush=True)
        codes = datamod.fetch_universe(args.market, args.top)
        names = datamod.fetch_names(args.market)
        uni: dict = {}
        for i, code in enumerate(codes, 1):
            try:
                df = datamod.fetch_ohlcv(code, args.start, args.end)
                if len(df) > 120:
                    uni[code] = df
            except Exception as e:  # noqa: BLE001
                print(f"  {code} 실패: {e}", file=sys.stderr)
        print(f"  → {len(uni)}종목 로드", flush=True)
        return uni, names
    print(f"[데이터] 합성 종목 {args.n}개 생성 (오프라인 검증)", flush=True)
    return odata.overnight_universe(n=args.n, days=args.days), {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "synthetic"], default="synthetic")
    ap.add_argument("--market", default="KOSPI")
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--n", type=int, default=40, help="합성 종목 수")
    ap.add_argument("--days", type=int, default=900, help="합성 봉 수")
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--cash", type=float, default=10_000_000)
    ap.add_argument("--regime", action="store_true", help="KOSPI 상승국면에서만 매수")
    ap.add_argument("--regime-ma", type=int, default=120)
    args = ap.parse_args()

    p = ClosingParams()
    universe, names = load_universe(args)
    if not universe:
        print("데이터가 비었습니다.", file=sys.stderr)
        return 1

    # 시장 국면 필터 (실데이터만): KOSPI 지수 상승국면에서만 매수
    regime = None
    if args.regime and args.source == "real":
        from swing.strategy import market_regime
        idx = datamod.fetch_index("KS11", args.start, args.end)
        regime = market_regime(idx["Close"], args.regime_ma)
        print(f"[국면] KOSPI risk-on 비중 {float(regime.mean())*100:.0f}%", flush=True)
    elif args.regime:
        print("[국면] 합성 데이터는 지수가 없어 국면 필터를 건너뜁니다.", flush=True)

    # 1) 갭상승 특징 분석 -----------------------------------------------------
    study = gap_feature_study(universe, p)
    print("\n" + format_study(study))

    # 2) 백테스트(오버나이트 매매) -------------------------------------------
    all_trades = []
    for code, df in universe.items():
        all_trades.extend(extract_trades(code, df, p, regime=regime))
    tstats = trade_stats(all_trades)
    eq = simulate_portfolio(all_trades, args.cash, args.max_positions)
    estats = equity_stats(eq)
    note = f"종가매매 · 종목 {len(universe)}개 · 소스={args.source} · 동시보유≤{args.max_positions}"
    print("\n" + format_report(tstats, estats, note))

    # 3) 결과 JSON 저장 -------------------------------------------------------
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "overnight_study.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "source": args.source, "market": args.market, "top": args.top,
                    "start": args.start, "end": args.end,
                    "universe_size": len(universe),
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                },
                "params": asdict(p),
                "study": study,
                "trade_stats": tstats,
                "equity_stats": estats,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"[저장] {RESULTS/'overnight_study.json'}")

    if all_trades:
        rows = [
            {
                "code": t.code,
                "buy_date": str(t.entry_date.date()), "buy_close": round(t.entry_price, 2),
                "sell_date": str(t.exit_date.date()), "sell_open": round(t.exit_price, 2),
                "ret_pct": round(t.ret * 100, 2),
            }
            for t in all_trades
        ]
        pd.DataFrame(rows).to_csv("overnight_trades.csv", index=False, encoding="utf-8-sig")
        print(f"[저장] overnight_trades.csv ({len(rows)}건)")

    # 4) 오늘의 종가매매 후보 (웹/알림용) ------------------------------------
    picks = latest_picks(universe, p, regime=regime)
    picks = [{**c, "name": names.get(c["code"], c["code"])} for c in picks]
    STATE.mkdir(exist_ok=True)
    with open(STATE / "overnight_latest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": args.source,
                "stats": {
                    "win_rate": tstats.get("win_rate"),
                    "expectancy": tstats.get("expectancy"),
                    "avg_ret": tstats.get("avg_ret"),
                    "trades": tstats.get("trades"),
                    "base_gap_up_rate": study.get("base_gap_up_rate"),
                    "signal_gap_up_rate": study.get("signal", {}).get("gap_up_rate"),
                    "lift": study.get("signal", {}).get("lift"),
                },
                "candidates": picks,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"[저장] {STATE/'overnight_latest.json'} — 오늘의 후보 {len(picks)}개")
    for c in picks[:15]:
        print(f"    · {c['name']}({c['code']})  종가 {c['close']}  "
              f"상승 {c['day_ret']}%  거래량 {c['vol_ratio']}배")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
