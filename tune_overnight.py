#!/usr/bin/env python3
"""종가매매법 파라미터 최적화 (과최적화 방지: 학습/검증 분리).

과정:
  1. 데이터를 날짜 기준으로 앞(학습 train)/뒤(검증 test) 로 나눈다.
  2. 파라미터 격자(GRID)를 학습 구간에서 백테스트해 기대값 순으로 정렬한다.
     (표본이 MIN_TRADES 미만인 조합은 신뢰 불가로 제외)
  3. 상위 조합을 **검증 구간(미래·미사용 데이터)** 에서 다시 평가한다.
     → 검증에서도 기대값이 (+)이고 학습과 크게 어긋나지 않는 조합이 '강건'하다.
  4. 최종 선택 조합의 승률·건당 기대수익률·연간 거래횟수를 보고한다.

사용법:
  python tune_overnight.py --source real --market KOSPI --top 150 --start 2019-01-01 \
      --slippage 0.001 --out results/overnight_validation.json
  python tune_overnight.py --source synthetic --n 40 --days 1200    # 오프라인 검증
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from swing import data as datamod
from swing.metrics import trade_stats
from overnight import data as odata
from overnight.engine import extract_trades
from overnight.strategy import ClosingParams
from overnight.study import gap_feature_study

# 탐색 격자 (곱으로 늘어나니 과하게 넓히지 말 것) — 108 조합
GRID = {
    "up_min": [0.015, 0.02, 0.03],
    "close_pos_min": [0.5, 0.6, 0.7],
    "vol_mult": [1.2, 1.5, 2.0],
    "breakout_lookback": [20, 60],
    "rsi_high": [75.0, 85.0],
}
MIN_TRADES = 30          # 이보다 적으면 통계 신뢰 불가
TOP_K = 12               # 학습 상위 몇 개를 검증할지


def load_universe(args) -> dict[str, pd.DataFrame]:
    if args.source == "real":
        print(f"[데이터] {args.market} 시총 상위 {args.top}종목 조회...", flush=True)
        codes = datamod.fetch_universe(args.market, args.top)
        uni: dict = {}
        for code in codes:
            try:
                df = datamod.fetch_ohlcv(code, args.start, args.end)
                if len(df) > 200:
                    uni[code] = df
            except Exception as e:  # noqa: BLE001
                print(f"  {code} 실패: {e}", flush=True)
        print(f"  → {len(uni)}종목 로드", flush=True)
        return uni
    print(f"[데이터] 합성 종목 {args.n}개 생성", flush=True)
    return odata.overnight_universe(n=args.n, days=args.days)


def split_by_date(universe: dict, ratio: float = 0.68) -> tuple[dict, dict, str]:
    """모든 종목에 공통인 분할 날짜를 잡아 train/test 로 나눈다."""
    all_dates = sorted({d for df in universe.values() for d in df.index})
    cut = all_dates[int(len(all_dates) * ratio)]
    train, test = {}, {}
    for code, df in universe.items():
        tr, te = df[df.index <= cut], df[df.index > cut]
        if len(tr) > 150:
            train[code] = tr
        if len(te) > 60:
            test[code] = te
    return train, test, str(pd.Timestamp(cut).date())


def evaluate(universe: dict, p: ClosingParams) -> dict:
    trades = []
    for code, df in universe.items():
        trades.extend(extract_trades(code, df, p))
    st = trade_stats(trades)
    st["trades_per_year"] = _trades_per_year(trades, universe)
    return st


def _years_span(universe: dict) -> float:
    dates = [d for df in universe.values() for d in df.index]
    if not dates:
        return 0.0
    return max((max(dates) - min(dates)).days / 365.25, 1e-9)


def _trades_per_year(trades: list, universe: dict) -> float:
    yrs = _years_span(universe)
    return round(len(trades) / yrs, 1) if yrs else 0.0


def grid_search(train: dict, test: dict, base: ClosingParams) -> dict:
    keys = list(GRID)
    results = []
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"[탐색] {len(combos)} 조합 × (학습·검증)", flush=True)
    for i, values in enumerate(combos, 1):
        overrides = dict(zip(keys, values))
        p = replace(base, **overrides)
        tr = evaluate(train, p)
        if tr.get("trades", 0) < MIN_TRADES:
            continue
        results.append({"params": overrides, "train": tr})
        if i % 20 == 0:
            print(f"  ...{i}/{len(combos)}", flush=True)

    # 학습 기대값 순 정렬 → 상위 K개만 검증
    results.sort(key=lambda r: r["train"]["expectancy"], reverse=True)
    for r in results[:TOP_K]:
        p = replace(base, **r["params"])
        r["test"] = evaluate(test, p)

    return {"ranked": results[:TOP_K], "n_evaluated": len(results)}


def pick_robust(ranked: list) -> dict | None:
    """검증 기대값이 (+)인 것 중 검증 기대값이 가장 큰 조합."""
    cand = [r for r in ranked if r.get("test", {}).get("trades", 0) >= 15
            and r["test"].get("expectancy", -1) > 0]
    if not cand:
        return None
    cand.sort(key=lambda r: r["test"]["expectancy"], reverse=True)
    return cand[0]


def _fmt(st: dict) -> str:
    if not st or st.get("trades", 0) == 0:
        return "매매 없음"
    return (f"거래 {st['trades']:>4} · 승률 {st['win_rate']*100:4.1f}% · "
            f"건당기대 {st['expectancy']*100:+5.2f}% · 손익비 {st['payoff']:.2f} · "
            f"연 {st.get('trades_per_year','?')}건")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "synthetic"], default="synthetic")
    ap.add_argument("--market", default="KOSPI")
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--days", type=int, default=1200)
    ap.add_argument("--slippage", type=float, default=0.0, help="매수 슬리피지(보수적)")
    ap.add_argument("--out", default="results/overnight_validation.json")
    args = ap.parse_args()

    base = ClosingParams(entry_slippage=args.slippage)
    universe = load_universe(args)
    if len(universe) < 5:
        print("데이터가 부족합니다.")
        return 1

    train, test, cut = split_by_date(universe)
    print(f"[분할] 학습 {len(train)}종목 / 검증 {len(test)}종목 · 경계일 {cut}", flush=True)

    search = grid_search(train, test, base)
    robust = pick_robust(search["ranked"])

    # 전체 구간(참고) + 선택 조합의 특징 분석
    chosen = replace(base, **robust["params"]) if robust else base
    full = evaluate(universe, chosen)
    study = gap_feature_study(universe, chosen)

    print("\n" + "=" * 66)
    print("  종가매매 파라미터 최적화 결과 (학습/검증 분리)")
    print("=" * 66)
    print(f"  소스={args.source} · 종목 {len(universe)} · 슬리피지 {args.slippage*100:.2f}%")
    print("-" * 66)
    print("  [학습 상위 조합 → 검증 성적]")
    for r in search["ranked"]:
        print(f"    {r['params']}")
        print(f"      학습: {_fmt(r['train'])}")
        print(f"      검증: {_fmt(r.get('test', {}))}")
    print("-" * 66)
    if robust:
        print(f"  ✅ 선택(강건) 조합: {robust['params']}")
        print(f"     검증(미래데이터): {_fmt(robust['test'])}")
        print(f"     전체구간 참고    : {_fmt(full)}")
        s = study.get("signal", {})
        if s.get("gap_up_rate") is not None:
            print(f"     갭상승 적중률    : {s['gap_up_rate']}% (무작위 {study['base_gap_up_rate']}% 대비 {s['lift']}배)")
    else:
        print("  ⚠️ 검증 구간에서 기대값(+)을 유지하는 강건한 조합이 없습니다.")
        print("     → 이 전략은 (이 표본에선) 비용 차감 후 실전 엣지가 약하다는 신호.")
    print("=" * 66)

    out = {
        "meta": {
            "source": args.source, "market": args.market, "top": args.top,
            "start": args.start, "end": args.end, "slippage": args.slippage,
            "universe_size": len(universe), "split_date": cut,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "grid": GRID,
        "ranked": search["ranked"],
        "chosen": {
            "params": robust["params"] if robust else None,
            "test_stats": robust["test"] if robust else None,
            "full_stats": full,
            "study": study,
        } if robust else {"params": None, "full_stats": full, "study": study},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"[저장] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
