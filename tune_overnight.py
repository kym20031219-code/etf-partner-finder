#!/usr/bin/env python3
"""종가매매법 파라미터 최적화 (과최적화 방지: 학습/검증 분리).

목표(objective) 선택:
  --objective expectancy : 건당 기대수익률을 최대화 (기본, 통계적으로 올바른 목표)
  --objective winrate    : 승률을 최대화 (단, 기대값이 (+)인 조합 중에서만)

승률만 높이면 '작게 자주 이기고 크게 지는' 함정에 빠지기 쉬우므로, winrate 모드도
**기대값(비용 반영) > 0** 인 조합 중에서만 승률 상위를 고른다.

과정:
  1. 데이터를 날짜 기준으로 앞(학습)/뒤(검증)로 나눈다.
  2. 파라미터 격자를 학습 구간에서 백테스트해 목표 기준으로 정렬한다.
  3. 상위 조합을 검증 구간(미래·미사용 데이터)에서 다시 평가한다.
  4. 최종 선택 조합의 승률·건당 기대수익률·연간 거래횟수를 보고한다.

--regime 을 켜면 KOSPI 지수 상승국면(risk-on)에서만 매수 → 통상 승률이 오른다.

사용법:
  python tune_overnight.py --source real --market KOSPI --top 150 --start 2019-01-01 \
      --objective winrate --regime --slippage 0.001 --out results/overnight_validation.json
  python tune_overnight.py --source synthetic --n 40 --days 1200 --objective winrate
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
from swing.strategy import market_regime
from swing.metrics import trade_stats
from overnight import data as odata
from overnight.engine import extract_trades
from overnight.strategy import ClosingParams
from overnight.study import gap_feature_study

# 탐색 격자 (승률 최적화에 여지를 주려 더 엄격한 값도 포함) — 72 조합
GRID = {
    "up_min": [0.02, 0.03, 0.05],
    "close_pos_min": [0.5, 0.7],
    "vol_mult": [1.5, 2.0, 3.0],
    "breakout_lookback": [20, 60],
    "rsi_high": [80.0, 90.0],
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


def build_regime(args) -> pd.Series | None:
    if not args.regime or args.source != "real":
        if args.regime:
            print("[국면] 합성 데이터는 지수가 없어 국면 필터를 건너뜁니다.", flush=True)
        return None
    idx = datamod.fetch_index("KS11", args.start, args.end)
    reg = market_regime(idx["Close"], args.regime_ma)
    print(f"[국면] KOSPI {args.regime_ma}일선 기준 risk-on 비중 "
          f"{float(reg.mean())*100:.0f}%", flush=True)
    return reg


def split_by_date(universe: dict, ratio: float = 0.68) -> tuple[dict, dict, str]:
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


def evaluate(universe: dict, p: ClosingParams, regime: pd.Series | None) -> dict:
    trades = []
    for code, df in universe.items():
        trades.extend(extract_trades(code, df, p, regime=regime))
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


def _rank_key(objective: str):
    """정렬 키. winrate 는 (기대값>0 우선, 그다음 승률) 로 정렬."""
    if objective == "winrate":
        return lambda st: (st["expectancy"] > 0, st["win_rate"], st["expectancy"])
    return lambda st: (st["expectancy"],)


def grid_search(train, test, base, regime, objective) -> dict:
    keys = list(GRID)
    results = []
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"[탐색] {len(combos)} 조합 × (학습·검증) · 목표={objective}", flush=True)
    for i, values in enumerate(combos, 1):
        overrides = dict(zip(keys, values))
        p = replace(base, **overrides)
        tr = evaluate(train, p, regime)
        if tr.get("trades", 0) < MIN_TRADES:
            continue
        results.append({"params": overrides, "train": tr})
        if i % 20 == 0:
            print(f"  ...{i}/{len(combos)}", flush=True)

    key = _rank_key(objective)
    results.sort(key=lambda r: key(r["train"]), reverse=True)
    for r in results[:TOP_K]:
        p = replace(base, **r["params"])
        r["test"] = evaluate(test, p, regime)
    return {"ranked": results[:TOP_K], "n_evaluated": len(results)}


def pick_robust(ranked: list, objective: str) -> dict | None:
    """검증 기대값이 (+)인 것 중, 목표에 맞는 최고 조합."""
    cand = [r for r in ranked if r.get("test", {}).get("trades", 0) >= 15
            and r["test"].get("expectancy", -1) > 0]
    if not cand:
        return None
    key = _rank_key(objective)
    cand.sort(key=lambda r: key(r["test"]), reverse=True)
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
    ap.add_argument("--objective", choices=["expectancy", "winrate"], default="expectancy")
    ap.add_argument("--regime", action="store_true", help="KOSPI 상승국면에서만 매수")
    ap.add_argument("--regime-ma", type=int, default=120)
    ap.add_argument("--out", default="results/overnight_validation.json")
    args = ap.parse_args()

    base = ClosingParams(entry_slippage=args.slippage)
    universe = load_universe(args)
    if len(universe) < 5:
        print("데이터가 부족합니다.")
        return 1
    regime = build_regime(args)

    train, test, cut = split_by_date(universe)
    print(f"[분할] 학습 {len(train)}종목 / 검증 {len(test)}종목 · 경계일 {cut}", flush=True)

    search = grid_search(train, test, base, regime, args.objective)
    robust = pick_robust(search["ranked"], args.objective)

    chosen = replace(base, **robust["params"]) if robust else base
    full = evaluate(universe, chosen, regime)
    study = gap_feature_study(universe, chosen)

    print("\n" + "=" * 66)
    print(f"  종가매매 파라미터 최적화 (목표={args.objective}, 국면필터={'ON' if regime is not None else 'OFF'})")
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
            "objective": args.objective, "regime": regime is not None,
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
