#!/usr/bin/env python3
"""모멘텀 돌파 전략 **백테스트 최적화** (과최적화 방지: 학습/검증 분리).

두 단계로 최적화한다.

  [1단계] 전략 파라미터(진입 돌파창·추세선·거래량·RSI·ATR 손절/트레일 등)를 격자로
          탐색한다. 각 종목 시계열을 앞(학습)/뒤(검증)로 나눠, **검증 구간에서도
          성적이 유지되는 강건한 조합**을 고른다. 목적함수(--objective)로 무엇을
          최대화할지 정한다: expectancy(기대값) / winrate(승률) / blend(승률 하한 +
          기대값).

  [2단계] 1단계에서 고른 전략으로, **랭킹 점수 가중치**(추세·돌파·모멘텀·거래량·RSI
          비율)를 탐색한다. '점수가 높을수록 실제 매매 수익이 좋았는가'를 학습구간의
          순위상관(Spearman)으로 평가하고 검증구간에서 확인한다. → 대시보드 순위가
          성과를 실제로 반영하도록 가중치를 정한다.

결과:
  results/momentum_best.json  — 최적 파라미터 + 가중치 + 학습/검증 성적
  results/momentum_opt.csv    — 1단계 전체 조합 성적

사용:
  # 실데이터 (네트워크 열린 환경: 내 PC / GitHub Actions)
  python optimize_momentum.py --source real --market KOSPI --top 150 --start 2018-01-01 \
      --objective blend --winrate-floor 0.45 --regime

  # 오프라인 코드 검증 (합성 데이터 — 수치는 참고 불가)
  python optimize_momentum.py --source synthetic --n 80 --days 1200 --max-combos 200

⚠️ 추천/백테스트 계산만 한다. 실제 매수/매도 주문·증권사 계좌 접근은 없다.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from dataclasses import asdict, replace
from datetime import date

import numpy as np
import pandas as pd

from swing import data as datamod
from swing.metrics import trade_stats
from swing.momentum import (
    MomentumParams,
    ScoreWeights,
    extract_trades_momentum,
    trades_with_features,
)

# 1단계 전략 파라미터 격자 (조합이 곱으로 늘어나니 과하게 넓히지 말 것)
GRID = {
    "entry_lookback": [20, 40, 55, 120],
    "ma_trend": [60, 120, 150],
    "vol_mult": [1.0, 1.3, 1.6],
    "rsi_min": [45, 50, 55],
    "init_stop_atr": [1.5, 2.0, 2.5],
    "trail_atr": [2.5, 3.5, 4.5],
    "max_hold": [30, 60],
}

MIN_TRADES = 30   # 이보다 표본이 적은 조합은 신뢰 불가로 제외


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def spearman(a: list[float], b: list[float]) -> float:
    """순위상관 (scipy 없이). 표본이 적으면 0."""
    n = len(a)
    if n < 5:
        return 0.0
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def objective_score(stats: dict, mode: str, winrate_floor: float) -> float:
    """목적함수 값 (클수록 좋음). 표본이 적으면 −inf."""
    t = stats.get("trades", 0)
    if t < MIN_TRADES:
        return float("-inf")
    wr = stats.get("win_rate", 0.0)
    ex = stats.get("expectancy", 0.0)
    if mode == "winrate":
        return wr * math.sqrt(t)
    if mode == "blend":
        if wr < winrate_floor:
            return float("-inf")     # 승률 하한 미달이면 탈락
        return ex * math.sqrt(t)
    # 기본: 기대값(=거래당 평균수익) × 표본 보정
    return ex * math.sqrt(t)


def split_universe(universe: dict, ratio: float = 0.6, min_len: int = 200) -> tuple[dict, dict]:
    train, test = {}, {}
    for code, df in universe.items():
        k = int(len(df) * ratio)
        if k > min_len and len(df) - k > min_len:
            train[code] = df.iloc[:k]
            test[code] = df.iloc[k:]
    return train, test


def run_stats(universe: dict, p: MomentumParams, regime=None) -> dict:
    trades = []
    for code, df in universe.items():
        trades.extend(extract_trades_momentum(code, df, p, regime=regime))
    return trade_stats(trades)


def load_universe(args) -> dict:
    if args.source == "real":
        codes = datamod.fetch_universe(args.market, args.top)
        uni = {}
        for code in codes:
            try:
                df = datamod.fetch_ohlcv(code, args.start, args.end)
                if len(df) > 400:
                    uni[code] = df
            except Exception:  # noqa: BLE001
                pass
        return uni
    return datamod.synthetic_universe(n=args.n, days=args.days)


# ---------------------------------------------------------------------------
# 2단계: 랭킹 점수 가중치 탐색
# ---------------------------------------------------------------------------
def optimize_weights(
    train: dict, test: dict, p: MomentumParams, regime, n_candidates: int, seed: int
) -> dict:
    """점수 가중치를 탐색해 '점수↔수익' 순위상관이 가장 높은 조합을 고른다."""
    rng = random.Random(seed)

    def collect(universe, w):
        scores, rets = [], []
        for code, df in universe.items():
            for feat, ret in trades_with_features(code, df, p, w, regime=regime):
                scores.append(feat["total"])
                rets.append(ret)
        return scores, rets

    # 후보 가중치: 기본값 + 디리클레 무작위 표본
    candidates = [ScoreWeights()]
    for _ in range(n_candidates):
        v = [rng.random() for _ in range(5)]
        s = sum(v) or 1.0
        candidates.append(ScoreWeights(
            w_trend=v[0] / s, w_breakout=v[1] / s, w_momentum=v[2] / s,
            w_volume=v[3] / s, w_rsi=v[4] / s,
        ))

    best = None
    for w in candidates:
        wn = w.normalized()
        s_tr, r_tr = collect(train, wn)
        corr_tr = spearman(s_tr, r_tr)
        if best is None or corr_tr > best["train_corr"]:
            s_te, r_te = collect(test, wn)
            best = {
                "weights": {k: round(v, 3) for k, v in asdict(wn).items()},
                "train_corr": round(corr_tr, 4),
                "test_corr": round(spearman(s_te, r_te), 4),
                "train_signals": len(s_tr),
            }
    return best or {}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "synthetic"], default="synthetic")
    ap.add_argument("--market", default="KOSPI")
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--days", type=int, default=1200)
    ap.add_argument("--objective", choices=["expectancy", "winrate", "blend"],
                    default="blend", help="1단계 목적함수")
    ap.add_argument("--winrate-floor", type=float, default=0.45,
                    help="blend 모드에서 요구할 최소 승률(0~1)")
    ap.add_argument("--max-combos", type=int, default=0,
                    help=">0 이면 격자에서 이 개수만 무작위 표본 (런타임 절약)")
    ap.add_argument("--weight-candidates", type=int, default=120)
    ap.add_argument("--regime", action="store_true", help="KOSPI 상승국면에서만 진입")
    ap.add_argument("--regime-ma", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--top-show", type=int, default=12)
    args = ap.parse_args()

    universe = load_universe(args)
    if not universe:
        print("데이터가 비었습니다.", flush=True)
        return 1

    regime = None
    if args.regime and args.source == "real":
        from swing.strategy import market_regime
        idx = datamod.fetch_index("KS11", args.start, args.end)
        regime = market_regime(idx["Close"], args.regime_ma)
        print(f"[국면] KOSPI {args.regime_ma}일선 risk-on 비중 {regime.mean()*100:.0f}%")

    train, test = split_universe(universe)
    print(f"종목 {len(universe)}개 · 학습 {len(train)} / 검증 {len(test)}", flush=True)

    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    if args.max_combos and len(combos) > args.max_combos:
        random.Random(args.seed).shuffle(combos)
        combos = combos[:args.max_combos]
    print(f"[1단계] 전략 조합 {len(combos)}개 평가 중 "
          f"(목적함수={args.objective}, 최소표본={MIN_TRADES})...", flush=True)

    base = MomentumParams()
    rows = []
    for n_done, combo in enumerate(combos, 1):
        ov = dict(zip(keys, combo))
        p = replace(base, **ov)
        tr = run_stats(train, p, regime=regime)
        te = run_stats(test, p, regime=regime)
        rows.append({
            **ov,
            "train_trades": tr.get("trades", 0),
            "train_winrate": round(tr.get("win_rate", 0) * 100, 1),
            "train_expect": round(tr.get("expectancy", 0) * 100, 2),
            "train_payoff": round(tr.get("payoff", 0), 2) if tr.get("trades") else 0,
            "test_trades": te.get("trades", 0),
            "test_winrate": round(te.get("win_rate", 0) * 100, 1),
            "test_expect": round(te.get("expectancy", 0) * 100, 2),
            "test_payoff": round(te.get("payoff", 0), 2) if te.get("trades") else 0,
            "test_avghold": round(te.get("avg_hold", 0), 1),
            "train_obj": objective_score(tr, args.objective, args.winrate_floor),
            "test_obj": objective_score(te, args.objective, args.winrate_floor),
        })
        if n_done % 50 == 0:
            print(f"  ...{n_done}/{len(combos)}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out_dir, "momentum_opt.csv"),
               index=False, encoding="utf-8-sig")

    robust = res[np.isfinite(res["train_obj"]) & np.isfinite(res["test_obj"])].copy()
    robust = robust.sort_values(["test_obj", "train_obj"], ascending=False)

    show_cols = keys + ["train_trades", "train_winrate", "train_expect",
                        "test_trades", "test_winrate", "test_expect",
                        "test_payoff", "test_avghold"]
    print("=" * 100)
    print(f"  [1단계] 검증 성적 상위 전략 (목적함수={args.objective}"
          + (f", 승률하한 {args.winrate_floor:.0%}" if args.objective == "blend" else "") + ")")
    print("=" * 100)
    if robust.empty:
        print("  조건을 만족하는 조합이 없습니다. --winrate-floor 를 낮추거나 종목/기간을 늘리세요.")
        # 그래도 결과 파일은 남긴다
        best_params = asdict(base)
        best_row = {}
    else:
        with pd.option_context("display.width", 220, "display.max_columns", None):
            print(robust[show_cols].head(args.top_show).to_string(index=False))
        best_row = robust.iloc[0].to_dict()
        best_params = asdict(replace(base, **{k: (int(best_row[k]) if isinstance(GRID[k][0], int)
                                                  else float(best_row[k])) for k in keys}))

    # 2단계: 가중치 최적화
    best_p = replace(base, **{k: best_params[k] for k in keys})
    print("\n[2단계] 랭킹 점수 가중치 탐색 중 (점수↔수익 순위상관)...", flush=True)
    weights = optimize_weights(train, test, best_p, regime,
                               args.weight_candidates, args.seed)
    if weights:
        print(f"  최적 가중치: {weights['weights']}")
        print(f"  점수↔수익 순위상관  학습 {weights['train_corr']}  검증 {weights['test_corr']}")

    out = {
        "meta": {
            "source": args.source, "market": args.market, "top": args.top,
            "start": args.start, "end": args.end, "objective": args.objective,
            "winrate_floor": args.winrate_floor, "regime": bool(args.regime),
            "min_trades": MIN_TRADES, "generated_at": date.today().isoformat(),
            "universe": len(universe), "train": len(train), "test": len(test),
        },
        "best_params": best_params,
        "best_stats": {k: best_row.get(k) for k in show_cols} if best_row else {},
        "best_weights": weights,
        "note": ("실데이터로 검증한 강건한 조합입니다. synthetic 결과는 코드 동작 확인용일 뿐 "
                 "실제 성과와 무관합니다."),
    }
    with open(os.path.join(args.out_dir, "momentum_best.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {args.out_dir}/momentum_best.json · {args.out_dir}/momentum_opt.csv")
    print("※ 검증(test)에서도 성적이 유지되고 학습과 크게 어긋나지 않는 조합을 신뢰하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
