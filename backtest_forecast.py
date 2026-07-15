#!/usr/bin/env python3
"""코스피 **전망 점수 백테스트** (워크포워드, 미래참조 없음).

무엇을 검증하나
  "지금의 6팩터 종합 전망 점수(기술+수급+PER/PBR+…)로 **매수/매도(=현금) 시그널**을
  줬다면 과거에 실제로 어땠는가?" 를 검증한다.

방식(정직성)
  - **미래참조 없음**: 각 날짜 t 의 팩터 점수는 t 까지의 데이터만으로 계산(slice_bundle)
    한다. 시그널은 t 종가로 판단하고 **다음 거래일(t+1)에 반영**한다(지연 체결).
  - **인샘플 과최적화 배제**: 종합 점수 가중치는 **기본값(합리적 초기값, prior)** 을 쓴다.
    (데이터로 튜닝한 가중치를 같은 기간 백테스트에 쓰면 미래참조가 되므로 기본적으로 제외)
  - **거래비용 반영**: 포지션이 바뀌는 날 왕복 비용을 차감한다.

시그널(롱/현금, 히스테리시스)
  - 종합점수 ≥ buy_thr → 매수(코스피 보유), ≤ sell_thr → 매도(현금), 그 사이는 유지.

리포트
  - 전략 vs 단순보유(buy-and-hold): 총수익·CAGR·**초과수익**
  - **MDD(최대낙폭)·승률(라운드트립)·샤프비율**·시장노출·거래횟수
  - results/kospi_backtest.json (대시보드가 자산곡선·지표를 읽음)

사용
  python backtest_forecast.py --source real --start 2016-01-01
  python backtest_forecast.py --source synthetic --days 1200   # 오프라인 코드검증

⚠️ 과거 성과는 미래를 보장하지 않는다. 단기 지수 타이밍은 노이즈가 크고, 표본기간·
   임계값에 민감하다. 매매 권유가 아니며 자동주문 기능은 없다.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from swing.kospi_forecast import FACTOR_KEYS, ForecastWeights, score_panel

RESULTS_DIR = Path("results")
BACKTEST_JSON = RESULTS_DIR / "kospi_backtest.json"
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 시그널 → 포지션 (히스테리시스, 롱/현금)
# ---------------------------------------------------------------------------
def signal_positions(score: pd.Series, buy_thr: float, sell_thr: float) -> pd.Series:
    """종합점수 → 목표 포지션(1=보유, 0=현금). 밴드 안에서는 직전 포지션 유지."""
    pos = np.zeros(len(score))
    cur = 0.0
    s = score.to_numpy()
    for i in range(len(s)):
        if s[i] >= buy_thr:
            cur = 1.0
        elif s[i] <= sell_thr:
            cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=score.index)


# ---------------------------------------------------------------------------
# 성과 지표
# ---------------------------------------------------------------------------
def _mdd(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _sharpe(daily_ret: np.ndarray) -> float:
    if len(daily_ret) < 2 or daily_ret.std() == 0:
        return 0.0
    return float(daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS))


def _cagr(equity_end: float, n_days: int) -> float:
    years = n_days / TRADING_DAYS
    if years <= 0 or equity_end <= 0:
        return 0.0
    return float(equity_end ** (1.0 / years) - 1.0)


def run_backtest(panel: pd.DataFrame, weights: ForecastWeights,
                 buy_thr: float, sell_thr: float, cost: float) -> dict:
    """점수 패널 → 전략/단순보유 성과. panel 은 종가 + 6팩터 점수(일별)."""
    wd = weights.as_dict()
    score = panel[FACTOR_KEYS].to_numpy() @ np.array([wd[k] for k in FACTOR_KEYS])
    score = pd.Series(score, index=panel.index)

    close = panel["close"].astype(float)
    ret = close.pct_change().fillna(0.0)               # 코스피 일간수익률

    target = signal_positions(score, buy_thr, sell_thr)
    pos = target.shift(1).fillna(0.0)                  # 지연 체결(미래참조 없음)
    turn = pos.diff().abs().fillna(pos.abs())          # 포지션 변경분
    strat_ret = pos * ret - turn * cost                # 전략 일간수익률(비용 차감)

    eq = (1.0 + strat_ret).cumprod()
    bh = (1.0 + ret).cumprod()
    n = len(panel)

    # 라운드트립 승률: 롱 보유 구간별 실현손익
    trades = []
    in_pos = False
    entry_eq = 1.0
    for i in range(n):
        p = pos.iloc[i]
        if p > 0 and not in_pos:
            in_pos, entry_eq = True, eq.iloc[i - 1] if i > 0 else 1.0
        elif p == 0 and in_pos:
            in_pos = False
            trades.append(eq.iloc[i - 1] / entry_eq - 1.0)
    if in_pos:
        trades.append(eq.iloc[-1] / entry_eq - 1.0)
    wins = [t for t in trades if t > 0]
    win_rate = len(wins) / len(trades) if trades else float("nan")

    def block(equity, dret):
        e = float(equity.iloc[-1])
        return {
            "total_return": round(e - 1.0, 4),
            "cagr": round(_cagr(e, n), 4),
            "mdd": round(_mdd(equity.to_numpy()), 4),
            "sharpe": round(_sharpe(np.asarray(dret)), 3),
        }

    strat = block(eq, strat_ret.to_numpy())
    hold = block(bh, ret.to_numpy())
    strat.update({
        "win_rate": round(win_rate, 4) if win_rate == win_rate else None,
        "n_trades": len(trades),
        "time_in_market": round(float(pos.mean()), 4),
    })

    # 대시보드용 자산곡선(약 200 포인트로 다운샘플)
    step = max(1, n // 200)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    curve = [{"d": str(panel.index[i].date()),
              "strat": round(float(eq.iloc[i]), 4),
              "bh": round(float(bh.iloc[i]), 4)} for i in idx]

    return {
        "strategy": strat,
        "buy_hold": hold,
        "excess": {
            "total_return": round(strat["total_return"] - hold["total_return"], 4),
            "cagr": round(strat["cagr"] - hold["cagr"], 4),
        },
        "equity": curve,
        "n_days": n,
        "date_range": [str(panel.index[0].date()), str(panel.index[-1].date())],
    }


def load_bundle(args):
    import run_kospi_forecast as rk
    if args.source == "synthetic":
        return rk.build_synthetic_bundle(days=args.days, seed=args.seed)
    try:
        return rk.build_real_bundle(args.start, args.end)
    except Exception as e:  # noqa: BLE001
        print(f"[경고] 실데이터 수집 실패 → 합성으로 대체: {e}", file=sys.stderr)
        return rk.build_synthetic_bundle(days=args.days, seed=args.seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "synthetic"], default="real")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--days", type=int, default=1200, help="합성 데이터 길이")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--buy-thr", type=float, default=54.0, help="이 점수 이상이면 매수")
    ap.add_argument("--sell-thr", type=float, default=46.0, help="이 점수 이하면 매도(현금)")
    ap.add_argument("--cost", type=float, default=0.001, help="포지션 변경당 왕복 비용(0.1%)")
    ap.add_argument("--step", type=int, default=1, help="패널 날짜 간격(1=일별)")
    ap.add_argument("--dry-run", action="store_true", help="JSON 저장 생략")
    args = ap.parse_args()

    bundle = load_bundle(args)
    print(f"[패널] 시점별 6팩터 점수 계산 중… (거래일 {len(bundle.kospi)}개, 미래참조 없음)",
          flush=True)
    panel = score_panel(bundle, step=args.step)
    if len(panel) < 60:
        print(f"패널 표본이 너무 적습니다({len(panel)}). 기간을 늘리세요.", file=sys.stderr)
        return 1

    weights = ForecastWeights()  # 기본값(prior) — 인샘플 튜닝 배제
    res = run_backtest(panel, weights, args.buy_thr, args.sell_thr, args.cost)

    s, h, x = res["strategy"], res["buy_hold"], res["excess"]
    print(f"\n=== 전망점수 백테스트 ({res['date_range'][0]} ~ {res['date_range'][1]}, "
          f"{res['n_days']}거래일) ===")
    print(f"  매수/매도 임계값 {args.buy_thr:.0f}/{args.sell_thr:.0f} · 비용 {args.cost*100:.1f}%/회 · 가중치 기본값")
    print(f"  {'':16}{'전략':>12}{'단순보유':>12}")
    print(f"  {'총수익':<14}{s['total_return']*100:>11.1f}%{h['total_return']*100:>11.1f}%")
    print(f"  {'연복리(CAGR)':<12}{s['cagr']*100:>11.1f}%{h['cagr']*100:>11.1f}%")
    print(f"  {'최대낙폭(MDD)':<11}{s['mdd']*100:>11.1f}%{h['mdd']*100:>11.1f}%")
    print(f"  {'샤프비율':<13}{s['sharpe']:>12.2f}{h['sharpe']:>12.2f}")
    wr = s['win_rate']
    print(f"  승률 {wr*100:.1f}% ({s['n_trades']}회 매매) · 시장노출 {s['time_in_market']*100:.0f}%"
          if wr is not None else f"  매매 없음")
    print(f"  → 단순보유 대비 초과수익 {x['total_return']*100:+.1f}%p (CAGR {x['cagr']*100:+.1f}%p)")

    if args.dry_run:
        print("\n[dry-run] JSON 저장 생략")
        return 0

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "source": args.source,
        "params": {"buy_thr": args.buy_thr, "sell_thr": args.sell_thr,
                   "cost": args.cost, "weights": "default_prior"},
        "note": ("워크포워드(미래참조 없음)·인샘플 튜닝 배제. 과거 성과는 미래를 "
                 "보장하지 않으며 표본기간·임계값에 민감함."),
        **res,
    }
    BACKTEST_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ {BACKTEST_JSON} 저장 — 대시보드가 자산곡선·지표를 표시")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
