#!/usr/bin/env python3
"""코스피 **전망 점수 백테스트** (워크포워드, 미래참조 없음).

무엇을 검증하나
  "지금의 6팩터 종합 전망 점수(기술+수급+PER/PBR+…)로 **매수/매도(=현금) 시그널**을
  줬다면 과거에 실제로 어땠는가?" 를 검증한다.

방식(정직성)
  - **미래참조 없음**: 각 날짜 t 의 팩터 점수는 t 까지의 데이터만으로 계산(slice_bundle).
    시그널은 t 종가로 판단하고 **다음 거래일(t+1)에 반영**한다(지연 체결).
  - **진짜 워크포워드(--mode walkforward, 기본)**: 매 리밸런싱 시점 t 에서 **t 이전
    데이터만으로 매수/매도 임계값을 재추정**(그리드 탐색, 인윈도 샤프 최대)하고 그
    임계값으로 t 시점 신호를 만든다. 최소 롤링윈도(기본 3년) 이후부터 성과를 집계한다.
  - **고정 임계값(--mode fixed)**: 전체 기간 동일 임계값(54/46). 비교 기준선.
  - 가중치는 항상 **기본값(prior)** — 인샘플 튜닝 배제(동적 가중치는 다음 단계).
  - **거래비용 반영**: 포지션이 바뀌는 날 왕복 비용을 차감.

리포트(results/kospi_backtest.json)
  - 전략 vs 단순보유(buy-and-hold): 총수익·CAGR·**초과수익**·MDD·**샤프**·**승률**·시장노출
  - **구간별 성과**(2016-2020 / 2020-2023 / 2023-2026): 초과수익이 한 구간에 쏠렸는지
  - **승률-손익비 분해**: 평균 익절/손절, 손익비, **상위 5개 거래의 총수익 기여도(%)**
  - 자산곡선(전략 vs 보유)

사용
  python backtest_forecast.py --source real --start 2016-01-01
  python backtest_forecast.py --source real --mode fixed          # 고정 임계값 기준선
  python backtest_forecast.py --source synthetic --days 1600      # 오프라인 코드검증

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

# 임계값 재추정 그리드 (매수 > 매도+2 조합만)
BUY_GRID = [52.0, 54.0, 56.0, 58.0, 60.0]
SELL_GRID = [40.0, 42.0, 44.0, 46.0, 48.0]

# 구간 분리 경계(완료기준: 초과수익 쏠림 확인)
SUBPERIOD_BOUNDS = ["2020-01-01", "2023-01-01"]


def composite_score(panel: pd.DataFrame, weights: ForecastWeights) -> pd.Series:
    wd = weights.as_dict()
    s = panel[FACTOR_KEYS].to_numpy() @ np.array([wd[k] for k in FACTOR_KEYS])
    return pd.Series(s, index=panel.index)


# ---------------------------------------------------------------------------
# 시그널 → 포지션
# ---------------------------------------------------------------------------
def positions_from_thresholds(score: pd.Series, buy: float, sell: float) -> pd.Series:
    """종합점수 → 목표 포지션(1=보유, 0=현금). 밴드 안에서는 직전 포지션 유지."""
    out = np.zeros(len(score))
    cur = 0.0
    s = score.to_numpy()
    for i in range(len(s)):
        if s[i] >= buy:
            cur = 1.0
        elif s[i] <= sell:
            cur = 0.0
        out[i] = cur
    return pd.Series(out, index=score.index)


# 하위호환 별칭(기존 테스트/코드)
def signal_positions(score: pd.Series, buy_thr: float, sell_thr: float) -> pd.Series:
    return positions_from_thresholds(score, buy_thr, sell_thr)


def walkforward_positions(score: pd.Series, ret: pd.Series, cost: float,
                          min_window: int, rebal: int) -> tuple[pd.Series, list[dict]]:
    """매 리밸런싱 시점에서 **과거 데이터만으로** 임계값을 재추정해 목표 포지션 생성.

    반환: (목표포지션 series, 리밸런싱별 선택 임계값 로그)
    각 리밸 시점 r 에서 [0, r) 구간의 (score, ret) 로 그리드 탐색해 인윈도 샤프가 가장
    높은 (buy, sell) 을 고르고, 다음 리밸까지 그 임계값을 쓴다(히스테리시스 상태 유지).
    """
    n = len(score)
    s = score.to_numpy()
    pos = np.zeros(n)
    cur = 0.0
    buy, sell = 54.0, 46.0        # 초기값(워밍업 동안만; 성과 집계 전 구간)
    log: list[dict] = []
    next_rebal = min_window
    for i in range(n):
        if i >= min_window and i >= next_rebal:
            buy, sell = _best_thresholds(score.iloc[:i], ret.iloc[:i], cost)
            log.append({"date": str(score.index[i].date()), "buy": buy, "sell": sell})
            next_rebal = i + rebal
        if s[i] >= buy:
            cur = 1.0
        elif s[i] <= sell:
            cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=score.index), log


def _best_thresholds(score: pd.Series, ret: pd.Series, cost: float) -> tuple[float, float]:
    """인윈도 (score, ret) 에서 전략 샤프가 최대인 (buy, sell) 그리드 조합."""
    best, best_sh = (54.0, 46.0), -1e9
    for buy in BUY_GRID:
        for sell in SELL_GRID:
            if buy <= sell + 2:
                continue
            pos = positions_from_thresholds(score, buy, sell).shift(1).fillna(0.0)
            turn = pos.diff().abs().fillna(pos.abs())
            sr = pos * ret - turn * cost
            sh = _sharpe(sr.to_numpy())
            if sh > best_sh:
                best_sh, best = sh, (buy, sell)
    return best


# ---------------------------------------------------------------------------
# 시뮬레이션 코어 + 지표
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


def _simulate(pos_target: pd.Series, ret: pd.Series, cost: float) -> dict:
    """목표포지션(지연 체결) → 전략 일간수익률·자산곡선."""
    pos = pos_target.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    strat_ret = pos * ret - turn * cost
    eq = (1.0 + strat_ret).cumprod()
    return {"pos": pos, "strat_ret": strat_ret, "eq": eq}


def _trades(pos: pd.Series, eq: pd.Series) -> list[float]:
    """롱 보유 구간별 실현손익 리스트."""
    trades, in_pos, entry = [], False, 1.0
    ev = eq.to_numpy()
    pv = pos.to_numpy()
    for i in range(len(pv)):
        if pv[i] > 0 and not in_pos:
            in_pos, entry = True, ev[i - 1] if i > 0 else 1.0
        elif pv[i] == 0 and in_pos:
            in_pos = False
            trades.append(ev[i - 1] / entry - 1.0)
    if in_pos:
        trades.append(ev[-1] / entry - 1.0)
    return trades


def _metrics(eq: pd.Series, dret: np.ndarray, n: int) -> dict:
    e = float(eq.iloc[-1])
    return {
        "total_return": round(e - 1.0, 4),
        "cagr": round(_cagr(e, n), 4),
        "mdd": round(_mdd(eq.to_numpy()), 4),
        "sharpe": round(_sharpe(dret), 3),
    }


def winloss_decomposition(trades: list[float]) -> dict:
    """승률-손익비 분해 + 상위 5개 거래 기여도(outlier 의존도)."""
    if not trades:
        return {"win_rate": None, "n_trades": 0}
    arr = np.array(trades)
    wins, losses = arr[arr > 0], arr[arr <= 0]
    total_win = float(wins.sum())
    # 상위 5개 '이익' 거래가 전체 이익합에서 차지하는 비중
    top5 = float(np.sort(wins)[::-1][:5].sum()) if len(wins) else 0.0
    return {
        "win_rate": round(len(wins) / len(arr), 4),
        "n_trades": len(arr),
        "avg_win": round(float(wins.mean()), 4) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 4) if len(losses) else 0.0,
        "payoff": round(float(wins.mean() / abs(losses.mean())), 2)
                  if len(wins) and len(losses) and losses.mean() != 0 else None,
        "top5_win_share": round(top5 / total_win, 4) if total_win > 0 else None,
        "best": round(float(arr.max()), 4),
        "worst": round(float(arr.min()), 4),
    }


def subperiod_performance(dates: pd.DatetimeIndex, strat_ret: pd.Series,
                          ret: pd.Series, bounds: list[str]) -> list[dict]:
    """구간별 전략 vs 단순보유 성과(초과수익 쏠림 확인)."""
    edges = [dates[0]] + [pd.Timestamp(b) for b in bounds] + [dates[-1] + pd.Timedelta(days=1)]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (dates >= a) & (dates < b)
        if m.sum() < 20:
            continue
        sr, rr = strat_ret[m], ret[m]
        se = float((1 + sr).prod() - 1)
        be = float((1 + rr).prod() - 1)
        out.append({
            "period": f"{a.date()} ~ {(b - pd.Timedelta(days=1)).date()}",
            "n_days": int(m.sum()),
            "strategy_return": round(se, 4),
            "buy_hold_return": round(be, 4),
            "excess_return": round(se - be, 4),
            "strategy_mdd": round(_mdd((1 + sr).cumprod().to_numpy()), 4),
            "buy_hold_mdd": round(_mdd((1 + rr).cumprod().to_numpy()), 4),
        })
    return out


def run_backtest(panel: pd.DataFrame, weights: ForecastWeights, cost: float,
                 mode: str = "walkforward", min_window: int = 756, rebal: int = 21) -> dict:
    """점수 패널 → 전략/단순보유 성과 + 구간별 + 승률/손익비 분해."""
    score = composite_score(panel, weights)
    close = panel["close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    if mode == "walkforward":
        target, thr_log = walkforward_positions(score, ret, cost, min_window, rebal)
        start_i = min_window                       # 워밍업 이후부터 집계
    else:
        target = positions_from_thresholds(score, 54.0, 46.0)
        thr_log, start_i = [], 0

    sim = _simulate(target, ret, cost)
    # 성과 집계 구간(워밍업 제외)으로 잘라 재기준화
    sl = slice(start_i, None)
    dates = panel.index[sl]
    sret = sim["strat_ret"].iloc[sl]
    rr = ret.iloc[sl]
    pos = sim["pos"].iloc[sl]
    eq = (1 + sret).cumprod()
    bh = (1 + rr).cumprod()
    n = len(dates)

    trades = _trades(pos, eq)
    strat = _metrics(eq, sret.to_numpy(), n)
    strat.update({"time_in_market": round(float(pos.mean()), 4)})
    strat.update(winloss_decomposition(trades))
    hold = _metrics(bh, rr.to_numpy(), n)

    step = max(1, n // 220)
    ci = list(range(0, n, step))
    if ci[-1] != n - 1:
        ci.append(n - 1)
    curve = [{"d": str(dates[i].date()), "strat": round(float(eq.iloc[i]), 4),
              "bh": round(float(bh.iloc[i]), 4)} for i in ci]

    return {
        "mode": mode,
        "min_window_days": min_window if mode == "walkforward" else 0,
        "rebalance_days": rebal if mode == "walkforward" else 0,
        "strategy": strat,
        "buy_hold": hold,
        "excess": {
            "total_return": round(strat["total_return"] - hold["total_return"], 4),
            "cagr": round(strat["cagr"] - hold["cagr"], 4),
        },
        "subperiods": subperiod_performance(dates, sret, rr, SUBPERIOD_BOUNDS),
        "threshold_reestimations": len(thr_log),
        "recent_thresholds": thr_log[-3:],
        "equity": curve,
        "n_days": n,
        "date_range": [str(dates[0].date()), str(dates[-1].date())],
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
    ap.add_argument("--days", type=int, default=1600, help="합성 데이터 길이")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mode", choices=["walkforward", "fixed"], default="walkforward")
    ap.add_argument("--cost", type=float, default=0.001, help="포지션 변경당 왕복 비용(0.1%)")
    ap.add_argument("--min-window", type=int, default=756, help="워크포워드 최소 롤링윈도(≈3년)")
    ap.add_argument("--rebalance", type=int, default=21, help="임계값 재추정 주기(거래일)")
    ap.add_argument("--step", type=int, default=1, help="패널 날짜 간격(1=일별)")
    ap.add_argument("--dry-run", action="store_true", help="JSON 저장 생략")
    args = ap.parse_args()

    bundle = load_bundle(args)
    print(f"[패널] 시점별 6팩터 점수 계산 중… (거래일 {len(bundle.kospi)}개, 미래참조 없음)",
          flush=True)
    panel = score_panel(bundle, step=args.step)
    if len(panel) < args.min_window + 60:
        print(f"패널 표본 부족({len(panel)}). 워크포워드엔 최소윈도+여유가 필요. 기간을 늘리세요.",
              file=sys.stderr)
        if args.mode == "walkforward":
            return 1

    res = run_backtest(panel, ForecastWeights(), args.cost, mode=args.mode,
                       min_window=args.min_window, rebal=args.rebalance)

    s, h, x = res["strategy"], res["buy_hold"], res["excess"]
    print(f"\n=== 전망점수 백테스트 [{res['mode']}] "
          f"({res['date_range'][0]} ~ {res['date_range'][1]}, {res['n_days']}거래일) ===")
    if res["mode"] == "walkforward":
        print(f"  임계값 재추정 {res['threshold_reestimations']}회 · 최소윈도 "
              f"{res['min_window_days']}일 · 리밸 {res['rebalance_days']}일")
    print(f"  {'':14}{'전략':>11}{'단순보유':>11}")
    print(f"  {'총수익':<12}{s['total_return']*100:>10.1f}%{h['total_return']*100:>10.1f}%")
    print(f"  {'CAGR':<13}{s['cagr']*100:>10.1f}%{h['cagr']*100:>10.1f}%")
    print(f"  {'MDD':<14}{s['mdd']*100:>10.1f}%{h['mdd']*100:>10.1f}%")
    print(f"  {'샤프':<13}{s['sharpe']:>11.2f}{h['sharpe']:>11.2f}")
    wr = s.get("win_rate")
    if wr is not None:
        print(f"  승률 {wr*100:.1f}% ({s['n_trades']}회) · 손익비 {s.get('payoff')} · "
              f"시장노출 {s['time_in_market']*100:.0f}% · 상위5거래 이익기여 "
              f"{(s.get('top5_win_share') or 0)*100:.0f}%")
    print(f"  → 단순보유 대비 초과수익 {x['total_return']*100:+.1f}%p")
    print("  [구간별 초과수익]")
    for sp in res["subperiods"]:
        print(f"    {sp['period']}: 전략 {sp['strategy_return']*100:+.1f}% vs 보유 "
              f"{sp['buy_hold_return']*100:+.1f}% → 초과 {sp['excess_return']*100:+.1f}%p")

    if args.dry_run:
        print("\n[dry-run] JSON 저장 생략")
        return 0
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "source": args.source,
        "params": {"cost": args.cost, "weights": "default_prior",
                   "mode": args.mode, "min_window": args.min_window, "rebalance": args.rebalance},
        "note": ("워크포워드(임계값 재추정, 미래참조 없음)·인샘플 튜닝 배제. 과거 성과는 "
                 "미래를 보장하지 않으며 표본기간·임계값에 민감함."),
        **res,
    }
    BACKTEST_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ {BACKTEST_JSON} 저장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
