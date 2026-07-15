#!/usr/bin/env python3
"""코스피 전망 **팩터 가중치 최적화** (단기 예측력 극대화, 과최적화 방지).

무엇을 푸나
  6팩터(거시·한국경제·실적·수급·밸류에이션·기술) 종합 점수가 **앞으로의 코스피
  수익률(단기, 기본 60거래일≈3개월)을 얼마나 잘 예측하는지**를 기준으로 팩터
  가중치를 캘리브레이션한다. 예측력 지표는 **순위 IC(Information Coefficient,
  종합점수와 미래수익의 스피어만 상관)** 를 쓴다.

어떻게
  1. 과거 각 날짜 t 에서 **그 시점 데이터만으로**(미래참조 없이) 6팩터 점수를 계산해
     패널을 만든다. 동시에 t→t+h 의 코스피 미래수익률을 라벨로 붙인다.
  2. 시계열을 앞(학습)/뒤(검증)로 나눈다(기본 65/35). **미래 구간이 검증**이므로
     진짜 아웃오브샘플이다.
  3. 6차원 심플렉스(가중치 합=1) 위에서 무작위(디리클레) 후보를 많이 뽑아, **학습
     구간 IC 상위** 후보 중 **검증 구간 IC 가 가장 높은** 조합을 고른다(워크포워드
     선택 → 과최적화 완화). 기본 가중치(합리적 초기값)의 IC 와 함께 리포트한다.

결과
  results/kospi_weights_best.json — 최적 가중치 + 학습/검증 IC + 방향 적중률 + 기준선
  → run_kospi_forecast.py 가 다음 실행부터 이 가중치를 자동으로 사용한다.

사용
  # 실데이터 (네트워크 열린 환경: 내 PC / GitHub Actions)
  python optimize_kospi.py --source real --start 2016-01-01 --horizon 60

  # 오프라인 코드 검증 (합성 데이터 — 수치는 참고 불가, 동작만 확인)
  python optimize_kospi.py --source synthetic --days 900 --samples 800

⚠️ 예측력 '캘리브레이션'만 한다. 어떤 백테스트도 미래 수익을 보장하지 않으며, 매매
   권유·자동주문 기능은 없다. 단기 지수 예측은 본질적으로 노이즈가 크고 IC 는 보통
   작다(0.05~0.15면 유의미). 낮은 IC 를 과신하지 말 것.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from swing.kospi_forecast import (
    ForecastWeights, MarketBundle, composite_from, factor_scores, slice_bundle,
)

RESULTS_DIR = Path("results")
WEIGHTS_JSON = RESULTS_DIR / "kospi_weights_best.json"
FACTORS = ["macro", "korea", "earnings", "flows", "valuation", "technical"]


# ---------------------------------------------------------------------------
# 예측력 지표
# ---------------------------------------------------------------------------
def rank_ic(scores: np.ndarray, fwd: np.ndarray) -> float:
    """순위 IC = 종합점수와 미래수익의 스피어만 상관(랭크 후 피어슨)."""
    if len(scores) < 8:
        return float("nan")
    rs = pd.Series(scores).rank().to_numpy()
    rf = pd.Series(fwd).rank().to_numpy()
    if rs.std() == 0 or rf.std() == 0:
        return 0.0
    return float(np.corrcoef(rs, rf)[0, 1])


def hit_rate(scores: np.ndarray, fwd: np.ndarray) -> float:
    """방향 적중률: sign(종합점수-50) == sign(미래수익) 비율(중립 근처는 제외)."""
    side = np.sign(scores - 50.0)
    mask = side != 0
    if mask.sum() == 0:
        return float("nan")
    return float((side[mask] == np.sign(fwd[mask])).mean())


# ---------------------------------------------------------------------------
# 패널 생성 (시점별 6팩터 점수 + 미래수익) — 미래참조 없음
# ---------------------------------------------------------------------------
def build_panel(b: MarketBundle, min_obs: int = 150, step: int = 1,
                warmup: int = 150) -> pd.DataFrame:
    """각 날짜의 6팩터 점수 패널을 만든다.

    step>1 이면 날짜를 건너뛰며 계산(속도↑). 미래수익은 후단계에서 horizon 별로 붙인다.
    """
    idx = b.kospi.index
    close = b.kospi["Close"]
    rows = []
    for i in range(warmup, len(idx), step):
        d = idx[i]
        sub = slice_bundle(b, d)
        if len(sub.kospi) < min_obs:
            continue
        sc = factor_scores(sub)
        rows.append({"date": d, "close": float(close.iloc[i]), **sc})
    panel = pd.DataFrame(rows).set_index("date")
    return panel


def attach_forward(panel: pd.DataFrame, full_close: pd.Series, h: int) -> pd.DataFrame:
    """패널에 t→t+h 코스피 미래수익률(fwd) 라벨을 붙인다(끝 h개는 라벨없음→drop)."""
    c = full_close.reindex(panel.index).astype(float)
    fut = full_close.shift(-h).reindex(panel.index).astype(float)
    out = panel.copy()
    out["fwd"] = fut / c - 1.0
    return out.dropna(subset=["fwd"])


# ---------------------------------------------------------------------------
# 가중치 탐색 (심플렉스 위 무작위 후보 → 워크포워드 선택)
# ---------------------------------------------------------------------------
def weights_to_dict(w: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip(FACTORS, w)}


def evaluate(panel: pd.DataFrame, w: np.ndarray) -> float:
    """가중치 w 로 패널의 종합점수 IC 계산."""
    S = panel[FACTORS].to_numpy()
    comp = S @ w
    return rank_ic(comp, panel["fwd"].to_numpy())


def optimize_horizon(panel_h: pd.DataFrame, samples: int, train_frac: float,
                     seed: int, top_frac: float = 0.10, max_weight: float = 0.45,
                     shrink: float = 0.35) -> dict:
    """한 horizon 패널에서 최적 가중치 탐색.

    학습 IC 상위 top_frac 후보 중 **검증 IC 최대** 조합을 고른다(워크포워드).
    과최적화·퇴화(단일 팩터 몰빵) 방지:
      - 후보를 균형쪽으로 바이어스(Dirichlet α=2)하고, **한 팩터 상한(max_weight)** 초과
        후보는 버린다(코너 해 차단).
      - 선택된 가중치를 **기본값(합리적 초기값)으로 shrink** 만큼 수축(축소추정)해 분산↓.
    """
    rng = np.random.default_rng(seed)
    n = len(panel_h)
    cut = int(n * train_frac)
    train, val = panel_h.iloc[:cut], panel_h.iloc[cut:]
    if len(train) < 20 or len(val) < 20:
        raise ValueError(f"표본 부족: train={len(train)} val={len(val)} (기간을 늘리세요)")

    default_w = np.array([ForecastWeights().as_dict()[k] for k in FACTORS])
    # 후보: 기본값 + 균형 바이어스 디리클레(상한 초과분은 거부)
    draws = rng.dirichlet(np.full(len(FACTORS), 2.0), size=samples * 2)
    draws = draws[draws.max(axis=1) <= max_weight][:samples]
    cand = np.vstack([default_w, draws]) if len(draws) else default_w[None, :]

    rec = [(w, evaluate(train, w)) for w in cand]
    rec = [r for r in rec if r[1] == r[1]]  # NaN 제거
    rec.sort(key=lambda r: r[1], reverse=True)

    k = max(1, int(len(rec) * top_frac))
    top = rec[:k]
    # 워크포워드 선택: 학습 상위군 중 검증 IC 최대
    best = max(top, key=lambda r: evaluate(val, r[0]))
    # 축소추정: 선택값을 기본 prior 쪽으로 shrink 만큼 당긴다
    w_sel = best[0] / best[0].sum()
    w_best = (1 - shrink) * w_sel + shrink * (default_w / default_w.sum())
    w_best = w_best / w_best.sum()

    ic_full = evaluate(panel_h, w_best)
    ic_tr = evaluate(train, w_best)
    ic_val = evaluate(val, w_best)
    ic_def_val = evaluate(val, default_w / default_w.sum())
    comp_val = val[FACTORS].to_numpy() @ w_best
    return {
        "weights": weights_to_dict(w_best),
        "ic_train": round(ic_tr, 4),
        "ic_val": round(ic_val, 4),
        "ic_full": round(ic_full, 4),
        "hit_val": round(hit_rate(comp_val, val["fwd"].to_numpy()), 4),
        "baseline_ic_val": round(ic_def_val, 4),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
    }


# ---------------------------------------------------------------------------
# 데이터 로드 (러너와 동일 소스 재사용)
# ---------------------------------------------------------------------------
def load_bundle(args) -> MarketBundle:
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
    ap.add_argument("--days", type=int, default=900, help="합성 데이터 길이")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--horizon", type=int, default=60,
                    help="최적화 목표 예측기간(거래일). 단기~3개월이면 60 근처")
    ap.add_argument("--eval-horizons", type=int, nargs="*", default=[20, 40, 60],
                    help="함께 리포트할 예측기간들(거래일)")
    ap.add_argument("--samples", type=int, default=3000, help="가중치 후보 표본 수")
    ap.add_argument("--train-frac", type=float, default=0.65)
    ap.add_argument("--max-weight", type=float, default=0.45,
                    help="한 팩터 가중치 상한(코너 해/몰빵 방지)")
    ap.add_argument("--shrink", type=float, default=0.35,
                    help="선택 가중치를 기본값으로 당기는 축소추정 비율(0~1, 과최적화 완화)")
    ap.add_argument("--step", type=int, default=1, help="패널 날짜 간격(속도용)")
    ap.add_argument("--dry-run", action="store_true", help="JSON 저장 생략")
    args = ap.parse_args()

    horizons = sorted(set(args.eval_horizons) | {args.horizon})
    bundle = load_bundle(args)
    print(f"[패널] 시점별 6팩터 점수 계산 중… (거래일 {len(bundle.kospi)}개)", flush=True)
    panel = build_panel(bundle, step=args.step)
    if len(panel) < 60:
        print(f"패널 표본이 너무 적습니다({len(panel)}). 기간을 늘리세요.", file=sys.stderr)
        return 1
    full_close = bundle.kospi["Close"]

    per_h = {}
    for h in horizons:
        ph = attach_forward(panel, full_close, h)
        try:
            per_h[h] = optimize_horizon(
                ph, args.samples, args.train_frac, args.seed,
                max_weight=args.max_weight, shrink=args.shrink,
            )
        except ValueError as e:
            print(f"[h={h}] 스킵: {e}", file=sys.stderr)
    if args.horizon not in per_h:
        print("목표 horizon 최적화 실패 — 기간/표본을 늘리세요.", file=sys.stderr)
        return 1

    chosen = per_h[args.horizon]
    print(f"\n=== 최적화 결과 (목표 {args.horizon}거래일 ≈ {args.horizon/21:.1f}개월) ===")
    print(f"검증 IC {chosen['ic_val']:+.3f} (기본가중치 {chosen['baseline_ic_val']:+.3f}) · "
          f"학습 IC {chosen['ic_train']:+.3f} · 방향적중 {chosen['hit_val']*100:.1f}%")
    print(f"표본 학습 {chosen['n_train']} / 검증 {chosen['n_val']}")
    print("가중치(합=100%):")
    for k, v in chosen["weights"].items():
        print(f"   {k:<10} {v*100:5.1f}%")
    print("\n[참고] 기간별 검증 IC:")
    for h in horizons:
        if h in per_h:
            print(f"   {h:>3}일: 최적 {per_h[h]['ic_val']:+.3f} vs 기본 {per_h[h]['baseline_ic_val']:+.3f}")

    if args.dry_run:
        print("\n[dry-run] JSON 저장 생략")
        return 0

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "objective": "rank_ic_forward_return",
        "target_horizon_days": args.horizon,
        "best_weights": chosen["weights"],
        "metrics": chosen,
        "by_horizon": {str(h): per_h[h] for h in per_h},
        "default_weights": ForecastWeights().as_dict(),
        "source": args.source,
        "samples": args.samples,
        "panel_rows": int(len(panel)),
        "date_range": [str(panel.index[0].date()), str(panel.index[-1].date())],
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "note": ("단기 지수 예측 IC 는 본질적으로 작다(0.05~0.15면 유의미). "
                 "합성 데이터 결과는 참고 불가. 미래수익 보장 아님."),
    }
    WEIGHTS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ {WEIGHTS_JSON} 저장 — 다음 전망 실행부터 이 가중치를 자동 사용")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
