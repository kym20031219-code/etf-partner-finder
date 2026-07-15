#!/usr/bin/env python3
"""코스피 전망 점수 **유효성 검증** (라벨 정의 + 팩터 IC).

무엇을 하나 (기능 추가가 아니라 '검증 강화')
  [항목2] 타겟(라벨)을 **"20거래일 후 KOSPI 수익률"** 로 명시적으로 정의하고,
          종합점수(0~100) ↔ 이 라벨의 관계를 **구간표 + 산점도**로 남긴다.
  [항목4] 6개 팩터 각각에 대해 라벨과의 **IC(피어슨)·Rank IC(스피어만)** 와 유의성
          (t-통계량)을 계산한다. 유의하지 않은 팩터는 '제거 후보'로 **표시만** 한다.

미래참조 없음: 각 시점 t 의 팩터 점수는 t 까지 데이터로 계산(score_panel), 라벨은
t→t+20 미래수익. 예측력을 정직하게 측정한다.

결과: results/kospi_validation.json (대시보드 '팩터 유효성'·'점수-미래수익' 섹션)

⚠️ 20일 라벨은 겹쳐서(overlapping) 자기상관이 있어 단순 t-통계량은 과대평가될 수 있다
   (엄밀히는 Newey-West 보정 필요). IC 자체가 작은 게 정상이며(0.05~0.1도 유의미),
   과거 관계가 미래를 보장하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from swing.kospi_forecast import FACTOR_KEYS, FACTOR_META, ForecastWeights, score_panel

RESULTS_DIR = Path("results")
VALIDATION_JSON = RESULTS_DIR / "kospi_validation.json"
HORIZON = 20  # 거래일 (약 1개월)


def forward_return(close: pd.Series, index: pd.DatetimeIndex, h: int) -> pd.Series:
    c = close.reindex(index).astype(float)
    fut = close.shift(-h).reindex(index).astype(float)
    return (fut / c - 1.0)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 8 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 8:
        return 0.0
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return _pearson(rx, ry)


def _tstat(ic: float, n: int) -> float:
    if n < 4 or abs(ic) >= 1.0:
        return 0.0
    return float(ic * np.sqrt((n - 2) / (1 - ic * ic)))


def factor_ic(panel: pd.DataFrame, fwd: pd.Series) -> list[dict]:
    """6팩터 + 종합점수의 IC/Rank IC/유의성."""
    df = panel.join(fwd.rename("fwd")).dropna(subset=["fwd"])
    y = df["fwd"].to_numpy()
    n = len(y)
    rows = []
    wd = ForecastWeights().as_dict()
    comp = df[FACTOR_KEYS].to_numpy() @ np.array([wd[k] for k in FACTOR_KEYS])
    for key in FACTOR_KEYS + ["_composite"]:
        x = comp if key == "_composite" else df[key].to_numpy()
        ic = _pearson(x, y)
        ric = _rank_ic(x, y)
        t = _tstat(ric, n)
        rows.append({
            "key": key,
            "label": "종합점수" if key == "_composite" else FACTOR_META[key][0],
            "ic": round(ic, 4),
            "rank_ic": round(ric, 4),
            "t_stat": round(t, 2),
            "significant": bool(abs(t) >= 1.96),
            "n": n,
            "removal_candidate": bool(key != "_composite" and abs(t) < 1.96),
        })
    # 종합점수를 맨 위로, 나머지는 |RankIC| 내림차순
    rows.sort(key=lambda r: (r["key"] != "_composite", -abs(r["rank_ic"])))
    return rows


def score_vs_forward(panel: pd.DataFrame, fwd: pd.Series) -> dict:
    """종합점수 ↔ 20일 후 수익률: 구간(bin) 통계표 + 산점도용 다운샘플."""
    wd = ForecastWeights().as_dict()
    comp = pd.Series(panel[FACTOR_KEYS].to_numpy() @ np.array([wd[k] for k in FACTOR_KEYS]),
                     index=panel.index, name="score")
    df = pd.concat([comp, fwd.rename("fwd")], axis=1).dropna()
    # 구간표: 점수대별 표본수·평균 미래수익·상승비율
    edges = [0, 40, 45, 50, 55, 60, 100]
    labels = ["<40", "40–45", "45–50", "50–55", "55–60", "≥60"]
    df["bucket"] = pd.cut(df["score"], bins=edges, labels=labels, include_lowest=True)
    table = []
    for lab in labels:
        g = df[df["bucket"] == lab]["fwd"]
        if len(g) == 0:
            continue
        table.append({
            "bucket": lab, "n": int(len(g)),
            "avg_fwd_return": round(float(g.mean()), 4),
            "median_fwd_return": round(float(g.median()), 4),
            "up_ratio": round(float((g > 0).mean()), 4),
        })
    # 산점도: 최대 ~300점 다운샘플
    step = max(1, len(df) // 300)
    scat = [{"score": round(float(r.score), 1), "fwd": round(float(r.fwd), 4)}
            for r in df.iloc[::step].itertuples()]
    ic = _pearson(df["score"].to_numpy(), df["fwd"].to_numpy())
    ric = _rank_ic(df["score"].to_numpy(), df["fwd"].to_numpy())
    return {"bucket_table": table, "scatter": scat, "n": int(len(df)),
            "composite_ic": round(ic, 4), "composite_rank_ic": round(ric, 4)}


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
    ap.add_argument("--horizon", type=int, default=HORIZON, help="라벨 예측기간(거래일)")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bundle = load_bundle(args)
    print(f"[패널] 시점별 6팩터 점수 계산 중… ({len(bundle.kospi)}거래일, 미래참조 없음)",
          flush=True)
    panel = score_panel(bundle, step=args.step)
    if len(panel) < 120:
        print(f"패널 표본 부족({len(panel)}).", file=sys.stderr)
        return 1
    fwd = forward_return(bundle.kospi["Close"], panel.index, args.horizon)

    ic_rows = factor_ic(panel, fwd)
    svf = score_vs_forward(panel, fwd)

    print(f"\n=== 라벨: {args.horizon}거래일 후 KOSPI 수익률 (표본 {svf['n']}) ===")
    print(f"  종합점수 IC {svf['composite_ic']:+.3f} · Rank IC {svf['composite_rank_ic']:+.3f}")
    print("  [점수대별 평균 미래수익 · 상승비율]")
    for r in svf["bucket_table"]:
        print(f"    {r['bucket']:>6}  n={r['n']:>4}  평균 {r['avg_fwd_return']*100:+5.1f}%  "
              f"상승 {r['up_ratio']*100:4.0f}%")
    print("\n  [팩터 유효성 — Rank IC / t / 유의]")
    for r in ic_rows:
        flag = "유의" if r["significant"] else ("제거후보" if r["removal_candidate"] else "")
        print(f"    {r['label']:<10} IC {r['ic']:+.3f} · RankIC {r['rank_ic']:+.3f} · "
              f"t {r['t_stat']:+5.2f}  {flag}")
    sig = sum(1 for r in ic_rows if r["key"] != "_composite" and r["significant"])
    print(f"\n  통계적으로 유의한 팩터: {sig}/6")

    if args.dry_run:
        print("\n[dry-run] JSON 저장 생략")
        return 0
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "source": args.source,
        "horizon_days": args.horizon,
        "label_definition": f"{args.horizon}거래일 후 KOSPI 종가 수익률",
        "date_range": [str(panel.index[0].date()), str(panel.index[-1].date())],
        "significant_factor_count": sig,
        "factor_ic": ic_rows,
        "score_vs_forward": svf,
        "note": ("20일 라벨은 겹쳐서(overlapping) 자기상관이 있어 단순 t-통계량은 과대평가"
                 " 가능(엄밀히 Newey-West 필요). 과거 관계가 미래를 보장하지 않음."),
    }
    VALIDATION_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ {VALIDATION_JSON} 저장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
