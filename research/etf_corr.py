#!/usr/bin/env python3
"""12개 ETF 일별 총수익(TR) 수집 + 구간별 상관 안정성 분석.

구간:
  low   저금리기        2010-2021
  high  고금리/긴축기   2022-2024
  c2003 위기: 2020년 3월
  c2022 위기: 2022년 전체

각 쌍의 상관을 구간별로 계산(겹치는 기간만, 표본수 명시)하고,
  - 불안정 쌍: 구간 간 상관 변동폭(max−min)이 큰 쌍
  - 안정적 저상관 쌍: 모든 구간에서 |상관|이 낮게 유지되는 쌍
을 구분해 보고한다. (survivorship 방지: 상장 이전 구간은 N/A로 명시, 조작 없음)
"""
from __future__ import annotations

import argparse
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["SPMO", "QQQ", "VOO", "GLD", "TLT", "IEF", "SGOV", "XLV", "XLE", "VNQ", "DBC", "SHY"]

WINDOWS = {
    "저금리(10-21)": ("2010-01-01", "2021-12-31"),
    "고금리(22-24)": ("2022-01-01", "2024-12-31"),
    "위기2020-03": ("2020-03-01", "2020-03-31"),
    "위기2022": ("2022-01-01", "2022-12-31"),
}
MIN_OBS = {"위기2020-03": 12}  # 짧은 위기창은 최소표본 완화
DEFAULT_MIN_OBS = 30


def load_returns(start, end) -> tuple[pd.DataFrame, dict]:
    import FinanceDataReader as fdr
    cols, cov = {}, {}
    for t in TICKERS:
        try:
            df = fdr.DataReader(t, start, end)
            s = (df["Adj Close"] if "Adj Close" in df.columns else df["Close"]).dropna()
            if len(s) < 30:
                cov[t] = "데이터부족"; continue
            cols[t] = s
            cov[t] = f"{s.index[0].date()}~{s.index[-1].date()} ({len(s)}일)"
        except Exception as e:  # noqa: BLE001
            cov[t] = f"실패:{e}"
    px = pd.concat(cols, axis=1)
    rets = px.pct_change()
    return rets, cov


def window_corr(rets: pd.DataFrame, a, b, min_obs) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = rets.loc[a:b]
    corr = sub.corr()
    notna = sub.notna().astype(int)
    counts = notna.T.dot(notna)          # 쌍별 공통 표본수
    corr = corr.where(counts >= min_obs)  # 표본 부족 쌍은 NaN
    return corr.round(3), counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    rets, cov = load_returns(args.start, args.end)
    print("=== 커버리지 (survivorship: 상장 시점 명시) ===")
    for t in TICKERS:
        print(f"  {t:5} {cov.get(t)}")

    corrs = {}
    for wname, (a, b) in WINDOWS.items():
        mo = MIN_OBS.get(wname, DEFAULT_MIN_OBS)
        c, _ = window_corr(rets, a, b, mo)
        corrs[wname] = c

    # 쌍별 구간 상관 표
    rows = []
    for x, y in itertools.combinations([t for t in TICKERS if t in rets.columns], 2):
        vals = {w: (None if pd.isna(corrs[w].loc[x, y]) else float(corrs[w].loc[x, y]))
                for w in WINDOWS}
        avail = [v for v in vals.values() if v is not None]
        if len(avail) < 2:
            rng = None
        else:
            rng = round(max(avail) - min(avail), 3)
        rows.append({"pair": f"{x}-{y}", **vals,
                     "range": rng,
                     "max_abs": round(max(abs(v) for v in avail), 3) if avail else None})
    df = pd.DataFrame(rows)

    # 불안정 쌍: range 큰 순
    unstable = df[df["range"].notna()].sort_values("range", ascending=False)
    # 안정적 저상관: 모든 가용 구간에서 |corr|<=0.25, 최소 3구간 이상 가용
    def n_avail(r):
        return sum(r[w] is not None for w in WINDOWS)
    stable = df[df.apply(lambda r: n_avail(r) >= 3 and (r["max_abs"] is not None and r["max_abs"] <= 0.25), axis=1)]
    stable = stable.sort_values("max_abs")

    wcols = list(WINDOWS)
    print("\n=== 상관 불안정 상위 15쌍 (구간 변동폭 큰 순) ===")
    print(unstable[["pair"] + wcols + ["range"]].head(15).to_string(index=False))
    print("\n=== 구간 무관 꾸준히 낮은 상관 쌍 (|corr|<=0.25, 3구간+ 가용) ===")
    if stable.empty:
        print("  해당 쌍 없음 (기준 완화 필요)")
    else:
        print(stable[["pair"] + wcols + ["max_abs"]].to_string(index=False))

    if args.report_json:
        out = {"coverage": cov, "windows": {k: list(v) for k, v in WINDOWS.items()},
               "corr_matrices": {w: corrs[w].where(pd.notna(corrs[w]), None).to_dict() for w in WINDOWS},
               "pairs": rows,
               "unstable_top": unstable[["pair"] + wcols + ["range"]].head(20).to_dict("records"),
               "stable_low": stable[["pair"] + wcols + ["max_abs"]].to_dict("records")}
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[저장] {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
