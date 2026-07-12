#!/usr/bin/env python3
"""S&P500(SPY 토탈리턴) 마켓타이밍 국면 필터 비교 연구.

전략:
  BH        Buy & Hold (항상 100% 주식)
  MA200     200일 이동평균선 위=주식, 아래=현금
  YC        장단기 금리차(10Y-2Y) 정상=주식, 역전=현금
  MOM       12개월(252일) 모멘텀 양수=주식, 음수=현금
  COMBO     위 3개 신호 중 2개 이상 매수면 100% 주식, 아니면 현금

원칙:
  - 신호는 당일 종가 정보로 산출 → 다음 날부터 적용(미래참조 없음, shift 1)
  - 현금은 3개월 국채(FRED DGS3MO) 수익률로 이자 발생
  - 전환 시 회전율 × 거래비용(기본 0.1%) 차감
  - 주가는 토탈리턴(배당 재투자) 기준

데이터: FinanceDataReader (네트워크 필요 — GitHub Actions/로컬)
  주가 : ^SP500TR (실패 시 SPY 수정종가)
  금리차: FRED:T10Y2Y
  현금 : FRED:DGS3MO (3개월 국채, 연율%)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# 분석 구간
BEAR_WINDOWS = {
    "2000-02 닷컴": ("2000-03-24", "2002-10-09"),
    "2008 금융위기": ("2007-10-09", "2009-03-09"),
    "2022 긴축": ("2022-01-03", "2022-10-12"),
}
REBOUND_WINDOWS = {
    "2009 반등": ("2009-03-09", "2009-12-31"),
    "2020 코로나반등": ("2020-03-23", "2020-08-31"),
}


# ---------------------------------------------------------------------------
def load_data(start: str, end: str) -> pd.DataFrame:
    import FinanceDataReader as fdr

    # 주가(토탈리턴)
    stock = None
    for tkr in ("^SP500TR", "US500TR"):
        try:
            df = fdr.DataReader(tkr := tkr, start, end)
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            s = df[col].dropna()
            if len(s) > 1000:
                stock = s.rename("stock")
                print(f"[주가] {tkr} ({col}) {s.index[0].date()}~{s.index[-1].date()} {len(s)}일")
                break
        except Exception as e:  # noqa: BLE001
            print(f"[주가] {tkr} 실패: {e}")
    if stock is None:  # 폴백: SPY 수정종가(배당반영 총수익 근사)
        df = fdr.DataReader("SPY", start, end)
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        stock = df[col].dropna().rename("stock")
        print(f"[주가] 폴백 SPY ({col}) {len(stock)}일")

    # 금리차 / 현금금리 (FRED)
    spread = fdr.DataReader("FRED:T10Y2Y", start, end).iloc[:, 0].rename("spread")
    tbill = fdr.DataReader("FRED:DGS3MO", start, end).iloc[:, 0].rename("tbill")

    df = pd.DataFrame(index=stock.index)
    df["stock"] = stock
    df["spread"] = spread.reindex(df.index).ffill()
    df["tbill"] = tbill.reindex(df.index).ffill()
    df = df.dropna(subset=["stock"])
    # 현금 일간수익률 (연율% → 일간)
    df["rf_daily"] = (1 + df["tbill"].clip(lower=0) / 100) ** (1 / TRADING_DAYS) - 1
    df["stock_ret"] = df["stock"].pct_change().fillna(0)
    return df


def signals(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=df.index)
    ma200 = df["stock"].rolling(200).mean()
    s["MA200"] = (df["stock"] > ma200).astype(float)
    s["YC"] = (df["spread"] >= 0).astype(float)
    mom = df["stock"] / df["stock"].shift(TRADING_DAYS) - 1
    s["MOM"] = (mom > 0).astype(float)
    s["COMBO"] = ((s["MA200"] + s["YC"] + s["MOM"]) >= 2).astype(float)
    # 워밍업(지표 미정) 구간은 100% 주식으로 간주하지 않도록 NaN → 이후 dropna 정렬
    warm = df["stock"].rolling(200).mean().notna() & df["stock"].shift(TRADING_DAYS).notna()
    return s.where(warm, np.nan)


def simulate(df: pd.DataFrame, target: pd.Series, cost: float) -> dict:
    """target: 당일 종가 기준 목표주식비중(0/1). 다음날부터 적용."""
    w = target.shift(1)  # 미래참조 방지
    # 유효 시작점부터
    valid = w.notna()
    w = w[valid].fillna(0.0)
    d = df.loc[w.index]
    turnover = w.diff().abs().fillna(w.iloc[0])
    port = w * d["stock_ret"] + (1 - w) * d["rf_daily"] - turnover * cost
    switches = int((w.diff().abs() > 1e-9).sum())
    cost_drag = float((turnover * cost).sum())
    return {"ret": port, "switches": switches, "cost_drag": cost_drag,
            "weight": w, "days_in_mkt": float(w.mean())}


def metrics(ret: pd.Series, rf_daily: pd.Series) -> dict:
    ret = ret.dropna()
    rf = rf_daily.reindex(ret.index).fillna(0)
    n = len(ret)
    eq = (1 + ret).cumprod()
    cagr = eq.iloc[-1] ** (TRADING_DAYS / n) - 1
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    excess = ret - rf
    sharpe = (excess.mean() * TRADING_DAYS) / vol if vol > 0 else 0.0
    downside = ret[ret < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (excess.mean() * TRADING_DAYS) / downside if downside > 0 else 0.0
    mdd = ((eq - eq.cummax()) / eq.cummax()).min()
    return {"cagr": float(cagr), "vol": float(vol), "sharpe": float(sharpe),
            "sortino": float(sortino), "mdd": float(mdd),
            "total_return": float(eq.iloc[-1] - 1)}


def window_return(ret: pd.Series, a: str, b: str) -> float:
    seg = ret.loc[a:b].dropna()
    if seg.empty:
        return float("nan")
    return float((1 + seg).prod() - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1999-06-01")  # 200일/12개월 워밍업 여유
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--cost", type=float, default=0.001)  # 0.1%
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    df = load_data(args.start, args.end)
    sig = signals(df)

    strategies = {
        "BH": pd.Series(1.0, index=df.index),
        "MA200": sig["MA200"],
        "YC": sig["YC"],
        "MOM": sig["MOM"],
        "COMBO": sig["COMBO"],
    }

    results = {}
    rets = {}
    for name, target in strategies.items():
        sim = simulate(df, target, args.cost if name != "BH" else 0.0)
        m = metrics(sim["ret"], df["rf_daily"])
        m.update({"switches": sim["switches"], "cost_drag_pct": round(sim["cost_drag"] * 100, 2),
                  "days_in_market_pct": round(sim["days_in_mkt"] * 100, 1)})
        results[name] = m
        rets[name] = sim["ret"]

    # 구간별 분석
    bh = rets["BH"]
    bears = {}
    for label, (a, b) in BEAR_WINDOWS.items():
        bh_r = window_return(bh, a, b)
        bears[label] = {"BH": round(bh_r * 100, 1)}
        for name in ("MA200", "YC", "MOM", "COMBO"):
            r = window_return(rets[name], a, b)
            bears[label][name] = round(r * 100, 1)
            bears[label][f"{name}_손실감소"] = round((r - bh_r) * 100, 1)
    rebounds = {}
    for label, (a, b) in REBOUND_WINDOWS.items():
        bh_r = window_return(bh, a, b)
        rebounds[label] = {"BH": round(bh_r * 100, 1)}
        for name in ("MA200", "YC", "MOM", "COMBO"):
            r = window_return(rets[name], a, b)
            rebounds[label][name] = round(r * 100, 1)
            rebounds[label][f"{name}_기회비용"] = round((bh_r - r) * 100, 1)

    out = {
        "meta": {"start": str(df.index[0].date()), "end": str(df.index[-1].date()),
                 "days": len(df), "cost_pct": args.cost * 100},
        "metrics": results, "bear_windows": bears, "rebound_windows": rebounds,
    }

    # 콘솔 출력
    print("\n================ 전략 비교 (2000~2025, 비용 %.1f%%) ================" % (args.cost * 100))
    hdr = f"{'전략':<7}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'Sortino':>9}{'MDD':>8}{'전환':>6}{'주식%':>7}"
    print(hdr); print("-" * len(hdr))
    for name, m in results.items():
        print(f"{name:<7}{m['cagr']*100:>7.1f}%{m['vol']*100:>7.1f}%{m['sharpe']:>8.2f}"
              f"{m['sortino']:>9.2f}{m['mdd']*100:>7.1f}%{m['switches']:>6}{m['days_in_market_pct']:>6.0f}%")
    print("\n[대세 하락장 — 각 필터 수익률(%) 및 B&H 대비 손실감소(%p)]")
    print(json.dumps(bears, ensure_ascii=False, indent=2))
    print("\n[급반등 구간 — 각 필터 수익률(%) 및 기회비용(%p)]")
    print(json.dumps(rebounds, ensure_ascii=False, indent=2))

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[저장] {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
