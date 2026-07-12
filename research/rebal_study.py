#!/usr/bin/env python3
"""리밸런싱 순수효과 연구 (자산조합 A/B/C 비교).

표준 전제는 research/config.py / METHODOLOGY.md 를 따른다:
  - 거래비용 왕복 0.1% (편도 0.05%)를 회전율에 비례 차감
  - look-ahead 방지: 비중은 전일 종가 기준, 리밸런싱은 종가 체결 → 익일 반영
  - survivorship 방지: 자산은 경제적 역할로 사전 선정, 상장 이전은 지수 스플라이싱

측정:
  각 조합에 대해 [리밸런싱 vs 드리프트(방치)] 를 동일 자산·동일(균등)비중으로 비교.
  순수효과 = 지표(리밸런싱) − 지표(드리프트). 분산수익·국면별·롤링 일관성 포함.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.config import ROUND_TRIP_COST, CALENDAR_RULES, THRESHOLD_BANDS

TD = 252
ONE_WAY = ROUND_TRIP_COST / 2  # 편도 비용률

SETS = {
    "A_주식+국채": ["EQ", "LT"],
    "B_주식+금": ["EQ", "GLD"],
    "C_주식+국채+금": ["EQ", "LT", "GLD"],
}
STRESS = {"2008_금융위기": ("2007-10-09", "2009-03-09"),
          "2022_긴축": ("2022-01-03", "2022-10-12")}


# ---------------------------------------------------------------------------
def _adj(df):
    return df["Adj Close"] if "Adj Close" in df.columns else df["Close"]


def load_prices(start, end) -> tuple[pd.DataFrame, dict]:
    import FinanceDataReader as fdr
    notes = {}

    def try_tickers(role, cands):
        for t in cands:
            try:
                s = _adj(fdr.DataReader(t, start, end)).dropna()
                if len(s) > 1000:
                    notes[role] = f"{t} ({s.index[0].date()}~{s.index[-1].date()}, {len(s)}일)"
                    return s.rename(role)
            except Exception as e:  # noqa: BLE001
                print(f"[{role}] {t} 실패: {e}")
        raise RuntimeError(f"{role} 데이터 확보 실패")

    eq = try_tickers("EQ", ["^SP500TR", "VFINX"])          # 주식 총수익
    lt = try_tickers("LT", ["VUSTX", "TLT"])                # 장기국채 총수익(뮤추얼펀드)
    gld = try_tickers("GLD", ["GC=F", "GLD", "XAUUSD=X"])   # 금 (무배당→가격≈총수익)
    px = pd.concat([eq, lt, gld], axis=1).dropna()
    return px, notes


def metrics(ret: pd.Series, rf: pd.Series) -> dict:
    ret = ret.dropna()
    rf = rf.reindex(ret.index).fillna(0)
    eq = (1 + ret).cumprod()
    n = len(ret)
    cagr = eq.iloc[-1] ** (TD / n) - 1
    vol = ret.std() * np.sqrt(TD)
    ex = ret - rf
    sharpe = ex.mean() * TD / vol if vol > 0 else 0.0
    dn = ret[ret < 0].std() * np.sqrt(TD)
    sortino = ex.mean() * TD / dn if dn > 0 else 0.0
    mdd = ((eq - eq.cummax()) / eq.cummax()).min()
    return {"cagr": float(cagr), "vol": float(vol), "sharpe": float(sharpe),
            "sortino": float(sortino), "mdd": float(mdd)}


def period_end_set(idx: pd.DatetimeIndex, freq: str) -> set:
    code = {"ME": "M", "QE": "Q", "YE": "Y"}[freq]
    s = pd.Series(idx, index=idx)
    return set(s.groupby(idx.to_period(code)).transform("last"))


def simulate(returns: pd.DataFrame, target: np.ndarray, rule) -> dict:
    """rule: 'drift' | 'ME'/'QE'/'YE' | ('band', x). 균등/지정 비중 target."""
    idx = returns.index
    rmat = returns.to_numpy()
    cur = target.copy()
    rebal_days = period_end_set(idx, rule) if rule in ("ME", "QE", "YE") else set()
    band = rule[1] if isinstance(rule, tuple) else None
    out = np.empty(len(idx))
    turnover_sum = 0.0
    n_rebal = 0
    for i in range(len(idx)):
        r = rmat[i]
        pr = float(np.dot(cur, r))                 # 전일 비중으로 당일 수익
        cur = cur * (1 + r)
        cur = cur / cur.sum()                      # 비중 드리프트
        do = False
        if rule != "drift":
            if band is not None:
                do = np.abs(cur - target).max() > band
            else:
                do = idx[i] in rebal_days
        if do:
            turn = float(np.abs(target - cur).sum())   # 총 매매(양방향)
            pr -= turn * ONE_WAY                        # 비용 차감(종가 체결)
            cur = target.copy()
            turnover_sum += turn
            n_rebal += 1
        out[i] = pr
    return {"ret": pd.Series(out, index=idx), "n_rebal": n_rebal,
            "turnover_oneway_total": turnover_sum / 2}


def diversification_return(returns: pd.DataFrame, w: np.ndarray) -> float:
    """≈ ½(Σ wᵢσᵢ² − σ_p²), 연율. 리밸런싱이 이론적으로 뽑아내는 분산수익."""
    cov = returns.cov().to_numpy() * TD
    var_assets = np.diag(cov)
    port_var = w @ cov @ w
    return float(0.5 * (np.dot(w, var_assets) - port_var))


def rolling_effect(reb: pd.Series, dft: pd.Series, win_y: int) -> dict:
    """롤링 win_y년 CAGR 차이(리밸런싱−드리프트) 분포로 일관성 측정."""
    w = win_y * TD
    er = (1 + reb).rolling(w).apply(lambda x: x.prod(), raw=True) ** (TD / w) - 1
    ed = (1 + dft).rolling(w).apply(lambda x: x.prod(), raw=True) ** (TD / w) - 1
    diff = (er - ed).dropna()
    if diff.empty:
        return {}
    return {"windows": int(len(diff)), "median_pp": round(float(diff.median()) * 100, 2),
            "pct_positive": round(float((diff > 0).mean()) * 100, 1),
            "p25_pp": round(float(diff.quantile(.25)) * 100, 2),
            "p75_pp": round(float(diff.quantile(.75)) * 100, 2)}


def window_stats(ret: pd.Series, a, b) -> dict:
    seg = ret.loc[a:b].dropna()
    if seg.empty:
        return {}
    eq = (1 + seg).cumprod()
    return {"total_pct": round(float(eq.iloc[-1] - 1) * 100, 1),
            "mdd_pct": round(float(((eq - eq.cummax()) / eq.cummax()).min()) * 100, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1999-06-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--primary", default="QE", help="주 리밸런싱 규칙")
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    import FinanceDataReader as fdr
    px, notes = load_prices(args.start, args.end)
    rets = px.pct_change().dropna()
    print("데이터:", json.dumps(notes, ensure_ascii=False))
    print(f"공통구간 {rets.index[0].date()} ~ {rets.index[-1].date()} ({len(rets)}일)")

    # 무위험(3M 국채) — Sharpe용
    try:
        tb = fdr.DataReader("FRED:DGS3MO", args.start, args.end).iloc[:, 0]
        rf = ((1 + tb.reindex(rets.index).ffill().clip(lower=0) / 100) ** (1 / TD) - 1).fillna(0)
    except Exception:  # noqa: BLE001
        rf = pd.Series(0.0, index=rets.index)

    # 상관(경제적 상보성 구조)
    corr_full = rets.corr().round(2)
    corr_2022 = rets.loc["2022-01-01":"2022-12-31"].corr().round(2)

    report = {"meta": {"start": str(rets.index[0].date()), "end": str(rets.index[-1].date()),
                       "days": len(rets), "cost_round_trip": ROUND_TRIP_COST,
                       "weights": "equal (사전등록)", "primary_rule": args.primary,
                       "data_notes": notes},
              "correlations": {"full": corr_full.to_dict(), "2022": corr_2022.to_dict()},
              "sets": {}}

    for sname, cols in SETS.items():
        sub = rets[cols]
        w = np.repeat(1 / len(cols), len(cols))
        drift = simulate(sub, w, "drift")
        reb = simulate(sub, w, args.primary)
        m_d = metrics(drift["ret"], rf)
        m_r = metrics(reb["ret"], rf)
        effect = {k: round((m_r[k] - m_d[k]) * (100 if k != "sharpe" and k != "sortino" else 1), 4)
                  for k in m_d}
        # 리밸런싱 규칙 민감도(사전등록 그리드 전부 보고)
        grid = {}
        for rule in CALENDAR_RULES + [("band", b) for b in THRESHOLD_BANDS]:
            key = rule if isinstance(rule, str) else f"band{int(rule[1]*100)}"
            sim = simulate(sub, w, rule)
            mm = metrics(sim["ret"], rf)
            grid[key] = {"cagr": round(mm["cagr"]*100, 2), "sharpe": round(mm["sharpe"], 2),
                         "mdd": round(mm["mdd"]*100, 1), "n_rebal": sim["n_rebal"]}
        # 국면(스트레스 구간) — 2자산 붕괴 vs 3자산 복원 확인
        stress = {}
        for label, (a, b) in STRESS.items():
            stress[label] = {"rebalanced": window_stats(reb["ret"], a, b),
                             "drift": window_stats(drift["ret"], a, b)}
        report["sets"][sname] = {
            "assets": cols,
            "drift": {k: round(v, 4) for k, v in m_d.items()},
            "rebalanced": {k: round(v, 4) for k, v in m_r.items()},
            "pure_effect": effect,
            "diversification_return_pct": round(diversification_return(sub, w) * 100, 2),
            "rebalances_per_year": round(reb["n_rebal"] / (len(rets) / TD), 1),
            "rule_sensitivity": grid,
            "rolling_effect": {f"{y}y": rolling_effect(reb["ret"], drift["ret"], y)
                               for y in (3, 5, 10)},
            "stress": stress,
        }

    # 출력
    print("\n=== 상관계수 (전체 / 2022) ===")
    print(corr_full.to_string()); print("---2022---"); print(corr_2022.to_string())
    for sname, d in report["sets"].items():
        print(f"\n===== {sname}  (분산수익 {d['diversification_return_pct']}%/년, "
              f"리밸런싱 {d['rebalances_per_year']}회/년) =====")
        print(f"{'':12}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'MDD':>8}")
        for lab, m in (("드리프트", d["drift"]), ("리밸런싱", d["rebalanced"])):
            print(f"{lab:12}{m['cagr']*100:>7.2f}%{m['sharpe']:>8.2f}{m['sortino']:>9.2f}{m['mdd']*100:>7.1f}%")
        pe = d["pure_effect"]
        print(f"순수효과   ΔCAGR {pe['cagr']:+.2f}%p · ΔSharpe {pe['sharpe']:+.2f} · ΔMDD {pe['mdd']:+.1f}%p")
        print("롤링 일관성(리밸>드리프트 %):",
              {y: v.get("pct_positive") for y, v in d["rolling_effect"].items()})

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[저장] {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
