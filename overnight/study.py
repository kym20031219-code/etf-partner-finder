"""갭상승 특징 분석(study).

질문: "다음날 갭상승한 종목은 **전날(=오늘)** 어떤 특징을 보였나?"

방법:
  1. 모든 종목·모든 날에 대해 특징(당일상승률·고가권마감·거래량배수·돌파·RSI 등)과
     라벨(다음날 시가 갭 ≥ gap_threshold 이면 '갭상승')을 만든다.
  2. 갭상승한 날 vs 아닌 날로 나눠 각 특징의 평균을 비교한다  → '전날 특징' 표
  3. 종가매매 규칙(strategy.generate_signals)에 걸린 날의 갭상승 확률이
     전체 평균(base rate)보다 얼마나 높은지(lift) 를 잰다  → 규칙의 유효성
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import ClosingParams, add_features, generate_signals

# 비교할 특징들: (컬럼명, 사람이 읽는 라벨, 퍼센트 표기 여부)
FEATURES = [
    ("day_ret", "당일 상승률", True),
    ("close_pos", "고가권 마감(0~1)", False),
    ("vol_ratio", "거래량 배수(20일평균 대비)", False),
    ("rsi", "RSI(14)", False),
    ("breakout", "20일 고가 돌파 비율", False),
]


def _panel(universe: dict[str, pd.DataFrame], p: ClosingParams) -> pd.DataFrame:
    """모든 종목의 특징+라벨을 한 판(panel)으로 쌓는다."""
    frames = []
    for code, df in universe.items():
        if len(df) < p.ma_mid + p.breakout_lookback + 5:
            continue
        d = add_features(df, p)
        d = generate_signals(df, p)[["entry"]].join(d)
        # 라벨(gap)이 정의된 봉만: 마지막 봉은 next_open 이 없어 제외
        d = d[d["gap"].notna()]
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def gap_feature_study(universe: dict[str, pd.DataFrame], p: ClosingParams) -> dict:
    """갭상승 특징 분석 결과를 dict 로 반환 (JSON 직렬화 가능)."""
    panel = _panel(universe, p)
    if panel.empty:
        return {"n_days": 0}

    up = panel[panel["gap_up"]]
    down = panel[~panel["gap_up"]]
    base_rate = float(panel["gap_up"].mean())

    # 1) 갭상승 vs 비갭상승 — 전날 특징 평균 비교
    feature_table = []
    for col, label, is_pct in FEATURES:
        up_mean = float(up[col].mean()) if len(up) else float("nan")
        dn_mean = float(down[col].mean()) if len(down) else float("nan")
        feature_table.append(
            {
                "feature": label,
                "gap_up_mean": round(up_mean * 100, 2) if is_pct else round(up_mean, 3),
                "no_gap_mean": round(dn_mean * 100, 2) if is_pct else round(dn_mean, 3),
                "is_pct": is_pct,
            }
        )

    # 2) 종가매매 규칙의 유효성: 규칙 통과 날의 갭상승 확률 vs 전체 평균
    sig = panel[panel["entry"]]
    sig_rate = float(sig["gap_up"].mean()) if len(sig) else float("nan")
    sig_gap_mean = float(sig["gap"].mean()) if len(sig) else float("nan")
    all_gap_mean = float(panel["gap"].mean())

    return {
        "n_days": int(len(panel)),
        "n_gap_up": int(len(up)),
        "base_gap_up_rate": round(base_rate * 100, 2),        # 아무 날이나 골랐을 때 갭상승 확률 %
        "avg_gap_all": round(all_gap_mean * 100, 3),          # 전체 평균 익일 갭 %
        "feature_table": feature_table,
        "signal": {
            "n_signals": int(len(sig)),
            "gap_up_rate": round(sig_rate * 100, 2) if len(sig) else None,   # 규칙 통과 날 갭상승 확률 %
            "avg_gap": round(sig_gap_mean * 100, 3) if len(sig) else None,   # 규칙 통과 날 평균 익일 갭 %
            "lift": round(sig_rate / base_rate, 2) if len(sig) and base_rate else None,
        },
    }


def format_study(study: dict) -> str:
    """스터디 결과를 콘솔 리포트 문자열로."""
    if study.get("n_days", 0) == 0:
        return "분석할 데이터가 없습니다."
    L = []
    L.append("=" * 56)
    L.append("  갭상승 특징 분석 — '전날 무슨 일이 있었나'")
    L.append("=" * 56)
    L.append(f"  표본(종목·일)     : {study['n_days']:,} 일")
    L.append(f"  다음날 갭상승 비율 : {study['base_gap_up_rate']:.1f} %  (기준선/base rate)")
    L.append(f"  전체 평균 익일 갭  : {study['avg_gap_all']:+.2f} %")
    L.append("-" * 56)
    L.append("  [전날 특징 평균]  갭상승한 날  vs  그렇지 않은 날")
    for row in study["feature_table"]:
        unit = "%" if row["is_pct"] else ""
        L.append(
            f"    {row['feature']:<24} {row['gap_up_mean']:>8}{unit}  vs {row['no_gap_mean']:>8}{unit}"
        )
    s = study["signal"]
    L.append("-" * 56)
    L.append("  [종가매매 규칙 유효성]")
    L.append(f"    규칙 통과 건수    : {s['n_signals']:,}")
    if s["gap_up_rate"] is not None:
        L.append(f"    규칙 통과 시 갭상승 : {s['gap_up_rate']:.1f} %  (기준선 대비 {s['lift']}배)")
        L.append(f"    규칙 통과 평균 갭  : {s['avg_gap']:+.2f} %")
    L.append("=" * 56)
    return "\n".join(L)
