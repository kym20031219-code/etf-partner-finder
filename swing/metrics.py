"""매매 통계 및 자산곡선 지표."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Trade


def trade_stats(trades: list[Trade]) -> dict:
    if not trades:
        return {"trades": 0}
    rets = np.array([t.ret for t in trades])
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    hold = np.array([t.bars_held for t in trades])

    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    win_rate = len(wins) / len(rets)
    # 기대값 = 승률*평균이익 + 패률*평균손실
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else float("inf")

    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1

    return {
        "trades": len(rets),
        "win_rate": win_rate,
        "avg_ret": rets.mean(),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff": payoff,
        "expectancy": expectancy,
        "avg_hold": hold.mean(),
        "best": rets.max(),
        "worst": rets.min(),
        "exit_reasons": reasons,
    }


def equity_stats(eq: pd.DataFrame) -> dict:
    if eq.empty or len(eq) < 2:
        return {}
    e = eq["equity"]
    total_return = e.iloc[-1] / e.iloc[0] - 1
    days = (eq.index[-1] - eq.index[0]).days or 1
    years = days / 365.25
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    roll_max = e.cummax()
    mdd = ((e - roll_max) / roll_max).min()
    return {
        "start_equity": float(e.iloc[0]),
        "end_equity": float(e.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "mdd": float(mdd),
        "years": years,
    }


def format_report(tstats: dict, estats: dict, params_note: str = "") -> str:
    if tstats.get("trades", 0) == 0:
        return "매매 없음 (신호가 발생하지 않았습니다). 파라미터를 완화해 보세요."
    L = []
    L.append("=" * 52)
    L.append("  눌림목 스윙 백테스트 결과")
    if params_note:
        L.append("  " + params_note)
    L.append("=" * 52)
    L.append(f"  총 매매 수      : {tstats['trades']}")
    L.append(f"  승률            : {tstats['win_rate']*100:5.1f} %")
    L.append(f"  평균 수익률/건  : {tstats['avg_ret']*100:+5.2f} %")
    L.append(f"  평균 이익 (승)  : {tstats['avg_win']*100:+5.2f} %")
    L.append(f"  평균 손실 (패)  : {tstats['avg_loss']*100:+5.2f} %")
    L.append(f"  손익비(Payoff)  : {tstats['payoff']:5.2f}")
    L.append(f"  기대값/건       : {tstats['expectancy']*100:+5.2f} %")
    L.append(f"  평균 보유일     : {tstats['avg_hold']:5.1f} 일")
    L.append(f"  최고/최악       : {tstats['best']*100:+.1f}% / {tstats['worst']*100:+.1f}%")
    L.append(f"  청산 사유       : {tstats['exit_reasons']}")
    if estats:
        L.append("-" * 52)
        L.append(f"  포트폴리오 기간 : {estats['years']:.1f} 년")
        L.append(f"  누적 수익률     : {estats['total_return']*100:+.1f} %")
        L.append(f"  연복리(CAGR)    : {estats['cagr']*100:+.1f} %")
        L.append(f"  최대낙폭(MDD)   : {estats['mdd']*100:.1f} %")
    L.append("=" * 52)
    return "\n".join(L)
