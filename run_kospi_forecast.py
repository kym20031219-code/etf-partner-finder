#!/usr/bin/env python3
"""코스피 **단기 전망** 점수 산출 → JSON/CSV 저장 (대시보드 kospi.html 용).

동작:
  1. 시장 데이터 수집 (거시·한국경제·실적·수급·밸류에이션·기술 재료)
     - real     : FinanceDataReader(지수·환율·해외지수) + pykrx(PER/PBR·투자자 수급)
     - synthetic: 네트워크가 막힌 환경에서 파이프라인/대시보드 검증용 합성 데이터
  2. swing.kospi_forecast 로 6팩터 전망 점수 계산
  3. results/kospi_forecast.json (대시보드 최신본) 저장
  4. results/kospi_history.csv (종합점수·코스피 종가 히스토리) 갱신
  5. TELEGRAM_* 있으면 전망 요약 전송 (없으면 스킵)

⚠️ **참고용 전망 점수**만 만든다. 매매 권유·자동주문 기능은 없다.

사용:
  python run_kospi_forecast.py                     # 실데이터
  python run_kospi_forecast.py --source synthetic  # 오프라인 검증
  python run_kospi_forecast.py --dry-run           # 텔레그램 미전송
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date, datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from swing.kospi_forecast import MarketBundle, compute_forecast

RESULTS_DIR = Path("results")
FORECAST_JSON = RESULTS_DIR / "kospi_forecast.json"
HISTORY_CSV = RESULTS_DIR / "kospi_history.csv"

# 반도체 대표주 (지수 핵심 엔진): 삼성전자 · SK하이닉스
SEMI_LEADERS = ["005930", "000660"]


# ---------------------------------------------------------------------------
# 실데이터 수집 (네트워크 필요) — 소스별로 방어적으로, 실패는 None 으로 둔다
# ---------------------------------------------------------------------------
def _fdr_series(code: str, start: str, end: str, col: str = "Close") -> pd.Series | None:
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(code, start, end)
        df = df.rename(columns=str.capitalize)
        if col not in df.columns:
            return None
        s = df[col].dropna()
        return s if len(s) else None
    except Exception as e:  # noqa: BLE001
        print(f"  [FDR] {code} 실패: {e}", file=sys.stderr)
        return None


def _first(*series: pd.Series | None) -> pd.Series | None:
    """None 이 아닌 첫 시계열. (pandas Series 는 `A or B` 가 불가하므로 이 헬퍼 사용)"""
    for s in series:
        if s is not None:
            return s
    return None


def _semis_index(start: str, end: str) -> pd.Series | None:
    """삼성전자+SK하이닉스를 동일가중 합성지수(기준100)로 만든다."""
    parts = []
    for code in SEMI_LEADERS:
        s = _fdr_series(code, start, end)
        if s is not None and len(s):
            parts.append(s / s.iloc[0])
    if not parts:
        return None
    df = pd.concat(parts, axis=1).dropna()
    return (df.mean(axis=1) * 100.0) if len(df) else None


def _pykrx_fundamental(start: str, end: str) -> tuple[pd.Series | None, pd.Series | None]:
    """코스피(1001) PER·PBR 시계열."""
    try:
        from pykrx import stock
        s, e = start.replace("-", ""), end.replace("-", "")
        df = stock.get_index_fundamental(s, e, "1001")
        if df is None or df.empty:
            return None, None
        df.index = pd.to_datetime(df.index)
        per = df["PER"].dropna() if "PER" in df.columns else None
        pbr = df["PBR"].dropna() if "PBR" in df.columns else None
        return per, pbr
    except Exception as e:  # noqa: BLE001
        print(f"  [pykrx] PER/PBR 실패: {e}", file=sys.stderr)
        return None, None


def _pykrx_flows(start: str, end: str) -> tuple[pd.Series | None, pd.Series | None]:
    """코스피 외국인·기관 일별 순매수 거래대금."""
    try:
        from pykrx import stock
        s, e = start.replace("-", ""), end.replace("-", "")
        # 투자자별 일별 순매수 거래대금 (KOSPI 시장 전체)
        df = stock.get_market_trading_value_by_date(s, e, "KOSPI")
        if df is None or df.empty:
            return None, None
        df.index = pd.to_datetime(df.index)
        cols = {c: c for c in df.columns}
        foreign = None
        for k in ("외국인", "외국인합계"):
            if k in cols:
                foreign = df[k].dropna()
                break
        inst = df["기관합계"].dropna() if "기관합계" in cols else None
        return foreign, inst
    except Exception as e:  # noqa: BLE001
        print(f"  [pykrx] 수급 실패: {e}", file=sys.stderr)
        return None, None


def build_real_bundle(start: str, end: str) -> MarketBundle:
    print("[수집] 실데이터 (FDR + pykrx)…", flush=True)
    ks = _fdr_series("KS11", start, end, "Close")
    if ks is None:
        raise RuntimeError("코스피 지수(KS11) 조회 실패 — 실데이터 모드를 쓸 수 없습니다.")
    # 코스피 OHLCV
    import FinanceDataReader as fdr
    kospi = fdr.DataReader("KS11", start, end).rename(columns=str.capitalize)
    kospi = kospi[["Open", "High", "Low", "Close", "Volume"]].dropna()

    per, pbr = _pykrx_fundamental(start, end)
    foreign, inst = _pykrx_flows(start, end)

    return MarketBundle(
        kospi=kospi,
        sp500=_first(_fdr_series("US500", start, end), _fdr_series("S&P500", start, end)),
        vix=_fdr_series("VIX", start, end),
        us10y=_first(_fdr_series("US10YT=RR", start, end), _fdr_series("US10YT", start, end)),
        usdkrw=_fdr_series("USD/KRW", start, end),
        china=_fdr_series("SSEC", start, end),
        kosdaq=_fdr_series("KQ11", start, end),
        semis=_semis_index(start, end),
        per=per, pbr=pbr, foreign=foreign, inst=inst,
    )


# ---------------------------------------------------------------------------
# 합성 데이터 (오프라인 검증용) — 대시보드가 실제로 렌더되도록 전 재료 생성
# ---------------------------------------------------------------------------
def _rw(n: int, start_val: float, mu: float, sig: float, rng, index) -> pd.Series:
    rets = rng.normal(mu, sig, n)
    vals = start_val * np.cumprod(1 + rets)
    return pd.Series(vals, index=index)


def build_synthetic_bundle(days: int = 400, seed: int = 7) -> MarketBundle:
    print("[수집] 합성 데이터 (오프라인 검증)…", flush=True)
    from swing import data as datamod

    kospi = datamod.make_synthetic("KOSPI", days=days, seed=seed)
    # 지수 스케일로 보정 (2000~2900 근처)
    kospi = kospi.copy()
    scale = 2600.0 / float(kospi["Close"].iloc[-1])
    for col in ("Open", "High", "Low", "Close"):
        kospi[col] = kospi[col] * scale
    idx = kospi.index
    rng = np.random.default_rng(seed)

    return MarketBundle(
        kospi=kospi,
        sp500=_rw(days, 5000, 0.0006, 0.010, rng, idx),
        vix=_rw(days, 18, 0.0, 0.05, rng, idx).clip(10, 45),
        us10y=_rw(days, 4.2, 0.0, 0.012, rng, idx).clip(2.5, 6.0),
        usdkrw=_rw(days, 1350, 0.0001, 0.005, rng, idx),
        china=_rw(days, 3100, 0.0003, 0.011, rng, idx),
        kosdaq=_rw(days, 850, 0.0005, 0.013, rng, idx),
        semis=_rw(days, 100, 0.0011, 0.018, rng, idx),
        per=_rw(days, 11.5, 0.0, 0.006, rng, idx).clip(8, 16),
        pbr=_rw(days, 1.05, 0.0, 0.005, rng, idx).clip(0.7, 1.5),
        foreign=pd.Series(rng.normal(400e8, 6000e8, days), index=idx),
        inst=pd.Series(rng.normal(-150e8, 5000e8, days), index=idx),
    )


# ---------------------------------------------------------------------------
# 저장 / 히스토리 / 알림
# ---------------------------------------------------------------------------
def save_forecast(result: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    result = {**result, "generated_at": datetime.now().isoformat(timespec="seconds")}
    FORECAST_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def update_history(result: dict, keep: int = 400) -> None:
    """(date, kospi_close, score, bias) 를 히스토리 CSV 에 upsert."""
    RESULTS_DIR.mkdir(exist_ok=True)
    rows: dict[str, dict] = {}
    if HISTORY_CSV.exists():
        with HISTORY_CSV.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[r["date"]] = r
    rows[result["as_of"]] = {
        "date": result["as_of"],
        "kospi_close": result["kospi_close"],
        "score": result["score"],
        "bias": result["bias"],
    }
    ordered = [rows[k] for k in sorted(rows)][-keep:]
    with HISTORY_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "kospi_close", "score", "bias"])
        w.writeheader()
        w.writerows(ordered)


def _slice_bundle(b: MarketBundle, end: pd.Timestamp) -> MarketBundle:
    """번들의 모든 시계열을 end 이하로 잘라 '그 날 시점' 번들을 만든다(소급 계산용)."""
    def cut(x):
        if x is None:
            return None
        return x[x.index <= end]
    return MarketBundle(
        kospi=cut(b.kospi), sp500=cut(b.sp500), vix=cut(b.vix), us10y=cut(b.us10y),
        usdkrw=cut(b.usdkrw), china=cut(b.china), kosdaq=cut(b.kosdaq), semis=cut(b.semis),
        per=cut(b.per), pbr=cut(b.pbr), foreign=cut(b.foreign), inst=cut(b.inst),
    )


def backfill_history(b: MarketBundle, n: int) -> int:
    """최근 n 거래일에 대해 전망 점수를 소급 계산해 히스토리 CSV 를 채운다.

    각 날짜 시점의 데이터만으로(미래 참조 없이) 종합점수를 재계산하므로, 대시보드가
    첫날부터 '점수 추이'를 보여줄 수 있다. 실데이터에도 그대로 쓸 수 있다.
    """
    idx = b.kospi.index
    dates = idx[-n:] if n < len(idx) else idx
    done = 0
    for d in dates:
        if d == idx[-1]:
            continue  # 최신일은 메인 결과가 이미 기록
        sub = _slice_bundle(b, d)
        if len(sub.kospi) < 120:
            continue
        try:
            update_history(compute_forecast(sub))
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [backfill] {d.date()} 스킵: {e}", file=sys.stderr)
    return done


def format_summary(r: dict) -> str:
    lines = [
        f"📊 <b>코스피 단기 전망</b>  ({escape(r['as_of'])})",
        f"코스피 {r['kospi_close']:,} ({r['kospi_change_pct']:+}%)",
        f"종합점수 <b>{r['score']}</b>/100 · <b>{escape(r['bias'])}</b> (신뢰도 {escape(r['confidence'])})",
        "",
    ]
    for fct in r["factors"]:
        lines.append(f"{fct['icon']} {fct['label']}: {fct['score']} ({fct['bias']})")
    lines.append("")
    lines.append(escape(r["outlook"]))
    lines.append("\n⚠️ 참고용 전망 점수일 뿐 매매 권유가 아닙니다.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "synthetic"], default="real")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--days", type=int, default=400, help="합성 데이터 길이")
    ap.add_argument("--backfill", type=int, default=0,
                    help="최근 N 거래일 종합점수를 소급 계산해 히스토리 채우기(첫 실행용)")
    ap.add_argument("--dry-run", action="store_true", help="텔레그램 전송하지 않음")
    args = ap.parse_args()

    if args.source == "synthetic":
        bundle = build_synthetic_bundle(days=args.days)
    else:
        try:
            bundle = build_real_bundle(args.start, args.end)
        except Exception as e:  # noqa: BLE001
            print(f"[경고] 실데이터 수집 실패 → 합성 데이터로 대체: {e}", file=sys.stderr)
            bundle = build_synthetic_bundle(days=args.days)

    result = compute_forecast(bundle)
    save_forecast(result)
    update_history(result)
    if args.backfill > 0:
        n = backfill_history(bundle, args.backfill)
        print(f"[backfill] 과거 {n}일 점수 소급 기록")

    print(f"\n✅ {FORECAST_JSON} 저장")
    print(f"   기준일 {result['as_of']} · 코스피 {result['kospi_close']:,} "
          f"({result['kospi_change_pct']:+}%)")
    print(f"   종합점수 {result['score']}/100 · {result['bias']} "
          f"(신뢰도 {result['confidence']})")
    for fct in result["factors"]:
        avail = sum(1 for s in fct["signals"] if s["available"])
        print(f"   {fct['icon']} {fct['label']:<10} {fct['score']:>5}  "
              f"({fct['bias']}) · 신호 {avail}/{len(fct['signals'])}")
    print(f"   → {result['outlook']}")

    if args.dry_run or not os.environ.get("TELEGRAM_TOKEN"):
        print("\n[dry-run 또는 토큰 없음] 텔레그램 전송 생략")
        return 0
    try:
        from swing.notify import TelegramError, send_message
        send_message(format_summary(result))
        print("✅ 텔레그램 전송 완료")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 텔레그램 전송 실패(비치명적): {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
