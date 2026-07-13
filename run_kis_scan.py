#!/usr/bin/env python3
"""실시간 종가매매 스캔 (KIS 분봉) — 15:10~15:20 실행용.

동작:
  1) 대상 종목의 과거 일봉으로 기준선(DailyContext)을 만든다 (장 시작 전에 안 변함)
  2) KIS 로 '오늘치 분봉'을 받아 오전/오후/막판 특징 + 종가매매 신호를 계산
  3) KOSPI 지수 상승국면(regime)일 때만 매수 후보로 인정
  4) state/overnight_latest.json 갱신 → 대시보드 '오늘의 후보' 에 실시간 반영

사용:
  # 실전 (KIS 키 필요, 국내 장중에 실행)
  export KIS_APP_KEY=... KIS_APP_SECRET=... KIS_ENV=real
  python run_kis_scan.py --market KOSPI --top 100 --snapshot 15:18

  # 오프라인 데모 (KIS/네트워크 불필요, 합성 분봉으로 파이프라인 점검)
  python run_kis_scan.py --source synthetic
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

from overnight.strategy import ClosingParams
from intraday.features import DailyContext, compute_features
from intraday.synthetic import make_minute_day

STATE = Path("state")


def _parse_snapshot(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def _candidate(feat: dict) -> dict:
    """대시보드 스키마(close/day_ret/close_pos/vol_ratio/rsi)에 맞춰 변환 + 분봉 특징 부가."""
    return {
        "code": feat["code"], "name": feat.get("name", feat["code"]),
        "date": str(date.today()),
        "close": feat["price"],                 # 현재가(=매수 예정가)
        "day_ret": feat["day_ret"], "close_pos": feat["close_pos"],
        "vol_ratio": feat["vol_ratio"], "rsi": feat["rsi"],
        # 분봉 특징(참고용)
        "morning_ret": feat["morning_ret"], "afternoon_ret": feat["afternoon_ret"],
        "last30_ret": feat["last30_ret"], "vwap_pos": feat["vwap_pos"],
    }


def run_real(args, p: ClosingParams, snapshot: time) -> tuple[list[dict], bool | None]:
    from swing import data as datamod
    from swing.strategy import market_regime
    from intraday.kis import KisClient

    # 시장 국면
    regime_on = None
    try:
        idx = datamod.fetch_index("KS11", args.start, str(date.today()))
        regime_on = bool(market_regime(idx["Close"], args.regime_ma).iloc[-1])
        print(f"[국면] KOSPI risk-on={regime_on}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[국면] 지수 조회 실패, 국면필터 없이 진행: {e}", flush=True)

    codes = datamod.fetch_universe(args.market, args.top)
    names = datamod.fetch_names(args.market)
    kis = KisClient()
    to_hhmmss = snapshot.strftime("%H%M%S")

    cands = []
    for i, code in enumerate(codes, 1):
        try:
            daily = datamod.fetch_ohlcv(code, args.start, str(date.today()))
            if len(daily) < 65:
                continue
            # 오늘 봉이 이미 일봉에 들어있다면 제외(기준선은 어제까지)
            daily = daily[daily.index.date < date.today()]
            ctx = DailyContext.from_daily(code, daily, p, name=names.get(code, code))
            minute = kis.minute_bars(code, to_hhmmss)
            feat = compute_features(minute, ctx, p, snapshot=snapshot,
                                    regime_on=(regime_on is not False))
            if feat.get("enough") and feat["signal"]:
                cands.append(_candidate(feat))
        except Exception as e:  # noqa: BLE001
            print(f"  ({i}/{len(codes)}) {code} 실패: {e}", flush=True)
    return cands, regime_on


def run_synthetic(p: ClosingParams, snapshot: time) -> tuple[list[dict], bool | None]:
    """KIS 없이 파이프라인 점검: 강한 하루 2종목 + 약한 하루 1종목."""
    demo = [
        ("005930", "삼성전자(합성)", 74000, 0.07, 0.7),   # 강함 → 신호 기대
        ("000660", "SK하이닉스(합성)", 190000, 0.06, 0.6),  # 강함
        ("068270", "셀트리온(합성)", 180000, 0.005, 0.3),  # 약함 → 신호 없음
    ]
    prior = [10000 + 30 * k for k in range(60)]   # 완만한 상승 과거선
    cands = []
    for code, name, prev, dret, late in demo:
        closes = [prev * (1 + 0.001 * k) for k in range(60)]
        daily = pd.DataFrame({
            "Open": closes, "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes], "Close": closes,
            "Volume": [1_000_000] * 60,
        }, index=pd.bdate_range("2026-04-01", periods=60))
        ctx = DailyContext.from_daily(code, daily, p, name=name)
        # 분봉의 전일종가는 기준선의 전일종가와 일치시켜야 day_ret 이 맞다.
        # 거래량은 20일평균의 3배 이상 나오도록 base 를 키운다.
        minute = make_minute_day(prev_close=ctx.prev_close, day_ret=dret, late_strength=late,
                                 base_min_vol=9000, seed=abs(hash(code)) % 1000)
        feat = compute_features(minute, ctx, p, snapshot=snapshot, regime_on=True)
        print(f"  {name}: day_ret={feat['day_ret']}% vol×{feat['vol_ratio']} "
              f"close_pos={feat['close_pos']}% signal={feat['signal']}")
        if feat.get("enough") and feat["signal"]:
            cands.append(_candidate(feat))
    return cands, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "synthetic"], default="real")
    ap.add_argument("--market", default="KOSPI")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--start", default=str(date.today().replace(year=date.today().year - 1)))
    ap.add_argument("--snapshot", default="15:18", help="판단 시각 HH:MM (매수 직전)")
    ap.add_argument("--regime-ma", type=int, default=120)
    args = ap.parse_args()

    p = ClosingParams()
    snapshot = _parse_snapshot(args.snapshot)

    if args.source == "real":
        cands, regime_on = run_real(args, p, snapshot)
    else:
        cands, regime_on = run_synthetic(p, snapshot)

    cands.sort(key=lambda c: c["vol_ratio"] * max(c["day_ret"], 0), reverse=True)
    STATE.mkdir(exist_ok=True)
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "kis-live" if args.source == "real" else "synthetic",
        "snapshot": args.snapshot,
        "market_regime": regime_on,
        "candidates": cands,
    }
    (STATE / "overnight_latest.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[저장] state/overnight_latest.json — 실시간 후보 {len(cands)}개")
    for c in cands:
        print(f"    · {c['name']}({c['code']}) {c['close']}  상승 {c['day_ret']}%  "
              f"거래량 {c['vol_ratio']}배  오전 {c['morning_ret']}% 오후 {c['afternoon_ret']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
