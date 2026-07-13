#!/usr/bin/env python3
"""장마감 후 하루 1회 **모멘텀 돌파** 종목 스코어링 → CSV 히스토리로 저장.

전략 방향(사용자 요구): 며칠~수주간 강한 상승을 노리고, 손실은 짧게(ATR 초기손절)
수익은 길게(샹들리에 트레일링) 가져가는 신고가 돌파 추세추종. (기존 눌림목 아님)

동작:
  1. 대상 종목의 일봉을 받아 장기추세 위 종목을 0~100점으로 줄세움 (swing/momentum.py)
  2. results/scan_YYYYMMDD.csv 로 그날의 순위표 저장 (히스토리 커밋)
  3. results/index.json (대시보드 드롭다운용 날짜 목록) 갱신
  4. TELEGRAM_TOKEN/TELEGRAM_CHAT_ID 있으면 상위 5종목만 요약 전송 (없으면 스킵)

최적화 결과(results/momentum_best.json)가 있으면 그 파라미터·가중치를 자동으로
사용하고, 없으면 코드 기본값을 쓴다. (optimize_momentum.py 로 생성)

⚠️ 종목 '추천 점수'만 만든다. 실제 매수/매도 주문·증권사 계좌 접근 코드는 없다.
   매매는 본인 판단·책임으로 별도 계좌에서 진행한다.

사용:
  python run_daily_scan.py --market KOSPI --top 150          # 실데이터
  python run_daily_scan.py --source synthetic --dry-run       # 오프라인/미전송 테스트
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace
from datetime import date, datetime
from html import escape
from pathlib import Path

from swing import data as datamod
from swing.momentum import MomentumParams, ScoreWeights, rank_universe
from swing.notify import TelegramError, send_message
from swing.strategy import market_regime

RESULTS_DIR = Path("results")
INDEX_JSON = RESULTS_DIR / "index.json"
BEST_JSON = RESULTS_DIR / "momentum_best.json"

# CSV 컬럼 순서 (대시보드 테이블도 이 순서를 따른다)
CSV_FIELDS = [
    "rank", "code", "name", "close", "total",
    "trend", "breakout", "momentum", "volume", "rsi_score",
    "rsi", "roc_pct", "high_gap_pct", "signal", "stop", "trail_hint",
]


def load_optimized() -> tuple[MomentumParams, ScoreWeights, bool]:
    """results/momentum_best.json 이 있으면 최적 파라미터·가중치를 로드."""
    p, w = MomentumParams(), ScoreWeights()
    if not BEST_JSON.exists():
        return p, w, False
    try:
        d = json.loads(BEST_JSON.read_text(encoding="utf-8"))
        bp = d.get("best_params") or {}
        fields = {f for f in MomentumParams.__dataclass_fields__}
        p = replace(p, **{k: v for k, v in bp.items() if k in fields})
        bw = (d.get("best_weights") or {}).get("weights") or {}
        wfields = {f for f in ScoreWeights.__dataclass_fields__}
        w = replace(w, **{k: v for k, v in bw.items() if k in wfields})
        return p, w, True
    except Exception as e:  # noqa: BLE001
        print(f"[경고] {BEST_JSON} 읽기 실패, 기본값 사용: {e}", file=sys.stderr)
        return MomentumParams(), ScoreWeights(), False


def build_universe(args) -> tuple[dict, dict]:
    if args.source == "synthetic":
        return datamod.synthetic_universe(n=args.n, days=args.days), {}
    codes = datamod.fetch_universe(args.market, args.top)
    names = datamod.fetch_names(args.market)
    uni: dict = {}
    for code in codes:
        try:
            df = datamod.fetch_ohlcv(code, args.start, args.end)
            if len(df) > 150:
                uni[code] = df
        except Exception as e:  # noqa: BLE001
            print(f"  {code} 실패: {e}", file=sys.stderr)
    return uni, names


def write_csv(path: Path, rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def update_index(scan_date: str, generated_at: str) -> list[str]:
    dates = sorted(
        {p.stem.replace("scan_", "") for p in RESULTS_DIR.glob("scan_*.csv")},
        reverse=True,
    )
    INDEX_JSON.write_text(
        json.dumps({"latest": dates[0] if dates else scan_date, "dates": dates,
                    "generated_at": generated_at, "strategy": "momentum-breakout"},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dates


def format_top(rows: list[dict], scan_date: str, n: int = 5) -> str:
    top = rows[:n]
    lines = [f"🚀 <b>모멘텀 돌파 스코어 TOP {len(top)}</b>  ({escape(scan_date)})", ""]
    for r in top:
        name = escape(str(r.get("name", r["code"])))
        tag = "🚀돌파" if r["signal"] == "buy" else "👀관심"
        lines.append(
            f"{r['rank']}. <b>{name}</b> (<code>{escape(r['code'])}</code>) · {tag}\n"
            f"   총점 <b>{r['total']}</b> · 종가 {r['close']:,} · {r['mom_lb']}일수익 {r['roc_pct']:+}%\n"
            f"   추세 {r['trend']} / 돌파 {r['breakout']} / 모멘텀 {r['momentum']} / 거래량 {r['volume']}"
        )
    lines.append("\n⚠️ 추천 점수일 뿐 매매 권유가 아닙니다. 매매는 본인 판단·책임.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "synthetic"], default="real")
    ap.add_argument("--market", default="KOSPI")
    ap.add_argument("--top", type=int, default=150, help="스코어링 대상 시총 상위 종목 수")
    ap.add_argument("--show", type=int, default=30, help="CSV 에 남길 상위 순위 개수")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--days", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true", help="텔레그램 전송하지 않고 출력만")
    ap.add_argument("--no-regime", action="store_true", help="시장 국면 필터 끄기")
    ap.add_argument("--regime-ma", type=int, default=120)
    args = ap.parse_args()

    p, w, optimized = load_optimized()
    print(f"[전략] 모멘텀 돌파 · 파라미터 {'최적화값' if optimized else '기본값'} 사용", flush=True)

    universe, names = build_universe(args)
    if not universe:
        print("데이터가 비었습니다.", file=sys.stderr)
        return 1

    regime = None
    regime_on = None
    if args.source == "real" and not args.no_regime:
        try:
            idx = datamod.fetch_index("KS11", args.start, args.end)
            regime = market_regime(idx["Close"], args.regime_ma)
            regime_on = bool(regime.iloc[-1])
            print(f"[국면] KOSPI risk-on={regime_on} (최근)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[국면] 지수 조회 실패, 필터 생략: {e}", file=sys.stderr)

    rows = rank_universe(universe, p=p, w=w, names=names, top=args.show, regime=regime)
    for r in rows:  # 알림 문구용 보조 값
        r["mom_lb"] = p.mom_lookback
    if not rows:
        print("장기 상승추세 후보가 없습니다 — 빈 결과 저장", file=sys.stderr)

    scan_date = rows[0]["date"] if rows else args.end
    stamp = scan_date.replace("-", "")
    generated_at = datetime.now().isoformat(timespec="seconds")

    csv_path = RESULTS_DIR / f"scan_{stamp}.csv"
    write_csv(csv_path, rows)
    dates = update_index(stamp, generated_at)
    print(f"✅ {csv_path} 저장 ({len(rows)}종목) · 히스토리 {len(dates)}일치")

    buys = sum(1 for r in rows if r["signal"] == "buy")
    print(f"돌파 진입신호 {buys}개 · 관심 {len(rows) - buys}개")
    for r in rows[:10]:
        print(f"  {r['rank']:2d}. {r['name']:<16} {r['code']}  총점 {r['total']:>5}  "
              f"(추세 {r['trend']} 돌파 {r['breakout']} 모멘텀 {r['momentum']} 거래량 {r['volume']})")

    if not rows:
        return 0
    if args.dry_run or not os.environ.get("TELEGRAM_TOKEN"):
        print("[dry-run 또는 토큰 없음] 텔레그램 전송 생략")
        return 0
    try:
        send_message(format_top(rows, scan_date, n=5))
        print("✅ 텔레그램 전송 완료")
    except TelegramError as e:
        print(f"⚠️ {e}", file=sys.stderr)  # 알림 실패가 저장을 망치지 않도록 비치명적
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
