#!/usr/bin/env python3
"""장중 눌림목 스캔 → 신규 신호만 텔레그램 알림.

동작:
  1. 대상 종목의 최신 일봉을 받아 '가장 최근 봉' 진입 신호를 스캔
  2. state/alerted.json 과 대조해 (종목·신호일) 기준 신규만 추림 → 중복 알림 방지
  3. TELEGRAM_TOKEN/TELEGRAM_CHAT_ID 있으면 신규 후보를 전송
  4. state/signals_latest.json(웹 대시보드용), state/alerted.json(중복방지) 갱신

사용:
  python run_scan.py --market KOSPI --top 100          # 실데이터 + 전송
  python run_scan.py --source synthetic --dry-run       # 오프라인/미전송 테스트

Secrets(GitHub Actions) 또는 환경변수로 토큰을 주입하세요. 코드에 넣지 마세요.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

from swing import data as datamod
from swing.notify import TelegramError, format_candidates, send_message
from swing.strategy import PullbackParams, current_picks

STATE_DIR = Path("state")
ALERTED = STATE_DIR / "alerted.json"
SIGNALS = STATE_DIR / "signals_latest.json"


def load_alerted() -> set[str]:
    if ALERTED.exists():
        try:
            return set(json.loads(ALERTED.read_text(encoding="utf-8")).get("keys", []))
        except Exception:  # noqa: BLE001
            return set()
    return set()


def save_alerted(keys: set[str]) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    # 오래된 키가 무한정 쌓이지 않도록 최근 400개만 유지
    ALERTED.write_text(
        json.dumps({"keys": sorted(keys)[-400:]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_universe(args) -> tuple[dict, dict]:
    """(universe, names) 반환."""
    if args.source == "synthetic":
        uni = datamod.synthetic_universe(n=args.n, days=args.days)
        return uni, {}
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "synthetic"], default="real")
    ap.add_argument("--market", default="KOSPI")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--dry-run", action="store_true", help="전송하지 않고 출력만")
    args = ap.parse_args()

    p = PullbackParams()
    universe, names = build_universe(args)
    if not universe:
        print("데이터가 비었습니다.", file=sys.stderr)
        return 1

    picks = current_picks(universe, p)
    candidates = picks["buy"]
    watch = picks["watch"]

    def named(items):
        return [{**c, "name": names.get(c["code"], c["code"])} for c in items]

    # 중복 방지: (종목|신호일) 키 — 매수 신호에만 적용
    already = load_alerted()
    new = [c for c in candidates if f"{c['code']}|{c['date']}" not in already]

    # 웹 대시보드용 최신 스냅샷은 항상 갱신 (매수 신호 + 관심 관찰)
    STATE_DIR.mkdir(exist_ok=True)
    SIGNALS.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": args.source,
                "candidates": named(candidates),
                "watch": named(watch),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"매수신호 {len(candidates)}개 (신규 {len(new)}) · 관심관찰 {len(watch)}개")
    if not new:
        print("신규 신호 없음 — 알림 생략")
        return 0

    text = format_candidates(new, names)
    print("\n" + text + "\n")

    if args.dry_run or not os.environ.get("TELEGRAM_TOKEN"):
        print("[dry-run 또는 토큰 없음] 전송 생략")
    else:
        try:
            send_message(text)
            print("✅ 텔레그램 전송 완료")
        except TelegramError as e:
            print(f"⚠️ {e}", file=sys.stderr)
            return 2

    # 신규 키를 기록해 다음 실행 때 중복 전송 방지
    already.update(f"{c['code']}|{c['date']}" for c in new)
    save_alerted(already)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
