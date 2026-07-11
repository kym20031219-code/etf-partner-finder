"""텔레그램 알림 전송 (표준 라이브러리만 사용).

봇 토큰과 chat_id 는 코드에 넣지 말고 환경변수/Secrets 로 주입한다:
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from html import escape


class TelegramError(RuntimeError):
    pass


def _get_creds(token: str | None, chat_id: str | None) -> tuple[str, str]:
    token = token or os.environ.get("TELEGRAM_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise TelegramError(
            "TELEGRAM_TOKEN / TELEGRAM_CHAT_ID 가 없습니다. 환경변수 또는 Secrets 로 설정하세요."
        )
    return token, chat_id


def send_message(text: str, token: str | None = None, chat_id: str | None = None,
                 timeout: int = 15) -> dict:
    """텔레그램으로 HTML 메시지 전송."""
    token, chat_id = _get_creds(token, chat_id)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }
    ).encode()
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except Exception as e:  # noqa: BLE001
        raise TelegramError(f"텔레그램 전송 실패: {e}") from e
    if not body.get("ok"):
        raise TelegramError(f"텔레그램 API 오류: {body}")
    return body


def format_candidates(candidates: list[dict], names: dict[str, str] | None = None) -> str:
    """오늘의 후보 종목을 사람이 읽기 좋은 알림 텍스트로."""
    names = names or {}
    if not candidates:
        return "📭 오늘 눌림목 신규 신호 종목이 없습니다."
    date = candidates[0].get("date", "")
    lines = [f"📈 <b>눌림목 스윙 신규 신호</b>  ({escape(date)})", ""]
    for c in candidates:
        code = c["code"]
        name = escape(names.get(code, code))
        up = ((c["target"] / c["close"]) - 1) * 100
        dn = (1 - (c["stop"] / c["close"])) * 100
        lines.append(
            f"• <b>{name}</b> (<code>{escape(code)}</code>)\n"
            f"   종가 {c['close']:,} · RSI {c['rsi']}\n"
            f"   🎯 목표 {c['target']:,} (+{up:.0f}%)  🛑 손절 {c['stop']:,} (-{dn:.0f}%)"
        )
    lines.append("\n⚠️ 투자 참고용 신호일 뿐 매매 권유가 아닙니다.")
    return "\n".join(lines)
