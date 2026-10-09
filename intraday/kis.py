"""한국투자증권(KIS) OpenAPI 클라이언트 — 토큰 발급 + 당일 분봉 조회.

표준 라이브러리(urllib)만 사용한다(HTTPS_PROXY 자동 반영). 앱키/시크릿은 반드시
환경변수로 주입한다(코드에 넣지 말 것):

  export KIS_APP_KEY="..."          # KIS 개발자센터에서 발급
  export KIS_APP_SECRET="..."
  export KIS_ENV="real"             # real(실전) 또는 mock(모의투자)

⚠️ 이 클라이언트는 KIS 공식 문서(https://apiportal.koreainvestment.com)의 엔드포인트·
tr_id 규격에 맞춰 작성했으나, 저장소를 만든 원격 세션에서는 KIS 접속이 차단돼 **실호출을
검증하지 못했다.** 회원님 키로 처음 실행할 때 응답 필드명을 한 번 확인하길 권한다.
"""
from __future__ import annotations

import json
import os
import time as _time
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

DOMAINS = {
    "real": "https://openapi.koreainvestment.com:9443",
    "mock": "https://openapivts.koreainvestment.com:29443",
}
TOKEN_CACHE = Path("state/kis_token.json")


class KisError(RuntimeError):
    pass


class KisClient:
    def __init__(self, app_key: str | None = None, app_secret: str | None = None,
                 env: str | None = None):
        self.app_key = app_key or os.environ.get("KIS_APP_KEY", "")
        self.app_secret = app_secret or os.environ.get("KIS_APP_SECRET", "")
        self.env = (env or os.environ.get("KIS_ENV", "real")).lower()
        self.domain = os.environ.get("KIS_DOMAIN") or DOMAINS.get(self.env, DOMAINS["real"])
        if not (self.app_key and self.app_secret):
            raise KisError("KIS_APP_KEY / KIS_APP_SECRET 환경변수가 필요합니다.")
        self._token: str | None = None

    # ------------------------------------------------------------------ token
    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.domain + path,
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    def token(self) -> str:
        """접근토큰(약 24시간 유효). 파일 캐시로 재사용(발급은 분당 1회 제한)."""
        if self._token:
            return self._token
        if TOKEN_CACHE.exists():
            try:
                c = json.loads(TOKEN_CACHE.read_text())
                if c.get("env") == self.env and c.get("expires_at", 0) > _time.time() + 60:
                    self._token = c["access_token"]
                    return self._token
            except Exception:  # noqa: BLE001
                pass
        data = self._post("/oauth2/tokenP", {
            "grant_type": "client_credentials",
            "appkey": self.app_key, "appsecret": self.app_secret,
        })
        tok = data.get("access_token")
        if not tok:
            raise KisError(f"토큰 발급 실패: {data}")
        TOKEN_CACHE.parent.mkdir(exist_ok=True)
        TOKEN_CACHE.write_text(json.dumps({
            "env": self.env, "access_token": tok,
            "expires_at": _time.time() + int(data.get("expires_in", 86400)),
        }))
        self._token = tok
        return tok

    # ----------------------------------------------------------------- quotes
    def _get(self, path: str, tr_id: str, params: dict) -> dict:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        req = urllib.request.Request(
            f"{self.domain}{path}?{qs}",
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.token()}",
                "appkey": self.app_key, "appsecret": self.app_secret,
                "tr_id": tr_id, "custtype": "P",
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    def minute_bars(self, code: str, to_hhmmss: str = "153000",
                    pages: int = 14, pause: float = 0.2) -> pd.DataFrame:
        """당일 1분봉을 조립해 반환 (DatetimeIndex, Open/High/Low/Close/Volume).

        KIS 는 한 번에 기준시각 이전 30건만 주므로, 가장 이른 시각을 다음 기준으로
        삼아 09:00 까지 거슬러 페이징한다.
        tr_id: FHKST03010200 (주식당일분봉조회).
        """
        rows: dict[str, dict] = {}
        cursor = to_hhmmss
        for _ in range(pages):
            data = self._get(
                "/uapi/domestic-stock/v1/quotes/inquire-time-itemchartprice",
                "FHKST03010200",
                {
                    "FID_ETC_CLS_CODE": "", "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": code, "FID_INPUT_HOUR_1": cursor,
                    "FID_PW_DATA_INCU_YN": "Y",
                },
            )
            out = data.get("output2") or []
            if not out:
                break
            for b in out:
                hhmmss = b.get("stck_cntg_hour")
                if not hhmmss:
                    continue
                rows[hhmmss] = b
            earliest = min(b["stck_cntg_hour"] for b in out)
            if earliest <= "090000" or earliest >= cursor:
                break
            cursor = earliest
            _time.sleep(pause)   # 초당 호출제한 보호

        if not rows:
            raise KisError(f"{code} 분봉 응답이 비었습니다 (장 시간/코드 확인).")

        today = datetime.now().strftime("%Y-%m-%d")
        recs = []
        for hhmmss, b in sorted(rows.items()):
            ts = pd.Timestamp(f"{today} {hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}")
            recs.append((ts, float(b["stck_oprc"]), float(b["stck_hgpr"]),
                         float(b["stck_lwpr"]), float(b["stck_prpr"]), float(b["cntg_vol"])))
        df = pd.DataFrame(recs, columns=["ts", "Open", "High", "Low", "Close", "Volume"]).set_index("ts")
        return df
