#!/usr/bin/env python3
"""미국 거시경제 지표를 모아 대시보드용 JSON(results/macro.json)으로 저장한다.

목적(사용자 요구):
  - 매달: 미국 CPI(물가), 고용지표(NFP·실업률), FOMC(연준 정책금리)
  - 분기: S&P 500 실적 흐름, GDP 성장률
  - 매일: 미국 10년물 국채금리, VIX(공포지수), 달러 인덱스
  위 지표를 한 화면에서 보고, "시장이 왜 움직이는지"를 체계적으로 판단하도록
  각 지표를 해석(regime 엔진)해 초보 투자자용 가이드까지 함께 만든다.

데이터 출처:
  세인트루이스 연준의 FRED 공개 CSV 엔드포인트
  (https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES> ). API 키 불필요.
  → GitHub Actions 러너(외부망 개방)에서 매일 실행해 결과를 커밋한다.

동작:
  1. 각 지표 시계열을 내려받아(실패해도 다른 지표에 영향 없음)
  2. 최신값·직전값·변화·간단한 히스토리(스파크라인용)를 뽑고
  3. regime 엔진으로 "시장 날씨"와 초보용 가이드를 계산해
  4. results/macro.json 으로 저장한다.

오프라인/초기 seed:
  네트워크가 막힌 환경(이 저장소 개발 세션 등)에서는 --seed 로 예시 데이터를
  만들어 페이지가 비지 않게 한다. seed 데이터는 source="seed"로 표시되고
  대시보드에 '예시(초기) 데이터' 배너가 뜬다. 최초 Actions 실행이 실제값으로 덮어쓴다.

⚠️ 이 스크립트는 지표를 '보여주고 해석'만 한다. 어떤 매수/매도 주문도 하지 않는다.
   가이드는 교육용이며 투자 권유가 아니다. 매매는 본인 판단·책임이다.

사용:
  python fetch_macro.py                 # 실데이터 → results/macro.json
  python fetch_macro.py --seed          # 오프라인 예시 데이터
  python fetch_macro.py --out out.json  # 저장 경로 지정
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone, date
from pathlib import Path

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={cosd}"
RESULTS_DIR = Path("results")
OUT_JSON = RESULTS_DIR / "macro.json"

# 대시보드가 쓰는 FRED 시리즈 ID
SERIES = {
    "us10y": "DGS10",       # 미국 10년물 국채금리 (일간, %)
    "vix": "VIXCLS",        # CBOE 변동성지수 VIX (일간)
    "dxy": "DTWEXBGS",      # 광의 달러 인덱스 (일간, 2006=100)
    "sp500": "SP500",       # S&P 500 지수 (일간, 약 10년치)
    "cpi": "CPIAUCSL",      # 소비자물가지수 CPI (월간, 계절조정)
    "nfp": "PAYEMS",        # 비농업부문 고용자수 (월간, 천명)
    "unrate": "UNRATE",     # 실업률 (월간, %)
    "fedfunds": "DFEDTARU", # 연방기금 목표금리 상단 (FOMC가 정함, %)
    "gdp": "A191RL1Q225SBEA",  # 실질 GDP 전분기比 연율 (분기, %)
}


def http_get(url: str, timeout: int = 40) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (macro-fetch)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def fetch_series(sid: str, cosd: str = "2014-01-01") -> list[tuple[str, float]]:
    """FRED CSV → [(YYYY-MM-DD, value)] (결측 '.' 제외, 오름차순)."""
    text = http_get(FRED_CSV.format(sid=sid, cosd=cosd))
    out: list[tuple[str, float]] = []
    for i, line in enumerate(text.splitlines()):
        if i == 0 or not line.strip():
            continue  # 헤더(observation_date,SID) 건너뜀
        parts = line.split(",")
        if len(parts) < 2:
            continue
        d, v = parts[0].strip(), parts[1].strip()
        if v in (".", "", "NA"):
            continue
        try:
            out.append((d, float(v)))
        except ValueError:
            continue
    return out


def pct_change(cur: float, prev: float) -> float | None:
    if prev in (0, None) or cur is None:
        return None
    return (cur - prev) / abs(prev) * 100.0


def spark(series: list[tuple[str, float]], n: int) -> list[float]:
    """스파크라인용 최근 n개 값(숫자 배열)."""
    return [round(v, 4) for _, v in series[-n:]]


def build_daily(key: str, series: list[tuple[str, float]], spark_n: int) -> dict:
    """일간 지표: 최신값·전일比 변화·최근 히스토리."""
    if not series:
        return {"available": False}
    d, v = series[-1]
    prev = series[-2][1] if len(series) > 1 else None
    return {
        "available": True,
        "value": round(v, 3),
        "date": d,
        "change": round(v - prev, 3) if prev is not None else None,
        "change_pct": round(pct_change(v, prev), 2) if prev is not None else None,
        "spark": spark(series, spark_n),
        # 200영업일 평균(장기추세 판단용) — 있으면
        "ma200": round(sum(x for _, x in series[-200:]) / min(200, len(series)), 3),
    }


def build_cpi(series: list[tuple[str, float]]) -> dict:
    """CPI: 전년동월比(YoY) 물가상승률과 직전월 YoY."""
    if len(series) < 13:
        return {"available": False}
    def yoy_at(i: int) -> float | None:
        if i - 12 < 0:
            return None
        return pct_change(series[i][1], series[i - 12][1])
    last = len(series) - 1
    cur = yoy_at(last)
    prev = yoy_at(last - 1)
    hist = [round(yoy_at(i), 2) for i in range(max(12, last - 23), last + 1) if yoy_at(i) is not None]
    return {
        "available": cur is not None,
        "value": round(cur, 2) if cur is not None else None,
        "prev": round(prev, 2) if prev is not None else None,
        "date": series[last][0][:7],
        "spark": hist,
        "target": 2.0,  # 연준 물가 목표
    }


def build_nfp(series: list[tuple[str, float]]) -> dict:
    """NFP: 비농업 고용자수의 전월比 증감(천명)."""
    if len(series) < 2:
        return {"available": False}
    diffs = [(series[i][0], series[i][1] - series[i - 1][1]) for i in range(1, len(series))]
    d, chg = diffs[-1]
    prev = diffs[-2][1] if len(diffs) > 1 else None
    return {
        "available": True,
        "value": round(chg, 0),           # 이번 달 신규고용(천명)
        "prev": round(prev, 0) if prev is not None else None,
        "date": d[:7],
        "spark": [round(c, 0) for _, c in diffs[-12:]],
    }


def build_monthly(series: list[tuple[str, float]], spark_n: int = 12, digits: int = 2) -> dict:
    """실업률·정책금리 등 월간 레벨 지표."""
    if not series:
        return {"available": False}
    d, v = series[-1]
    prev = series[-2][1] if len(series) > 1 else None
    return {
        "available": True,
        "value": round(v, digits),
        "prev": round(prev, digits) if prev is not None else None,
        "date": d[:7],
        "spark": spark(series, spark_n),
    }


def build_fedfunds(series: list[tuple[str, float]]) -> dict:
    """연방기금 목표금리 상단(일간)의 최신값과 최근 변경."""
    if not series:
        return {"available": False}
    d, v = series[-1]
    # 마지막으로 값이 바뀐 시점(=최근 FOMC 인상/인하) 찾기
    prev_level = None
    prev_date = None
    for i in range(len(series) - 2, -1, -1):
        if abs(series[i][1] - v) > 1e-9:
            prev_level = series[i][1]
            prev_date = series[i + 1][0]  # 새 금리가 적용된 첫날
            break
    return {
        "available": True,
        "value": round(v, 2),
        "date": d,
        "last_change_date": prev_date,
        "last_change": round(v - prev_level, 2) if prev_level is not None else None,
        "spark": spark(series, 24),
    }


def build_gdp(series: list[tuple[str, float]]) -> dict:
    """실질 GDP 전분기比 연율(%) 최신값."""
    if not series:
        return {"available": False}
    d, v = series[-1]
    prev = series[-2][1] if len(series) > 1 else None
    # 분기 라벨(예: 2026 Q2)
    y, m, _ = d.split("-")
    q = (int(m) - 1) // 3 + 1
    return {
        "available": True,
        "value": round(v, 1),
        "prev": round(prev, 1) if prev is not None else None,
        "date": f"{y} Q{q}",
        "spark": [round(x, 1) for _, x in series[-8:]],
    }


def build_sp500_quarterly(series: list[tuple[str, float]]) -> dict:
    """S&P 500 '실적 시즌' 요약: 분기 수익률 + 현재 실적발표 시즌 여부.
    (무료·무키 소스에는 개별 기업 EPS가 없어, 지수 흐름과 실적시즌 캘린더로 대체)."""
    if not series:
        return {"available": False}
    d, v = series[-1]
    # 대략 63영업일(약 1분기) 전 대비
    base = series[-64][1] if len(series) > 64 else series[0][1]
    qret = pct_change(v, base)
    today = date.fromisoformat(d)
    # 실적 시즌: 각 분기 결과를 다음 달부터 발표 (1·4·7·10월 시작)
    season_map = {1: "Q4 (전년)", 2: "Q4 (전년)", 4: "Q1", 5: "Q1",
                  7: "Q2", 8: "Q2", 10: "Q3", 11: "Q3"}
    season = season_map.get(today.month)
    return {
        "available": True,
        "level": round(v, 1),
        "date": d,
        "quarter_return": round(qret, 1) if qret is not None else None,
        "earnings_season": season,           # 지금 발표 중인 분기 실적(없으면 null)
        "spark": spark(series, 63),
    }


# ─────────────────────────── regime(시장 날씨) 엔진 ───────────────────────────
def build_regime(daily: dict, monthly: dict, quarterly: dict) -> dict:
    """각 지표를 점수화해 '시장 날씨'와 초보용 해석·가이드를 만든다.

    점수: -2(매우 부정) ~ +2(매우 긍정). 합산 후 5단계 날씨로 매핑.
    ⚠️ 규칙기반 참고 지표일 뿐 예측·권유가 아니다.
    """
    signals: list[dict] = []

    def add(name, score, reading, why):
        signals.append({"name": name, "score": score, "reading": reading, "why": why})

    # VIX (공포지수): 낮을수록 안정
    vix = daily.get("vix", {})
    if vix.get("available"):
        v = vix["value"]
        if v < 15: s, r = 2, "매우 안정"
        elif v < 20: s, r = 1, "안정"
        elif v < 27: s, r = -1, "경계"
        elif v < 35: s, r = -2, "불안"
        else: s, r = -2, "공포"
        add("VIX 변동성", s, f"{v} ({r})",
            "VIX가 낮으면 시장이 편안하다는 뜻(위험자산 우호). 20을 넘으면 불안, 30 이상은 공포 국면.")

    # 10년물 금리 방향: 급등은 주식·성장주에 부담
    us10y = daily.get("us10y", {})
    if us10y.get("available"):
        v, ma = us10y["value"], us10y.get("ma200")
        if ma:
            gap = v - ma
            if gap > 0.4: s, r = -1, "상승 압력"
            elif gap < -0.4: s, r = 1, "하락(완화)"
            else: s, r = 0, "횡보"
        else:
            s, r = 0, "중립"
        add("10년물 금리", s, f"{v}% ({r})",
            "금리가 오르면 대출·밸류에이션 부담↑(특히 성장주·부동산). 내리면 위험자산에 우호적.")

    # S&P 500 추세: 200일선 위/아래
    sp = daily.get("sp500", {})
    if sp.get("available"):
        v, ma = sp["value"], sp.get("ma200")
        if ma:
            if v > ma * 1.02: s, r = 2, "상승추세"
            elif v > ma: s, r = 1, "추세 위"
            elif v > ma * 0.98: s, r = -1, "추세 이탈 경계"
            else: s, r = -2, "하락추세"
            add("S&P 500 추세", s, f"{r}",
                "지수가 200일 평균선 위에 있으면 장기 상승추세(위험 감수 우호), 아래면 방어적으로.")

    # CPI: 목표(2%)에 가까울수록·내려갈수록 좋음
    cpi = monthly.get("cpi", {})
    if cpi.get("available"):
        v, p = cpi["value"], cpi.get("prev")
        cooling = (p is not None and v < p)
        if v <= 2.5: s, r = 2, "안정"
        elif v <= 3.5: s, r = (1 if cooling else 0), ("둔화 중" if cooling else "다소 높음")
        elif v <= 5: s, r = (0 if cooling else -1), ("높지만 둔화" if cooling else "높음")
        else: s, r = -2, "과열"
        add("CPI 물가", s, f"{v}% YoY ({r})",
            "물가가 높으면 연준이 금리를 높게 유지→위험자산 부담. 2%로 낮아지면 금리인하 여지↑.")

    # 실업률: 낮고 안정적이면 좋음, 급등은 침체 신호
    un = monthly.get("unrate", {})
    if un.get("available"):
        v, p = un["value"], un.get("prev")
        rising = (p is not None and v > p + 0.05)
        if v < 4.3 and not rising: s, r = 1, "견조"
        elif rising: s, r = -1, "상승(둔화)"
        else: s, r = 0, "보통"
        add("실업률", s, f"{v}% ({r})",
            "고용이 탄탄하면 소비·기업이익에 우호적. 실업률이 빠르게 오르면 경기침체 경고.")

    # NFP: 신규고용 강도
    nfp = monthly.get("nfp", {})
    if nfp.get("available"):
        v = nfp["value"]
        if v is None: pass
        elif v > 150: s, r = 1, "강한 고용"
        elif v > 0: s, r = 0, "완만한 고용"
        else: s, r = -2, "고용 감소"
        if v is not None:
            add("신규고용(NFP)", s, f"{int(v):+,}천명 ({r})",
                "매달 새로 늘어난 일자리 수. 꾸준히 늘면 경제 확장, 마이너스면 경기 위축 신호.")

    # GDP 성장률
    gdp = quarterly.get("gdp", {})
    if gdp.get("available"):
        v = gdp["value"]
        if v >= 2.5: s, r = 1, "견조한 성장"
        elif v >= 0: s, r = 0, "완만한 성장"
        else: s, r = -2, "역성장"
        add("GDP 성장률", s, f"{v}% (연율, {r})",
            "경제 규모의 성장 속도. 플러스면 확장, 2회 연속 마이너스면 기술적 침체.")

    total = sum(s["score"] for s in signals)
    n = len(signals)
    # 정규화(-100~+100)
    norm = round(total / (2 * n) * 100) if n else 0

    if norm >= 45:
        weather, emoji = "맑음", "☀️"
        summary = "위험자산에 우호적인 환경입니다. 지표 대부분이 안정·확장을 가리킵니다."
        stance = "공격적(성장 자산 비중을 평소보다 조금 높여도 되는 국면)"
    elif norm >= 15:
        weather, emoji = "구름 조금", "🌤️"
        summary = "대체로 양호하나 일부 경계 신호가 섞여 있습니다. 균형이 필요합니다."
        stance = "중립~약간 공격적(분산 유지, 무리한 집중은 자제)"
    elif norm > -15:
        weather, emoji = "흐림", "⛅"
        summary = "긍정·부정 신호가 팽팽합니다. 방향이 잡힐 때까지 방어와 공격을 반반으로."
        stance = "중립(분산·현금 비중 유지, 큰 베팅 금지)"
    elif norm > -45:
        weather, emoji = "비", "🌧️"
        summary = "부정적 신호가 우세합니다. 변동성 확대에 대비해 방어적으로."
        stance = "방어적(안전자산·현금 비중↑, 위험자산은 분할·소액)"
    else:
        weather, emoji = "폭풍", "⛈️"
        summary = "위험 회피 국면입니다. 자본 보전이 최우선입니다."
        stance = "매우 방어적(현금·단기채 위주, 반등 확인 전 신규 위험자산 자제)"

    # 초보용 예시 배분(교육용 · 권유 아님) — 날씨에 따른 '위험자산 비중' 예시 밴드
    if norm >= 45:   alloc = {"위험자산(주식 등)": "60~70%", "안전자산(채권·현금)": "30~40%"}
    elif norm >= 15: alloc = {"위험자산(주식 등)": "50~60%", "안전자산(채권·현금)": "40~50%"}
    elif norm > -15: alloc = {"위험자산(주식 등)": "40~50%", "안전자산(채권·현금)": "50~60%"}
    elif norm > -45: alloc = {"위험자산(주식 등)": "25~40%", "안전자산(채권·현금)": "60~75%"}
    else:            alloc = {"위험자산(주식 등)": "10~25%", "안전자산(채권·현금)": "75~90%"}

    return {
        "score": norm,
        "weather": weather,
        "emoji": emoji,
        "summary": summary,
        "stance": stance,
        "example_allocation": alloc,
        "signals": signals,
    }


# ─────────────────────────── 실행/조립 ───────────────────────────
def gather_live() -> dict:
    """FRED에서 실데이터를 모아 대시보드 payload를 만든다."""
    raw: dict[str, list] = {}
    errors: dict[str, str] = {}
    for key, sid in SERIES.items():
        # 일간 지표는 최근 2년, 월/분기는 더 길게
        cosd = "2018-01-01" if key in ("cpi", "nfp", "unrate", "gdp", "fedfunds") else "2023-01-01"
        try:
            raw[key] = fetch_series(sid, cosd)
            print(f"  ✓ {key:9s} {sid:16s} {len(raw[key])}개")
        except Exception as e:
            errors[key] = f"{type(e).__name__}: {e}"
            raw[key] = []
            print(f"  ✗ {key:9s} {sid:16s} 실패 — {errors[key]}", file=sys.stderr)

    daily = {
        "us10y": build_daily("us10y", raw["us10y"], 120),
        "vix": build_daily("vix", raw["vix"], 120),
        "dxy": build_daily("dxy", raw["dxy"], 120),
        "sp500": build_daily("sp500", raw["sp500"], 120),
    }
    monthly = {
        "cpi": build_cpi(raw["cpi"]),
        "nfp": build_nfp(raw["nfp"]),
        "unrate": build_monthly(raw["unrate"], 18, 1),
        "fedfunds": build_fedfunds(raw["fedfunds"]),
    }
    quarterly = {
        "gdp": build_gdp(raw["gdp"]),
        "sp500_earnings": build_sp500_quarterly(raw["sp500"]),
    }
    regime = build_regime(daily, monthly, quarterly)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "fred",
        "errors": errors,
        "daily": daily,
        "monthly": monthly,
        "quarterly": quarterly,
        "regime": regime,
    }


def gather_seed() -> dict:
    """오프라인용 예시 데이터. source='seed' 로 표시되어 UI에 경고 배너가 뜬다.
    최초 Actions 실행이 실제값으로 덮어쓴다. (수치는 예시일 뿐 실제값 아님)"""
    def sp(base, n, step):  # 단순한 예시 스파크라인
        return [round(base + i * step, 2) for i in range(n)]

    daily = {
        "us10y": {"available": True, "value": 4.21, "date": "2026-07-29", "change": -0.02,
                  "change_pct": -0.47, "ma200": 4.35, "spark": sp(4.5, 20, -0.015)},
        "vix": {"available": True, "value": 16.2, "date": "2026-07-29", "change": -0.4,
                "change_pct": -2.4, "ma200": 17.5, "spark": sp(18, 20, -0.09)},
        "dxy": {"available": True, "value": 121.3, "date": "2026-07-29", "change": 0.1,
                "change_pct": 0.08, "ma200": 122.0, "spark": sp(123, 20, -0.08)},
        "sp500": {"available": True, "value": 6380.0, "date": "2026-07-29", "change": 22.0,
                  "change_pct": 0.35, "ma200": 6050.0, "spark": sp(6100, 20, 14)},
    }
    monthly = {
        "cpi": {"available": True, "value": 2.9, "prev": 3.1, "date": "2026-06",
                "target": 2.0, "spark": [3.5, 3.4, 3.3, 3.2, 3.1, 2.9]},
        "nfp": {"available": True, "value": 165, "prev": 190, "date": "2026-06",
                "spark": [210, 185, 170, 190, 190, 165]},
        "unrate": {"available": True, "value": 4.1, "prev": 4.1, "date": "2026-06",
                   "spark": [3.9, 4.0, 4.0, 4.1, 4.1, 4.1]},
        "fedfunds": {"available": True, "value": 4.5, "date": "2026-07-29",
                     "last_change_date": "2026-06-18", "last_change": -0.25,
                     "spark": [5.5, 5.5, 5.25, 5.0, 4.75, 4.5]},
    }
    quarterly = {
        "gdp": {"available": True, "value": 2.3, "prev": 2.8, "date": "2026 Q1",
                "spark": [2.1, 3.4, 2.8, 2.3]},
        "sp500_earnings": {"available": True, "level": 6380.0, "date": "2026-07-29",
                           "quarter_return": 5.4, "earnings_season": "Q2",
                           "spark": sp(6050, 20, 16)},
    }
    regime = build_regime(daily, monthly, quarterly)
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "seed",
        "errors": {},
        "daily": daily,
        "monthly": monthly,
        "quarterly": quarterly,
        "regime": regime,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="미국 거시지표 → results/macro.json")
    ap.add_argument("--seed", action="store_true", help="오프라인 예시 데이터 생성")
    ap.add_argument("--out", default=str(OUT_JSON), help="저장 경로")
    args = ap.parse_args()

    if args.seed:
        print("예시(seed) 데이터 생성 중…")
        payload = gather_seed()
    else:
        print("FRED에서 실데이터 수집 중…")
        payload = gather_live()
        # 실행 결과가 전부 실패했으면(네트워크 차단 등) 종료코드로 알림
        if all(not payload["daily"][k].get("available") for k in payload["daily"]):
            print("모든 일간 지표 수집 실패 — 네트워크/정책 확인 필요", file=sys.stderr)
            # seed가 이미 있으면 덮어쓰지 않도록 실패 처리
            return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    r = payload["regime"]
    print(f"저장 완료 → {out}")
    print(f"시장 날씨: {r['emoji']} {r['weather']} (점수 {r['score']:+d}) · 신호 {len(r['signals'])}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
