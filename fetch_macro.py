#!/usr/bin/env python3
"""미국 거시경제 지표를 모아 대시보드용 JSON(results/macro.json)으로 저장한다.

목적(사용자 요구):
  미국 경제가 지금 어떤 상태인지 '종합적으로 판단'해 주는 대시보드용 데이터.
  - 매일: 10년물 국채금리, VIX(공포지수), 달러 인덱스, S&P 500
  - 매달: CPI(전년비 물가), 신규고용 NFP, 실업률, FOMC 정책금리
  - 분기: 실질 GDP 성장률, S&P 500 분기 흐름
  각 지표의 '현재 수치 + 직전 대비 변화 + 과거 시계열(1일~5년 그래프용)'을 담고,
  regime 엔진으로 '시장 날씨'와 '지금 지표가 어떤지 쉽게 풀어주는 설명'을 만든다.

데이터 출처:
  세인트루이스 연준의 FRED 공개 CSV (API 키 불필요).
  → GitHub Actions 러너(외부망 개방)에서 매일 자동 실행해 결과를 커밋한다.

⚠️ 지표를 '보여주고 해석'만 한다. 어떤 매수/매도 주문도 하지 않는다.
   설명·판단은 교육용 참고이며 투자 권유가 아니다.

사용:
  python fetch_macro.py            # 실데이터 → results/macro.json
  python fetch_macro.py --seed     # 오프라인 예시 데이터(과거 시계열도 합성 생성)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone, date, timedelta
from pathlib import Path

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={cosd}"
RESULTS_DIR = Path("results")
OUT_JSON = RESULTS_DIR / "macro.json"

# 지표 정의. cat=화면 구획, invert=값이 오르면 '나쁨'인지(색상),
# transform=원자료 가공(yoy=전년비, diff=전월증감), years=받아올 과거 연수.
INDICATORS = [
    dict(key="us10y", sid="DGS10", label="미국 10년물 국채금리", unit="%",
         cat="daily", invert=True, dec=2, years=5,
         sub="돈의 값. 오르면 대출·주식 밸류에이션 부담↑"),
    dict(key="vix", sid="VIXCLS", label="VIX 공포지수", unit="",
         cat="daily", invert=True, dec=2, years=5,
         sub="시장의 불안 온도. 20↑ 경계, 30↑ 공포"),
    dict(key="dxy", source="yahoo", symbols=["DX-Y.NYB", "DX=F"],
         label="달러 인덱스 (DXY)", unit="",
         cat="daily", invert=True, dec=2, years=5,
         sub="ICE 달러지수(DXY). 강달러=신흥국·원자재·수출주엔 역풍"),
    dict(key="sp500", sid="SP500", label="S&P 500", unit="",
         cat="daily", invert=False, dec=0, years=5,
         sub="미국 대표 500대 기업 주가지수"),
    dict(key="cpi", sid="CPIAUCSL", label="CPI 물가 (전년비)", unit="%",
         cat="monthly", invert=True, dec=2, years=11, transform="yoy",
         sub="연준 목표 2% 대비. 낮아질수록 금리인하 여지↑"),
    dict(key="nfp", sid="PAYEMS", label="신규고용 NFP", unit="천명",
         cat="monthly", invert=False, dec=0, years=11, transform="diff",
         sub="매달 늘어난 일자리 수. 경기 확장/위축의 척도"),
    dict(key="unrate", sid="UNRATE", label="실업률", unit="%",
         cat="monthly", invert=True, dec=1, years=11,
         sub="급등하면 경기침체 경고 신호"),
    dict(key="fedfunds", sid="DFEDTARU", label="FOMC 정책금리 (상단)", unit="%",
         cat="monthly", invert=True, dec=2, years=5,
         sub="연준이 정하는 기준금리"),
    dict(key="gdp", sid="A191RL1Q225SBEA", label="실질 GDP 성장률", unit="%",
         cat="quarterly", invert=False, dec=1, years=8,
         sub="연율 환산. 2회 연속 마이너스면 기술적 침체"),
    # S&P 500 분기 흐름은 sp500 시계열을 재사용(아래에서 처리)
]

FREQ_LABEL = {"daily": "어제보다", "monthly": "지난달보다", "quarterly": "지난분기보다"}


def http_get(url: str, timeout: int = 40) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (macro-fetch)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def fetch_yahoo(symbols: list[str], years: int) -> list[tuple[str, float]]:
    """Yahoo Finance 차트 API → [(YYYY-MM-DD, close)]. 심볼을 순서대로 시도.
    DXY(ICE 달러지수)는 FRED에 무료로 없어 여기서 받는다."""
    rng = "5y" if years <= 5 else "10y"
    last_err = None
    for sym in symbols:
        try:
            url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
                   f"{urllib.parse.quote(sym)}?range={rng}&interval=1d")
            j = json.loads(http_get(url))
            res = j["chart"]["result"][0]
            ts = res["timestamp"]
            closes = res["indicators"]["quote"][0]["close"]
            out = []
            for t, c in zip(ts, closes):
                if c is None:
                    continue
                d = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
                out.append((d, float(c)))
            if len(out) >= 2:
                return out
        except Exception as e:  # noqa: BLE001 — 다음 심볼로 폴백
            last_err = e
            continue
    raise RuntimeError(f"yahoo fetch 실패 {symbols}: {last_err}")


def fetch_series(sid: str, years: int) -> list[tuple[str, float]]:
    """FRED CSV → [(YYYY-MM-DD, value)] 오름차순. 결측('.') 제외."""
    cosd = (date.today() - timedelta(days=365 * years + 40)).isoformat()
    text = http_get(FRED_CSV.format(sid=sid, cosd=cosd))
    out: list[tuple[str, float]] = []
    for i, line in enumerate(text.splitlines()):
        if i == 0 or not line.strip():
            continue
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


def transform(kind: str | None, raw: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """원자료 → 표시용 시계열."""
    if kind == "yoy":  # 전년 동월 대비 % (CPI)
        out = []
        for i in range(12, len(raw)):
            base = raw[i - 12][1]
            if base:
                out.append((raw[i][0], (raw[i][1] - base) / abs(base) * 100.0))
        return out
    if kind == "diff":  # 전월 대비 증감 (NFP)
        return [(raw[i][0], raw[i][1] - raw[i - 1][1]) for i in range(1, len(raw))]
    return raw


def clip_years(series: list[tuple[str, float]], years: int) -> list[tuple[str, float]]:
    cutoff = (date.today() - timedelta(days=365 * years + 5)).isoformat()
    return [(d, v) for d, v in series if d >= cutoff]


def pct_change(cur, prev):
    if not prev or cur is None:
        return None
    return (cur - prev) / abs(prev) * 100.0


def fmt_date(iso: str, cat: str) -> str:
    if cat == "monthly":
        return iso[:7]
    if cat == "quarterly":
        y, m, _ = iso.split("-")
        return f"{y} Q{(int(m) - 1) // 3 + 1}"
    return iso


def build_indicator(meta: dict, series: list[tuple[str, float]]) -> dict:
    """현재값·직전 대비 변화·과거 시계열(그래프용)을 담은 카드 데이터."""
    if len(series) < 2:
        return {"available": False, "label": meta["label"]}
    dts = [d for d, _ in series]
    vals = [round(v, 4) for _, v in series]
    cur, prev = vals[-1], vals[-2]
    ma200 = round(sum(vals[-200:]) / min(200, len(vals)), 4)
    return {
        "available": True,
        "key": meta["key"],
        "label": meta["label"],
        "unit": meta["unit"],
        "cat": meta["cat"],
        "invert": meta["invert"],
        "dec": meta["dec"],
        "sub": meta["sub"],
        "value": round(cur, meta["dec"]),
        "prev": round(prev, meta["dec"]),
        "change": round(cur - prev, max(meta["dec"], 2)),
        "change_pct": round(pct_change(cur, prev), 2) if pct_change(cur, prev) is not None else None,
        "change_label": FREQ_LABEL[meta["cat"]],
        "date": fmt_date(dts[-1], meta["cat"]),
        "ma200": ma200,
        # 과거 그래프용 시계열(ISO 날짜, 프런트에서 1일/1주/1달/1년/5년 슬라이스)
        "series": {"d": dts, "v": vals},
    }


# ─────────────────────────── regime(시장 종합판단) 엔진 ───────────────────────────
def score_indicator(key: str, ind: dict) -> tuple[int, str, str]:
    """지표별 (점수 -2~+2, 짧은 판정, 지금 상태를 쉽게 푼 설명)."""
    v = ind.get("value")
    prev = ind.get("prev")
    if v is None:
        return 0, "-", ""

    if key == "vix":
        if v < 15: s, r = 2, "매우 안정"
        elif v < 20: s, r = 1, "안정"
        elif v < 27: s, r = -1, "경계"
        else: s, r = -2, "불안/공포"
        plain = (f"지금 시장 심리는 {'차분한' if v < 20 else '불안한'} 편이에요. "
                 f"공포지수(VIX)가 {v}로 " +
                 ("20 아래라 투자자들이 크게 겁먹지 않은 상태" if v < 20
                  else "20을 넘어 경계심이 커진 상태") + "입니다.")
        return s, r, plain
    if key == "us10y":
        ma = ind.get("ma200")
        gap = (v - ma) if ma else 0
        if gap > 0.4: s, r = -1, "상승 압력"
        elif gap < -0.4: s, r = 1, "하락(완화)"
        else: s, r = 0, "횡보"
        plain = (f"돈의 값인 10년물 금리는 {v}%예요. "
                 + ("최근 평균보다 높아 주식·대출에 부담을 주는 흐름" if gap > 0.4
                    else "최근 평균보다 낮아 위험자산에 우호적인 흐름" if gap < -0.4
                    else "큰 방향 없이 횡보하는 흐름") + "입니다.")
        return s, r, plain
    if key == "sp500":
        ma = ind.get("ma200")
        if ma and v > ma * 1.02: s, r = 2, "상승추세"
        elif ma and v > ma: s, r = 1, "추세 위"
        elif ma and v > ma * 0.98: s, r = -1, "추세 경계"
        else: s, r = -2, "하락추세"
        plain = ("미국 대표지수 S&P 500은 " +
                 ("장기 평균선 위에서 오르는 상승추세예요. 위험을 감수하기 좋은 국면입니다."
                  if ma and v > ma else
                  "장기 평균선 아래로 내려와 방어가 필요한 국면입니다."))
        return s, r, plain
    if key == "cpi":
        cooling = prev is not None and v < prev
        if v <= 2.5: s, r = 2, "안정"
        elif v <= 3.5: s, r = (1 if cooling else 0), ("둔화 중" if cooling else "다소 높음")
        elif v <= 5: s, r = (0 if cooling else -1), ("높지만 둔화" if cooling else "높음")
        else: s, r = -2, "과열"
        plain = (f"물가는 1년 전보다 {v}% 올랐어요(목표 2%). "
                 + (f"지난달({prev}%)보다 낮아져 진정되는 중" if cooling
                    else "아직 목표보다 높은 편") + "이라 "
                 + ("연준이 금리를 내릴 여지가 생기고 있어요." if v <= 3.0 or cooling
                    else "연준이 금리를 쉽게 못 내리는 상황이에요."))
        return s, r, plain
    if key == "unrate":
        rising = prev is not None and v > prev + 0.05
        if v < 4.3 and not rising: s, r = 1, "견조"
        elif rising: s, r = -1, "상승(둔화)"
        else: s, r = 0, "보통"
        plain = (f"실업률은 {v}%예요. "
                 + ("낮게 유지돼 고용이 탄탄한 편" if v < 4.3 and not rising
                    else "조금씩 오르고 있어 경기 둔화 신호를 살펴야 할 때"
                    if rising else "보통 수준") + "입니다.")
        return s, r, plain
    if key == "nfp":
        if v > 150: s, r = 1, "강한 고용"
        elif v > 0: s, r = 0, "완만한 고용"
        else: s, r = -2, "고용 감소"
        plain = (f"지난달 새 일자리는 {int(v):+,}천 개예요. "
                 + ("일자리가 꾸준히 늘며 경제가 확장 중" if v > 150
                    else "고용 증가세가 완만해진 상태" if v > 0
                    else "일자리가 줄어 경기 위축이 우려되는 상태") + "입니다.")
        return s, r, plain
    if key == "gdp":
        if v >= 2.5: s, r = 1, "견조한 성장"
        elif v >= 0: s, r = 0, "완만한 성장"
        else: s, r = -2, "역성장"
        plain = (f"경제 성장 속도(GDP)는 연 {v}%예요. "
                 + ("건강하게 확장 중" if v >= 2.5
                    else "성장하고 있지만 속도는 완만" if v >= 0
                    else "마이너스 성장이라 침체 신호") + "입니다.")
        return s, r, plain
    if key == "fedfunds":
        plain = f"연준 기준금리는 {v}%예요. 물가·고용을 보며 이 금리를 올리거나 내려 시장 전체에 영향을 줍니다."
        return 0, f"{v}%", plain
    if key == "dxy":
        plain = f"달러 가치(달러 인덱스)는 {v}예요. 달러가 강하면 신흥국·원자재·수출 기업에는 역풍이 됩니다."
        return 0, str(v), plain
    return 0, "-", ""


def build_regime(inds: dict) -> dict:
    signals = []
    # 종합점수에 넣을 핵심 지표 순서
    for key in ["vix", "us10y", "sp500", "cpi", "unrate", "nfp", "gdp"]:
        ind = inds.get(key)
        if not ind or not ind.get("available"):
            continue
        s, r, plain = score_indicator(key, ind)
        signals.append({
            "key": key, "name": ind["label"], "score": s,
            "reading": f"{ind['value']}{ind['unit']} · {r}",
            "value": ind["value"], "unit": ind["unit"],
            "change": ind["change"], "change_pct": ind["change_pct"],
            "change_label": ind["change_label"], "invert": ind["invert"],
            "plain": plain,
        })

    total = sum(x["score"] for x in signals)
    n = len(signals) or 1
    norm = round(total / (2 * n) * 100)

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

    # 지금 상태를 쉽게 풀어주는 한 문단(초보용)
    pos = [s["name"] for s in signals if s["score"] > 0]
    neg = [s["name"] for s in signals if s["score"] < 0]
    parts = [f"지금 미국 경제는 종합적으로 '{weather}'입니다."]
    if pos:
        parts.append("긍정적인 부분은 " + "·".join(pos[:3]) + " 쪽이고,")
    if neg:
        parts.append("조심할 부분은 " + "·".join(neg[:3]) + " 쪽이에요.")
    else:
        parts.append("특별히 위험한 신호는 두드러지지 않아요.")
    narrative = " ".join(parts)

    return {
        "score": norm, "weather": weather, "emoji": emoji,
        "summary": summary, "stance": stance,
        "narrative": narrative, "signals": signals,
    }


# ─────────────────────────── 실행/조립 ───────────────────────────
def assemble(inds: dict, source: str, errors: dict) -> dict:
    daily = {k: v for k, v in inds.items() if v.get("cat") == "daily"}
    monthly = {k: v for k, v in inds.items() if v.get("cat") == "monthly"}
    quarterly = {k: v for k, v in inds.items() if v.get("cat") == "quarterly"}
    # S&P 500 분기 흐름 카드(sp500 재사용, 분기 수익률 계산)
    sp = inds.get("sp500")
    if sp and sp.get("available"):
        vals = sp["series"]["v"]
        base = vals[-64] if len(vals) > 64 else vals[0]
        qret = pct_change(vals[-1], base)
        today = date.today()
        season = {1: "Q4(전년)", 2: "Q4(전년)", 4: "Q1", 5: "Q1",
                  7: "Q2", 8: "Q2", 10: "Q3", 11: "Q3"}.get(today.month)
        quarterly["sp500_q"] = {
            **{k: sp[k] for k in ("series", "invert", "dec")},
            "available": True, "key": "sp500_q", "cat": "quarterly",
            "label": "S&P 500 분기 흐름", "unit": "%",
            "value": round(qret, 1) if qret is not None else None,
            "change": None, "change_pct": None, "change_label": "지난분기보다",
            "date": sp["date"], "sub": f"최근 분기 지수 등락 · {'지금 '+season+' 실적시즌' if season else '실적 비수기'}",
            "level": sp["value"],
        }
    regime = build_regime(inds)
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "errors": errors,
        "daily": daily, "monthly": monthly, "quarterly": quarterly,
        "regime": regime,
    }


def gather_live() -> dict:
    inds, errors = {}, {}
    for meta in INDICATORS:
        try:
            if meta.get("source") == "yahoo":
                raw = fetch_yahoo(meta["symbols"], meta["years"])
            else:
                raw = fetch_series(meta["sid"], meta["years"])
            ser = clip_years(transform(meta.get("transform"), raw), meta["years"])
            inds[meta["key"]] = build_indicator(meta, ser)
            print(f"  ✓ {meta['key']:9s} {meta['sid']:16s} {len(ser)}pts")
        except Exception as e:
            errors[meta["key"]] = f"{type(e).__name__}: {e}"
            inds[meta["key"]] = {"available": False, "label": meta["label"], "cat": meta["cat"]}
            print(f"  ✗ {meta['key']:9s} 실패 — {errors[meta['key']]}", file=sys.stderr)
    return assemble(inds, "fred", errors)


def gather_seed() -> dict:
    """오프라인 예시 데이터. 과거 시계열도 합성 생성(그래프 확인용)."""
    import random
    random.seed(7)
    today = date.today()
    inds = {}

    def daily_series(years, start, drift, vol, floor=None):
        n = int(252 * years)
        out, v = [], start
        for i in range(n):
            d = today - timedelta(days=int((n - i) * 365 / 252))
            v = v + drift + random.uniform(-vol, vol)
            if floor is not None:
                v = max(floor, v)
            out.append((d.isoformat(), round(v, 4)))
        return out

    def monthly_series(months, start, drift, vol, floor=None):
        out, v = [], start
        for i in range(months):
            d = (today.replace(day=1) - timedelta(days=30 * (months - i)))
            v = v + drift + random.uniform(-vol, vol)
            if floor is not None:
                v = max(floor, v)
            out.append((d.isoformat(), round(v, 4)))
        return out

    specs = {
        "us10y": daily_series(5, 2.5, 0.0016, 0.05, 0.5),
        "vix": daily_series(5, 22, -0.002, 1.2, 9),
        "dxy": daily_series(5, 96, 0.003, 0.25),
        "sp500": daily_series(5, 4200, 2.4, 30),
        "cpi": monthly_series(60, 5.5, -0.03, 0.15, 0.5),
        "nfp": monthly_series(60, 180, -1, 60),
        "unrate": monthly_series(60, 3.6, 0.008, 0.08, 3.4),
        "fedfunds": daily_series(5, 1.0, 0.0025, 0.0, 0.25),
        "gdp": [(d, round(v, 1)) for d, v in monthly_series(20, 2.5, 0.0, 1.4)],
    }
    for meta in INDICATORS:
        inds[meta["key"]] = build_indicator(meta, specs[meta["key"]])
    return assemble(inds, "seed", {})


def main() -> int:
    ap = argparse.ArgumentParser(description="미국 거시지표 → results/macro.json")
    ap.add_argument("--seed", action="store_true", help="오프라인 예시 데이터 생성")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    if args.seed:
        print("예시(seed) 데이터 생성 중…")
        payload = gather_seed()
    else:
        print("FRED에서 실데이터 수집 중…")
        payload = gather_live()
        if all(not v.get("available") for v in payload["daily"].values()):
            print("모든 일간 지표 수집 실패 — 네트워크/정책 확인 필요", file=sys.stderr)
            return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    r = payload["regime"]
    print(f"저장 완료 → {out}  ({len(json.dumps(payload))//1024} KB)")
    print(f"시장 날씨: {r['emoji']} {r['weather']} (점수 {r['score']:+d}) · 신호 {len(r['signals'])}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
