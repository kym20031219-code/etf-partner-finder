"""코스피 **단기 전망** 점수 엔진 (6팩터 프레임워크).

사용자 분석 프레임을 그대로 코드화한다:

  1. 거시 (Macro)       : 미 증시(위험선호), VIX, 미 국채금리, 달러/원 환율
  2. 한국경제 (Korea)   : 원화 흐름, 중국 증시, 코스닥 위험선호, 반도체 상대강도
  3. 실적 (Earnings)    : 삼성전자·SK하이닉스 실적사이클 프록시 + 코스피 PER 방향
  4. 수급 (Flows)       : 외국인·기관 순매수(대금) 방향/강도
  5. 밸류에이션 (Value) : 코스피 PER·PBR 역사적 백분위, ERP(주식위험프리미엄)
  6. 기술 (Technical)   : 추세(정배열)·RSI·MACD·모멘텀·거래량·52주 위치

각 팩터는 0~100 점(50 = 중립)으로 환산되고, 가중 평균이 **종합 점수**가 된다.
종합 점수를 다시 방향(강세/중립-강세/중립/중립-약세/약세)·신뢰도로 매핑한다.

설계 원칙
  - **투명성**: 모든 하위 신호가 원시값과 함께 점수로 남는다(대시보드에서 근거 확인).
  - **견고성**: 입력이 없으면(오프라인·조회 실패) 해당 신호를 중립 50으로 두고
    `available=False` 로 표기 → 파이프라인이 죽지 않는다.
  - 반도체(삼성전자+SK하이닉스) 비중이 코스피에 절대적이므로 실적·한국경제 팩터에서
    반도체 상대강도를 핵심 변수로 함께 본다.

⚠️ 이 엔진은 **참고용 전망 점수**만 계산한다. 매매 권유가 아니며, 어떤 자동
   매수/매도 주문 기능도 없다. 전망은 불확실하며 빗나갈 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategy import rsi, sma

# ---------------------------------------------------------------------------
# 점수 헬퍼 (모두 0~100, 50 = 중립)
# ---------------------------------------------------------------------------
def _lin(v: float, at0: float, at100: float) -> float:
    """v 를 [at0→0, at100→100] 로 선형 매핑(클램프). at0>at100 이면 반전."""
    if v != v or at0 == at100:  # NaN 방어
        return 50.0
    t = (v - at0) / (at100 - at0)
    return 100.0 * max(0.0, min(1.0, t))


def _centered(x: float, span: float) -> float:
    """x=0 → 50, x=+span → 100, x=-span → 0 (변화율/기울기 신호용)."""
    if x != x or span <= 0:
        return 50.0
    return 50.0 + 50.0 * max(-1.0, min(1.0, x / span))


def _pct_rank(series: pd.Series, value: float) -> float:
    """value 가 series(과거값) 안에서 차지하는 백분위(0~100)."""
    s = pd.Series(series).dropna()
    if s.empty or value != value:
        return 50.0
    return float((s <= value).mean() * 100.0)


def _last(series: pd.Series | None) -> float:
    if series is None:
        return float("nan")
    s = pd.Series(series).dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def _roc(series: pd.Series | None, lb: int) -> float:
    """lb 거래일 전 대비 변화율(소수). 데이터 부족 시 NaN."""
    if series is None:
        return float("nan")
    s = pd.Series(series).dropna()
    if len(s) <= lb:
        return float("nan")
    prev = float(s.iloc[-1 - lb])
    return float(s.iloc[-1]) / prev - 1.0 if prev else float("nan")


# ---------------------------------------------------------------------------
# 입력 번들 / 가중치
# ---------------------------------------------------------------------------
@dataclass
class MarketBundle:
    """전망 계산에 쓰는 시계열 묶음. 필수는 코스피 OHLCV 뿐, 나머지는 선택."""

    kospi: pd.DataFrame                       # Open/High/Low/Close/Volume (필수)
    sp500: pd.Series | None = None            # 미 S&P500 종가 (위험선호)
    vix: pd.Series | None = None              # VIX 지수 (공포)
    us10y: pd.Series | None = None            # 미 10년 국채 금리(%) 종가
    usdkrw: pd.Series | None = None           # 달러/원 환율
    china: pd.Series | None = None            # 상하이종합 등 중국 증시
    kosdaq: pd.Series | None = None           # 코스닥 종가
    semis: pd.Series | None = None            # 반도체 대표주(삼성+하이닉스) 합성지수 종가
    per: pd.Series | None = None              # 코스피 PER 시계열
    pbr: pd.Series | None = None              # 코스피 PBR 시계열
    foreign: pd.Series | None = None          # 외국인 순매수대금(일별)
    inst: pd.Series | None = None             # 기관 순매수대금(일별)


@dataclass
class ForecastWeights:
    """팩터별 가중치. 단기 전망이라 기술·수급 비중을 높게 둔다(합=1로 정규화)."""

    macro: float = 0.20
    korea: float = 0.15
    earnings: float = 0.15
    flows: float = 0.20
    valuation: float = 0.10
    technical: float = 0.20

    def as_dict(self) -> dict[str, float]:
        d = {
            "macro": self.macro, "korea": self.korea, "earnings": self.earnings,
            "flows": self.flows, "valuation": self.valuation, "technical": self.technical,
        }
        s = sum(d.values()) or 1.0
        return {k: v / s for k, v in d.items()}


# 팩터 표시용 메타 (대시보드 라벨/설명)
FACTOR_META = {
    "macro":     ("거시", "🌎", "미 증시·VIX·금리·환율로 본 글로벌 위험선호"),
    "korea":     ("한국경제", "🇰🇷", "원화·중국 경기·코스닥·반도체 상대강도"),
    "earnings":  ("기업실적", "🏭", "반도체 실적사이클 프록시 + 코스피 PER 방향"),
    "flows":     ("수급", "💰", "외국인·기관 순매수 방향과 강도"),
    "valuation": ("밸류에이션", "⚖️", "PER·PBR 역사적 백분위와 ERP"),
    "technical": ("기술적", "📈", "추세·RSI·MACD·모멘텀·거래량·52주 위치"),
}


def _sig(name: str, detail: str, score: float, available: bool = True) -> dict:
    return {
        "name": name, "detail": detail,
        "score": round(float(score), 1), "available": bool(available),
    }


# ---------------------------------------------------------------------------
# 팩터 계산기 — 각자 {score, signals[]} 반환
# ---------------------------------------------------------------------------
def factor_macro(b: MarketBundle) -> list[dict]:
    sigs: list[dict] = []

    # 미 증시(S&P500)가 50일선 위 = 글로벌 위험선호 ON
    sp = pd.Series(b.sp500).dropna() if b.sp500 is not None else pd.Series(dtype=float)
    if len(sp) >= 50:
        gap = float(sp.iloc[-1]) / float(sma(sp, 50).iloc[-1]) - 1.0
        sigs.append(_sig("미 S&P500 추세", f"50일선 대비 {gap*100:+.1f}%", _centered(gap, 0.05)))
    else:
        sigs.append(_sig("미 S&P500 추세", "데이터 없음", 50, available=False))

    # VIX: 낮을수록 안정(위험선호). 13→100, 20→중립, 32→0
    vix = _last(b.vix)
    if vix == vix:
        sigs.append(_sig("VIX(변동성)", f"{vix:.1f}", _lin(vix, 32, 13)))
    else:
        sigs.append(_sig("VIX(변동성)", "데이터 없음", 50, available=False))

    # 미 10년물 금리: 최근 20일 상승이면 역풍(반전 매핑)
    dy = _roc(b.us10y, 20)  # 금리의 % 변화율
    y = _last(b.us10y)
    if dy == dy:
        # 금리 20일 +10% 상승 → 역풍(0점), -10% 하락 → 순풍(100점)
        sigs.append(_sig("미 10년 국채금리", f"{y:.2f}% ({dy*100:+.0f}% / 20일)",
                         _centered(-dy, 0.10)))
    else:
        sigs.append(_sig("미 10년 국채금리", "데이터 없음", 50, available=False))

    # 달러/원: 원화 약세(환율 상승)는 외국인 유입에 단기 역풍(반전)
    dfx = _roc(b.usdkrw, 20)
    fx = _last(b.usdkrw)
    if dfx == dfx:
        sigs.append(_sig("달러/원 환율", f"{fx:,.0f} ({dfx*100:+.1f}% / 20일)",
                         _centered(-dfx, 0.03)))
    else:
        sigs.append(_sig("달러/원 환율", "데이터 없음", 50, available=False))

    return sigs


def factor_korea(b: MarketBundle) -> list[dict]:
    sigs: list[dict] = []

    # 원화 레벨: 최근 1년 범위 안 위치 — 강세(낮은 환율)일수록 자금유입 우호(반전)
    fx = pd.Series(b.usdkrw).dropna() if b.usdkrw is not None else pd.Series(dtype=float)
    if len(fx) >= 60:
        rank = _pct_rank(fx.tail(250), float(fx.iloc[-1]))  # 높을수록 원화 약세
        sigs.append(_sig("원화 강도(1년 범위)", f"환율 상위 {rank:.0f}%ile", 100 - rank))
    else:
        sigs.append(_sig("원화 강도(1년 범위)", "데이터 없음", 50, available=False))

    # 중국 증시 20일 모멘텀 (대중 수출 민감)
    dc = _roc(b.china, 20)
    if dc == dc:
        sigs.append(_sig("중국 증시 모멘텀", f"{dc*100:+.1f}% / 20일", _centered(dc, 0.06)))
    else:
        sigs.append(_sig("중국 증시 모멘텀", "데이터 없음", 50, available=False))

    # 코스닥 상대 위험선호 (코스닥이 코스피보다 강하면 위험선호 ON)
    kq = _roc(b.kosdaq, 20)
    kp = _roc(b.kospi["Close"], 20)
    if kq == kq and kp == kp:
        rel = kq - kp
        sigs.append(_sig("코스닥 상대강도", f"코스피 대비 {rel*100:+.1f}%p", _centered(rel, 0.04)))
    else:
        sigs.append(_sig("코스닥 상대강도", "데이터 없음", 50, available=False))

    # 반도체 상대강도 (삼성+하이닉스 vs 코스피, 60일) — 지수의 핵심 엔진
    ds = _roc(b.semis, 60)
    dp = _roc(b.kospi["Close"], 60)
    if ds == ds and dp == dp:
        rel = ds - dp
        sigs.append(_sig("반도체 상대강도", f"코스피 대비 {rel*100:+.1f}%p / 60일",
                         _centered(rel, 0.08)))
    else:
        sigs.append(_sig("반도체 상대강도", "데이터 없음", 50, available=False))

    return sigs


def factor_earnings(b: MarketBundle) -> list[dict]:
    sigs: list[dict] = []

    # 반도체 실적사이클 프록시: 삼성+하이닉스 60일 모멘텀(실적 기대 선반영)
    ds = _roc(b.semis, 60)
    if ds == ds:
        sigs.append(_sig("반도체 실적사이클", f"대표주 {ds*100:+.1f}% / 60일 (기대 선반영)",
                         _centered(ds, 0.15)))
    else:
        sigs.append(_sig("반도체 실적사이클", "데이터 없음", 50, available=False))

    # 코스피 이익 방향: 가격은 버티는데 PER 하락 → 이익(E) 개선(긍정)
    # (PER 은 양수만 유효 — 0/음수 미확정값이 -100% 급락으로 오인되지 않게 거른다)
    per = pd.Series(b.per).dropna() if b.per is not None else pd.Series(dtype=float)
    per = per[per > 0]
    if len(per) >= 40:
        dper = float(per.iloc[-1]) / float(per.iloc[-21]) - 1.0  # 20일 PER 변화
        price_up = _roc(b.kospi["Close"], 20)
        price_up = 0.0 if price_up != price_up else price_up
        # PER 하락(+가격 유지↑)은 이익개선 신호 → -dper 를 점수화, 가격상승분 보정
        earn_dir = -dper + 0.5 * price_up
        sigs.append(_sig("코스피 이익 방향", f"PER {dper*100:+.1f}% / 20일",
                         _centered(earn_dir, 0.08)))
    else:
        sigs.append(_sig("코스피 이익 방향", "데이터 없음", 50, available=False))

    # 지수 중기 추세(120일)도 실적 기대의 종합 반영으로 함께 본다
    dp = _roc(b.kospi["Close"], 120)
    if dp == dp:
        sigs.append(_sig("지수 중기추세(120일)", f"{dp*100:+.1f}%", _centered(dp, 0.12)))
    else:
        sigs.append(_sig("지수 중기추세(120일)", "데이터 없음", 50, available=False))

    return sigs


def factor_flows(b: MarketBundle) -> list[dict]:
    sigs: list[dict] = []

    def flow_score(series: pd.Series | None, label: str, lb: int = 20) -> dict:
        s = pd.Series(series).dropna() if series is not None else pd.Series(dtype=float)
        if len(s) < lb:
            return _sig(label, "데이터 없음", 50, available=False)
        recent = s.tail(lb)
        cum = float(recent.sum())
        # 순매수 규모를 최근 변동성(절대값 평균)으로 정규화 → 방향+강도
        scale = float(recent.abs().mean()) * lb or 1.0
        z = cum / scale
        won = cum / 1e8  # 억원 표기(대금이 원 단위라면)
        return _sig(label, f"{lb}일 누적 {won:+,.0f}억", _centered(z, 1.0))

    f = flow_score(b.foreign, "외국인 순매수(20일)")
    i = flow_score(b.inst, "기관 순매수(20일)")
    sigs.append(f)
    sigs.append(i)

    # 최근 5일 외국인 단기 방향(가속/둔화)
    if b.foreign is not None and len(pd.Series(b.foreign).dropna()) >= 20:
        s = pd.Series(b.foreign).dropna()
        short = float(s.tail(5).sum())
        scale = float(s.tail(20).abs().mean()) * 5 or 1.0
        sigs.append(_sig("외국인 단기(5일)", f"{short/1e8:+,.0f}억",
                         _centered(short / scale, 1.0)))
    else:
        sigs.append(_sig("외국인 단기(5일)", "데이터 없음", 50, available=False))

    return sigs


def factor_valuation(b: MarketBundle) -> list[dict]:
    sigs: list[dict] = []

    # PER 백분위(3년): 낮을수록 저평가(긍정) → 반전.
    # PER·PBR 은 반드시 양수여야 유효(0/음수는 미확정·결측값이므로 버린다).
    per = pd.Series(b.per).dropna() if b.per is not None else pd.Series(dtype=float)
    per = per[per > 0]
    if len(per) >= 60:
        rank = _pct_rank(per.tail(750), float(per.iloc[-1]))
        sigs.append(_sig("PER 백분위(3년)", f"{float(per.iloc[-1]):.1f}배 · 상위 {rank:.0f}%ile",
                         100 - rank))
    else:
        sigs.append(_sig("PER 백분위(3년)", "데이터 없음", 50, available=False))

    # PBR 백분위(3년): 낮을수록 저평가(긍정) → 반전
    pbr = pd.Series(b.pbr).dropna() if b.pbr is not None else pd.Series(dtype=float)
    pbr = pbr[pbr > 0]
    if len(pbr) >= 60:
        rank = _pct_rank(pbr.tail(750), float(pbr.iloc[-1]))
        sigs.append(_sig("PBR 백분위(3년)", f"{float(pbr.iloc[-1]):.2f}배 · 상위 {rank:.0f}%ile",
                         100 - rank))
    else:
        sigs.append(_sig("PBR 백분위(3년)", "데이터 없음", 50, available=False))

    # ERP(주식위험프리미엄) = 이익수익률(1/PER) - 미10년물 금리. 높을수록 매력(긍정)
    per_last = float(per.iloc[-1]) if len(per) else float("nan")
    y = _last(b.us10y)
    if per_last == per_last and per_last > 0 and y == y:
        erp = (1.0 / per_last) * 100.0 - y  # %p
        sigs.append(_sig("ERP(위험프리미엄)", f"{erp:+.1f}%p (이익수익률−금리)",
                         _lin(erp, 0.0, 6.0)))
    else:
        sigs.append(_sig("ERP(위험프리미엄)", "데이터 없음", 50, available=False))

    return sigs


def factor_technical(b: MarketBundle) -> list[dict]:
    sigs: list[dict] = []
    c = b.kospi["Close"].dropna()
    close = float(c.iloc[-1])

    # 정배열(20>60>120) + 종가의 이평선 위치
    ma20, ma60, ma120 = sma(c, 20), sma(c, 60), sma(c, 120)
    if len(c) >= 120 and pd.notna(ma120.iloc[-1]):
        order = 0.0
        order += 1 if close > ma20.iloc[-1] else -1
        order += 1 if ma20.iloc[-1] > ma60.iloc[-1] else -1
        order += 1 if ma60.iloc[-1] > ma120.iloc[-1] else -1
        detail = "정배열" if order == 3 else ("역배열" if order == -3 else "혼조")
        sigs.append(_sig("이평선 배열", detail, _lin(order, -3, 3)))
    else:
        sigs.append(_sig("이평선 배열", "데이터 없음", 50, available=False))

    # 종가 vs 60일선 이격
    if len(c) >= 60 and pd.notna(ma60.iloc[-1]):
        gap = close / float(ma60.iloc[-1]) - 1.0
        sigs.append(_sig("60일선 이격", f"{gap*100:+.1f}%", _centered(gap, 0.05)))
    else:
        sigs.append(_sig("60일선 이격", "데이터 없음", 50, available=False))

    # RSI(14): 모멘텀. 50 중립, 상승할수록 강하되 75+ 과열은 소폭 감점
    if len(c) >= 15:
        r = float(rsi(c, 14).iloc[-1])
        score = _centered(r - 50, 20)
        if r > 75:
            score = max(40.0, score - (r - 75))  # 과열 경계
        sigs.append(_sig("RSI(14)", f"{r:.0f}", score))
    else:
        sigs.append(_sig("RSI(14)", "데이터 없음", 50, available=False))

    # MACD 히스토그램 부호/추세 (12,26,9)
    if len(c) >= 35:
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = float(macd.iloc[-1] - signal.iloc[-1])
        hist_norm = hist / close  # 가격 대비 정규화
        state = "상방" if hist > 0 else "하방"
        sigs.append(_sig("MACD 히스토그램", state, _centered(hist_norm, 0.01)))
    else:
        sigs.append(_sig("MACD 히스토그램", "데이터 없음", 50, available=False))

    # 20일 모멘텀(ROC)
    dp = _roc(c, 20)
    if dp == dp:
        sigs.append(_sig("20일 모멘텀", f"{dp*100:+.1f}%", _centered(dp, 0.05)))
    else:
        sigs.append(_sig("20일 모멘텀", "데이터 없음", 50, available=False))

    # 거래량 추세(최근 5일 vs 60일 평균) — 상승 시 거래 실림이 긍정
    v = b.kospi["Volume"].dropna()
    if len(v) >= 60:
        ratio = float(v.tail(5).mean()) / float(v.tail(60).mean() or 1)
        up = dp if dp == dp else 0.0
        # 거래량 증가가 상승과 동반이면 가점, 하락과 동반이면 감점
        vscore = _centered((ratio - 1.0) * np.sign(up if up != 0 else 1), 0.5)
        sigs.append(_sig("거래량 추세", f"5일/60일 {ratio:.2f}배", vscore))
    else:
        sigs.append(_sig("거래량 추세", "데이터 없음", 50, available=False))

    # 52주(250일) 범위 내 위치
    if len(c) >= 200:
        lo, hi = float(c.tail(250).min()), float(c.tail(250).max())
        pos = (close - lo) / (hi - lo) * 100 if hi > lo else 50.0
        sigs.append(_sig("52주 범위 위치", f"{pos:.0f}%ile", pos))
    else:
        sigs.append(_sig("52주 범위 위치", "데이터 없음", 50, available=False))

    return sigs


_FACTOR_FNS = {
    "macro": factor_macro, "korea": factor_korea, "earnings": factor_earnings,
    "flows": factor_flows, "valuation": factor_valuation, "technical": factor_technical,
}


def _factor_score(sigs: list[dict]) -> float:
    """사용 가능한 신호만 평균. 전부 없으면 50."""
    vals = [s["score"] for s in sigs if s["available"]]
    return float(np.mean(vals)) if vals else 50.0


def slice_bundle(b: MarketBundle, end: pd.Timestamp) -> MarketBundle:
    """번들의 모든 시계열을 end 시점 이하로 자른 '그 날 시점' 번들(미래참조 없음)."""
    def cut(x):
        return None if x is None else x[x.index <= end]
    return MarketBundle(
        kospi=cut(b.kospi), sp500=cut(b.sp500), vix=cut(b.vix), us10y=cut(b.us10y),
        usdkrw=cut(b.usdkrw), china=cut(b.china), kosdaq=cut(b.kosdaq), semis=cut(b.semis),
        per=cut(b.per), pbr=cut(b.pbr), foreign=cut(b.foreign), inst=cut(b.inst),
    )


def factor_scores(b: MarketBundle) -> dict[str, float]:
    """최신 봉 기준 6팩터 점수(0~100)만 계산. 최적화/백테스트용 경량 경로."""
    return {k: _factor_score(fn(b)) for k, fn in _FACTOR_FNS.items()}


def composite_from(scores: dict[str, float], w: ForecastWeights | None = None) -> float:
    """팩터 점수 dict + 가중치 → 종합 점수(0~100)."""
    wd = (w or ForecastWeights()).as_dict()
    return float(sum(scores.get(k, 50.0) * wd[k] for k in wd))


FACTOR_KEYS = list(_FACTOR_FNS)


def score_panel(b: MarketBundle, min_obs: int = 150, step: int = 1,
                warmup: int = 150) -> pd.DataFrame:
    """각 날짜의 6팩터 점수 + 종가 패널(미래참조 없음). 최적화·백테스트 공용.

    step>1 이면 날짜를 건너뛰며 계산(속도↑). 각 시점 t 의 점수는 t 까지의 데이터만
    사용(slice_bundle)하므로 미래참조가 없다.
    """
    idx = b.kospi.index
    close = b.kospi["Close"]
    rows = []
    for i in range(warmup, len(idx), step):
        d = idx[i]
        sub = slice_bundle(b, d)
        if len(sub.kospi) < min_obs:
            continue
        rows.append({"date": d, "close": float(close.iloc[i]), **factor_scores(sub)})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 방향/신뢰도 매핑
# ---------------------------------------------------------------------------
def bias_label(score: float) -> str:
    if score >= 65:
        return "강세"
    if score >= 57:
        return "중립-강세"
    if score > 43:
        return "중립"
    if score > 35:
        return "중립-약세"
    return "약세"


def _confidence(score: float, factor_scores: dict[str, float]) -> str:
    """방향 강도 + 팩터 합치(같은 방향 팩터 비율)로 신뢰도 산정."""
    strength = abs(score - 50)
    side = np.sign(score - 50)
    agree = sum(1 for v in factor_scores.values() if np.sign(v - 50) == side and side != 0)
    n = len(factor_scores)
    if strength >= 12 and agree >= max(4, n - 1):
        return "높음"
    if strength >= 6 and agree >= n // 2:
        return "보통"
    return "낮음"


def _outlook_text(score: float, bias: str, factor_scores: dict[str, float]) -> str:
    """상위 기여/역풍 팩터를 짚는 한 줄 코멘트."""
    ranked = sorted(factor_scores.items(), key=lambda kv: kv[1], reverse=True)
    up = [FACTOR_META[k][0] for k, v in ranked if v >= 55][:2]
    down = [FACTOR_META[k][0] for k, v in ranked[::-1] if v <= 45][:2]
    parts: list[str] = []
    if "강세" in bias:
        parts.append("단기 코스피는 상방에 무게가 실립니다.")
    elif "약세" in bias:
        parts.append("단기 코스피는 하방 압력이 우세합니다.")
    else:
        parts.append("단기 코스피는 뚜렷한 방향성이 약한 중립 국면입니다.")
    if up:
        parts.append(f"버팀목: {', '.join(up)}.")
    if down:
        parts.append(f"역풍: {', '.join(down)}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def compute_forecast(b: MarketBundle, w: ForecastWeights | None = None) -> dict:
    """번들 → 6팩터 전망 결과 dict (대시보드/알림/JSON 저장용)."""
    w = (w or ForecastWeights()).as_dict()

    factors: list[dict] = []
    factor_scores: dict[str, float] = {}
    avail_factors = 0
    for key, fn in _FACTOR_FNS.items():
        sigs = fn(b)
        n_total = len(sigs)
        n_avail = sum(1 for s in sigs if s["available"])
        raw = _factor_score(sigs)
        # 결측 비례 축소: 신호가 일부만 있으면 그만큼만 확신을 주고 중립(50)으로 당긴다.
        # (전체 신호가 있으면 avail=1 → 변화 없음. 프록시 하나로 과한 방향성 주장 방지)
        avail = n_avail / n_total if n_total else 0.0
        fscore = 50.0 + (raw - 50.0) * avail
        if n_avail:
            avail_factors += 1
        factor_scores[key] = fscore
        label, icon, desc = FACTOR_META[key]
        factors.append({
            "key": key, "label": label, "icon": icon, "desc": desc,
            "weight": round(w[key], 3), "score": round(fscore, 1),
            "bias": bias_label(fscore), "signals": sigs,
            "available_signals": n_avail, "total_signals": n_total,
        })

    composite = float(sum(factor_scores[k] * w[k] for k in factor_scores))
    bias = bias_label(composite)
    conf = _confidence(composite, factor_scores)

    c = b.kospi["Close"].dropna()
    close = float(c.iloc[-1])
    chg = (close / float(c.iloc[-2]) - 1.0) * 100.0 if len(c) >= 2 else 0.0
    as_of = str(b.kospi.index[-1].date())

    return {
        "as_of": as_of,
        "kospi_close": round(close, 2),
        "kospi_change_pct": round(chg, 2),
        "score": round(composite, 1),
        "bias": bias,
        "confidence": conf,
        "outlook": _outlook_text(composite, bias, factor_scores),
        "data_factors_available": avail_factors,
        "data_factors_total": len(factors),
        "factors": factors,
        "disclaimer": (
            "본 전망 점수는 공개 시장데이터에 기반한 규칙 기반 참고 지표이며 "
            "매매 권유가 아닙니다. 전망은 불확실하며 언제든 빗나갈 수 있고, "
            "본 시스템에는 어떤 자동 매수/매도 기능도 없습니다. 투자 판단과 책임은 "
            "전적으로 본인에게 있습니다."
        ),
    }
