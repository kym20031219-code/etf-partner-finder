"""데이터 수집 계층.

- 네트워크가 열린 환경(회원님 PC, GitHub Actions 등): FinanceDataReader 로 실제 국내
  주식 일봉을 받아옵니다.
- 네트워크가 막힌 환경(이 원격 샌드박스 등): 상승추세 + 눌림목이 반복되는 현실적인
  합성 OHLCV 를 생성해 엔진 검증에 사용합니다.

두 경우 모두 동일한 컬럼 규격을 돌려줍니다: Open, High, Low, Close, Volume (DatetimeIndex).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 실제 데이터 (네트워크 필요)
# ---------------------------------------------------------------------------
def fetch_ohlcv(code: str, start: str, end: str) -> pd.DataFrame:
    """FinanceDataReader 로 종목 일봉을 받아옵니다.

    code : 6자리 종목코드 (예: '005930' 삼성전자)
    """
    import FinanceDataReader as fdr  # 지연 임포트 (오프라인 환경 보호)

    df = fdr.DataReader(code, start, end)
    df = df.rename(columns=str.capitalize)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[cols].dropna()
    df = df[df["Volume"] > 0]
    return df


def fetch_universe(market: str = "KOSPI", top_n: int = 100) -> list[str]:
    """시가총액 상위 종목코드 목록 (네트워크 필요)."""
    import FinanceDataReader as fdr

    listing = fdr.StockListing(market)
    # 컬럼명이 버전마다 다를 수 있어 방어적으로 처리
    cap_col = next((c for c in ("Marcap", "MarketCap", "Market_Cap") if c in listing.columns), None)
    code_col = next((c for c in ("Code", "Symbol") if c in listing.columns), "Code")
    if cap_col:
        listing = listing.sort_values(cap_col, ascending=False)
    return listing[code_col].astype(str).str.zfill(6).head(top_n).tolist()


def fetch_delisted(market: str = "KOSPI", start: str = "2018-01-01",
                   limit: int = 200) -> list[str]:
    """상장폐지 종목코드 목록 (백테스트 **생존편향 완화**용).

    현재 상장 종목만으로 백테스트하면 그동안 폐지된 '패자'가 빠져 성과가 부풀려진다.
    `start` 이후에 폐지된 종목(=백테스트 기간 중 거래됐던 종목)을 되살려 유니버스에
    합친다.

    FDR 버전/네트워크에 따라 폐지 목록 조회가 실패할 수 있으므로 **방어적으로** 처리하고,
    실패하면 빈 리스트를 돌려준다(그 경우 생존편향 주의가 그대로 남는다).
    """
    try:
        import FinanceDataReader as fdr
    except Exception:  # noqa: BLE001
        return []

    lst = None
    for key in ("KRX-DELISTING", "KRX-Delisting", "KRX_DELISTING"):
        try:
            lst = fdr.StockListing(key)
            if lst is not None and len(lst):
                break
        except Exception:  # noqa: BLE001
            lst = None
    if lst is None or len(lst) == 0:
        return []

    try:
        df = lst.copy()
        low = {c.lower(): c for c in df.columns}
        code_col = next((low[c] for c in ("symbol", "code", "isucode") if c in low), None)
        if code_col is None:
            return []
        df["_code"] = df[code_col].astype(str).str.extract(r"(\d{6})")[0]
        df = df.dropna(subset=["_code"])

        # 폐지일 컬럼(있으면) 기준으로 start 이후만
        date_col = next((c for c in df.columns if "delist" in c.lower()), None)
        if date_col is None:
            date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col is not None:
            dd = pd.to_datetime(df[date_col], errors="coerce")
            df = df[dd >= pd.to_datetime(start)]

        # 시장 필터(컬럼 있으면)
        mkt_col = low.get("market")
        if mkt_col is not None:
            df = df[df[mkt_col].astype(str).str.upper().str.contains(market.upper(), na=False)]

        return df["_code"].dropna().unique().tolist()[:limit]
    except Exception:  # noqa: BLE001
        return []


def fetch_index(code: str = "KS11", start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """지수 일봉 (기본 KS11 = KOSPI 종합지수). 네트워크 필요."""
    import FinanceDataReader as fdr

    df = fdr.DataReader(code, start, end)
    df = df.rename(columns=str.capitalize)
    return df[["Close"]].dropna()


def fetch_names(market: str = "KOSPI") -> dict[str, str]:
    """종목코드 → 종목명 매핑 (네트워크 필요)."""
    import FinanceDataReader as fdr

    listing = fdr.StockListing(market)
    code_col = next((c for c in ("Code", "Symbol") if c in listing.columns), "Code")
    name_col = next((c for c in ("Name", "Korean Name") if c in listing.columns), "Name")
    codes = listing[code_col].astype(str).str.zfill(6)
    return dict(zip(codes, listing[name_col].astype(str)))


# ---------------------------------------------------------------------------
# 합성 데이터 (오프라인 검증용)
# ---------------------------------------------------------------------------
def make_synthetic(
    code: str,
    days: int = 750,
    start: str = "2022-01-03",
    seed: int | None = None,
) -> pd.DataFrame:
    """상승 국면과 눌림목(조정)이 번갈아 나오는 현실적인 일봉을 생성.

    레짐 스위칭: 강한 상승 → 눌림(조정) → 상승 ... 을 반복하도록 드리프트를 바꿔가며
    누적. 눌림목 전략이 실제로 매매를 잡을 수 있는 시계열을 만든다.
    """
    rng = np.random.default_rng(seed if seed is not None else abs(hash(code)) % (2**32))

    price = 10000 * (1 + rng.uniform(-0.3, 3.0))  # 종목마다 다른 시작가
    closes = []
    # 레짐: (일평균수익률, 일변동성, 지속일수 범위)
    regimes = [
        (0.0022, 0.018, (25, 55)),   # 상승 추세
        (-0.0016, 0.020, (8, 20)),   # 눌림/조정
        (0.0002, 0.014, (10, 25)),   # 횡보
        (0.0035, 0.024, (15, 35)),   # 강한 상승
        (-0.0040, 0.030, (6, 14)),   # 급락
    ]
    # 상승 국면 비중을 높게 두어 전반적 우상향
    weights = np.array([0.34, 0.22, 0.16, 0.20, 0.08])

    while len(closes) < days:
        mu, sig, (lo, hi) = regimes[rng.choice(len(regimes), p=weights)]
        n = int(rng.integers(lo, hi))
        rets = rng.normal(mu, sig, n)
        for r in rets:
            price *= (1 + r)
            price = max(price, 500)
            closes.append(price)
    closes = np.array(closes[:days])

    # 종가로부터 시/고/저 재구성
    idx = pd.bdate_range(start=start, periods=days)
    prev = np.concatenate([[closes[0]], closes[:-1]])
    opens = prev * (1 + rng.normal(0, 0.004, days))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.008, days)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.008, days)))
    base_vol = rng.integers(200_000, 3_000_000)
    # 변동성이 큰 날 거래량 증가
    move = np.abs(np.concatenate([[0], np.diff(closes) / closes[:-1]]))
    vol = (base_vol * (1 + 6 * move) * rng.uniform(0.6, 1.4, days)).astype(int)

    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vol},
        index=idx,
    )


def synthetic_universe(n: int = 20, **kw) -> dict[str, pd.DataFrame]:
    """검증용 합성 종목 묶음."""
    return {f"SYN{i:03d}": make_synthetic(f"SYN{i:03d}", seed=i, **kw) for i in range(n)}
