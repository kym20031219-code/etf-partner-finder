# 국내 주식 스윙(눌림목) 추천 시스템

상승 추세인 종목이 **20일 이동평균선까지 눌렸다가 반등하는 첫 자리**(눌림목)를
자동으로 스캔하고, 향후 **텔레그램 알림 + 웹 대시보드**로 실시간 추천하기 위한
프로젝트입니다.

> ⚠️ **이 시스템은 종목 "추천"만 제공하며 실제 매매는 본인의 판단과 책임으로
> 별도 계좌에서 진행해야 합니다.** 어떤 자동 매수/매도 주문 기능도, 증권사 계좌
> 접근 코드(KIS API 등)도 포함하지 않습니다. 표시되는 점수·목표가·손절가는 전략
> 규칙에 따른 참고용 예시일 뿐 매매 권유가 아니며, 실제 투자 전 반드시 백테스트로
> 검증하고 본인 책임하에 판단하세요.

## 현재 단계: 전략 + 백테스트 ✅

| 구성 | 파일 |
|------|------|
| 데이터 수집 (실데이터/합성) | `swing/data.py` |
| 눌림목 전략·신호 생성 | `swing/strategy.py` |
| 백테스트 엔진 (매매 추출·포트폴리오) | `swing/engine.py` |
| 성과 지표·리포트 | `swing/metrics.py` |
| 실행기 | `run_backtest.py` |
| 무결성 테스트 | `tests/test_engine.py` |

### 눌림목 매수 규칙 (요약)

**진입 조건** (종가 확정 후 판단 → 다음 봉 시가 체결, 미래참조 없음)
1. 추세: 종가 > 60일선, 20일선 > 60일선(정배열), 60일선 우상향
2. 눌림: 당일 저가가 20일선 +2% 이내로 근접(단, 60일선은 안 깨짐), 직전 20일 고점이 현재가보다 8%+ 높음
3. 반등: 종가가 20일선 위로 회복 + 양봉 + RSI 40~68 + 거래량 ≥ 20일 평균×0.8

**청산 우선순위**: 손절 −5% → 익절 +10% → 20일선 −3% 이탈 → 15거래일 초과
왕복 비용(수수료+세금+슬리피지) 약 0.35% 반영.

모든 수치는 `swing/strategy.py`의 `PullbackParams`에서 조정할 수 있습니다.

## 실행 방법

```bash
pip install -r requirements.txt

# 1) 실제 국내 데이터로 백테스트  (네트워크 열린 환경: 내 PC / GitHub Actions)
python run_backtest.py --source real --market KOSPI --top 100 --start 2020-01-01

# 2) 오프라인 검증  (합성 데이터, 네트워크 불필요)
python run_backtest.py --source synthetic --n 40 --days 900

# 테스트
python tests/test_engine.py
```

출력물:
- 콘솔 성과 리포트 (승률·손익비·기대값·CAGR·MDD 등)
- `trades.csv` — 개별 매매 내역
- `signals_latest.json` — 가장 최근 봉에서 신호가 뜬 **오늘의 후보** (알림/웹 연동용)

### ⚠️ 네트워크 관련 참고
이 저장소를 만든 원격(웹) 세션에서는 보안 정책상 네이버·KRX·야후 등 증시 데이터
호스트 접근이 차단되어 **실데이터 백테스트는 여기서 돌릴 수 없습니다.** 그래서
합성 데이터로 엔진을 검증했습니다. `--source real`은 **네트워크가 열린 회원님 PC나
GitHub Actions에서** 정상 동작합니다.

## 2단계: 텔레그램 알림 + 자동화 ✅

| 구성 | 파일 |
|------|------|
| 텔레그램 전송 (표준 라이브러리) | `swing/notify.py` |
| 장중 스캔 + 신규 알림 (중복 방지) | `run_scan.py` |
| 무료 스케줄러 | `.github/workflows/swing-scan.yml` |
| 상태(중복방지·최신 신호) | `state/alerted.json`, `state/signals_latest.json` |

### 텔레그램 봇 설정 (5분)

1. 텔레그램에서 **@BotFather** → `/newbot` → **봇 토큰**(`123456:ABC...`) 발급
2. 만든 봇과 대화 시작 후 **@userinfobot** 으로 본인 **chat_id**(숫자) 확인
3. 로컬 테스트:
   ```bash
   export TELEGRAM_TOKEN="봇토큰"
   export TELEGRAM_CHAT_ID="내chat_id"
   python run_scan.py --market KOSPI --top 100          # 실데이터 + 전송
   python run_scan.py --source synthetic --dry-run       # 미전송 테스트
   ```

### GitHub Actions 자동 실행 (무료)

1. 저장소 **Settings → Secrets and variables → Actions** 에 두 개 등록:
   `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`
2. 워크플로우가 **평일 장중(KST 09:00~15:30) 30분마다** 자동 스캔 → 신규 신호를 텔레그램 전송
3. 스캔 결과는 `state/` 에 커밋되어 중복 알림을 방지하고, 이후 웹 대시보드가 이를 읽습니다.

> ⚠️ **예약(schedule) 실행은 기본 브랜치(`main`)에 워크플로우가 있어야만 동작**합니다.
> 지금은 작업 브랜치에 있으므로, `main` 에 병합하면 자동 실행이 시작됩니다.
> 병합 전에는 Actions 탭의 **Run workflow(workflow_dispatch)** 로 수동 실행할 수 있습니다.

## 3단계: 웹 대시보드 ✅

`dashboard.html` — 휴대폰/PC 어디서든 보기 좋게 만든 반응형 웹 대시보드입니다.
**5단계에서 일일 스코어링 순위표 + 날짜 드롭다운** 형태로 업그레이드되었습니다
(아래 5단계 참고). `results/` 의 CSV 히스토리를 읽어 최신 스캔과 과거 스캔을
모두 볼 수 있습니다.

> 장중 실시간 신호(`run_scan.py`)는 여전히 텔레그램으로 전송되며
> `state/signals_latest.json` 에 최신 스냅샷이 남습니다. 웹 순위표는 장마감 후
> 하루 1회 집계되는 일일 스코어(5단계)를 기준으로 합니다.

### GitHub Pages 로 공개하기 (무료)

1. 저장소 **Settings → Pages** → Source: `Deploy from a branch` → Branch: `main` / `/ (root)` 저장
2. 잠시 후 `https://<사용자명>.github.io/etf-partner-finder/dashboard.html` 로 접속
3. 일일 스캔(5단계)이 `results/` 를 갱신할 때마다 대시보드에 자동 반영됩니다

> 로컬에서 미리 보려면: `python -m http.server 8000` 실행 후
> `http://localhost:8000/dashboard.html` 접속 (파일 직접 열기는 fetch 제약으로 동작 안 함).

## 4단계: 전략 튜닝 (과최적화 방지) ✅

`tune.py` — 파라미터 조합을 격자 탐색하되, 데이터를 **앞(학습)/뒤(검증)** 로 나눠
**검증 구간에서도 성적이 유지되는 강건한 조합**만 상위로 올립니다. 학습에서만 좋고
검증에서 무너지는 조합은 과최적화이므로 걸러집니다.

```bash
# 실데이터 (네트워크 열린 환경) — 기간을 길게 줄수록 신뢰도 ↑
python tune.py --source real --market KOSPI --top 100 --start 2019-01-01

# 오프라인 검증
python tune.py --source synthetic --n 60 --days 1000
```

- 탐색 격자는 `tune.py`의 `GRID` 에서 조정 (조합 수가 곱으로 늘어나니 과하게 넓히지 말 것)
- 표본이 `MIN_TRADES`(기본 20) 미만인 조합은 신뢰 불가로 제외
- **고르는 법**: 검증(test) 기대값이 (+)이고 학습·검증 성적이 크게 어긋나지 않는 조합

### 튜닝 레버 (전략에 추가된 선택 옵션)

`PullbackParams` 에 ATR 기반 청산과 트레일링 스톱을 넣어 탐색 폭을 넓힐 수 있습니다
(기본은 비활성 = 기존 고정 % 청산):

| 파라미터 | 의미 |
|----------|------|
| `use_atr_exits` | 손절/익절을 진입시점 ATR 배수로 산정 |
| `atr_stop_mult` / `atr_target_mult` | ATR 손절/익절 배수 |
| `trail_atr_mult` | >0 이면 이익 구간에서 최고종가 대비 ATR 배수 이탈 시 청산 |

## 5단계: 매일 자동 종목 스코어링 + 히스토리 대시보드 ✅

이진(매수/관심) 신호를 넘어, **상승추세 종목을 0~100점으로 줄세워** 매일 순위표를
남깁니다. 장마감 후 하루 1회 자동 실행되어 결과를 CSV 히스토리로 커밋합니다.

| 구성 | 파일 |
|------|------|
| 후보 스코어링(총점 + 하위점수) | `swing/score.py` |
| 일일 스캔 실행기 (CSV 저장 + 상위5 알림) | `run_daily_scan.py` |
| 매일 자동 실행 (평일 15:35 KST) | `.github/workflows/daily-scan.yml` |
| 결과 히스토리 | `results/scan_YYYYMMDD.csv`, `results/index.json` |
| 순위표 대시보드 (날짜 선택) | `dashboard.html` |

### 총점은 어떻게 매기나 (`ScoreParams`)

상승추세(종가>60일선·정배열·60일선 우상향)인 종목만 대상으로, 네 개 하위 점수를
가중 평균해 **총점(0~100)** 을 냅니다.

| 하위 점수 | 의미 | 기본 가중치 |
|-----------|------|:-----------:|
| **추세(trend)** | 60일선 위 여유 + 정배열 간격 + 60일선 기울기 | 0.35 |
| **눌림(pullback)** | 종가가 20일선에 근접할수록 높음(눌림 자리) | 0.30 |
| **모멘텀(momentum)** | RSI 가 반등 여력 구간(과열·과매도 아님)인지 | 0.20 |
| **거래량(volume)** | 최근 거래량이 20일 평균 대비 실린 정도 | 0.15 |

가중치·기준값은 `swing/score.py` 의 `ScoreParams` 에서 조정합니다.

### 실행 방법

```bash
# 실데이터 (네트워크 열린 환경) — KOSPI 시총 상위 100종목을 스코어링해 상위 30개 저장
python run_daily_scan.py --market KOSPI --top 100 --show 30

# 오프라인 검증 (합성 데이터, 미전송)
python run_daily_scan.py --source synthetic --n 40 --days 500 --dry-run
```

- `results/scan_YYYYMMDD.csv` — 그날의 순위표 (티커·종가·총점·하위점수·신호)
- `results/index.json` — 대시보드 드롭다운용 날짜 목록 (자동 갱신)
- `TELEGRAM_TOKEN`/`TELEGRAM_CHAT_ID` 가 있으면 **상위 5종목**을 요약 전송, 없으면 조용히 스킵

### 자동 실행 (GitHub Actions)

`.github/workflows/daily-scan.yml` 이 **평일 한국시간 15:35(KRX 장마감 직후)** 에 하루
한 번 스캔을 돌려 `results/` 에 커밋합니다. 텔레그램 Secret 은 2단계와 동일하게 등록하면
되고, 없어도 CSV 저장은 정상 동작합니다.

> ⚠️ 예약(schedule) 실행은 기본 브랜치(`main`)에 워크플로우가 있어야만 시작됩니다.
> 병합 전에는 Actions 탭의 **Run workflow(workflow_dispatch)** 로 수동 실행하세요.

### 순위표 대시보드

`dashboard.html` 은 `results/` 의 CSV 를 읽어 **최신 스캔 날짜의 상위 종목 테이블**
(티커·종가·총점·하위 점수)을 보여주고, **드롭다운으로 과거 날짜**도 선택할 수 있습니다.
GitHub Pages 배포 방법은 3단계와 동일합니다
(`https://<사용자명>.github.io/etf-partner-finder/dashboard.html`).

## 다음 단계 (예정)

- **커버리지 확대** — KOSDAQ·섹터별 스코어링, 점수 이력의 시계열 추적

> 이 프로젝트는 **추천(스코어링)** 에서 끝납니다. 실제 주문 집행·증권사 계좌 연동
> (KIS API 등)은 의도적으로 범위에서 제외했습니다. 매매는 본인 판단·책임으로
> 별도 계좌에서 진행하세요.

---
※ 저장소에 남아 있는 `etf_data.json` / `index.html`은 이전 ETF 포트폴리오
데모이며 이번 스윙 시스템과는 별개입니다.
