# 국내 주식 스윙(눌림목) 추천 시스템

상승 추세인 종목이 **20일 이동평균선까지 눌렸다가 반등하는 첫 자리**(눌림목)를
자동으로 스캔하고, 향후 **텔레그램 알림 + 웹 대시보드**로 실시간 추천하기 위한
프로젝트입니다.

> ⚠️ 투자 참고용 신호일 뿐 매매 권유가 아닙니다. 실제 투자 전 반드시 백테스트로
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

`dashboard.html` — `state/signals_latest.json` 을 읽어 현재 추천 종목을 카드로 표시합니다.
휴대폰/PC 어디서든 보기 좋게 반응형으로 만들었고, **3분마다 자동 새로고침**됩니다.

### GitHub Pages 로 공개하기 (무료)

1. 저장소 **Settings → Pages** → Source: `Deploy from a branch` → Branch: `main` / `/ (root)` 저장
2. 잠시 후 `https://<사용자명>.github.io/etf-partner-finder/dashboard.html` 로 접속
3. 스캔(2단계)이 `state/` 를 갱신할 때마다 대시보드에 자동 반영됩니다

> 로컬에서 미리 보려면: `python -m http.server 8000` 실행 후
> `http://localhost:8000/dashboard.html` 접속 (파일 직접 열기는 fetch 제약으로 동작 안 함).

## 다음 단계 (예정)

- **실시간 강화** — 한국투자증권 KIS API 연동(장중 실시간 시세)

---
※ 저장소에 남아 있는 `etf_data.json` / `index.html`은 이전 ETF 포트폴리오
데모이며 이번 스윙 시스템과는 별개입니다.
