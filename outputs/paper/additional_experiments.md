# 추가 실험 기획: 문헌 조사 기반 Gap 분석

> 문헌 조사 결과를 바탕으로, 논문의 완성도를 높이기 위한 추가 실험을 **우선순위 순**으로 정리합니다.

---

## 🔴 Priority 1: 논문 완성에 반드시 필요한 실험

### 1.1 Seasonality 자동 레이블링 개선 (Real-World)

**문제**: 현재 FFT 기반 auto-labeling으로 실세계 seasonality R²가 전 모델에서 음수 (특히 Traffic: -6.8e14). 이는 프로빙 자체의 실패가 아니라 **레이블 품질 문제**일 가능성이 높음.

**실험 계획**:
1. FFT 대신 **ACF (Autocorrelation Function)** 기반 주기 추정으로 교체
2. **STL decomposition** (statsmodels) 으로 seasonal component 강도를 레이블로 사용
3. 주기 존재 여부를 **이진 분류**로 변환 (regression → classification) — R² 문제 회피

**구현**:
- `src/datasets/real_world.py`의 `_compute_seasonality()` 수정
- 영향 범위: 실세계 데이터 5종 × 4 모델 재평가 (기존 representations 재사용 가능)

**예상 소요**: 구현 2시간 + 실행 1시간

**논문 기여**: Table 3의 seasonality 행이 모두 음수인 현재 상태는 리뷰어에게 치명적 약점. 개선하면 실세계 결과의 신뢰도가 크게 향상.

---

### 1.2 PCA 없는 MOMENT/GPT4TS Full-Dimensional 프로빙 (Subset)

**문제**: MOMENT(65536D)과 GPT4TS(24576D)에 PCA512를 적용했는데, seasonality 정보가 파괴됨. "PCA가 공정하지 않다"는 리뷰어 코멘트가 예상됨.

**실험 계획**:
1. MOMENT/GPT4TS의 **원본 표현**으로 최소 2개 속성(seasonality, anomaly) 프로빙
2. 샘플 수를 줄이거나(1000개) Ridge regression 사용하여 고차원 프로빙 안정화
3. PCA512 vs Full-D 비교표 작성

**구현**:
- `train_probe.py`에 `--regularization ridge --alpha 1.0` 옵션 추가
- 또는 scikit-learn `RidgeClassifier`/`Ridge` 직접 사용 (빠름)

**예상 소요**: 구현 1시간 + 실행 2시간 (고차원이라 느림)

**논문 기여**: PCA 정보 손실 정도를 정량화. "우리의 PCA 결과는 하한(lower bound)이며, full-D에서는 더 좋다"는 주장 뒷받침.

---

### 1.3 시각화 보강: LEACE/LDA 결과 Figure

**문제**: 현재 LEACE/LDA 결과는 테이블만 존재. 논문에는 시각적 figure가 필수.

**실험 계획**:
1. **LEACE Before/After Bar Chart**: 4 모델 × 6 속성, 그룹화된 막대 그래프
2. **LDA Steering Alpha Sweep Curve**: x=alpha, y=accuracy. GPT4TS와 MOMENT의 대표적 속성 3개
3. **Cross-model CKA 통합 Figure**: 6개 쌍을 2×3 subplot으로 배치

**구현**:
- `scripts/plot_leace_results.py` 신규 작성
- `scripts/plot_steering_curves.py` 신규 작성
- matplotlib/seaborn 사용

**예상 소요**: 3시간

**논문 기여**: Fig 3, Fig 4 생성. 논문의 시각적 완성도 대폭 향상.

---

## 🟡 Priority 2: 논문 강화에 크게 도움되는 실험

### 2.1 Cross-Property Interaction (LEACE)

**문제**: Wiliński et al.이 다루지 않는 독창적 실험. "Trend를 지우면 Stationarity 프로빙도 영향을 받는가?" — 속성 간 의존성 분석.

**실험 계획**:
1. 속성 A를 LEACE로 지운 후, 속성 B의 프로빙 정확도 변화 측정
2. 6×6 interaction matrix 생성 (대각선 = 기존 LEACE, 비대각선 = cross-property)
3. 대표 모델 1-2개로 수행 (MOMENT, GPT4TS)

**구현**:
- `scripts/run_leace_erasure.py` 수정: `--target_property`(지울 것) + `--eval_property`(평가할 것) 분리
- 기존 representations + probes 재사용

**예상 소요**: 구현 2시간 + 실행 3시간

**논문 기여**: 완전히 새로운 분석. "어떤 시간적 속성이 다른 속성과 표현을 공유하는가"에 대한 답변. 매우 독창적.

---

### 2.2 Layer-wise LEACE 분석

**문제**: 현재 LEACE는 best layer에서만 수행. 레이어별로 LEACE 효과가 어떻게 변하는지 모름.

**실험 계획**:
1. 모든 레이어에서 LEACE 적용 + 재프로빙
2. Layer progression plot with LEACE (기존 plot에 overlaid)
3. "어느 레이어에서 개념이 가장 강하게 linearly 인코딩되는가"의 정밀 측정

**구현**:
- `run_leace_erasure.py`를 반복문으로 모든 레이어 순회
- MOMENT (24L), GPT4TS (12L) 대상

**예상 소요**: 구현 1시간 + 실행 4시간

**논문 기여**: 기존 layer progression + LEACE = 더 풍부한 스토리. "Layer 5에서는 프로빙 100%이지만 LEACE drop은 0% → 정보는 있지만 linearly 제거 불가"

---

### 2.3 Hard Synthetic 확장: 나머지 속성

**문제**: Hard variant가 trend_hard, frequency_hard, anomaly_hard만 존재. Stationarity와 change_point도 hard version이 있으면 더 완전한 분석.

**실험 계획**:
1. `stationarity_hard`: 미세한 분산 변화 (현재는 명확한 drift)
2. `change_point_hard`: 미세한 분포 변화 (mean shift 줄이기)
3. 기존 파이프라인으로 추출 → 프로빙 → 평가

**구현**:
- `src/datasets/synthetic.py`에 2개 생성기 추가
- `scripts/extract_representations.py`에 등록

**예상 소요**: 구현 2시간 + 실행 3시간

**논문 기여**: Table 8 완성. 현재 "—" 셀들을 채움.

---

## 🟢 Priority 3: 있으면 좋지만 필수는 아닌 실험

### 3.1 Attention Pattern 분석

**문제**: Kalnāre et al. (2025)이 attention saliency를 분석. 우리도 추가하면 mechanistic 관점 보완.

**실험 계획**:
1. 각 모델의 attention weights 추출 (forward hook)
2. 시간적 속성별 attention 분포 시각화 (어느 패치에 attention이 집중되는가)
3. 추세 있는 신호 vs 없는 신호의 attention 차이

**예상 소요**: 5시간

**논문 기여**: Probing (what) + Attention (where) + Intervention (why) 삼위일체. 다만 scope가 넓어질 수 있음.

---

### 3.2 Structural Probe (Distance Probe)

**문제**: Hewitt & Manning (2019)의 structural probe는 NLP에서 syntax tree를 복원. 시계열에서의 아날로그는?

**실험 계획**:
1. 시계열의 "구조": 시간적 거리 (temporal distance matrix)
2. 표현 공간의 거리가 시간적 거리를 보존하는지 테스트
3. 주기적 신호에서 한 주기 내 위치가 표현 거리로 복원 가능한지

**예상 소요**: 4시간

**논문 기여**: 새로운 방법론. 다만 해석이 까다롭고 "시계열의 구조"를 정의하기 어려움.

---

### 3.3 추가 모델: TimesBERT / Timer

**문제**: 최근 TimesBERT (Zhang et al., 2025)가 BERT-style TS 모델 제안. Timer (Liu et al., 2024)도 유사.

**실험 계획**:
1. TimesBERT wrapper 구현 + 전체 파이프라인 실행
2. 5번째 pre-trained 모델로 추가

**예상 소요**: 8시간 (API 조사 + wrapper + 전체 파이프라인)

**논문 기여**: 모델 커버리지 확대. 다만 4개 모델이면 충분할 수 있음.

---

### 3.4 Fine-tuning 전후 비교

**문제**: 현재는 frozen 표현만 분석. Fine-tuning 후 표현이 어떻게 변하는지는 별도 연구 질문.

**실험 계획**:
1. MOMENT/PatchTST를 ETTh1 forecasting으로 fine-tuning
2. Fine-tuned 표현 추출 → 동일 프로빙
3. "Fine-tuning이 시간적 속성 인코딩을 강화하는가 약화하는가?"

**예상 소요**: 6시간

**논문 기여**: 중요한 질문이지만 scope를 크게 넓힘. 후속 연구로 남겨도 됨.

---

## 실행 계획 요약

| Priority | 실험 | 예상 소요 | 논문 영향 |
|----------|------|-----------|-----------|
| 🔴 1.1 | Seasonality 레이블링 개선 | 3h | Table 3 치명적 약점 해결 |
| 🔴 1.2 | Full-D 프로빙 (MOMENT/GPT4TS) | 3h | PCA 공정성 반론 차단 |
| 🔴 1.3 | LEACE/LDA 시각화 Figure | 3h | Fig 3, 4 생성 |
| 🟡 2.1 | Cross-Property LEACE | 5h | 독창적 실험, 차별화 |
| 🟡 2.2 | Layer-wise LEACE | 5h | 분석 깊이 ↑ |
| 🟡 2.3 | Hard Synthetic 확장 | 5h | Table 8 완성 |
| 🟢 3.1 | Attention 분석 | 5h | Mechanistic 관점 보완 |
| 🟢 3.2 | Structural Probe | 4h | 방법론적 참신성 |
| 🟢 3.3 | TimesBERT 추가 | 8h | 모델 커버리지 |
| 🟢 3.4 | Fine-tuning 비교 | 6h | Scope 확장 |

### 추천 실행 순서
1. **즉시**: 1.1 + 1.3 (병렬 실행 가능)
2. **다음**: 1.2 + 2.1 (병렬 실행 가능)
3. **시간 여유 시**: 2.2, 2.3
4. **후속 연구**: 3.x 시리즈

### 총 예상 소요 시간
- Priority 1 (필수): ~9시간
- Priority 2 (강화): ~15시간
- Priority 3 (선택): ~23시간
