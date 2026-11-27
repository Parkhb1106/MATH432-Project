# 특징 추출 (A 파트)

박동 간격 수열 한 구간(ECG의 RR 또는 PPG의 PPI, 초 단위)에서 특징을 계산하는 모듈의 설계를 기록한다. 데이터 준비는 [DESIGN.md](../DESIGN.md)를 본다.

## 1. 구성

| 경로 | 내용 |
| --- | --- |
| `src/features/scaling.py` | DFA α₁ |
| `src/features/entropy.py` | Sample entropy, COSEn, (multiscale) permutation entropy |
| `src/features/topology.py` | Takens 임베딩, persistent homology, 요약 통계, persistence image, landscape, diagram 캐시 |
| `src/features/hrv.py` | 기준선 HRV 통계 |
| `src/features/__init__.py` | `scalar_features(rr)`, 특징 계열별 열 이름 (`family_columns`) |
| `scripts/build_features.py` | 구간 데이터셋 전체의 특징 CSV와 diagram 캐시 생성 |
| `tests/test_features.py` | 합성 신호로 검증 (19개) |

**인터페이스**: 모든 함수는 1차원 간격 배열 하나(초 단위)를 받아 `{특징 이름: 값}` 딕셔너리를 돌려준다. 그래서 AFDB, Zenodo ECG, Zenodo PPG에 같은 코드를 쓴다.

| 계열 | 열 이름 접두사 | 개수 |
| --- | --- | --- |
| 척도 법칙 (i) | `dfa_` | 1 |
| 엔트로피 (ii) | `ent_` | 5 (+ 진단용 `_undefined` 2개) |
| 위상 (iii) | `top_` | 요약 9개 + fold별로 맞추는 PI, landscape |
| HRV 기준선 (v) | `hrv_` | 9 |

`_undefined`로 끝나는 열은 진단용이며 모델 입력에서 뺀다 (`family_columns`가 자동으로 제외).

## 2. 특징 정의와 파라미터

### 2.1 척도 법칙: DFA α₁

1. 누적합 프로파일 y(k) = Σ (xᵢ − x̄)
2. 상자 크기 n마다 앞에서부터, 그리고 뒤에서부터 겹치지 않는 상자로 나누고(총 2⌊N/n⌋개), 상자마다 직선을 맞춘 잔차의 RMS를 F(n)으로 한다.
3. α = log F(n)과 log n의 기울기

- **상자 크기 4–16박동** (α₁). α₂(16–64박동 이상)는 60박동 구간으로는 계산할 수 없다.
- 백색잡음 0.5, 1/f 잡음 1.0, 브라운 운동 1.5. 정상 동율동은 1 근처, AF는 0.5 쪽으로 간다.

### 2.2 엔트로피

| 특징 | 파라미터 | 비고 |
| --- | --- | --- |
| `ent_sampen` | m = 2, r = 0.2 × 구간 표준편차 | 길이 m과 m+1에서 같은 N−m개 템플릿 사용, Chebyshev 거리 |
| `ent_cosen` | m = 1, r = 30 ms 고정 | COSEn = SampEn + ln(2r) − ln(평균 RR). 짧은 RR 수열의 AF 검출용으로 제안된 지표 (Lake & Moorman, 2011) |
| `ent_mpe_s1`–`s3` | 순서 n = 3, 척도 1–3 | 척도 s는 s개씩 평균(coarse-graining). 척도 3에서 20개 값, 패턴 18개 |

- **SampEn이 정의되지 않는 경우** (일치하는 쌍 A = 0): 이 길이의 수열이 보일 수 있는 최댓값 ln(C(N−m, 2))을 돌려주고 `ent_sampen_undefined = 1`로 표시한다. 정상 동율동처럼 변동이 작은 구간에서 생길 수 있다.
- **동점 처리**: AFDB RR은 4 ms 단위라 같은 값이 연속으로 나올 수 있다. Permutation entropy는 위치 순서로 동점을 깬다(stable sort). 그래서 아주 규칙적인 리듬에서는 PE가 조금 낮게 나온다.

### 2.3 위상

1. **Takens 임베딩**: xᵢ = (sᵢ, sᵢ₊τ, …, sᵢ₊₍d−1₎τ). 기본값 **d = 3, τ = 1**이다. 60박동이면 58개 점이 된다. d = 2, τ = 1은 Poincaré plot과 같다.
2. **크기 정규화** (`normalize`)
   - `none` (기본): 초 단위 그대로 쓴다. H0 수명이 점들 사이의 간격, 즉 RMSSD나 SD1과 같은 "퍼짐 정도"를 잰다.
   - `zscore`: 구간마다 표준화해서 모양만 본다. 퍼짐 정도는 HRV와 엔트로피 특징에 맡긴다.
3. **Persistent homology**: ripser로 Vietoris–Rips filtration의 H0, H1을 계산한다. 끝나지 않는 H0 막대 하나는 뺀다.
4. **요약 통계** (`top_*`): H0의 총·최대·평균 수명, persistence entropy / H1의 개수, 총·최대 수명, persistence entropy / H1 최대 수명 ÷ H0 최대 수명
5. **벡터화** (fold의 학습 데이터로만 맞춤)
   - `PersistenceImage`: (birth, persistence) 좌표, 기본 10×10 격자, 가우시안 폭 1픽셀, 수명에 비례하는 선형 가중치. 격자 범위는 학습 diagram의 99% 분위수로 정한다. Rips의 H0는 birth가 모두 0이라 1×10으로 자동으로 줄어든다.
   - `PersistenceLandscape`: λ₁–λ₃을 학습 데이터 범위의 20개 점에서 샘플링한다 (60차원).

### 2.4 HRV 기준선

평균 RR, SDNN, RMSSD, pNN50, CV(= SDNN/평균), nRMSSD(= RMSSD/평균), SD1, SD2, SD1/SD2

- **SD1과 SD2를 넣은 이유**: H0 수명(`normalize=none`)은 Poincaré plot의 퍼짐 정도와 같은 양을 잰다. 기준선에 SD1을 넣어 두어야 "위상 특징이 HRV를 넘어서는 정보를 주는가"를 공정하게 비교할 수 있다.

## 3. 실행

```bash
python scripts/build_features.py                              # 기본: d=3, tau=1, normalize=none
python scripts/build_features.py --d 2 --normalize zscore     # 다른 임베딩 설정
```

| 결과물 | 내용 |
| --- | --- |
| `data/processed/features_afdb_{tag}.csv` | 구간마다 한 행: `record`, `label_50`, `label_excl`, `af_frac`와 스칼라 특징 |
| `data/processed/diagrams_afdb_{tag}.npz` | H0, H1 diagram 캐시 (이어 붙인 점 배열과 offsets). PI와 landscape를 fold별로 맞출 때 쓴다 |

`tag`는 `d{d}_tau{tau}_{normalize}`이다 (예: `d3_tau1_none`). 위상 열만 `tag`에 따라 달라지고 나머지 특징은 같다.

계산 시간은 구간당 약 2 ms다 (AFDB 18,925개 구간이면 1분 이내).

## 4. 검증 (`tests/test_features.py`)

| 대상 | 확인 내용 |
| --- | --- |
| HRV | 손으로 계산한 값과 일치 |
| DFA | 백색잡음 α ≈ 0.5, 브라운 운동 α ≈ 1.5, 합성 동율동 > 1 > 합성 AF |
| SampEn | 이중 for문으로 센 A, B와 일치, 규칙적 신호 < 무작위 신호, 정의되지 않을 때 상한값 반환 |
| COSEn | 합성 AF > 합성 동율동 |
| PE | 단조 수열 0, 긴 백색잡음 ≈ 1 |
| Takens | d = 2이면 Poincaré 쌍과 같음, 점 개수 |
| PH | 원 위의 점 → 긴 H1 하나, 가우시안 덩어리 → 긴 H1 없음, H0 막대 n−1개 |
| 정규화 | `zscore`이면 크기와 위치를 바꿔도 diagram이 같음 |
| PI, landscape | 출력 크기, H0 격자 축소, 빈 diagram은 0, 점 하나의 landscape 최댓값 = (d−b)/2 |

## 5. 파일럿에서 정할 것

| 항목 | 후보 | 판단 기준 |
| --- | --- | --- |
| 임베딩 차원 | d = 2 (Poincaré), 3 | H1이 의미 있게 생기는지, AF와 정상의 분리 정도 |
| 크기 정규화 | `none`, `zscore` | HRV만 쓴 모델에 위상 특징을 더했을 때 성능이 오르는지 |
| SampEn | m = 2, r = 0.2σ | 정의되지 않는 구간 비율. 높으면 m = 1이나 r을 키움 |
| MPE | 척도 1–3, n = 3 | 척도별 AF와 정상의 분포 차이 |

## 6. 파일럿 결과 (AFDB, 2026-11-27)

`python scripts/pilot_features.py`로 실행한다. 결과는 `results/pilot_afdb/`(표)와 `results/figures/pilot_*.png`(그림)에 있다. 18,925개 구간(AF 40.8%), 팀 공용 fold(`results/folds_afdb.json`)를 썼고, 라벨은 `label_50`이다.

### 6.1 특징 하나씩 본 AUROC

| 특징 | AUROC | 해석 |
| --- | --- | --- |
| `ent_cosen` | **0.995** | 특징 하나만으로 거의 완벽하게 구분한다. r을 30 ms로 고정하고 평균 RR 항이 들어가서 크기 정보까지 담는다 |
| `top_h1_total` (d=3, 초 단위) | 0.974 | |
| `hrv_pnn50` | 0.956 | |
| `top_h0_total` (d=3, 초 단위) | 0.952 | |
| `ent_mpe_s1` | 0.788 | |
| `ent_sampen` | 0.728 | r = 0.2σ라 크기 정보가 빠져 있어 단독으로는 약하다 |
| `dfa_alpha1` | 0.551 | 아래 6.3 참고 |

### 6.2 특징 계열별 교차검증 (5-fold, 평균 AUROC)

| 계열 | LR | RF |
| --- | --- | --- |
| HRV 기준선 (9개) | 0.987 | 0.990 |
| DFA α₁ (1개) | 0.369 | 0.753 |
| 엔트로피 (5개) | 0.996 | 0.996 |
| DFA + 엔트로피 | 0.996 | 0.997 |
| 위상 요약, d=3, 초 단위 (9개) | 0.985 | 0.993 |
| 위상 요약, d=3, zscore | 0.916 | 0.910 |
| 위상 요약, d=2, zscore | 0.975 | 0.968 |
| PI (H0+H1), d=3, 초 단위 (110개) | 0.994 | 0.994 |
| DFA + 엔트로피 + 위상 (d=3, 초 단위) | 0.997 | 0.997 |
| HRV + 위상 (d=3, 초 단위) | 0.995 | 0.997 |

전체 결과(F1, 민감도, 특이도, 임베딩 4가지)는 `results/pilot_afdb/family_cv.csv`에 있다.

### 6.3 발견한 것

1. **AFDB의 AF 검출은 거의 천장에 닿아 있다.** HRV 기준선도 AUROC 0.99다. 그래서 1단계에서 특징 계열 사이의 차이는 작게 나올 것이다. 연구의 차별점은 2단계(전이)와 3단계(움직임 강건성)에서 나와야 한다.
2. **간격 수열의 위상 특징은 대부분 "크기" 정보다.** 초 단위 그대로 임베딩하면 AF의 점구름이 넓게 퍼져 H0 수명이 길어진다. H1 점은 대각선 바로 옆에 있어 수명이 짧은 잡음 고리다 (`pilot_example_segments.png`). 크기를 지운 zscore 버전은 성능이 0.91–0.97로 떨어진다. 즉 계획 단계에서 우려한 대로, 간격 PH는 RMSSD나 SD1과 비슷한 정보를 많이 담는다. **위상 특징의 고유한 기여는 3단계의 PPG 파형 임베딩에서 보여야 한다.**
3. **그래도 모양 정보가 조금은 있다.** zscore 위상 특징만으로 AUROC 0.97(d=2)이 나오고, HRV에 위상을 더하면 RF 기준 0.990에서 0.997로 오른다. 효과가 작으므로 1단계 본 실험에서 fold별 대응 검정으로 유의한지 확인해야 한다.
4. **DFA α₁은 단조적인 특징이 아니다.** AF 구간은 환자와 상관없이 α₁ ≈ 0.6(무작위에 가까움)으로 일정하다. 그런데 non-AF 구간은 환자마다 0.2–1.2로 크게 다르다. AFDB 환자는 AF 사이에도 조기수축이 잦은 심장질환자라서, 건강한 심장의 1/f 법칙(α ≈ 1)이 non-AF 구간에서도 성립하지 않는다. 그래서 선형 모델(LR)은 실패하고(0.37), 비선형인 RF만 일부 구분한다(0.75). 리포트에서 "척도 법칙이 AF에서 무너진다"를 다룰 때 이 점을 함께 써야 한다.
5. **SampEn이 정의되지 않는 구간은 6.2%다** (non-AF 6.9%, AF 5.1%). 상한값으로 대체하면 된다. COSEn은 정의되지 않는 경우가 없다.

### 6.4 결정 (제안)

| 항목 | 결정 | 근거 |
| --- | --- | --- |
| 위상 기본 설정 | **d = 3, τ = 1, 초 단위** | 위상 요약 성능이 가장 좋다 |
| zscore 버전 | 보조 분석으로 유지 | "크기를 뺀 모양 정보"가 HRV를 보완하는지 보이는 데 쓴다 |
| SampEn | m = 2, r = 0.2σ 유지 | 정의되지 않는 구간이 6%로 적다 |
| DFA | α₁ 유지, 비단조성을 리포트에 서술 | 선형 모델에서 쓸 때는 주의 |
| 1단계 평가 | AUROC 외에 F1, 민감도, 특이도와 fold별 대응 검정을 같이 보고 | AUROC가 모두 0.99 근처라 차이가 잘 안 보인다 |

## 7. 환경 메모

- `ripser==0.6.15`를 `requirements.txt`에 추가했다.
- Windows의 스마트 앱 컨트롤이 켜져 있으면 scipy 1.18.1과 scikit-learn 1.9.1(및 1.8.0)의 DLL이 차단된다. 그런 PC에서는 scipy 1.16.3과 scikit-learn 1.7.2로 설치하면 동작한다 (2026-11-27 확인, 기존 테스트 16개와 새 테스트 19개 통과).
- **주의**: scikit-learn 1.7.2에서 `build_afdb.py`를 실행하면 `StratifiedGroupKFold`의 배정이 달라져 `results/folds_afdb.json`이 바뀐다. 이 파일은 팀 공용이므로 커밋된 버전으로 되돌려서(`git checkout -- results/folds_afdb.json`) 쓴다. 구간 데이터(`T_afdb.npz`)는 버전과 상관없이 같다.
